"""Tests for clipper.ai.transcribe_openai (OpenAITranscriber) and clipper.ai.factory.

No real OpenAI calls: a fake client captures the SDK call, or a real ``openai.OpenAI`` client
talks to an in-process ``httpx2.MockTransport``. Responses are the SDK's own
``TranscriptionVerbose`` models built from a realistic verbose_json payload.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import httpx2
import openai
import pytest
from openai.types.audio import TranscriptionVerbose
from pydantic import SecretStr

from clipper.ai import openai_client as oc
from clipper.ai import transcribe_openai as to
from clipper.ai.factory import get_selector, get_transcriber
from clipper.ai.interfaces import ClipSelector, RawWord, Transcriber
from clipper.ai.transcribe_openai import MAX_UPLOAD_BYTES, OpenAITranscriber, sanitize_raw
from clipper.ai.transcript import merge_chunks, validate_transcript
from clipper.config import Settings
from clipper.errors import CancelledError, MediaError, ProviderAuthError, ProviderError, ProviderNotConfiguredError
from clipper.models import TranscriptChunk
from clipper.procs import CancelToken, run
from conftest import requires_ffmpeg

FAKE_KEY = "sk-proj-TESTKEYabcdefghijklmnop0123456789"
REQUEST = httpx2.Request("POST", "https://api.openai.com/v1/audio/transcriptions")

VERBOSE: dict[str, Any] = {
    "task": "transcribe",
    "language": "english",
    "duration": 4.25,
    "text": "Hello there, friend. How are you?",
    "segments": [
        {"id": 0, "seek": 0, "start": 0.0, "end": 1.9, "text": " Hello there, friend.", "tokens": [50364, 2425, 456],
         "temperature": 0.0, "avg_logprob": -0.21, "compression_ratio": 0.9, "no_speech_prob": 0.012},
        {"id": 1, "seek": 0, "start": 2.4, "end": 4.1, "text": " How are you?", "tokens": [1012, 366, 291],
         "temperature": 0.0, "avg_logprob": -0.3, "compression_ratio": 0.9, "no_speech_prob": 0.02},
    ],
    "words": [
        {"word": "Hello", "start": 0.0, "end": 0.42}, {"word": "there", "start": 0.48, "end": 0.86},
        {"word": "friend", "start": 0.98, "end": 1.9}, {"word": "How", "start": 2.4, "end": 2.66},
        {"word": "are", "start": 2.7, "end": 2.9}, {"word": "you", "start": 2.94, "end": 4.1},
    ],
    "usage": {"type": "duration", "seconds": 5},
}


def verbose(data: dict[str, Any] | None = None, *, request_id: str | None = "req_ok") -> TranscriptionVerbose:
    resp = TranscriptionVerbose.model_validate(data or VERBOSE)
    resp._request_id = request_id          # what the SDK sets from the x-request-id header
    return resp


def status_error(cls: type[openai.APIStatusError], status: int, *, body: Any = None,
                 headers: dict[str, str] | None = None, request_id: str = "req_err") -> openai.APIStatusError:
    response = httpx2.Response(status, request=REQUEST, headers={"x-request-id": request_id, **(headers or {})})
    return cls(f"Error code: {status}", response=response, body=body)


class FakeTranscriptions:
    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = outcomes
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        name, fh, mime = kwargs["file"]
        self.calls.append({**kwargs, "file": (name, fh.read(), mime)})
        out = self.outcomes.pop(0)
        if isinstance(out, BaseException):
            raise out
        return out


class FakeClient:
    def __init__(self, *outcomes: Any) -> None:
        self.transcriptions = FakeTranscriptions(list(outcomes))
        self.audio = SimpleNamespace(transcriptions=self.transcriptions)

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.transcriptions.calls


class Usage:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def record(self, **kwargs: Any) -> None:
        self.rows.append(kwargs)


@pytest.fixture
def keyed(settings: Settings) -> Settings:
    return settings.model_copy(update={"openai_api_key": SecretStr(FAKE_KEY)})


@pytest.fixture
def chunk(tmp_path: Path) -> Path:
    path = tmp_path / "chunk_000.flac"
    path.write_bytes(b"fLaC" + bytes(range(256)) * 8)
    return path


@pytest.fixture
def slept(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Make backoff sleeps instant (recorded) inside the transcriber's call_openai."""
    record: list[float] = []
    monkeypatch.setattr(to, "call_openai", functools.partial(oc.call_openai, sleep=record.append))
    return record


def transcribe(t: OpenAITranscriber, path: Path, *, usage: Usage | None = None,
               cancel: CancelToken | None = None, audio_seconds: float | None = 4.25):
    return t.transcribe_chunk(path, language="en", cancel=cancel or CancelToken(), usage=usage,
                              audio_seconds=audio_seconds)


# ============================================================================ success path
def test_request_parameters_and_conversion(keyed: Settings, chunk: Path) -> None:
    client = FakeClient(verbose())
    usage = Usage()
    t = OpenAITranscriber(keyed, client=client)
    assert (t.provider, t.model) == ("openai", "whisper-1")
    result = transcribe(t, chunk, usage=usage)

    (call,) = client.calls
    assert call["model"] == "whisper-1"
    assert call["response_format"] == "verbose_json"
    assert call["timestamp_granularities"] == ["word", "segment"]
    assert call["language"] == "en"
    assert call["temperature"] == 0
    name, data, mime = call["file"]
    assert name == "chunk_000.flac" and Path(name).suffix == ".flac"
    assert mime == "audio/flac" and data == chunk.read_bytes()

    assert result.words == [RawWord("Hello", 0.0, 0.42), RawWord("there", 0.48, 0.86), RawWord("friend", 0.98, 1.9),
                            RawWord("How", 2.4, 2.66), RawWord("are", 2.7, 2.9), RawWord("you", 2.94, 4.1)]
    assert [s.text for s in result.segments] == ["Hello there, friend.", "How are you?"]
    assert result.segments[0].no_speech_prob == 0.012 and result.segments[1].avg_logprob == -0.3
    assert (result.language, result.duration, result.request_id) == ("english", 4.25, "req_ok")
    assert result.raw["segments"][0]["tokens"] == [50364, 2425, 456]
    assert result.raw["text"] == VERBOSE["text"]
    json.dumps(result.raw)                                     # cacheable as JSON

    (row,) = usage.rows
    assert row["stage"] == "transcribe" and row["provider"] == "openai" and row["model"] == "whisper-1"
    assert row["status"] == "ok" and row["request_id"] == "req_ok" and row["error_code"] is None
    assert row["audio_seconds"] == 5.0                        # billed seconds reported by the API
    assert row["duration_s"] >= 0


def test_audio_seconds_falls_back_to_caller_value(keyed: Settings, chunk: Path) -> None:
    data = {k: v for k, v in VERBOSE.items() if k != "usage"}
    usage = Usage()
    transcribe(OpenAITranscriber(keyed, client=FakeClient(verbose(data))), chunk, usage=usage, audio_seconds=7.5)
    assert usage.rows[0]["audio_seconds"] == 7.5


@pytest.mark.parametrize("suffix,mime", [(".wav", "audio/wav"), (".mp3", "audio/mpeg"), (".FLAC", "audio/flac")])
def test_mime_follows_extension(keyed: Settings, tmp_path: Path, suffix: str, mime: str) -> None:
    path = tmp_path / f"chunk_001{suffix}"
    path.write_bytes(b"RIFF0000")
    client = FakeClient(verbose())
    transcribe(OpenAITranscriber(keyed, client=client), path)
    assert client.calls[0]["file"][0] == path.name and client.calls[0]["file"][2] == mime


def test_silent_chunk_returns_empty_result(keyed: Settings, chunk: Path) -> None:
    data = {"task": "transcribe", "language": "english", "duration": 30.0, "text": "", "segments": [], "words": []}
    result = transcribe(OpenAITranscriber(keyed, client=FakeClient(verbose(data))), chunk)
    assert result.words == [] and result.segments == []


def test_raw_response_is_sanitized() -> None:
    raw = {"text": "hi", "headers": {"authorization": f"Bearer {FAKE_KEY}"}, "Authorization": "x",
           "x-request-id": "req_1", "api_key": FAKE_KEY, "openai-organization": "org",
           "segments": [{"text": " hi", "tokens": [1, 2], "set-cookie": "a=b"}],
           "words": [{"word": "hi", "start": 0.0, "end": 0.2}]}
    clean = sanitize_raw(raw)
    assert FAKE_KEY not in json.dumps(clean)
    assert clean == {"text": "hi", "segments": [{"text": " hi", "tokens": [1, 2]}],
                     "words": [{"word": "hi", "start": 0.0, "end": 0.2}]}


def test_dict_response_is_sanitized_too(keyed: Settings, chunk: Path) -> None:
    data = {**VERBOSE, "headers": {"Authorization": f"Bearer {FAKE_KEY}"}}
    result = transcribe(OpenAITranscriber(keyed, client=FakeClient(data)), chunk)
    assert "headers" not in result.raw and FAKE_KEY not in json.dumps(result.raw)
    assert result.request_id is None and len(result.words) == 6


# ============================================================================ refusals before upload
def test_models_without_word_timestamps_are_refused(keyed: Settings, chunk: Path) -> None:
    s = keyed.model_copy(update={"transcribe_model": "gpt-transcribe"})
    client = FakeClient(verbose())
    with pytest.raises(ProviderError) as ei:
        transcribe(OpenAITranscriber(s, client=client), chunk)
    e = ei.value
    assert e.code == "provider_model_unsupported" and e.retryable is False
    assert "whisper-1" in e.user_message and "CLIPPER_TRANSCRIBE_MODEL" in e.user_message
    assert "word timestamps" in e.user_message
    assert client.calls == []


def test_missing_empty_and_unsupported_files(keyed: Settings, tmp_path: Path) -> None:
    client = FakeClient()
    t = OpenAITranscriber(keyed, client=client)
    empty = tmp_path / "chunk_002.flac"
    empty.write_bytes(b"")
    odd = tmp_path / "chunk_003.aac"
    odd.write_bytes(b"data")
    for path in (tmp_path / "missing.flac", empty, odd):
        with pytest.raises(MediaError) as ei:
            transcribe(t, path)
        assert str(tmp_path) not in ei.value.user_message and "/" not in ei.value.user_message
    assert client.calls == []


def test_upload_limit_is_enforced_before_upload(keyed: Settings, tmp_path: Path) -> None:
    big = tmp_path / "chunk_009.flac"
    with big.open("wb") as fh:                     # sparse file: no 25 MB actually written
        fh.truncate(MAX_UPLOAD_BYTES + 1)
    client = FakeClient(verbose())
    with pytest.raises(MediaError) as ei:
        transcribe(OpenAITranscriber(keyed, client=client), big)
    assert "25 MB" in ei.value.user_message and "CLIPPER_TRANSCRIBE_CHUNK_SECONDS" in ei.value.user_message
    assert client.calls == []
    at_limit = tmp_path / "chunk_010.flac"
    with at_limit.open("wb") as fh:
        fh.truncate(MAX_UPLOAD_BYTES)
    assert to._check_audio_file(at_limit) == "audio/flac"


def test_cancel_before_upload(keyed: Settings, chunk: Path) -> None:
    token = CancelToken()
    token.cancel()
    client = FakeClient(verbose())
    with pytest.raises(CancelledError):
        transcribe(OpenAITranscriber(keyed, client=client), chunk, cancel=token)
    assert client.calls == []


# ============================================================================ error mapping
def test_401_maps_to_auth_error_and_records_usage(keyed: Settings, chunk: Path, slept: list[float]) -> None:
    body = {"message": f"Incorrect API key provided: {FAKE_KEY}", "code": "invalid_api_key"}
    client = FakeClient(status_error(openai.AuthenticationError, 401, body=body, request_id="req_401"))
    usage = Usage()
    with pytest.raises(ProviderAuthError) as ei:
        transcribe(OpenAITranscriber(keyed, client=client), chunk, usage=usage)
    assert ei.value.user_message.startswith("OpenAI rejected the API key")
    assert FAKE_KEY not in ei.value.user_message
    assert len(client.calls) == 1 and slept == []
    (row,) = usage.rows
    assert row["status"] == "error" and row["error_code"] == "provider_auth" and row["request_id"] == "req_401"
    assert row["audio_seconds"] == 4.25 and row["stage"] == "transcribe"


def test_429_retries_then_succeeds(keyed: Settings, chunk: Path, slept: list[float]) -> None:
    err = status_error(openai.RateLimitError, 429, body={"message": "Rate limit reached"}, headers={"retry-after": "2"})
    client = FakeClient(err, verbose())
    usage = Usage()
    result = transcribe(OpenAITranscriber(keyed, client=client), chunk, usage=usage)
    assert len(result.words) == 6
    assert len(client.calls) == 2
    assert all(c["file"][1] == chunk.read_bytes() for c in client.calls)    # file re-sent in full on retry
    assert sum(slept) == pytest.approx(2.0)
    assert [r["status"] for r in usage.rows] == ["ok"]


def test_500_exhaustion_is_retryable_provider_error(keyed: Settings, chunk: Path, slept: list[float]) -> None:
    errs = [status_error(openai.InternalServerError, 500, request_id=f"req_{i}") for i in range(4)]
    client = FakeClient(*errs)
    usage = Usage()
    with pytest.raises(ProviderError) as ei:
        transcribe(OpenAITranscriber(keyed, client=client), chunk, usage=usage)
    assert len(client.calls) == keyed.openai_max_retries + 1 == 4
    assert ei.value.retryable is True
    assert ei.value.user_message == "OpenAI is unavailable or rate-limited right now; try again later."
    assert usage.rows[0]["status"] == "error" and usage.rows[0]["request_id"] == "req_3"
    assert len(slept) > 0


def test_400_is_not_retried(keyed: Settings, chunk: Path, slept: list[float]) -> None:
    body = {"message": "Invalid file format. Supported formats: ['flac', 'mp3', 'wav']", "type": "invalid_request_error"}
    client = FakeClient(status_error(openai.BadRequestError, 400, body=body))
    with pytest.raises(ProviderError) as ei:
        transcribe(OpenAITranscriber(keyed, client=client), chunk)
    assert ei.value.retryable is False and len(client.calls) == 1
    assert "Invalid file format" in ei.value.user_message


def test_404_names_the_transcription_model(keyed: Settings, chunk: Path, slept: list[float]) -> None:
    body = {"message": "The model `whisper-1` does not exist or you do not have access to it."}
    client = FakeClient(status_error(openai.NotFoundError, 404, body=body))
    with pytest.raises(ProviderError) as ei:
        transcribe(OpenAITranscriber(keyed, client=client), chunk)
    assert "'whisper-1'" in ei.value.user_message and "CLIPPER_TRANSCRIBE_MODEL" in ei.value.user_message


def test_timeout_then_success(keyed: Settings, chunk: Path, slept: list[float]) -> None:
    client = FakeClient(openai.APITimeoutError(request=REQUEST), verbose())
    assert len(transcribe(OpenAITranscriber(keyed, client=client), chunk).words) == 6
    assert len(client.calls) == 2


def test_cancel_during_backoff_raises_promptly(keyed: Settings, chunk: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    token = CancelToken()
    steps: list[float] = []

    def sleep(seconds: float) -> None:
        steps.append(seconds)
        token.cancel()

    monkeypatch.setattr(to, "call_openai", functools.partial(oc.call_openai, sleep=sleep))
    err = status_error(openai.RateLimitError, 429, headers={"retry-after": "60"})
    client = FakeClient(err, verbose())
    usage = Usage()
    with pytest.raises(CancelledError):
        transcribe(OpenAITranscriber(keyed, client=client), chunk, usage=usage, cancel=token)
    assert len(client.calls) == 1 and len(steps) == 1
    assert usage.rows == []                     # nothing billable happened


def test_text_without_word_timings_is_an_explicit_error(keyed: Settings, chunk: Path) -> None:
    data = {k: v for k, v in VERBOSE.items() if k != "words"}
    usage = Usage()
    with pytest.raises(ProviderError) as ei:
        transcribe(OpenAITranscriber(keyed, client=FakeClient(verbose(data, request_id="req_nowords"))), chunk,
                   usage=usage)
    assert ei.value.code == "provider_invalid_output" and "word timings" in ei.value.user_message
    assert usage.rows[0]["request_id"] == "req_nowords" and usage.rows[0]["audio_seconds"] == 5.0


def test_unexpected_response_type(keyed: Settings, chunk: Path) -> None:
    with pytest.raises(ProviderError) as ei:
        transcribe(OpenAITranscriber(keyed, client=FakeClient("plain text transcript")), chunk)
    assert ei.value.code == "provider_invalid_output"


def test_usage_sink_failure_does_not_break_transcription(keyed: Settings, chunk: Path) -> None:
    class Broken:
        def record(self, **_: Any) -> None:
            raise RuntimeError("db locked")

    result = OpenAITranscriber(keyed, client=FakeClient(verbose())).transcribe_chunk(
        chunk, language="en", cancel=CancelToken(), usage=Broken())
    assert len(result.words) == 6


# ============================================================================ real SDK over a mock transport
def _sdk_client(handler: Callable[[httpx2.Request], httpx2.Response]) -> openai.OpenAI:
    return openai.OpenAI(api_key=FAKE_KEY, base_url="http://openai.test/v1", max_retries=0,
                         http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))


def _assert_multipart(body: bytes, *, filename: str, mime: str, audio: bytes) -> None:
    assert f'filename="{filename}"'.encode() in body
    assert f"Content-Type: {mime}".encode() in body
    assert audio in body
    for name, value in [("model", "whisper-1"), ("response_format", "verbose_json"), ("language", "en"),
                        ("temperature", "0"), ("timestamp_granularities[]", "word"),
                        ("timestamp_granularities[]", "segment")]:
        assert f'name="{name}"\r\n\r\n{value}\r\n'.encode() in body, name


def test_real_sdk_request_and_request_id(keyed: Settings, chunk: Path, slept: list[float]) -> None:
    seen: list[bytes] = []
    replies = [httpx2.Response(503, json={"error": {"message": "overloaded"}}, headers={"retry-after": "0"}),
               httpx2.Response(200, json=VERBOSE, headers={"x-request-id": "req_sdk"})]

    def handler(request: httpx2.Request) -> httpx2.Response:
        assert request.url.path == "/v1/audio/transcriptions"
        seen.append(request.read())
        return replies.pop(0)

    usage = Usage()
    result = transcribe(OpenAITranscriber(keyed, client=_sdk_client(handler)), chunk, usage=usage)
    assert len(seen) == 2
    for body in seen:
        _assert_multipart(body, filename="chunk_000.flac", mime="audio/flac", audio=chunk.read_bytes())
    assert result.request_id == "req_sdk" and len(result.words) == 6
    assert usage.rows[-1]["request_id"] == "req_sdk"


def test_real_sdk_auth_failure(keyed: Settings, chunk: Path) -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(401, json={"error": {"message": "Incorrect API key provided: sk-proj-****6789",
                                                    "code": "invalid_api_key"}}, headers={"x-request-id": "req_x"})

    usage = Usage()
    with pytest.raises(ProviderAuthError):
        transcribe(OpenAITranscriber(keyed, client=_sdk_client(handler)), chunk, usage=usage)
    assert usage.rows[0]["request_id"] == "req_x" and usage.rows[0]["error_code"] == "provider_auth"


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_real_flac_chunk_upload_and_merge(keyed: Settings, tmp_path: Path) -> None:
    """A real FFmpeg-encoded FLAC chunk goes through the real SDK multipart encoder unchanged, and the
    chunk-relative result lands at the right source times after merging."""
    flac = tmp_path / "chunk_001.flac"
    run([keyed.ffmpeg, "-hide_banner", "-nostdin", "-v", "error", "-y", "-f", "lavfi",
         "-i", "sine=frequency=330:sample_rate=16000:duration=4.25", "-ac", "1", "-c:a", "flac", str(flac)],
        timeout=60)
    audio = flac.read_bytes()
    assert audio[:4] == b"fLaC" and len(audio) < MAX_UPLOAD_BYTES

    def handler(request: httpx2.Request) -> httpx2.Response:
        _assert_multipart(request.read(), filename="chunk_001.flac", mime="audio/flac", audio=audio)
        return httpx2.Response(200, json=VERBOSE, headers={"x-request-id": "req_flac"})

    t = get_transcriber(keyed)
    assert isinstance(t, OpenAITranscriber)
    t._client = _sdk_client(handler)
    raw = transcribe(t, flac, audio_seconds=4.25)

    plan = SimpleNamespace(index=1, start=30.0, end=34.25, keep_start=30.0, keep_end=35.25)  # like ChunkPlan
    transcript = merge_chunks([(plan, raw)], duration=34.25, provider=t.provider, model=t.model, language="en",
                              chunk_meta=[TranscriptChunk(index=1, start=30.0, end=34.25, keep_start=30.0,
                                                          keep_end=35.25, model=t.model, request_id=raw.request_id)])
    validate_transcript(transcript)
    assert [w.text for w in transcript.words] == ["Hello", "there,", "friend.", "How", "are", "you?"]
    assert transcript.words[0].start == pytest.approx(30.0) and transcript.words[-1].end == pytest.approx(34.1)
    assert [u.text for u in transcript.units] == ["Hello there, friend.", "How are you?"]
    assert transcript.chunks[0].request_id == "req_flac"


# ============================================================================ factory
def test_factory_requires_key(settings: Settings) -> None:
    for factory in (get_transcriber, get_selector):
        with pytest.raises(ProviderNotConfiguredError) as ei:
            factory(settings)
        assert "OPENAI_API_KEY" in ei.value.user_message and ".env" in ei.value.user_message
        assert ei.value.retryable is False


def test_factory_builds_openai_adapters(keyed: Settings) -> None:
    t = get_transcriber(keyed)
    assert isinstance(t, OpenAITranscriber) and isinstance(t, Transcriber)
    assert (t.provider, t.model) == ("openai", keyed.transcribe_model)
    selection = pytest.importorskip("clipper.ai.selection")
    s = get_selector(keyed)
    assert isinstance(s, selection.OpenAISelector) and isinstance(s, ClipSelector)
    assert (s.select_model, s.review_model) == (keyed.select_model, keyed.review_model)
