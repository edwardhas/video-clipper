"""OpenAI speech-to-text adapter implementing :class:`~clipper.ai.interfaces.Transcriber`.

Uses ``POST /v1/audio/transcriptions`` with ``response_format="verbose_json"`` and
``timestamp_granularities=["word", "segment"]``. As of 2026-10-05 ``whisper-1`` is the
only OpenAI transcription model that returns word timestamps; it is deprecated with
removal scheduled for 2027-02-26 (see docs/RESEARCH_NOTES.md). Its successors return
no word timings, so moving to them needs a forced-alignment (or local timestamping)
adapter behind the same ``Transcriber`` protocol — this module refuses other models
instead of silently degrading captions.

Returned times are **chunk-relative** seconds; ``transcript.merge_chunks`` offsets them.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Any

from ..config import Settings
from ..errors import CancelledError, ClipperError, InterruptedError_, MediaError, ProviderError
from ..procs import NEVER_CANCELLED, CancelToken
from .interfaces import RawChunkResult, RawSegment, RawWord, UsageSink
from .openai_client import attempts_from_settings, call_openai, make_client

log = logging.getLogger(__name__)

PROVIDER = "openai"
STAGE = "transcribe"
WORD_TIMESTAMP_MODELS = frozenset({"whisper-1"})
# The API limit is "25 MB" per upload (the server enforces 26,214,400 bytes on the whole
# multipart body); 25,000,000 bytes satisfies both readings and leaves room for form fields.
MAX_UPLOAD_BYTES = 25_000_000
AUDIO_MIME_TYPES = {
    ".flac": "audio/flac",
    ".wav": "audio/wav",
    ".mp3": "audio/mpeg",
    ".mpga": "audio/mpeg",
    ".mpeg": "audio/mpeg",
    ".m4a": "audio/mp4",
    ".mp4": "audio/mp4",
    ".ogg": "audio/ogg",
    ".webm": "audio/webm",
}
_SENSITIVE_KEY_PARTS = ("authorization", "api_key", "apikey", "api-key", "header", "cookie", "secret",
                        "password", "bearer", "credential", "organization", "openai-project")


def _is_sensitive_key(key: object) -> bool:
    k = str(key).lower()
    return k.startswith("x-") or any(part in k for part in _SENSITIVE_KEY_PARTS)


def sanitize_raw(value: Any) -> Any:
    """Drop header/auth-like keys anywhere in a JSON-like structure (the API never returns the key)."""
    if isinstance(value, dict):
        return {str(k): sanitize_raw(v) for k, v in value.items() if not _is_sensitive_key(k)}
    if isinstance(value, (list, tuple)):
        return [sanitize_raw(v) for v in value]
    return value


def _float_or_none(value: Any) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _response_dict(resp: Any) -> dict[str, Any]:
    if hasattr(resp, "model_dump"):
        data = resp.model_dump(mode="json")
    elif isinstance(resp, dict):
        data = resp
    else:
        data = None
    if not isinstance(data, dict):
        raise ProviderError("OpenAI returned a transcription in an unexpected format. Please try again.",
                            code="provider_invalid_output", detail=f"transcription response type {type(resp).__name__}")
    return data


def to_raw_chunk_result(resp: Any, *, request_id: str | None) -> RawChunkResult:
    """Convert a ``TranscriptionVerbose`` (or its dict form) into a provider-neutral result."""
    data = _response_dict(resp)
    words: list[RawWord] = []
    for w in data.get("words") or []:
        if not isinstance(w, dict):
            continue
        start, end = _float_or_none(w.get("start")), _float_or_none(w.get("end"))
        if start is None or end is None or not isinstance(w.get("word"), str):
            log.debug("skipping malformed word entry")
            continue
        words.append(RawWord(text=w["word"], start=start, end=end))
    segments: list[RawSegment] = []
    for s in data.get("segments") or []:
        if not isinstance(s, dict):
            continue
        start, end = _float_or_none(s.get("start")), _float_or_none(s.get("end"))
        if start is None or end is None:
            continue
        segments.append(RawSegment(text=str(s.get("text") or "").strip(), start=start, end=end,
                                   no_speech_prob=_float_or_none(s.get("no_speech_prob")),
                                   avg_logprob=_float_or_none(s.get("avg_logprob"))))
    text = str(data.get("text") or "") + " ".join(s.text for s in segments)
    if not words and any(ch.isalnum() for ch in text):
        raise ProviderError(
            "OpenAI returned a transcript without word timings, which captions and clip timing need. "
            "Please try again.",
            code="provider_invalid_output", detail="verbose_json response has text but no words")
    language = data.get("language")
    return RawChunkResult(words=words, segments=segments,
                          language=language if isinstance(language, str) else None,
                          duration=_float_or_none(data.get("duration")), request_id=request_id,
                          raw=sanitize_raw(data))


def _billed_seconds(resp: Any, fallback: float | None) -> float | None:
    usage = getattr(resp, "usage", None)
    seconds = _float_or_none(getattr(usage, "seconds", None))
    if seconds is None:
        seconds = fallback if fallback is not None else _float_or_none(getattr(resp, "duration", None))
    return seconds


def _check_audio_file(path: Path) -> str:
    """Return the MIME type for ``path``; ``MediaError`` if missing, empty, too large or unsupported."""
    mime = AUDIO_MIME_TYPES.get(path.suffix.lower())
    if mime is None:
        raise MediaError("This audio chunk has a format the transcription service does not accept.",
                         detail=f"unsupported transcription chunk extension {path.suffix!r}")
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise MediaError("An audio chunk for transcription is missing. Run the analysis again.",
                         detail=f"chunk {path.name} not readable: {exc.__class__.__name__}") from exc
    if not path.is_file() or size == 0:
        raise MediaError("An audio chunk for transcription is empty. Run the analysis again.",
                         detail=f"chunk {path.name} size {size}")
    if size > MAX_UPLOAD_BYTES:
        raise MediaError(
            f"An audio chunk is {size / 1e6:.1f} MB, above OpenAI's 25 MB upload limit. "
            "Lower CLIPPER_TRANSCRIBE_CHUNK_SECONDS in the .env file and try again.",
            detail=f"chunk {path.name} size {size}")
    return mime


class OpenAITranscriber:
    """``Transcriber`` backed by OpenAI ``whisper-1`` (word + segment timestamps)."""

    provider = PROVIDER

    def __init__(self, settings: Settings, client: Any | None = None) -> None:
        self.settings = settings
        self.model = settings.transcribe_model
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            self._client = make_client(self.settings)
        return self._client

    def _require_word_timestamps(self) -> None:
        if self.model not in WORD_TIMESTAMP_MODELS:
            raise ProviderError(
                f"The transcription model '{self.model}' is not supported: only whisper-1 returns the "
                "word timestamps that captions and clip timing need (as of 2026-10-05). Other OpenAI "
                "transcription models would need a separate alignment step, which this release does "
                "not include. Set CLIPPER_TRANSCRIBE_MODEL=whisper-1 in the .env file and restart.",
                code="provider_model_unsupported", retryable=False,
                detail=f"transcribe model {self.model!r} lacks word timestamps")

    def transcribe_chunk(self, audio_path: Path, *, language: str, cancel: CancelToken | None,
                         usage: UsageSink | None = None, audio_seconds: float | None = None) -> RawChunkResult:
        """Transcribe one chunk; times in the result are relative to the start of ``audio_path``."""
        token = cancel or NEVER_CANCELLED
        self._require_word_timestamps()
        token.raise_if_cancelled()
        path = Path(audio_path)
        mime = _check_audio_file(path)
        client = self.client

        def _send() -> Any:
            try:
                fh = path.open("rb")
            except OSError as exc:
                raise MediaError("An audio chunk for transcription could not be read. Run the analysis again.",
                                 detail=f"open {path.name}: {exc.__class__.__name__}") from exc
            with fh:
                return client.audio.transcriptions.create(
                    file=(path.name, fh, mime), model=self.model, response_format="verbose_json",
                    timestamp_granularities=["word", "segment"], temperature=0,
                    **({"language": language} if language else {}))

        started = time.monotonic()
        try:
            resp = call_openai(_send, what="transcription", cancel=token, model=self.model,
                               attempts=attempts_from_settings(self.settings))
        except (CancelledError, InterruptedError_):
            raise                     # nothing billable completed (only failed attempts, if any)
        except ClipperError as exc:
            self._record(usage, request_id=getattr(exc, "request_id", None), status="error",
                         error_code=exc.code, audio_seconds=audio_seconds, started=started)
            raise
        rid = getattr(resp, "_request_id", None)
        request_id = rid if isinstance(rid, str) else None
        billed = _billed_seconds(resp, audio_seconds)
        try:
            result = to_raw_chunk_result(resp, request_id=request_id)
        except ClipperError as exc:
            self._record(usage, request_id=request_id, status="error", error_code=exc.code,
                         audio_seconds=billed, started=started)
            raise
        self._record(usage, request_id=request_id, status="ok", error_code=None, audio_seconds=billed,
                     started=started)
        log.info("transcribed %s (%d words, request %s)", path.name, len(result.words), result.request_id)
        return result

    def _record(self, usage: UsageSink | None, *, request_id: str | None, status: str, error_code: str | None,
                audio_seconds: float | None, started: float) -> None:
        if usage is None:
            return
        try:
            usage.record(stage=STAGE, provider=PROVIDER, model=self.model, request_id=request_id, status=status,
                         audio_seconds=audio_seconds, duration_s=round(time.monotonic() - started, 3),
                         error_code=error_code)
        except Exception:  # usage accounting must never break transcription
            log.warning("could not record transcription usage", exc_info=True)
