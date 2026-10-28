"""Tests for clipper.ai.openai_client: client construction, retries, cancellation and error mapping.

SDK exceptions are real ``openai`` exception objects built on ``httpx2`` responses (the same
constructor the SDK uses); a few tests go through a real ``openai.OpenAI`` client whose HTTP
transport is an in-process ``httpx2.MockTransport`` (no network).
"""

from __future__ import annotations

import logging
import threading
import time
from email.utils import format_datetime
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import httpx2
import openai
import pytest
from pydantic import SecretStr

from clipper.ai import openai_client as oc
from clipper.config import Settings
from clipper.errors import (
    CancelledError,
    InterruptedError_,
    ProviderAuthError,
    ProviderError,
    ProviderNotConfiguredError,
)
from clipper.procs import CancelToken

FAKE_KEY = "sk-proj-TESTKEYabcdefghijklmnop0123456789"
REQUEST = httpx2.Request("POST", "https://api.openai.com/v1/audio/transcriptions")


def status_error(cls: type[openai.APIStatusError], status: int, *, body: Any = None,
                 headers: dict[str, str] | None = None, request_id: str = "req_test") -> openai.APIStatusError:
    """Build an SDK status error exactly like ``OpenAI._make_status_error`` does."""
    response = httpx2.Response(status, request=REQUEST, headers={"x-request-id": request_id, **(headers or {})},
                               json={"error": body} if isinstance(body, dict) else None)
    return cls(f"Error code: {status}", response=response, body=body)


class Flaky:
    """Callable raising the given exceptions in turn, then returning ``result``."""

    def __init__(self, *errors: BaseException, result: Any = "ok") -> None:
        self.errors = list(errors)
        self.result = result
        self.calls = 0

    def __call__(self) -> Any:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return self.result


class SleepRecorder:
    def __init__(self, on_sleep: Callable[[float], None] | None = None) -> None:
        self.slept: list[float] = []
        self.on_sleep = on_sleep

    def __call__(self, seconds: float) -> None:
        self.slept.append(seconds)
        if self.on_sleep:
            self.on_sleep(seconds)

    @property
    def total(self) -> float:
        return sum(self.slept)


def call(fn: Callable[[], Any], *, sleep: SleepRecorder | None = None, cancel: CancelToken | None = None,
         attempts: int = 4, model: str | None = None, what: str = "transcription") -> Any:
    return oc.call_openai(fn, what=what, cancel=cancel or CancelToken(), attempts=attempts, model=model,
                          sleep=sleep or SleepRecorder())


# ============================================================================ make_client
def test_make_client_requires_key(settings: Settings) -> None:
    with pytest.raises(ProviderNotConfiguredError) as ei:
        oc.make_client(settings)
    assert ei.value.user_message == (
        "OpenAI API key is not configured. Add OPENAI_API_KEY to the .env file in the project folder "
        "(see .env.example) and restart.")
    assert ei.value.retryable is False
    blank = settings.model_copy(update={"openai_api_key": SecretStr("   ")})
    with pytest.raises(ProviderNotConfiguredError):
        oc.make_client(blank)


def test_make_client_disables_sdk_retries(settings: Settings) -> None:
    s = settings.model_copy(update={"openai_api_key": SecretStr(FAKE_KEY), "openai_timeout_seconds": 123.0})
    client = oc.make_client(s)
    assert isinstance(client, openai.OpenAI)
    assert client.max_retries == 0
    assert client.timeout == 123.0
    assert client.api_key == FAKE_KEY
    assert FAKE_KEY not in repr(s)


def test_attempts_derived_from_settings(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    assert oc.attempts_from_settings(settings) == 4                      # default 3 retries → 4 attempts
    assert oc.attempts_from_settings(settings.model_copy(update={"openai_max_retries": 0})) == 1
    monkeypatch.setattr(oc, "get_settings", lambda: settings.model_copy(update={"openai_max_retries": 1}))
    err = status_error(openai.InternalServerError, 500)
    fn = Flaky(err, err, err)
    with pytest.raises(ProviderError):
        oc.call_openai(fn, what="transcription", cancel=CancelToken(), sleep=SleepRecorder())
    assert fn.calls == 2

    def broken() -> Settings:
        raise RuntimeError("bad config")
    monkeypatch.setattr(oc, "get_settings", broken)
    assert oc.default_attempts() == oc.DEFAULT_ATTEMPTS == 4


# ============================================================================ retries
def test_success_first_try_does_not_sleep() -> None:
    sleep = SleepRecorder()
    assert call(Flaky(result=42), sleep=sleep) == 42
    assert sleep.slept == []


def test_rate_limit_retries_respecting_retry_after_then_succeeds() -> None:
    err = status_error(openai.RateLimitError, 429, body={"message": "Rate limit reached", "code": "rate_limit_exceeded"},
                       headers={"retry-after": "3"})
    fn = Flaky(err, err, result="done")
    sleep = SleepRecorder()
    assert call(fn, sleep=sleep) == "done"
    assert fn.calls == 3
    assert sleep.total == pytest.approx(6.0)
    assert max(sleep.slept) <= oc.SLEEP_STEP + 1e-9          # slept in small, cancellable steps


@pytest.mark.parametrize("headers,expected", [
    ({"retry-after-ms": "1500"}, 1.5),
    ({"retry-after": "0"}, 0.0),
    ({"retry-after": "9999"}, oc.MAX_DELAY),                   # capped
])
def test_retry_after_header_variants(headers: dict[str, str], expected: float) -> None:
    err = status_error(openai.RateLimitError, 429, body={"message": "slow down"}, headers=headers)
    sleep = SleepRecorder()
    call(Flaky(err), sleep=sleep)
    assert sleep.total == pytest.approx(expected)


def test_retry_after_http_date() -> None:
    when = datetime.now(timezone.utc) + timedelta(seconds=5)
    err = status_error(openai.RateLimitError, 429, headers={"retry-after": format_datetime(when, usegmt=True)})
    assert 3.0 <= (oc.retry_after_seconds(err) or 0) <= 5.0


def test_exponential_backoff_with_jitter_bounds() -> None:
    assert oc.backoff_delay(0, rng=lambda: 0.0) == pytest.approx(1.0)
    assert oc.backoff_delay(0, rng=lambda: 1.0) == pytest.approx(2.0)
    assert oc.backoff_delay(2, rng=lambda: 1.0) == pytest.approx(8.0)
    assert oc.backoff_delay(10, rng=lambda: 1.0) == pytest.approx(oc.MAX_DELAY)
    assert oc.backoff_delay(3, retry_after=2.5) == 2.5
    for k in range(6):
        d = oc.backoff_delay(k)
        step = min(oc.MAX_DELAY, oc.BASE_DELAY * 2 ** k)
        assert step / 2 <= d <= step


def test_server_errors_exhaust_into_retryable_unavailable_error() -> None:
    err = status_error(openai.InternalServerError, 500, request_id="req_500")
    fn = Flaky(*([err] * 10))
    sleep = SleepRecorder()
    with pytest.raises(ProviderError) as ei:
        call(fn, sleep=sleep, attempts=4)
    assert fn.calls == 4
    assert sleep.slept
    e = ei.value
    assert e.user_message == "OpenAI is unavailable or rate-limited right now; try again later."
    assert e.retryable is True and e.code == "provider_error"
    assert e.request_id == "req_500"
    assert isinstance(e.__cause__, openai.InternalServerError)


@pytest.mark.parametrize("make_err", [
    lambda: openai.APITimeoutError(request=REQUEST),
    lambda: openai.APIConnectionError(request=REQUEST),
    lambda: status_error(openai.ConflictError, 409),
    lambda: status_error(openai.APIStatusError, 408),
    lambda: status_error(openai.InternalServerError, 503),
])
def test_transient_errors_are_retried(make_err: Callable[[], Exception]) -> None:
    fn = Flaky(make_err(), result="ok")
    assert call(fn) == "ok"
    assert fn.calls == 2


def test_timeouts_exhausted_map_to_unavailable() -> None:
    fn = Flaky(*[openai.APITimeoutError(request=REQUEST) for _ in range(3)])
    with pytest.raises(ProviderError) as ei:
        call(fn, attempts=3)
    assert fn.calls == 3 and ei.value.retryable is True
    assert "Request timed out" not in ei.value.user_message


# ============================================================================ non-retryable mapping
def test_authentication_error_maps_to_provider_auth_without_key(caplog: pytest.LogCaptureFixture) -> None:
    body = {"message": f"Incorrect API key provided: {FAKE_KEY}. You can find your API key at "
                       "https://platform.openai.com/account/api-keys.", "type": "invalid_request_error",
            "code": "invalid_api_key"}
    fn = Flaky(status_error(openai.AuthenticationError, 401, body=body, request_id="req_401"))
    caplog.set_level(logging.DEBUG)
    with pytest.raises(ProviderAuthError) as ei:
        call(fn)
    assert fn.calls == 1
    e = ei.value
    assert e.user_message.startswith("OpenAI rejected the API key")
    assert e.retryable is False and e.code == "provider_auth"
    assert e.request_id == "req_401"
    assert FAKE_KEY not in e.user_message and FAKE_KEY not in (e.detail or "")
    assert FAKE_KEY not in caplog.text


@pytest.mark.parametrize("cls,status", [(openai.PermissionDeniedError, 403), (openai.NotFoundError, 404)])
def test_model_access_errors_name_the_model(cls: type[openai.APIStatusError], status: int) -> None:
    body = {"message": "The model `gpt-6.1-sol` does not exist or you do not have access to it.",
            "type": "invalid_request_error", "code": "model_not_found"}
    fn = Flaky(status_error(cls, status, body=body))
    with pytest.raises(ProviderError) as ei:
        call(fn, model="gpt-6.1-sol", what="clip selection")
    e = ei.value
    assert fn.calls == 1 and e.retryable is False
    assert "'gpt-6.1-sol'" in e.user_message and "clip selection" in e.user_message
    assert f"HTTP {status}" in e.user_message
    assert "CLIPPER_SELECT_MODEL" in e.user_message and "CLIPPER_TRANSCRIBE_MODEL" in e.user_message


def test_model_access_error_without_model_uses_provider_text() -> None:
    body = {"message": "Project proj_123 does not have access to model whisper-1"}
    with pytest.raises(ProviderError) as ei:
        call(Flaky(status_error(openai.PermissionDeniedError, 403, body=body)))
    assert "whisper-1" in ei.value.user_message
    assert "the model used for transcription" in ei.value.user_message


def test_bad_request_includes_short_sanitized_reason() -> None:
    long_tail = "x" * 1000
    body = {"message": f"Invalid file format. Supported formats: flac, mp3. File /Users/someone/data/chunk_000.flac "
                       f"rejected; key {FAKE_KEY} Bearer abc.def {long_tail}", "type": "invalid_request_error"}
    fn = Flaky(status_error(openai.BadRequestError, 400, body=body))
    with pytest.raises(ProviderError) as ei:
        call(fn)
    e = ei.value
    assert fn.calls == 1 and e.retryable is False
    assert e.user_message.startswith("OpenAI rejected the transcription request (HTTP 400): Invalid file format.")
    assert "/Users/" not in e.user_message and FAKE_KEY not in e.user_message and "abc.def" not in e.user_message
    reason = e.user_message.split(": ", 1)[1]
    assert len(reason) <= oc.MAX_ERROR_TEXT


def test_bad_request_without_body() -> None:
    with pytest.raises(ProviderError) as ei:
        call(Flaky(status_error(openai.BadRequestError, 400, body=None)))
    assert ei.value.user_message == "OpenAI rejected the transcription request (HTTP 400)."


def test_unprocessable_and_other_4xx_are_not_retried() -> None:
    for err in (status_error(openai.UnprocessableEntityError, 422, body={"message": "bad"}),
                status_error(openai.APIStatusError, 413, body={"message": "too large"})):
        fn = Flaky(err)
        with pytest.raises(ProviderError) as ei:
            call(fn)
        assert fn.calls == 1 and ei.value.retryable is False


def test_insufficient_quota_is_not_retried() -> None:
    err = status_error(openai.RateLimitError, 429, body={"message": "You exceeded your current quota",
                                                         "code": "insufficient_quota"})
    fn = Flaky(err, err)
    with pytest.raises(ProviderError) as ei:
        call(fn)
    assert fn.calls == 1 and ei.value.retryable is False
    assert "billing" in ei.value.user_message


def test_non_openai_exceptions_propagate_unchanged() -> None:
    def boom() -> None:
        raise ValueError("local bug")
    with pytest.raises(ValueError):
        call(boom)


def test_sanitize_error_text() -> None:
    text = f"see /var/data/projects/p_1/work/x.json and C:\\Users\\me\\a.wav with {FAKE_KEY} and Bearer tok123"
    out = oc.sanitize_error_text(text)
    assert "/var/data" not in out and "C:\\" not in out and FAKE_KEY not in out and "tok123" not in out
    assert "https://platform.openai.com/x" in oc.sanitize_error_text("https://platform.openai.com/x")
    assert len(oc.sanitize_error_text("y" * 1000)) == oc.MAX_ERROR_TEXT


# ============================================================================ cancellation
def test_cancel_before_first_attempt() -> None:
    token = CancelToken()
    token.cancel()
    fn = Flaky(result="never")
    with pytest.raises(CancelledError):
        call(fn, cancel=token)
    assert fn.calls == 0


def test_cancel_during_backoff_stops_promptly_with_fake_sleep() -> None:
    token = CancelToken()
    err = status_error(openai.RateLimitError, 429, headers={"retry-after": "30"})
    fn = Flaky(err, err, result="never")
    sleep = SleepRecorder(on_sleep=lambda s: token.cancel() if len(sleep.slept) >= 2 else None)
    with pytest.raises(CancelledError):
        call(fn, sleep=sleep, cancel=token)
    assert fn.calls == 1
    assert sleep.total <= 2 * oc.SLEEP_STEP + 1e-9


def test_shutdown_during_backoff_interrupts() -> None:
    token = CancelToken()
    err = status_error(openai.InternalServerError, 500)
    sleep = SleepRecorder(on_sleep=lambda s: token.shutdown.set())
    with pytest.raises(InterruptedError_):
        call(Flaky(err, result="never"), sleep=sleep, cancel=token)


def test_cancel_during_real_backoff_sleep_is_prompt() -> None:
    """Real time.sleep: a 30 s Retry-After is abandoned within a fraction of a second of cancelling."""
    token = CancelToken()
    err = status_error(openai.RateLimitError, 429, headers={"retry-after": "30"})
    fn = Flaky(err, result="never")
    timer = threading.Timer(0.3, token.cancel)
    timer.start()
    started = time.monotonic()
    try:
        with pytest.raises(CancelledError):
            oc.call_openai(fn, what="transcription", cancel=token, attempts=4)
    finally:
        timer.cancel()
    assert time.monotonic() - started < 2.0
    assert fn.calls == 1


def test_cancel_via_polled_check_function() -> None:
    """The worker's token polls the job row (at most every 0.5 s); backoff sleeps keep re-checking it."""
    flag = {"cancel": False}
    token = CancelToken(check=lambda: flag["cancel"])

    def real_sleep(seconds: float) -> None:
        flag["cancel"] = True
        time.sleep(seconds)

    err = status_error(openai.InternalServerError, 502, headers={"retry-after": "20"})
    fn = Flaky(err, err, result="never")
    sleep = SleepRecorder(on_sleep=real_sleep)
    with pytest.raises(CancelledError):
        call(fn, sleep=sleep, cancel=token)
    assert fn.calls == 1 and sleep.total < 1.5


# ============================================================================ real SDK client, mock transport
def _sdk_client(handler: Callable[[httpx2.Request], httpx2.Response]) -> openai.OpenAI:
    return openai.OpenAI(api_key=FAKE_KEY, base_url="http://openai.test/v1", max_retries=0,
                         http_client=httpx2.Client(transport=httpx2.MockTransport(handler)))


def test_real_sdk_errors_are_mapped_and_retried() -> None:
    responses = [
        httpx2.Response(429, json={"error": {"message": "Rate limit", "code": "rate_limit_exceeded"}},
                        headers={"retry-after-ms": "0", "x-request-id": "req_a"}),
        httpx2.Response(503, json={"error": {"message": "overloaded"}}, headers={"retry-after": "0"}),
        httpx2.Response(200, json={"object": "model", "id": "whisper-1", "created": 1, "owned_by": "openai"},
                        headers={"x-request-id": "req_ok"}),
    ]
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        return responses.pop(0)

    client = _sdk_client(handler)
    model = oc.call_openai(lambda: client.models.retrieve("whisper-1"), what="model check",
                           cancel=CancelToken(), attempts=4, sleep=SleepRecorder())
    assert model.id == "whisper-1" and model._request_id == "req_ok"
    assert len(seen) == 3
    assert seen[0].headers["authorization"] == f"Bearer {FAKE_KEY}"     # key only ever goes to the API


def test_real_sdk_401_maps_to_auth_error() -> None:
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(401, json={"error": {"message": f"Incorrect API key provided: {FAKE_KEY[:12]}****",
                                                    "code": "invalid_api_key"}},
                               headers={"x-request-id": "req_401"})

    client = _sdk_client(handler)
    with pytest.raises(ProviderAuthError) as ei:
        oc.call_openai(lambda: client.models.retrieve("whisper-1"), what="model check", cancel=CancelToken())
    assert ei.value.request_id == "req_401"
    assert FAKE_KEY[:12] not in ei.value.user_message


def test_real_sdk_timeout_is_retried() -> None:
    calls = {"n": 0}

    def handler(request: httpx2.Request) -> httpx2.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            raise httpx2.ReadTimeout("timed out", request=request)
        return httpx2.Response(200, json={"object": "model", "id": "whisper-1", "created": 1, "owned_by": "openai"})

    client = _sdk_client(handler)
    model = oc.call_openai(lambda: client.models.retrieve("whisper-1"), what="model check",
                           cancel=CancelToken(), sleep=SleepRecorder())
    assert model.id == "whisper-1" and calls["n"] == 2
