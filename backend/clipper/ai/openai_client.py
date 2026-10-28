"""OpenAI client construction and a bounded, cancellable retry wrapper.

``make_client`` builds an ``openai.OpenAI`` with the SDK's own retries disabled;
``call_openai`` retries transient failures itself so it can honour cancellation
(checked before every attempt and in small steps while backing off) and map every
SDK exception onto a typed :class:`~clipper.errors.ClipperError` with a
plain-language message.

Messages and log lines never contain the API key: provider error text (which
can echo a partially masked key on 401) is never used for 401, and for other client
errors (400/403/404/422) it is redacted (key-like tokens, bearer tokens, host paths)
and truncated to 300 characters first.
"""

from __future__ import annotations

import email.utils
import logging
import math
import random
import re
import time
from typing import Callable, TypeVar

import openai

from ..config import Settings, get_settings
from ..errors import ProviderAuthError, ProviderError, ProviderNotConfiguredError
from ..procs import NEVER_CANCELLED, CancelToken

log = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_ATTEMPTS = 4
BASE_DELAY = 2.0
MAX_DELAY = 60.0
SLEEP_STEP = 0.25          # cancellation is re-checked at least this often while backing off
MAX_ERROR_TEXT = 300

NOT_CONFIGURED_MESSAGE = (
    "OpenAI API key is not configured. Add OPENAI_API_KEY to the .env file in the project folder "
    "(see .env.example) and restart."
)
AUTH_MESSAGE = (
    "OpenAI rejected the API key. Check OPENAI_API_KEY in the .env file in the project folder "
    "(it may be mistyped, revoked or belong to a project without API access) and restart."
)
UNAVAILABLE_MESSAGE = "OpenAI is unavailable or rate-limited right now; try again later."
QUOTA_MESSAGE = (
    "Your OpenAI account has no remaining credit or reached its spending limit. "
    "Check billing in the OpenAI dashboard, then try again."
)
MODEL_SETTINGS_HINT = (
    "Choose a model your account can use by setting CLIPPER_TRANSCRIBE_MODEL, CLIPPER_SELECT_MODEL "
    "or CLIPPER_REVIEW_MODEL in the .env file, then restart."
)

_KEY_RE = re.compile(r"\b(?:sk|rk|pk|sess)-[A-Za-z0-9_\-*]{4,}")
_BEARER_RE = re.compile(r"(?i)bearer\s+\S+")
_POSIX_PATH_RE = re.compile(r"(?<![\w.:/])/(?:[^\s/'\"]+/)+[^\s'\"]*")
_WIN_PATH_RE = re.compile(r"\b[A-Za-z]:\\[^\s'\"]*")


def make_client(settings: Settings) -> openai.OpenAI:
    """Return an SDK client with our timeout and SDK retries disabled (we retry ourselves)."""
    if not settings.openai_configured or settings.openai_api_key is None:
        raise ProviderNotConfiguredError(NOT_CONFIGURED_MESSAGE)
    return openai.OpenAI(
        api_key=settings.openai_api_key.get_secret_value().strip(),
        timeout=settings.openai_timeout_seconds,
        max_retries=0,
    )


def attempts_from_settings(settings: Settings) -> int:
    """Total attempts (first try + retries) derived from ``CLIPPER_OPENAI_MAX_RETRIES`` (default 3 → 4)."""
    return max(1, int(settings.openai_max_retries) + 1)


def default_attempts() -> int:
    """Attempts from the process settings; ``DEFAULT_ATTEMPTS`` if settings cannot be loaded."""
    try:
        return attempts_from_settings(get_settings())
    except Exception:  # invalid/missing configuration must not break error handling
        return DEFAULT_ATTEMPTS


def sanitize_error_text(text: str, *, limit: int = MAX_ERROR_TEXT) -> str:
    """Redact key-like tokens and host paths, collapse whitespace and truncate."""
    cleaned = _KEY_RE.sub("[redacted]", text)
    cleaned = _BEARER_RE.sub("Bearer [redacted]", cleaned)
    cleaned = _WIN_PATH_RE.sub("[path]", cleaned)
    cleaned = _POSIX_PATH_RE.sub("[path]", cleaned)
    cleaned = " ".join(cleaned.split())
    if len(cleaned) > limit:
        cleaned = cleaned[: limit - 1].rstrip() + "…"
    return cleaned


def request_id_from_error(exc: BaseException | None) -> str | None:
    """Find the OpenAI request id on ``exc`` or anything in its cause/context chain."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        rid = getattr(exc, "request_id", None)
        if isinstance(rid, str) and rid:
            return rid
        exc = exc.__cause__ or exc.__context__
    return None


def retry_after_seconds(exc: BaseException) -> float | None:
    """Server-requested delay from ``retry-after-ms`` / ``retry-after`` headers, if any."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    for name, divisor in (("retry-after-ms", 1000.0), ("retry-after", 1.0)):
        value = headers.get(name)
        if value is None:
            continue
        try:
            seconds = float(value) / divisor
        except ValueError:
            continue
        if math.isfinite(seconds) and seconds >= 0:
            return seconds
    value = headers.get("retry-after")
    if value:
        try:
            when = email.utils.parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None
        if when is not None:
            seconds = when.timestamp() - time.time()
            return max(0.0, seconds)
    return None


def _body_message(exc: BaseException) -> str:
    body = getattr(exc, "body", None)
    text: str | None = None
    if isinstance(body, dict):
        msg = body.get("message")
        if isinstance(msg, str):
            text = msg
        elif isinstance(body.get("error"), dict) and isinstance(body["error"].get("message"), str):
            text = body["error"]["message"]
    elif isinstance(body, str):
        text = body
    return sanitize_error_text(text) if text else ""


def _status(exc: BaseException) -> int | None:
    status = getattr(exc, "status_code", None)
    return status if isinstance(status, int) else None


def _detail(exc: BaseException, what: str) -> str:
    parts = [f"{what}: {type(exc).__name__}"]
    status = _status(exc)
    if status is not None:
        parts.append(f"status={status}")
    rid = request_id_from_error(exc)
    if rid:
        parts.append(f"request_id={rid}")
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code:
        parts.append(f"code={sanitize_error_text(code, limit=60)}")
    return " ".join(parts)


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, openai.RateLimitError):
        return getattr(exc, "code", None) != "insufficient_quota"
    if isinstance(exc, (openai.APIConnectionError, openai.InternalServerError, openai.ConflictError)):
        return True   # APITimeoutError is a subclass of APIConnectionError
    if isinstance(exc, openai.APIStatusError):
        status = _status(exc)
        return status is not None and (status == 408 or status >= 500)
    return False


def _model_phrase(what: str, model: str | None) -> str:
    return f"the model '{model}' ({what})" if model else f"the model used for {what}"


def map_openai_error(exc: BaseException, *, what: str, model: str | None = None) -> ProviderError:
    """Translate a non-retried SDK exception into a typed, user-safe ``ProviderError``.

    The returned error carries ``request_id`` (OpenAI's id for the failed request, or None).
    """
    err = _map_openai_error(exc, what=what, model=model)
    err.request_id = request_id_from_error(exc)  # type: ignore[attr-defined]
    return err


def _map_openai_error(exc: BaseException, *, what: str, model: str | None) -> ProviderError:
    detail = _detail(exc, what)
    if isinstance(exc, openai.AuthenticationError):
        return ProviderAuthError(AUTH_MESSAGE, detail=detail)
    if isinstance(exc, openai.RateLimitError) and getattr(exc, "code", None) == "insufficient_quota":
        return ProviderError(QUOTA_MESSAGE, retryable=False, detail=detail)
    if isinstance(exc, (openai.PermissionDeniedError, openai.NotFoundError)):
        status = _status(exc)
        reason = _body_message(exc)
        return ProviderError(
            f"OpenAI says {_model_phrase(what, model)} is not available to this API key "
            f"(HTTP {status}). The account or project may lack access to it, or it was retired. "
            + (f"OpenAI said: {reason} " if reason else "")
            + MODEL_SETTINGS_HINT,
            retryable=False,
            detail=detail,
        )
    if isinstance(exc, (openai.BadRequestError, openai.UnprocessableEntityError)) or (
        isinstance(exc, openai.APIStatusError) and not _is_retryable(exc)
    ):
        status = _status(exc)
        reason = _body_message(exc)
        message = f"OpenAI rejected the {what} request (HTTP {status})"
        message += f": {reason}" if reason else "."
        return ProviderError(message, retryable=False, detail=detail)
    if isinstance(exc, openai.APIResponseValidationError):
        return ProviderError(f"OpenAI returned a response in an unexpected format for {what}.",
                             retryable=True, detail=detail)
    if _is_retryable(exc):
        return ProviderError(UNAVAILABLE_MESSAGE, retryable=True, detail=detail)
    return ProviderError(f"The OpenAI request for {what} failed unexpectedly. Try again later.",
                         retryable=True, detail=detail)


def backoff_delay(retry_index: int, *, base_delay: float = BASE_DELAY, max_delay: float = MAX_DELAY,
                  retry_after: float | None = None, rng: Callable[[], float] = random.random) -> float:
    """Delay before retry number ``retry_index`` (0-based).

    A server ``Retry-After`` wins (capped at ``max_delay``); otherwise exponential
    backoff ``base_delay * 2**retry_index`` capped at ``max_delay`` with "equal
    jitter" (uniform in [50 %, 100 %] of the step).
    """
    if retry_after is not None:
        return min(max_delay, max(0.0, retry_after))
    step = min(max_delay, base_delay * (2.0 ** retry_index))
    return step * (0.5 + 0.5 * rng())


def cancellable_sleep(seconds: float, cancel: CancelToken, *, sleep: Callable[[float], None] = time.sleep,
                      step: float = SLEEP_STEP) -> None:
    """Sleep ``seconds`` in increments of ``step``, raising promptly if ``cancel`` fires."""
    remaining = max(0.0, seconds)
    while remaining > 1e-9:
        cancel.raise_if_cancelled()
        chunk = min(step, remaining)
        sleep(chunk)
        remaining -= chunk
    cancel.raise_if_cancelled()


def call_openai(
    fn: Callable[[], T],
    *,
    what: str,
    cancel: CancelToken | None,
    attempts: int | None = None,
    model: str | None = None,
    sleep: Callable[[float], None] = time.sleep,
    base_delay: float = BASE_DELAY,
    max_delay: float = MAX_DELAY,
) -> T:
    """Run ``fn`` (one SDK request) with bounded retries and typed error mapping.

    ``what`` is a short noun phrase used in messages ("transcription", "clip selection");
    ``model`` (optional) is named when OpenAI refuses access to the model. ``attempts``
    defaults to ``CLIPPER_OPENAI_MAX_RETRIES + 1`` (4 unless configured otherwise).
    Raised ``ProviderError`` objects carry ``request_id`` of the last failed request (or None).
    Retries: 429 (except ``insufficient_quota``), timeouts, connection errors, 409, 408 and 5xx.
    Raises ``CancelledError``/``InterruptedError_`` from the token, ``ProviderAuthError`` on 401,
    ``ProviderError(retryable=False)`` for other client errors and ``ProviderError(retryable=True)``
    when retries are exhausted. The original SDK exception is chained as ``__cause__``.
    """
    token = cancel or NEVER_CANCELLED
    total = max(1, default_attempts() if attempts is None else attempts)
    for attempt in range(total):
        token.raise_if_cancelled()
        try:
            return fn()
        except openai.OpenAIError as exc:
            if not _is_retryable(exc):
                raise map_openai_error(exc, what=what, model=model) from exc
            if attempt + 1 >= total:
                log.warning("OpenAI %s failed after %d attempts (%s)", what, total, _detail(exc, what))
                err = ProviderError(UNAVAILABLE_MESSAGE, retryable=True, detail=_detail(exc, what))
                err.request_id = request_id_from_error(exc)  # type: ignore[attr-defined]
                raise err from exc
            delay = backoff_delay(attempt, base_delay=base_delay, max_delay=max_delay,
                                  retry_after=retry_after_seconds(exc))
            log.info("OpenAI %s attempt %d/%d failed (%s); retrying in %.1fs",
                     what, attempt + 1, total, _detail(exc, what), delay)
            cancellable_sleep(delay, token, sleep=sleep)
    raise AssertionError("unreachable")  # pragma: no cover
