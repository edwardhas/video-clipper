"""Typed errors carrying a stable code and a user-facing message.

Worker stages raise these; the job runner stores ``code``/``user_message``/``retryable``
on the job row and the UI shows ``user_message``. Never put secrets, API keys or
absolute host paths into ``user_message``.
"""

from __future__ import annotations


class ClipperError(Exception):
    code: str = "internal_error"
    retryable: bool = False

    def __init__(self, user_message: str, *, code: str | None = None, retryable: bool | None = None,
                 detail: str | None = None) -> None:
        super().__init__(user_message)
        self.user_message = user_message
        if code is not None:
            self.code = code
        if retryable is not None:
            self.retryable = retryable
        # Developer detail for logs (must also be free of secrets).
        self.detail = detail


class InputError(ClipperError):
    """Invalid user input (bad URL, unsupported file, too long, etc.)."""
    code = "invalid_input"


class SourceUnavailableError(ClipperError):
    """Remote source cannot be fetched (private, removed, geo-blocked, sign-in required)."""
    code = "source_unavailable"


class MediaError(ClipperError):
    """Corrupt/unsupported media or an FFmpeg failure."""
    code = "media_error"


class NoSpeechError(ClipperError):
    """The source has no audio or not enough speech for analysis."""
    code = "insufficient_speech"


class ProviderError(ClipperError):
    """AI provider failure (network, rate limit, invalid output)."""
    code = "provider_error"
    retryable = True


class ProviderAuthError(ProviderError):
    code = "provider_auth"
    retryable = False


class ProviderNotConfiguredError(ProviderError):
    code = "provider_not_configured"
    retryable = False


class DiskSpaceError(ClipperError):
    code = "disk_full"
    retryable = True


class MissingDependencyError(ClipperError):
    code = "missing_dependency"


class CancelledError(ClipperError):
    """Raised inside a job when cancellation was requested. Not an error for the user."""
    code = "cancelled"


class InterruptedError_(ClipperError):
    """Raised when the worker is shutting down; job becomes resumable."""
    code = "interrupted"
    retryable = True
