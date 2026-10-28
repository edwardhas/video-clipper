"""Provider factories: the only place that decides which AI adapters the pipeline uses.

A future provider (e.g. a Claude adapter) plugs in here by returning another object that
implements the same ``Transcriber`` / ``ClipSelector`` protocols.
"""

from __future__ import annotations

from ..config import Settings
from ..errors import ProviderNotConfiguredError
from .interfaces import ClipSelector, Transcriber
from .openai_client import NOT_CONFIGURED_MESSAGE


def _require_openai_key(settings: Settings) -> None:
    if not settings.openai_configured:
        raise ProviderNotConfiguredError(NOT_CONFIGURED_MESSAGE)


def get_transcriber(settings: Settings) -> Transcriber:
    """Speech-to-text adapter for ``settings`` (OpenAI ``whisper-1``)."""
    _require_openai_key(settings)
    from .transcribe_openai import OpenAITranscriber

    return OpenAITranscriber(settings)


def get_selector(settings: Settings) -> ClipSelector:
    """Clip selection/review adapter for ``settings`` (OpenAI Responses API)."""
    _require_openai_key(settings)
    from .selection import OpenAISelector   # imported lazily: heavier module, owned separately

    return OpenAISelector(settings)
