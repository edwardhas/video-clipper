"""Provider-neutral interfaces for transcription and clip selection.

The OpenAI adapters implement these; a future Claude (or other) adapter only has
to implement the same protocols — the pipeline, rendering and storage code do not
change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..models import Candidate, ProjectOptions, ReviewDecision, Transcript
from ..procs import CancelToken


@dataclass
class RawWord:
    text: str
    start: float          # chunk-relative seconds
    end: float            # chunk-relative seconds


@dataclass
class RawSegment:
    text: str
    start: float          # chunk-relative seconds
    end: float
    no_speech_prob: float | None = None
    avg_logprob: float | None = None


@dataclass
class RawChunkResult:
    words: list[RawWord]
    segments: list[RawSegment]
    language: str | None = None
    duration: float | None = None
    request_id: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)   # sanitized provider response


@runtime_checkable
class UsageSink(Protocol):
    def record(self, *, stage: str, provider: str, model: str, request_id: str | None, status: str,
               input_tokens: int | None = None, output_tokens: int | None = None,
               reasoning_tokens: int | None = None, cached_tokens: int | None = None,
               audio_seconds: float | None = None, duration_s: float | None = None,
               error_code: str | None = None) -> None: ...


class NullUsageSink:
    def record(self, **_: Any) -> None:  # pragma: no cover - trivial
        return None


@runtime_checkable
class JsonCache(Protocol):
    """Durable key -> JSON value store for completed paid sub-requests (selection windows, review batches).

    ``get`` returns ``None`` on a miss. Selectors build keys covering every request input and
    ``put`` each result as soon as it is paid for — before honouring a cancel — so a resume or
    retry never repeats a completed API call. ``put(key, None)`` clears an entry (it reads back
    as a miss); selectors use it so a retry of a window/batch that failed asks afresh.
    """

    def get(self, key: str) -> Any | None: ...

    def put(self, key: str, value: Any) -> None: ...


@runtime_checkable
class Transcriber(Protocol):
    provider: str
    model: str

    def transcribe_chunk(self, audio_path: Path, *, language: str, cancel: CancelToken,
                         usage: UsageSink | None = None, audio_seconds: float | None = None) -> RawChunkResult: ...


@runtime_checkable
class ClipSelector(Protocol):
    provider: str
    select_model: str
    review_model: str

    def find_candidates(self, transcript: Transcript, options: ProjectOptions, *, cancel: CancelToken,
                        usage: UsageSink, cache: JsonCache | None = None) -> list[Candidate]: ...

    def review(self, transcript: Transcript, candidates: list[Candidate],
               frames: dict[str, list[tuple[float, Path]]], options: ProjectOptions, *,
               cancel: CancelToken, usage: UsageSink, cache: JsonCache | None = None) -> list[ReviewDecision]: ...
