"""Domain models shared by the worker, the pipeline modules and the API.

Time conventions (important):
* ``src_*`` / ``start``/``end`` on transcript objects are **source seconds**,
  measured on the imported media timeline (0 = first decoded sample after any
  container start offset is normalised away).
* ``rel_*`` values are **clip-relative seconds** (0 = first frame of the rendered
  clip). Convert with ``rel = src - clip.src_start``. Never mix the two.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

TRANSCRIPT_VERSION = 2   # 2: words hallucinated inside detected silences are dropped (merge_chunks)

# Clip length limits enforced everywhere (AI selection, API trim validation; mirrored in
# frontend/src/lib/trim.ts). Seconds.
MIN_CLIP_SECONDS = 3.0
MAX_CLIP_SECONDS = 180.0


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ----------------------------------------------------------------------------- project
class ProjectOptions(_Model):
    clip_count: int = Field(5, ge=1, le=10)
    min_seconds: float = Field(20.0, ge=5.0, le=170.0)
    max_seconds: float = Field(60.0, ge=10.0, le=180.0)
    captions_enabled: bool = True
    language: Literal["en"] = "en"

    @model_validator(mode="after")
    def _check_range(self) -> "ProjectOptions":
        if self.max_seconds - self.min_seconds < 5.0:
            raise ValueError("max_seconds must be at least 5 seconds above min_seconds")
        return self


class MediaInfo(_Model):
    duration: float                  # seconds (container/stream duration after normalisation)
    size_bytes: int
    container: str                   # ffprobe format_name
    has_video: bool
    has_audio: bool
    width: int = 0                   # display width (rotation applied)
    height: int = 0                  # display height (rotation applied)
    coded_width: int = 0
    coded_height: int = 0
    rotation: int = 0                # degrees clockwise to display (0/90/180/270)
    fps: float = 0.0                 # average frame rate of the video stream
    video_codec: str | None = None
    audio_codec: str | None = None
    audio_channels: int | None = None
    audio_sample_rate: int | None = None
    start_time: float = 0.0          # container start_time (informational)
    pix_fmt: str | None = None


# ----------------------------------------------------------------------------- transcript
class Word(_Model):
    id: str                          # "w000001" – stable, in time order
    text: str                        # display text (punctuation/casing aligned from segment text)
    raw: str                         # token as returned by the provider
    start: float                     # source seconds
    end: float                       # source seconds
    segment_id: str
    chunk_index: int


class Segment(_Model):
    id: str                          # "s00001"
    start: float
    end: float
    text: str
    word_start: int                  # index into Transcript.words (inclusive); -1 when no words
    word_end: int                    # index into Transcript.words (inclusive); -1 when no words
    chunk_index: int
    no_speech_prob: float | None = None
    avg_logprob: float | None = None


class Unit(_Model):
    """Sentence-like unit used for clip selection (the LLM chooses unit ranges)."""
    id: str                          # "u0001"
    start: float                     # = words[word_start].start
    end: float                       # = words[word_end].end
    text: str
    word_start: int
    word_end: int


class TranscriptChunk(_Model):
    index: int
    start: float                     # source seconds where the chunk audio begins (incl. overlap)
    end: float                       # source seconds where the chunk audio ends (incl. overlap)
    keep_start: float                # words whose midpoint is in [keep_start, keep_end) are kept
    keep_end: float
    model: str
    request_id: str | None = None
    raw_path: str | None = None      # relative path of the stored raw provider response
    cached: bool = False


class Transcript(_Model):
    version: int = TRANSCRIPT_VERSION
    provider: str
    model: str
    language: str
    duration: float
    words: list[Word]
    segments: list[Segment]
    units: list[Unit]
    chunks: list[TranscriptChunk]
    fixes: dict[str, int] = Field(default_factory=dict)   # counts of normalisation fixes applied
    created_at: float


# ----------------------------------------------------------------------------- selection
class CandidateScores(_Model):
    hook: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    value: int = Field(ge=1, le=5)
    emotion: int = Field(ge=1, le=5)


class Candidate(_Model):
    id: str                          # "c001"
    window_index: int
    start_unit_id: str
    end_unit_id: str
    src_start: float                 # derived from transcript, never from the model
    src_end: float
    title: str
    hook: str
    payoff: str
    reason: str
    topic: str
    scores: CandidateScores
    model_score: float               # 0..100 derived from scores in code


class ReviewDecision(_Model):
    candidate_id: str
    keep: bool
    final_score: float               # 0..100
    title: str
    reason: str
    opening: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    interest: int = Field(ge=1, le=5)
    visual: int = Field(ge=1, le=5)
    duplicate_of: str | None = None
    start_unit_id: str
    end_unit_id: str


class ClipSelection(_Model):
    """The AI's original choice for a clip (immutable after selection)."""
    candidate_id: str
    start_unit_id: str
    end_unit_id: str
    src_start: float
    src_end: float
    title: str
    hook: str
    reason: str
    topic: str
    score: float
    sub_scores: dict[str, float] = Field(default_factory=dict)
    select_model: str
    review_model: str
    prompt_version: str
    # Caption style the clip was first rendered with (restored by "Restore original edit").
    original_style: "CaptionStyle | None" = None


# ----------------------------------------------------------------------------- editing
CaptionSize = Literal["S", "M", "L"]
CaptionColor = Literal["white", "yellow", "boxed"]
CaptionPosition = Literal["top", "middle", "bottom"]


class CaptionStyle(_Model):
    size: CaptionSize = "M"
    color: CaptionColor = "white"
    position: CaptionPosition = "bottom"
    highlight: bool = True           # highlight the spoken word (only where real word timings exist)
    uppercase: bool = False


class FramingChoice(_Model):
    mode: Literal["auto", "crop", "fit"] = "auto"
    # Manual horizontal crop centre (0..1 of source width) for mode == "crop".
    # None = follow the automatic face track / centre.
    crop_x: float | None = Field(default=None, ge=0.0, le=1.0)


class CueOverride(_Model):
    """User text for a caption cue whose token count differs from the spoken words.

    The range is rendered as one cue with the user's text and *no* word highlight,
    because per-word timings for the new tokens are unknown (never fabricated).
    """
    first_word_id: str
    last_word_id: str
    text: str = Field(max_length=300)


class ClipEdit(_Model):
    src_start: float
    src_end: float
    captions_enabled: bool = True
    style: CaptionStyle = Field(default_factory=CaptionStyle)
    word_overrides: dict[str, str] = Field(default_factory=dict)   # word_id -> corrected text
    cue_overrides: list[CueOverride] = Field(default_factory=list)
    framing: FramingChoice = Field(default_factory=FramingChoice)

    @model_validator(mode="after")
    def _check(self) -> "ClipEdit":
        if self.src_end <= self.src_start:
            raise ValueError("clip end must be after clip start")
        return self


# ----------------------------------------------------------------------------- framing
class FramingPlan(_Model):
    version: int
    layout: Literal["crop", "fit"]
    src_start: float                 # analysed source range
    src_end: float
    src_w: int                       # display dims of source
    src_h: int
    crop_w: int = 0                  # crop window size in source pixels (layout == crop)
    crop_h: int = 0
    crop_y: int = 0
    # (rel_t, x_left_px) keyframes, piecewise-linear, clip-relative seconds.
    keyframes: list[tuple[float, float]] = Field(default_factory=list)
    # Crop plans only: clip-relative [start, end) seconds (same time base as keyframes) shown in the
    # full-frame "fit" layout instead of the crop (shots without a single clear face, e.g. slides).
    fit_spans: list[tuple[float, float]] = Field(default_factory=list)
    reason: str                      # user-facing explanation of the choice
    detector: str                    # e.g. "yunet-2023mar" or "none"
    stats: dict[str, float] = Field(default_factory=dict)


# ----------------------------------------------------------------------------- captions
class CaptionWord(_Model):
    word_id: str
    text: str
    rel_start: float
    rel_end: float


class Cue(_Model):
    id: str                          # first word id of the cue
    first_word_id: str
    last_word_id: str
    text: str
    words: list[CaptionWord]
    rel_start: float
    rel_end: float
    highlight: bool                  # False when text overridden with different token count
    edited: bool = False


ClipSelection.model_rebuild()
