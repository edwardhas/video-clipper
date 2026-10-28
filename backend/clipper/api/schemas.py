"""HTTP API contract (request/response bodies).

The frontend mirrors these in ``frontend/src/api/types.ts`` – keep both in sync.
All times are seconds (floats). ``src_*`` = source timeline, ``rel_*`` = clip timeline.
URLs returned to the browser are always relative (same-origin), never absolute.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from ..models import CaptionStyle, ClipEdit, Cue, FramingChoice, ProjectOptions


class Out(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ----------------------------------------------------------------------------- auth
class LoginIn(BaseModel):
    username: str = Field(max_length=100)
    password: str = Field(max_length=1000)


class SessionOut(Out):
    authenticated: bool            # True when a valid session exists OR auth is disabled
    auth_required: bool
    mode: Literal["local", "hosted"]
    username: str | None = None
    csrf_token: str | None = None  # send as X-CSRF-Token on every mutating request
    expires_at: float | None = None


# ----------------------------------------------------------------------------- system/settings
class StorageInfo(Out):
    location_label: str            # e.g. "This computer" / "Server data volume"
    data_dir_display: str          # local: absolute path; hosted: container path (no host secrets)
    free_bytes: int
    used_bytes: int
    reserve_bytes: int


class OpenAIInfo(Out):
    configured: bool               # key present (value never returned)
    transcribe_model: str
    select_model: str
    review_model: str
    data_sent: list[str]           # human-readable disclosure lines
    transcription_note: str        # e.g. whisper-1 removal date note


class LimitsInfo(Out):
    max_source_minutes: float
    max_upload_mb: int
    max_clips: int
    min_seconds_floor: float
    max_seconds_cap: float


class ToolsInfo(Out):
    ffmpeg_version: str | None
    ytdlp_version: str | None
    js_runtime: str | None         # e.g. "deno 2.8.2"
    encoder: str | None            # encoder the worker validated, e.g. "libx264"
    face_detector: str | None      # e.g. "YuNet (OpenCV)" or None if model missing


class WorkerInfo(Out):
    online: bool
    last_heartbeat_age_s: float | None


class SystemOut(Out):
    mode: Literal["local", "hosted"]
    version: str
    public_origin: str
    storage: StorageInfo
    openai: OpenAIInfo
    limits: LimitsInfo
    tools: ToolsInfo
    worker: WorkerInfo


class DefaultsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    clip_count: int = Field(5, ge=1, le=10)
    min_seconds: float = Field(20.0, ge=5.0, le=170.0)
    max_seconds: float = Field(60.0, ge=10.0, le=180.0)
    captions_enabled: bool = True
    caption_style: CaptionStyle = Field(default_factory=CaptionStyle)


# ----------------------------------------------------------------------------- uploads/projects
class UploadOut(Out):
    id: str
    filename: str
    size_bytes: int


class YoutubeSource(BaseModel):
    kind: Literal["youtube"]
    url: str = Field(max_length=500)


class UploadSource(BaseModel):
    kind: Literal["upload"]
    upload_id: str = Field(max_length=64)


class ProjectCreateIn(BaseModel):
    source: YoutubeSource | UploadSource = Field(discriminator="kind")
    options: ProjectOptions | None = None    # None -> saved defaults


class ReselectIn(BaseModel):
    """Re-run moment selection only (transcript is reused from cache)."""
    options: ProjectOptions


StageKey = Literal["import", "probe", "audio", "transcribe", "candidates", "review", "framing", "render"]
StageState = Literal["pending", "running", "done", "failed", "skipped"]


class StageOut(Out):
    key: StageKey
    label: str
    state: StageState
    detail: str | None = None
    duration_s: float | None = None


JobState = Literal["queued", "running", "succeeded", "failed", "cancelled", "interrupted"]


class JobOut(Out):
    id: str
    kind: Literal["analyze", "render"]
    state: JobState
    stage: str | None
    stage_detail: str | None
    progress_current: int | None   # real step counts only, e.g. chunk 2 of 5
    progress_total: int | None
    cancel_requested: bool
    error_code: str | None
    error_message: str | None
    retryable: bool | None
    attempts: int
    created_at: float
    started_at: float | None
    finished_at: float | None


ProjectStatus = Literal["queued", "running", "ready", "no_clips", "failed", "cancelled", "interrupted"]


class ProjectSummaryOut(Out):
    id: str
    title: str
    source_kind: Literal["youtube", "upload"]
    source_url: str | None
    status: ProjectStatus
    status_detail: str | None
    created_at: float
    updated_at: float
    duration: float | None
    clip_count: int
    thumb_url: str | None


class MediaSummary(Out):
    duration: float
    width: int
    height: int
    fps: float
    has_audio: bool


class UsageOut(Out):
    requests: int
    input_tokens: int
    output_tokens: int
    audio_seconds: float
    estimated_cost_usd: float | None   # only when based on configured, dated rates
    cost_note: str


class ProjectActions(Out):
    can_cancel: bool
    can_resume: bool               # interrupted/failed/cancelled -> re-queue, reusing finished stages
    can_reselect: bool             # transcript exists -> re-run selection with new options
    can_delete: bool


class RenderInfoOut(Out):
    width: int
    height: int
    duration: float
    size_bytes: int
    video_codec: str
    audio_codec: str | None


class ClipRenderOut(Out):
    state: Literal["none", "queued", "rendering", "ready", "failed", "stale"]
    error: str | None
    video_url: str | None          # /media/clips/{id}/video.mp4?v={render_hash}
    download_url: str | None       # /media/clips/{id}/download.mp4
    srt_url: str | None            # /media/clips/{id}/captions.srt
    thumb_url: str | None
    render_hash: str | None
    info: RenderInfoOut | None
    stale: bool                    # current edit differs from the ready render


class FramingSummary(Out):
    layout: Literal["crop", "fit"]
    reason: str
    detector: str


class ClipOut(Out):
    id: str
    project_id: str
    rank: int
    title: str
    reason: str
    score: float                   # 0..100 editorial heuristic (not a virality probability)
    score_detail: dict[str, float]
    src_start: float               # current edit range (source seconds)
    src_end: float
    duration: float
    ai_src_start: float            # the AI's original range
    ai_src_end: float
    hook: str
    topic: str
    edit: ClipEdit
    render: ClipRenderOut
    framing: FramingSummary | None
    job: JobOut | None             # active render job, if any


class ProjectOut(ProjectSummaryOut):
    options: ProjectOptions
    media: MediaSummary | None
    stages: list[StageOut]
    job: JobOut | None             # latest analyze job
    clips: list[ClipOut]
    usage: UsageOut
    actions: ProjectActions


class TranscriptWordOut(Out):
    id: str
    text: str
    start: float                   # source seconds
    end: float


class ClipDetailOut(ClipOut):
    cues: list[Cue]                # caption cues for the current edit (clip-relative times)
    words: list[TranscriptWordOut] # transcript words around the clip (±context) for trimming
    context_start: float
    context_end: float
    source_duration: float
    source_width: int
    source_height: int


class ClipEditIn(BaseModel):
    """Partial update; omitted fields keep their current value."""
    model_config = ConfigDict(extra="forbid")
    src_start: float | None = None
    src_end: float | None = None
    captions_enabled: bool | None = None
    style: CaptionStyle | None = None
    framing: FramingChoice | None = None


class CaptionEditIn(BaseModel):
    """Correct the text of one cue (identified by its word range)."""
    model_config = ConfigDict(extra="forbid")
    first_word_id: str = Field(max_length=16)
    last_word_id: str = Field(max_length=16)
    text: str = Field(max_length=300)


class ErrorOut(Out):
    detail: str
    code: str | None = None
