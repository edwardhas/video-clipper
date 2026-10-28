/**
 * Mirror of backend/clipper/api/schemas.py (and the shared models it embeds
 * from backend/clipper/models.py). Field names and types must stay identical.
 *
 * Times are seconds. `src_*` = source timeline, `rel_*` = clip timeline.
 * Every URL the API returns is relative (same-origin).
 */

// ---------------------------------------------------------------- shared models (clipper/models.py)
export type Language = 'en'

export interface ProjectOptions {
  clip_count: number
  min_seconds: number
  max_seconds: number
  captions_enabled: boolean
  language: Language
}

export type CaptionSize = 'S' | 'M' | 'L'
export type CaptionColor = 'white' | 'yellow' | 'boxed'
export type CaptionPosition = 'top' | 'middle' | 'bottom'

export interface CaptionStyle {
  size: CaptionSize
  color: CaptionColor
  position: CaptionPosition
  highlight: boolean
  uppercase: boolean
}

export type FramingMode = 'auto' | 'crop' | 'fit'

export interface FramingChoice {
  mode: FramingMode
  /** Manual horizontal crop centre (0..1 of source width); null = automatic. */
  crop_x: number | null
}

export interface CueOverride {
  first_word_id: string
  last_word_id: string
  text: string
}

export interface ClipEdit {
  src_start: number
  src_end: number
  captions_enabled: boolean
  style: CaptionStyle
  word_overrides: Record<string, string>
  cue_overrides: CueOverride[]
  framing: FramingChoice
}

export interface CaptionWord {
  word_id: string
  text: string
  rel_start: number
  rel_end: number
}

export interface Cue {
  id: string
  first_word_id: string
  last_word_id: string
  text: string
  words: CaptionWord[]
  rel_start: number
  rel_end: number
  highlight: boolean
  edited: boolean
}

// ---------------------------------------------------------------- auth
export type AppMode = 'local' | 'hosted'

export interface LoginIn {
  username: string
  password: string
}

export interface SessionOut {
  authenticated: boolean
  auth_required: boolean
  mode: AppMode
  username: string | null
  csrf_token: string | null
  expires_at: number | null
}

// ---------------------------------------------------------------- system / settings
export interface StorageInfo {
  location_label: string
  data_dir_display: string
  free_bytes: number
  used_bytes: number
  reserve_bytes: number
}

export interface OpenAIInfo {
  configured: boolean
  transcribe_model: string
  select_model: string
  review_model: string
  data_sent: string[]
  transcription_note: string
}

export interface LimitsInfo {
  max_source_minutes: number
  max_upload_mb: number
  max_clips: number
  min_seconds_floor: number
  max_seconds_cap: number
}

export interface ToolsInfo {
  ffmpeg_version: string | null
  ytdlp_version: string | null
  js_runtime: string | null
  encoder: string | null
  face_detector: string | null
}

export interface WorkerInfo {
  online: boolean
  last_heartbeat_age_s: number | null
}

export interface SystemOut {
  mode: AppMode
  version: string
  public_origin: string
  storage: StorageInfo
  openai: OpenAIInfo
  limits: LimitsInfo
  tools: ToolsInfo
  worker: WorkerInfo
}

export interface DefaultsModel {
  clip_count: number
  min_seconds: number
  max_seconds: number
  captions_enabled: boolean
  caption_style: CaptionStyle
}

// ---------------------------------------------------------------- uploads / projects
export interface UploadOut {
  id: string
  filename: string
  size_bytes: number
}

export interface YoutubeSource {
  kind: 'youtube'
  url: string
}

export interface UploadSource {
  kind: 'upload'
  upload_id: string
}

export interface ProjectCreateIn {
  source: YoutubeSource | UploadSource
  /** null/omitted -> saved defaults */
  options?: ProjectOptions | null
}

export interface ReselectIn {
  options: ProjectOptions
}

export type StageKey =
  | 'import'
  | 'probe'
  | 'audio'
  | 'transcribe'
  | 'candidates'
  | 'review'
  | 'framing'
  | 'render'

export type StageState = 'pending' | 'running' | 'done' | 'failed' | 'skipped'

export interface StageOut {
  key: StageKey
  label: string
  state: StageState
  detail: string | null
  duration_s: number | null
}

export type JobState = 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled' | 'interrupted'

export interface JobOut {
  id: string
  kind: 'analyze' | 'render'
  state: JobState
  stage: string | null
  stage_detail: string | null
  /** Real step counts only, e.g. chunk 2 of 5. */
  progress_current: number | null
  progress_total: number | null
  cancel_requested: boolean
  error_code: string | null
  error_message: string | null
  retryable: boolean | null
  attempts: number
  created_at: number
  started_at: number | null
  finished_at: number | null
}

export type ProjectStatus =
  | 'queued'
  | 'running'
  | 'ready'
  | 'no_clips'
  | 'failed'
  | 'cancelled'
  | 'interrupted'

export type SourceKind = 'youtube' | 'upload'

export interface ProjectSummaryOut {
  id: string
  title: string
  source_kind: SourceKind
  source_url: string | null
  status: ProjectStatus
  status_detail: string | null
  created_at: number
  updated_at: number
  duration: number | null
  clip_count: number
  thumb_url: string | null
}

export interface MediaSummary {
  duration: number
  width: number
  height: number
  fps: number
  has_audio: boolean
}

export interface UsageOut {
  requests: number
  input_tokens: number
  output_tokens: number
  audio_seconds: number
  estimated_cost_usd: number | null
  cost_note: string
}

export interface ProjectActions {
  can_cancel: boolean
  can_resume: boolean
  can_reselect: boolean
  can_delete: boolean
}

export interface RenderInfoOut {
  width: number
  height: number
  duration: number
  size_bytes: number
  video_codec: string
  audio_codec: string | null
}

export type RenderState = 'none' | 'queued' | 'rendering' | 'ready' | 'failed' | 'stale'

export interface ClipRenderOut {
  state: RenderState
  error: string | null
  video_url: string | null
  download_url: string | null
  srt_url: string | null
  thumb_url: string | null
  render_hash: string | null
  info: RenderInfoOut | null
  stale: boolean
}

export interface FramingSummary {
  layout: 'crop' | 'fit'
  reason: string
  detector: string
}

export interface ClipOut {
  id: string
  project_id: string
  rank: number
  title: string
  reason: string
  /** 0..100 editorial heuristic (not a virality probability). */
  score: number
  score_detail: Record<string, number>
  src_start: number
  src_end: number
  duration: number
  ai_src_start: number
  ai_src_end: number
  hook: string
  topic: string
  edit: ClipEdit
  render: ClipRenderOut
  framing: FramingSummary | null
  job: JobOut | null
}

export interface ProjectOut extends ProjectSummaryOut {
  options: ProjectOptions
  media: MediaSummary | null
  stages: StageOut[]
  job: JobOut | null
  clips: ClipOut[]
  usage: UsageOut
  actions: ProjectActions
}

export interface TranscriptWordOut {
  id: string
  text: string
  /** source seconds */
  start: number
  end: number
}

export interface ClipDetailOut extends ClipOut {
  cues: Cue[]
  words: TranscriptWordOut[]
  context_start: number
  context_end: number
  source_duration: number
  source_width: number
  source_height: number
}

/** Partial update; omitted fields keep their current value. */
export interface ClipEditIn {
  src_start?: number | null
  src_end?: number | null
  captions_enabled?: boolean | null
  style?: CaptionStyle | null
  framing?: FramingChoice | null
}

export interface CaptionEditIn {
  first_word_id: string
  last_word_id: string
  text: string
}

export interface ErrorOut {
  detail: string
  code: string | null
}
