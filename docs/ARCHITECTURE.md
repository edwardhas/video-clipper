# Local Clipper — architecture and module contracts

This is the binding contract between modules. If you change a signature here,
update every caller. Shared code that already exists and must be **used, not
re-implemented**: `clipper/config.py`, `clipper/errors.py`, `clipper/procs.py`,
`clipper/storage.py`, `clipper/db.py`, `clipper/models.py`,
`clipper/jobs/queue.py`, `clipper/api/schemas.py`.

## Processes

```
browser ──HTTP(S)──> [reverse proxy, hosted only] ──> API (uvicorn/FastAPI, clipper.api.app:create_app)
                                                     │  serves frontend/dist + /api + /media
                                                     └─ SQLite (data/clipper.db, WAL) <── worker (python -m clipper.worker)
                                                                                            runs one job at a time
                                                                                            ffmpeg / yt-dlp / OpenAI
```

* `python -m clipper.supervisor` starts API + worker as child processes (one command), prints the URL,
  forwards SIGINT/SIGTERM, and stops both cleanly. `scripts/start.sh` wraps it.
* The worker never depends on an HTTP request: closing the browser does not stop jobs.
* Frontend calls are relative (`/api/...`, `/media/...`) — never hard-coded hosts.

## Time conventions

* `src_*`: seconds on the imported source timeline. Transcript words/segments/units use source time.
* `rel_*`: seconds on the rendered clip timeline (`rel = src - edit.src_start`).
* Never pass one where the other is expected; name variables accordingly.

## Data layout

See `clipper/storage.py` docstring. DB stores *relative* paths; resolve with `Storage.resolve`.
Original user files are never referenced: uploads are streamed into `uploads/<id>/` and then
moved into the project's `source/` dir. Deletion only touches app-managed directories.

## Analyze pipeline (job kind `analyze`) — `clipper/jobs/pipeline.py`

Stages (keys are fixed; labels in parentheses are shown in the UI):

| key | label | work | cache key inputs |
|-----|-------|------|------------------|
| import | Import source | YouTube: metadata check (duration limit, live, availability) then download ≤1080p. Upload: move from uploads/ into source/. sha256 of the file. | youtube_id / upload_id |
| probe | Inspect media | ffprobe → `MediaInfo`; reject corrupt, no streams, too long. No audio → `NoSpeechError`. | source_sha256 + PROBE_VERSION |
| audio | Extract audio | 16 kHz mono WAV for analysis, silencedetect, chunk plan, FLAC chunks | source_sha256 + AUDIO_VERSION + chunk params |
| transcribe | Transcribe speech | per-chunk provider call (cached per chunk in `cache/transcribe-chunk/`), merge with offsets/overlap reconciliation, punctuation alignment, units, validation. Insufficient speech → project `no_clips` with explanation (not a failure). | audio key + transcribe model + language + TRANSCRIPT_VERSION |
| candidates | Find moments | overlapping windows → structured candidates by unit IDs → validate → dedupe | transcript key + select model + PROMPT_VERSION + options(min,max,count) |
| review | Review & rank | shortlist → frames at candidate times → structured review with images → validate → global dedupe → top N above threshold | candidates key + review model + REVIEW_PROMPT_VERSION |
| framing | Frame clips | per selected clip: face analysis → `FramingPlan` | source + range + FRAMING_VERSION |
| render | Render clips | per clip: captions + ffmpeg render + validation + atomic finalize | render hash |

Each stage records a row in `stages` (state, cache_key, artifact JSON, timings). On resume a stage
is skipped when its row is `done` and its cache_key matches the recomputed key and its artifacts
exist. Expensive provider results are additionally cached content-addressed under
`data/cache/<namespace>/<key>.json` so another project with the same source reuses them.

Clips are published (inserted into `clips`) inside `queue.commit_success(... publish=...)` — or, for
the render stage, each clip's render is published with a guarded update that checks the job is still
running and not cancelled. A cancelled job never flips to succeeded and never publishes.

Project status: `queued → running → ready | no_clips | failed | cancelled | interrupted`.
`status_detail` carries the user-facing explanation (e.g. "Only 40 words of speech were found…").

## Render job (job kind `render`) — `clipper/jobs/render_job.py`

Triggered by the editor. Inputs: clip row (edit JSON). Steps: if the edit's range differs from the
stored framing plan's range → re-run framing analysis (local, no API). Build cues from the stored
transcript (no API). Compute render hash; if a ready render with the same hash exists → done.
Else render to temp, validate, atomically move to `clips/<clip_id>/<hash>.mp4` (+ `.srt`, thumb),
publish with a guarded update, delete the previous render files. **Never** calls transcription or
selection. Tests assert the provider is not invoked.

## Module contracts

### `clipper/media/probe.py`
```python
PROBE_VERSION: int
def probe(path: Path, *, settings: Settings, cancel: CancelToken | None = None) -> MediaInfo
    # ffprobe -show_format -show_streams -of json (arg list). Display w/h account for rotation
    # (stream side_data_list displaymatrix "rotation" and legacy tags.rotate). fps from avg_frame_rate.
    # Raises MediaError("…not a readable video…") for corrupt input / no streams.
def tool_version(binary: str) -> str | None        # first line of `<bin> -version`
```

### `clipper/media/audio.py`
```python
AUDIO_VERSION: int
@dataclass(frozen=True) class ChunkPlan: index:int; start:float; end:float; keep_start:float; keep_end:float
def extract_analysis_audio(src: Path, out_wav: Path, *, settings, cancel) -> float      # 16k mono s16 WAV; returns duration
    # time 0 = container start (-copyts + asetpts), the same origin -ss seeking uses, so MPEG-TS/PS with
    # late-starting audio stay aligned with renders (AUDIO_VERSION 2).
def detect_silences(wav: Path, *, settings, cancel, noise_db: float = -35.0, min_silence: float = 0.35) -> list[tuple[float, float]]
def plan_chunks(duration: float, silences: list[tuple[float, float]], *, target: float, overlap: float,
                search: float = 60.0) -> list[ChunkPlan]   # PURE. cuts at silence midpoints near multiples of target;
                # chunk i audio = [cut_{i-1} - overlap, cut_i + overlap] clamped; keep = [cut_{i-1}, cut_i);
                # first keep_start = 0, last keep_end = +inf represented as duration + 1.
def encode_chunk(wav: Path, out_path: Path, start: float, end: float, *, settings, cancel) -> None  # FLAC, sample-accurate
def mean_volume_db(wav: Path, *, settings, cancel) -> float | None   # volumedetect, for "silent audio" detection
```

### `clipper/media/frames.py`
```python
def extract_frame(src: Path, t: float, out_jpg: Path, *, max_width: int = 640, settings, cancel=None) -> Path
def extract_frames(src: Path, times: list[float], out_dir: Path, *, max_width: int = 512, settings, cancel=None) -> list[Path]
def sample_frames(src: Path, src_start: float, src_end: float, *, fps: float, width: int, settings, cancel,
                  max_frames: int = 400) -> tuple[list["np.ndarray"], list[float]]   # BGR arrays + source times
```

### `clipper/media/youtube.py`
```python
def parse_youtube_url(url: str) -> tuple[str, str]   # (video_id, canonical "https://www.youtube.com/watch?v=ID"); InputError otherwise.
    # accept youtube.com/watch?v=, youtu.be/, /shorts/, /live/, /embed/, m./music. hosts; reject playlists-only, other hosts.
def fetch_metadata(url: str, *, settings, cancel) -> dict   # yt-dlp -J --no-playlist; returns {id,title,duration,is_live,live_status,availability,...}
def download(url: str, out_dir: Path, *, settings, cancel, progress_cb=None) -> Path   # ≤1080p, merged; returns media file path
def classify_error(stderr: str) -> ClipperError       # SourceUnavailableError with friendly text + "upload the file instead"
def ytdlp_version() -> str | None
```
Never use cookies / --cookies-from-browser. JS runtime per `settings.ytdlp_js_runtime`.

### `clipper/media/encoders.py`
```python
@dataclass class EncoderChoice: name: str; video_args: list[str]; hardware: bool; note: str
def choose_encoder(settings, *, cancel=None) -> EncoderChoice   # tests h264_videotoolbox (macOS) with a tiny
    # real encode+decode when settings.hw_encode == "auto"; falls back to libx264. Cached per process.
def cpu_encoder(settings) -> EncoderChoice
```

### `clipper/ai/interfaces.py`
```python
@dataclass class RawWord: text: str; start: float; end: float            # chunk-relative seconds
@dataclass class RawSegment: text: str; start: float; end: float; no_speech_prob: float|None; avg_logprob: float|None
@dataclass class RawChunkResult: words: list[RawWord]; segments: list[RawSegment]; language: str|None;
                               duration: float|None; request_id: str|None; raw: dict   # raw = sanitized provider JSON
class Transcriber(Protocol):
    provider: str; model: str
    def transcribe_chunk(self, audio_path: Path, *, language: str, cancel: CancelToken) -> RawChunkResult
class ClipSelector(Protocol):
    provider: str; select_model: str; review_model: str
    def find_candidates(self, transcript: Transcript, options: ProjectOptions, *, cancel, usage: UsageSink,
                        cache: JsonCache | None = None) -> list[Candidate]   # cache: per-window paid results
    def review(self, transcript: Transcript, candidates: list[Candidate], frames: dict[str, list[tuple[float, Path]]],
               options: ProjectOptions, *, cancel, usage: UsageSink,
               cache: JsonCache | None = None) -> list[ReviewDecision]   # cache: per-batch paid results
class JsonCache(Protocol):   # get(key) -> Any | None; put(key, value); keys are sha256 hex from the selector
class UsageSink(Protocol):
    def record(self, *, stage: str, provider: str, model: str, request_id: str|None, status: str,
               input_tokens: int|None = None, output_tokens: int|None = None, reasoning_tokens: int|None = None,
               cached_tokens: int|None = None, audio_seconds: float|None = None, duration_s: float|None = None,
               error_code: str|None = None) -> None
```
Provider factories: `clipper/ai/factory.py`: `get_transcriber(settings) -> Transcriber`, `get_selector(settings) -> ClipSelector`
(raise `ProviderNotConfiguredError` with setup instructions when `OPENAI_API_KEY` is missing).
A future Claude adapter implements the same protocols; nothing downstream changes.

### `clipper/ai/openai_client.py`
```python
def make_client(settings) -> openai.OpenAI          # timeout + max_retries=0 (we retry ourselves)
def call_openai(fn: Callable[[], T], *, what: str, cancel: CancelToken, attempts: int = 4) -> T
    # bounded exponential backoff with jitter for RateLimitError / APIConnectionError / APITimeoutError /
    # InternalServerError (5xx); AuthenticationError -> ProviderAuthError; PermissionDeniedError/NotFoundError
    # (model access) -> ProviderError(non-retryable, names the model); BadRequestError -> ProviderError
    # (non-retryable). Checks cancel before each attempt and while sleeping. Never logs the key.
```

### `clipper/ai/transcript.py` (pure, heavily unit-tested)
```python
def merge_chunks(results: list[tuple[ChunkPlan, RawChunkResult]], *, duration: float, provider: str, model: str,
                 language: str, chunk_meta: list[TranscriptChunk]) -> Transcript
    # offset chunk-relative times by plan.start; keep a word iff its midpoint ∈ [keep_start, keep_end);
    # same rule for segments; clamp to [0, duration]; enforce monotonic non-overlapping word times
    # (count fixes in Transcript.fixes); drop empty words; align punctuation from segment text onto words;
    # assign ids w000001.., s00001.., build units; never invent timings.
def align_punctuation(raw_words: list[str], segment_text: str) -> list[str]
def build_units(words: list[Word], *, max_words: int = 45, pause_break: float = 1.2) -> list[Unit]
def validate_transcript(t: Transcript) -> None      # ValueError on non-monotonic/out-of-bounds/id issues
def words_between(t: Transcript, src_start: float, src_end: float) -> list[Word]   # by midpoint
def unit_index(t: Transcript) -> dict[str, int]
def range_for_units(t: Transcript, start_unit_id: str, end_unit_id: str, *, pad: float = 0.25) -> tuple[float, float]
    # start = first word start - min(pad, half gap to previous word); end likewise; clamped to [0, duration]
def speech_stats(t: Transcript) -> dict   # word_count, speech_seconds, coverage, words_per_minute
def insufficient_speech_reason(t: Transcript, media_duration: float) -> str | None
```

### `clipper/ai/transcribe_openai.py`
`OpenAITranscriber(settings, client=None)` implementing `Transcriber` with `whisper-1`,
`response_format="verbose_json"`, `timestamp_granularities=["word","segment"]`, `language`.
Sanitize raw JSON (it never contains the key; still drop any header-like fields). Respect the 25 MB
upload limit (assert file size before upload). Model ID from settings; document whisper-1 removal date.

### `clipper/ai/selection.py` + `clipper/ai/prompts.py`
```python
PROMPT_VERSION: str; REVIEW_PROMPT_VERSION: str
@dataclass class Window: index:int; unit_start:int; unit_end:int; context_before:int; context_after:int  # unit indices
def build_windows(t: Transcript, *, window_seconds=900.0, overlap_seconds=120.0, context_seconds=60.0) -> list[Window]
def validate_candidates(raw: list[dict], window: Window, t: Transcript, options: ProjectOptions) -> tuple[list[Candidate], list[tuple[dict, str]]]
def dedupe_candidates(c: list[Candidate], *, overlap_ratio: float = 0.5) -> list[Candidate]
def shortlist(c: list[Candidate], options: ProjectOptions) -> list[Candidate]
def apply_review(cands: list[Candidate], decisions: list[ReviewDecision], t: Transcript, options: ProjectOptions,
                 *, min_score: float = 55.0) -> list[ClipSelection]    # validates ids/ranges, dedupes, ranks, ≤ clip_count
class OpenAISelector(ClipSelector)   # Responses API + Structured Outputs (strict JSON schema); refusal /
                                     # incomplete / invalid → bounded retry then explicit ProviderError.
```
Rules: model output references **existing unit IDs only**; times are always derived in code.
Durations must fall in `[max(min_seconds*0.85, 3), min(max_seconds*1.15, 180)]` after snapping
(`MIN_CLIP_SECONDS`/`MAX_CLIP_SECONDS` in `models.py`, shared with the API and editor). Reject ad reads,
housekeeping ("subscribe", sponsor), context-dependent fragments, sensational titles not supported by
the transcript, near-duplicates. Transcript text is untrusted data, wrapped in delimiters; the model has
no tools; instructions inside media text are ignored.

### `clipper/framing/` (`analyze.py`, `faces.py`)
```python
FRAMING_VERSION: int
def analyze_framing(src: Path, media: MediaInfo, src_start: float, src_end: float, *, settings, cancel) -> FramingPlan
    # YuNet (cv2.FaceDetectorYN, model in clipper/assets/models/) on frames sampled at ~4 fps (bounded);
    # scene cuts by histogram distance; per shot classify single/multi/none; single dominant face for most
    # of the clip → layout "crop" with smoothed, dead-zoned, per-shot-reset keyframes; otherwise "fit".
    # Model missing → "fit"/centre crop with reason "face detector unavailable".
def plan_for_edit(auto: FramingPlan | None, choice: FramingChoice, media: MediaInfo, src_start: float, src_end: float) -> FramingPlan
def crop_x_expr(plan: FramingPlan) -> str     # ffmpeg expression of t (piecewise linear, clamped), ≤ ~60 segments
def detector_available() -> bool
```
Face detection ≠ active speaker detection; reasons must say "single clear face" / "multiple people"
and never claim speaker tracking.

### `clipper/render/captions.py`
```python
def build_cues(words: list[Word], edit: ClipEdit, *, max_chars: int = 32, max_lines: int = 2,
               max_cue_seconds: float = 3.0, pause_break: float = 0.6) -> list[Cue]
    # words with midpoint in [edit.src_start, edit.src_end); rel times = src - edit.src_start, clamped to
    # [0, duration]; word_overrides replace word text (timing kept); cue_overrides become their own cue
    # with highlight=False; never fabricate per-word timing.
def apply_caption_edit(edit: ClipEdit, words: list[Word], first_word_id: str, last_word_id: str, text: str) -> ClipEdit
    # same token count → word_overrides; otherwise a CueOverride; empty text → hide those words (override "")
def to_srt(cues: list[Cue]) -> str
```

### `clipper/render/caption_image.py`
```python
def render_caption_track(cues: list[Cue], style: CaptionStyle, out_dir: Path, *, duration: float,
                         width: int = 1080, height: int = 1920) -> CaptionTrack | None
    # Pillow + bundled OFL font (clipper/assets/fonts). One RGBA PNG per visual state (cue × highlighted
    # word). Writes ffconcat file with exact durations starting at t=0 (transparent gaps) and returns
    # CaptionTrack(concat_path, x, y, w, h). Safe margins keep text out of TikTok UI zones.
```

### `clipper/render/renderer.py`
```python
RENDER_VERSION: int
def render_hash(*, source_sha256: str, edit: ClipEdit, plan: FramingPlan, cues: list[Cue], encoder: str) -> str
def render_clip(*, src: Path, media: MediaInfo, edit: ClipEdit, plan: FramingPlan, cues: list[Cue],
                out_mp4: Path, work_dir: Path, settings, cancel, encoder: EncoderChoice) -> dict
    # 1080x1920 H.264 yuv420p (limited range; full-range input converted; HLG/PQ HDR tone-mapped to BT.709)
    # + AAC 48k stereo, +faststart; fast seek (-ss before -i, -noaccurate_seek) then fps=…:start_time=0 and
    # atrim=start=0 so VFR sources start with the frame on screen; ≥23 fps output (30 below); last frame held;
    # short audio fades at cut points, captions overlaid from the PNG track. Writes to a temp path,
    # validates (ffprobe streams/dims/duration + full decode), then atomically renames. Returns render_info.
def validate_output(path: Path, *, expected_duration: float, settings) -> dict   # raises MediaError
def make_thumbnail(mp4: Path, out_jpg: Path, *, t: float, settings) -> None
```

### `clipper/jobs/pipeline.py`, `clipper/jobs/render_job.py`, `clipper/worker.py`, `clipper/supervisor.py`
Worker loop: register in `workers`, `recover_interrupted` at startup AND every 30 s from the heartbeat thread
(a stored worker with our own PID on the same host that registered before this process started counts as dead,
which covers container restarts that reuse PIDs), choose encoder once, then
`claim_next` → run → `commit_success`/`finish`. Heartbeat thread every 5 s. CancelToken wired to
`queue.is_cancel_requested`. SIGTERM: set `cancel.shutdown`, current job → `interrupted`.
Render jobs get priority 10, analyze jobs 0.

### `clipper/api/` (FastAPI)
Routes (all JSON unless noted; all except health/auth require a session when auth is enabled):

| method | path | notes |
|--------|------|-------|
| GET | /api/health | `{"status":"ok"}` only |
| GET | /api/auth/session | `SessionOut` |
| POST | /api/auth/login | `LoginIn` → `SessionOut` + cookie; throttled; Origin checked |
| POST | /api/auth/logout | clears session |
| GET | /api/system | `SystemOut` |
| GET/PUT | /api/settings | `DefaultsModel` |
| POST | /api/uploads | raw streamed body (`Content-Type: application/octet-stream`, header `X-Filename`), size-limited → `UploadOut` |
| GET/POST | /api/projects | list `ProjectSummaryOut[]` / create `ProjectCreateIn` → `ProjectOut` |
| GET/DELETE | /api/projects/{id} | `ProjectOut` / delete project's app-managed data |
| POST | /api/projects/{id}/cancel, /resume, /reselect, /cleanup | job controls; cleanup removes work files only |
| GET | /api/clips/{id} | `ClipDetailOut` |
| PATCH | /api/clips/{id} | `ClipEditIn` → `ClipDetailOut` (marks render stale; does not render) |
| POST | /api/clips/{id}/captions | `CaptionEditIn` → `ClipDetailOut` |
| POST | /api/clips/{id}/captions/reset | drop caption overrides |
| POST | /api/clips/{id}/reset | restore AI edit |
| POST | /api/clips/{id}/render | enqueue render job (no-op if hash unchanged) → `ClipDetailOut` |
| DELETE | /api/clips/{id} | remove clip + its renders (409 while an analysis job is queued/running) |
| GET | /media/clips/{id}/video.mp4 | inline, Range support |
| GET | /media/clips/{id}/download.mp4 | attachment |
| GET | /media/clips/{id}/captions.srt | attachment |
| GET | /media/clips/{id}/thumb.jpg, /media/projects/{id}/thumb.jpg | images |
| GET | /media/projects/{id}/frame.jpg?t= | source frame (for framing UI), cached |
| GET | /media/projects/{id}/export.zip?clips=a,b&srt=1 | ZIP of selected ready clips |

Security middleware: Host header allow-list (DNS-rebinding defence), Origin/Referer check +
`X-CSRF-Token` (session-bound; also issued when auth is disabled) on every non-GET request,
no CORS by default, security headers (CSP default-src 'self'; media-src 'self' blob:; img-src 'self' data: blob:;
frame-ancestors 'none'; X-Content-Type-Options nosniff; Referrer-Policy same-origin).
Cookies: `clipper_session`, HttpOnly, SameSite=Lax, host-only (no Domain), Secure under HTTPS.
Errors never include absolute paths, stack traces or secrets.

### Frontend (`frontend/`)
Vue 3 + TS + vue-router. Views: Login, Import (home), Projects, Project (status + results), ClipEditor,
Settings. `src/api/client.ts` wraps fetch with relative URLs, CSRF header, 401 → login redirect.
Real progress only (upload bytes via XHR, job stage + step counts); no invented percentages.
Palette: primary `#591E4A`, amber accent `#F2A33A` (text on amber must be dark), background `#F7F1F4`.

## Artifact locations (relative to `data/projects/<project_id>/`)

| artifact | path |
|----------|------|
| imported media | `source/source.<ext>` |
| project thumbnail | `source/thumb.jpg` |
| analysis audio (16 kHz mono WAV) | `work/audio/analysis.wav` |
| transcription chunks | `work/audio/chunk_000.flac` … |
| raw provider responses (sanitised) | `work/transcribe_raw/chunk_000.json` … |
| normalised transcript (`Transcript` JSON) | `work/transcript.json` |
| candidates / review results | `work/candidates.json`, `work/review.json` |
| review frames | `work/frames/<candidate_id>_<n>.jpg` |
| render scratch (caption PNGs, ffconcat) | `work/render/<clip_id>/…` (deleted after a successful render) |
| rendered clip | `clips/<clip_id>/<render_hash>.mp4`, `.srt`, `.json` (render info), `<render_hash>.jpg` (thumbnail) |

Global content-addressed caches: `data/cache/transcribe-chunk/<key>.json` (RawChunkResult as JSON),
`data/cache/candidates-window/<key>.json` and `data/cache/review-batch/<key>.json` (each paid selection/review
request, written before a cancel is honoured),
`data/cache/candidates/<key>.json`, `data/cache/review/<key>.json`.

## Shared helpers owned by specific modules
* `clipper/jobs/usage.py`: `DbUsageSink(db_path, project_id, job_id)` implements `UsageSink`.
* `clipper/ai/pricing.py`: dated list prices used only for clearly-labelled estimates.
* `clipper/media/encoders.py`: `EncoderChoice`, `choose_encoder`, `cpu_encoder` (already implemented).
* `clipper/procs.run(..., stdout_bytes_cb=...)` streams binary stdout (rawvideo frame sampling).

## Revisions after the defect review (2026-10-07)

A seven-dimension review (50 findings, 48 independently reproduced) led to these contract changes:

* `ClipSelection.original_style` stores the caption style a clip was first rendered with;
  `POST /api/clips/{id}/reset` restores it (older clips fall back to the saved defaults).
* `build_cues(..., include_hidden=True)` (editor detail only) returns hidden captions as edited cues with
  empty text so they can be restored; render, SRT and render hashes use the default `False`.
* `merge_chunks` recovers a seam word that neither chunk owned (opposite-side midpoints) and treats copies of
  one word heard by both chunks as a single word.
* Reselect jobs carry `previous_options`; a cancelled/failed/interrupted reselect restores them.
* Upload import is idempotent (an upload already moved into `source/` is reused, never deleted).
* `probe`: display size applies the sample aspect ratio; WebM/MKV without duration metadata are measured from
  packet timestamps (PROBE_VERSION 2). FRAMING_VERSION 2, RENDER_VERSION 2, AUDIO_VERSION 2.
* Caption PNG timeline uses `option framerate 1000` per concat entry (switches are not quantised to 40 ms).
* Login throttling is atomic (attempt reserved inside `BEGIN IMMEDIATE` before the password check).
* `auth=auto` turns sign-in on whenever any configured host/origin is not loopback.
* Deletions never follow symlinks; tar restores drop links.

## Revisions after the second review (2026-10-07)

* **Word ids** are derived from source start times (`w` + 9-digit milliseconds, `-2`/`-3` suffix for words
  starting in the same millisecond), so a re-merge that drops a word never shifts the ids that caption
  corrections refer to. `validate_transcript` checks ids against `word_ids_for(starts)`.
* **Silence hallucinations**: the audio stage stores detected silences; `merge_chunks(..., silences=...)` drops a
  word only when its whole span lies inside a silence AND its segment has `no_speech_prob >= 0.5`
  (whisper-1 returns e.g. "you" for digital silence). `TRANSCRIPT_VERSION` 2.
* **Chunk transcription cache** is keyed by the chunk's PCM content (`pcm_sha256`) + model + language + format,
  so version bumps that do not change the audio never re-bill; the audio artifact carries `silences` and
  per-chunk `pcm_sha256`.
* **Per-shot framing**: `FramingPlan.fit_spans` (clip-relative `[start, end)`, auto crop plans only). Shots
  without a single clear face lasting ≥ 1 s show the full frame (blurred background) inside a face-crop clip; a
  clip is cropped when a single face covers ≥ 65 % of the clip or of the time outside fit spans.
  Manual "Face crop" / "Full frame" have no spans. `FRAMING_VERSION` 3, `RENDER_VERSION` 3; editor re-renders
  re-analyse plans from older framing versions.
* Captions/highlights appear on the first output frame at or after their start (`settb=AVTB` on the overlay's
  main input). Output frame rates near a standard rate are tagged with the exact rational (e.g. 30000/1001).
* Reselect that finds no new clips (previous clips kept) restores the previous options.
* `client_ip` honours `CLIPPER_TRUSTED_PROXIES` in local mode too (same-host proxy/tunnel setups).
