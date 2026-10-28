# Verified external facts (checked 2026-10-05)

Sources: developers.openai.com (platform.openai.com/docs redirects there), github.com/openai/openai-python
v3.24.0 source, PyPI, github.com/yt-dlp/yt-dlp (tag 2026.08.19) README/wiki/source. Re-verify before
changing defaults. Raw research reports are summarised here; nothing below is from memory.

## OpenAI — selection models (Responses API + Structured Outputs + image input)
- Default `gpt-6.1-sol` (in $2.00 / cached $0.10 / out $10.00 per 1M tokens; 1.05M context; 128k max output;
  reasoning effort: low, medium (default), high, xhigh, max — **no `none`/`minimal`**).
- Cheaper alternative `gpt-6-luna` ($0.10 / $0.01 / $0.50; effort none…max).
- Deprecated (avoid): `gpt-5-mini-2025-08-07`/`gpt-5-nano` (shutdown 2026-12-11), `gpt-5.1` (2027-04-01),
  `gpt-5.4-nano` (2027-04-01), `o4-mini`, `gpt-4.1-nano` (2026-10-23).
- When reasoning effort is not `none`, do **not** send `temperature`/`top_p`.
- SDK: `openai==3.24.0` (uses `httpx2`, OS certificate store → Docker images need `ca-certificates`).
  - `client.responses.parse(model=, input=[...], text_format=PydanticModel, reasoning={"effort": "low"},
    max_output_tokens=N)` → `resp.output_parsed` (or None). Cannot combine `text_format` with `text={"format":...}`.
  - Raw alternative: `client.responses.create(..., text={"format": {"type": "json_schema", "name": "x",
    "schema": {...}, "strict": True}})` → `json.loads(resp.output_text)`.
  - Images: `{"type": "input_image", "image_url": "data:image/jpeg;base64,...", "detail": "low"}`
    (pass `detail` explicitly; low ≈ fits 512×512).
  - Refusal: output item `type == "message"` with content item `type == "refusal"` (`.refusal` text).
  - Truncation: `resp.status == "incomplete"` and `resp.incomplete_details.reason == "max_output_tokens"`
    (also `content_filter`). `max_output_tokens` covers reasoning + visible output; reserve ≥ 25k.
    `responses.parse` raises `pydantic.ValidationError` if the final JSON is cut off.
  - Usage: `resp.usage.input_tokens`, `.input_tokens_details.cached_tokens`, `.output_tokens`,
    `.output_tokens_details.reasoning_tokens`.
  - Request id: `resp._request_id`; on errors `exc.request_id` (APIStatusError).
  - Exceptions: `APITimeoutError` ⊂ `APIConnectionError` (catch first); `APIStatusError` subclasses:
    400 BadRequestError, 401 AuthenticationError, 403 PermissionDeniedError, 404 NotFoundError,
    409 ConflictError, 422 UnprocessableEntityError, 429 RateLimitError, ≥500 InternalServerError.
  - Client: `OpenAI(api_key=..., timeout=..., max_retries=0)`; per call `client.with_options(...)`.
  - Strict schema: every property required; optional → `["string","null"]`; `additionalProperties: false`
    everywhere; no `allOf/not/if`; supported: enum, pattern, minimum/maximum, minItems/maxItems.
    Avoid `dict[str, X]` fields. Output key order follows schema order (put `reason` before `score`).

## OpenAI — transcription
- `whisper-1` is the **only** OpenAI model returning word timestamps
  (`response_format="verbose_json"`, `timestamp_granularities=["word","segment"]`).
- `whisper-1` (and gpt-4o-transcribe family) are **deprecated; removal 2027-02-26**; replacement
  `gpt-transcribe`/`gpt-live-transcribe` — **no word timestamps**. Migration requires forced alignment or a
  local timestamping model behind the `Transcriber` interface; never spread words evenly.
- verbose_json: `text, language, duration, segments[{id, seek, start, end, text, tokens, temperature,
  avg_logprob, compression_ratio, no_speech_prob}], words[{word, start, end}]`, `usage{seconds}`.
  Words appear **without punctuation** (inferred from the official example) → align from segment text.
  Segment text has a leading space.
- Limit: 25 MB per file. Formats: API reference/SDK list `flac, mp3, mp4, mpeg, mpga, m4a, ogg, wav, webm`;
  the narrative guide omits flac/ogg. Send an extension-bearing filename. Prompt ≤ 224 tokens. No streaming.
- Price: whisper-1 $0.006/min; gpt-transcribe $0.0045/min.

## yt-dlp
- `yt-dlp[default]==2026.8.19` includes `yt-dlp-ejs==0.8.0` (must match). Python ≥ 3.10.
- YouTube needs an external JS runtime: **deno ≥ 2.3.0** is the only one enabled by default
  (`--js-runtimes deno[:PATH]`). No `--remote-components` needed with yt-dlp-ejs installed.
- Server flags: `--ignore-config --no-playlist -t mp4 -S "res:1080" -P home:DIR -P temp:DIR
  -o "%(id)s.%(ext)s" --restrict-filenames --socket-timeout 30 -R 5 --fragment-retries 5
  --retry-sleep exp=1:30 --progress --newline --progress-delta 1
  --progress-template "download:PROG %(progress.{status,downloaded_bytes,total_bytes,total_bytes_estimate,speed,eta})j"
  --print after_move:"FILE %(filepath)s"`. Metadata: `-J --no-playlist` (simulate). `--match-filters`
  (documented name). Check `filesize_approx`, `duration`, `live_status` (`is_live`, `is_upcoming`, `post_live`) from `-J`.
- Error substrings (lowercase match): "not a bot" (bot check, curly apostrophe), "confirm your age"/"age-restricted",
  "private video", "video unavailable"/"this video is unavailable", "members-only"/"join this channel",
  "live event will begin"/"premieres in", "not made this video available in your country"/"geo restriction",
  "http error 429", "isn't available, try again later", "drm protected", "unsupported url",
  "requested format is not available" (often JS runtime problem), "no supported javascript runtime".
  Strip the cookies advice yt-dlp appends; never use cookies.
- Docker: `COPY --from=denoland/deno:bin-2.9.7 /deno /usr/local/bin/deno`.

## OpenCV / YuNet
- Pinned `opencv-python-headless>=4.10,<5` (4.14.0.94). `face_detection_yunet_2023mar.onnx`
  (MIT, sha256 8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4) has a fixed input shape;
  OpenCV 5 needs the 2026may model, so 5.x is avoided for now. YuNet works for faces ~10–300 px:
  downscale frames (e.g. 640 px wide) before detection.
  API: `cv2.FaceDetectorYN.create(model, "", (w, h), score_threshold, nms_threshold, top_k)`;
  `det.setInputSize((w, h))`; `n, faces = det.detect(bgr)`; faces rows = [x, y, w, h, 10 landmark coords, score].

## Fonts
- Montserrat ExtraBold/Black static TTF from github.com/JulietaUla/Montserrat (SIL OFL 1.1), bundled in
  `backend/clipper/assets/fonts/` with `OFL.txt`.

## Test media (openly licensed)
- Local-file test: NASA astronaut Don Pettit ISS interview with KOIN-TV (public domain, 15:55, 480p, 70.7 MB)
  https://upload.wikimedia.org/wikipedia/commons/transcoded/e/e2/NASA_Astronaut_Don_Pettit_Talks_with_KOIN-TV%2C_Portland%2C_Oregon_-_April_16%2C_2025_%28iss072m261061434%29.webm/NASA_Astronaut_Don_Pettit_Talks_with_KOIN-TV%2C_Portland%2C_Oregon_-_April_16%2C_2025_%28iss072m261061434%29.webm.480p.vp9.webm
- Speaker + slides test: William Fisher, CopyrightX Lecture 3.1 (CC BY 3.0/4.0; third-party excerpts excluded), 21:09, 720p, 34.6 MB.
- YouTube test: https://www.youtube.com/watch?v=nKyWusznRgQ (CopyrightX Lecture 2.1, "Creative Commons Attribution license (reuse allowed)", 10:02).
- Wikimedia requires a descriptive User-Agent and rate-limits (HTTP 429) → retry with backoff.
