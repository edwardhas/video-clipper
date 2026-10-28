# Local Clipper

A personal tool that turns long YouTube videos or local recordings into short, captioned,
vertical (9:16, 1080×1920) clips for TikTok and similar platforms.

Paste a link or upload a file, click **Generate clips**, and you get several ranked clips with
burned-in captions. Preview them, adjust the start and end, correct captions, change framing and
caption style, then download individual MP4s, SRT files or a ZIP. Projects survive restarts.

## Product goal (keep this in mind when changing anything)

The value is **finding moments worth watching and presenting them well**, not producing arbitrary
excerpts. A valid MP4 of a random 40 seconds is a failure. Optimise for:

- complete ideas with a strong opening and a clear payoff, understandable without the rest of the video;
- accurate captions with real word timing (never fabricated);
- sensible framing (follow a single clear face; otherwise keep the full frame);
- a short path from "paste a link" to "download", with the owner spending time choosing and making
  small corrections rather than searching the source or configuring an AI pipeline.

Fewer good clips beat more weak ones: the app returns fewer clips than requested when the source
does not justify more. The score shown on each clip is an editorial heuristic, not a probability of
going viral. The products this is inspired by (OpusClip, Choppity, Klap, Riverside Magic Clips,
Vizard, Ssemble) are references for the experience, not features to copy.

## First-release scope

- **Validated for speech-led videos:** interviews, podcasts, commentary, talks and presentations in English.
- Other videos are accepted, but analysis depends on speech. A video with no audio track, silent
  audio or too little speech ends with a clear "no clips" explanation instead of forced output.
  Action-only content (sports, gaming) has **not** been validated.
- Up to **120 minutes** per source (configurable via `CLIPPER_MAX_SOURCE_MINUTES`); the limit is
  checked before processing. Long sources have not been load-tested beyond the test media listed
  in `VALIDATION.md`.
- Single owner. No accounts, billing, teams, scheduling, social publishing or generative B-roll.

## Hosting context

Local Clipper runs on your own computer. Hosting is optional: once it works locally, the **same**
application can run on a persistent server under a hostname you control (example:
`clipper.example.com`). The hostname only routes to that server; it does not provide compute. Hosting
the frontend alone while calling a laptop is a different architecture and is not supported. See
[DEPLOYMENT.md](DEPLOYMENT.md).

## Quick start (macOS or Linux, no Docker needed)

Requirements: Python 3.12, Node 20.19+ (or 22.12+), FFmpeg/ffprobe with libx264, deno ≥ 2.3 (for
YouTube). `scripts/doctor.sh` checks all of them.

```bash
scripts/setup.sh          # venv + locked Python deps, frontend build, model/font checks, creates .env
# put your key into .env:  OPENAI_API_KEY=...   (never paste it anywhere else)
scripts/start.sh          # starts the web app + background worker; prints http://127.0.0.1:8765
scripts/stop.sh           # clean shutdown (a running job is marked "interrupted" and can be resumed)
```

`scripts/start.sh --detach` runs it in the background (logs in `data/logs/`). Closing the browser
never stops processing; the worker is a separate process.

Other scripts: `scripts/doctor.sh` (diagnostics), `scripts/backup.sh` / `scripts/restore.sh`,
`scripts/hash-password.sh` (owner password for hosted mode), `scripts/fetch-test-media.sh`
(openly licensed test videos).

## Where things are stored and what leaves the machine

- **Local mode:** everything lives in `data/` in this folder: `clipper.db` (SQLite: projects, jobs,
  edits), `projects/<id>/source/` (a copy of the imported video), `work/` (audio, transcript,
  intermediate files), `clips/` (rendered MP4/SRT). Uploads are copies; your original files are never
  modified or deleted. Settings shows the exact location, disk use and cleanup controls.
- **Hosted mode:** the same layout on the server's persistent `/data` volume.
- **Sent to OpenAI:** extracted audio (for transcription), transcript text (to find and rank
  moments) and a few still frames per shortlisted clip (to review framing and quality). Source
  videos and rendered clips are not uploaded to OpenAI. The API key is read only by the backend.

## Models

Transcription `whisper-1` (word timestamps; OpenAI removes it on **2027-02-26** — see
[docs/MODELS.md](docs/MODELS.md) for the migration plan), moment finding and review `gpt-6.1-sol`
(Responses API, strict structured outputs, image input). All configurable in `.env`.
Facts and dates were verified against OpenAI's documentation on 2026-10-05
([docs/RESEARCH_NOTES.md](docs/RESEARCH_NOTES.md)).

## How it works

1. **Import** (yt-dlp for YouTube, ≤1080p; or a streamed upload) → **probe** (ffprobe: duration,
   streams, rotation, frame rate).
2. **Audio**: 16 kHz mono analysis track; split into ≤10-minute FLAC chunks at pauses with 3 s overlap.
3. **Transcribe** each chunk (cached by audio content); merge with exact source offsets, no duplicated
   or lost words at seams; words whisper invents inside measured silence are dropped; punctuation
   aligned onto words; sentence-like units; word IDs derived from time so caption fixes stay attached.
4. **Find moments**: overlapping transcript windows → structured candidates that may only reference
   existing unit IDs; cut times are derived from stored word timings, never from model output, and
   are moved to whole-sentence boundaries.
5. **Review & rank** shortlisted candidates with surrounding context and frames; validate,
   de-duplicate, keep only those above the quality bar.
6. **Frame**: YuNet face detection with scene-cut resets and smoothing → face-following 9:16 crop for
   a single clear face; shots without one (slides, several people) switch to the full frame on a
   blurred background, per shot, inside the same clip; clips with no clear face stay full frame.
   Face detection is not speaker detection, and the app never claims otherwise.
7. **Render**: FFmpeg H.264/AAC 1080×1920, fast-start, accurate cuts, short audio fades; captions
   rendered with Pillow (bundled Montserrat, SIL OFL) and composited frame-accurately; each output is
   probed and fully decoded before it is marked ready.

Every stage records its cache key; resume/retry skips finished stages, and editing captions, style,
trim or framing re-renders only that clip with zero API calls. A cancelled job can never become
successful. Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Development

```bash
cd backend && .venv/bin/python -m pytest -q          # unit + real FFmpeg integration tests
cd frontend && npm test && npm run build              # frontend unit tests + type-checked build
cd frontend && npx playwright test                    # browser acceptance (see VALIDATION.md)
```

Implementation progress and decisions: [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md).
Verification evidence and known limitations: [VALIDATION.md](VALIDATION.md).
