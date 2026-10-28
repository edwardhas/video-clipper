# Implementation status — Local Clipper

Last updated: 2026-10-07 (build sessions 1–2). **State: locally accepted (live), hosting prepared, not deployed.**

## Environment (detected)
- macOS 26.0.1, Apple M4, 16 GB RAM, arm64. Python 3.12.7 (pyenv) → `backend/.venv`; uv inside the venv for
  locking (`backend/requirements*.lock`, hashed, universal). Node 25.2.1; Vite 8, Vue 3.5, vue-router 5, TS ~6.0.
- FFmpeg/ffprobe 9.0.1 (Homebrew, **no libass/freetype**) → captions rendered with Pillow and composited with
  `overlay` from a concat timeline. libx264 + h264_videotoolbox (VT passes a real capability test; CPU fallback).
- deno 2.8.2 for yt-dlp 2026.8.19 (+ yt-dlp-ejs 0.8.0). Docker Desktop installed (was stopped; left stopped).
- `OPENAI_API_KEY` in `.env` (git-ignored) since 2026-10-07.

## Decisions
- Models: selection + review `gpt-6.1-sol` (effort low), transcription `whisper-1` (removal 2027-02-26).
- OpenCV pinned `<5` (YuNet 2023mar). Port 8765. FLAC chunks ≤10 min at silences, 3 s overlap.
- Deterministic clip ids per (project, range); time-based word ids; content-addressed caches for every paid call.
- Per-shot framing (fit spans) for mixed speaker/slide clips; sentence-boundary completion of AI ranges.
- Docker image on Debian trixie (FFmpeg 7.1); compose subnet/proxy IP/port configurable.

## Milestones
- [x] Environment, contracts, all modules (9 parallel agents), orchestration (lead)
- [x] Integration, setup.sh end to end, browser suites, restart persistence
- [x] Cancel / SIGKILL recovery / resume on the real app
- [x] Docker image + compose stack with Caddy HTTPS (local), hosted browser suite
- [x] Review round 1 (48 confirmed findings) + round 2 (13) — all fixed with regression tests
- [x] Live OpenAI acceptance: smoke test + 3 full analyses (ISS interview, YouTube lecture, slides lecture)
- [x] VALIDATION.md, README.md, DEPLOYMENT.md, docs/MODELS.md, docs/ARCHITECTURE.md

## Test status
backend 1295 passed / 2 skipped (host); 1209 passed in the container (host-only tests excluded);
frontend 74; Playwright: synthetic 8, restart 1, hosted 5, live 2.

## Next steps (owner decisions)
1. Choose the host (persistent Linux server, Docker or native) and the real hostname (example:
   `clipper.example.com`). Then follow DEPLOYMENT.md (DNS A/AAAA, `deploy/.env.hosted`, `docker compose up -d`,
   smoke test).
2. Plan the whisper-1 replacement before 2027-02-26 (alignment adapter behind `Transcriber`).
3. Optional: cosmetic stage-row reset after cancelled reselect (finding 33).
