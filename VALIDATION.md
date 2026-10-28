# Validation — Local Clipper

Date: 2026-10-07. Machine: Apple M4, macOS 26.0.1, Python 3.12.7, Node 25.2.1, FFmpeg 9.0.1 (Homebrew,
no libass), deno 2.8.2. Container: `local-clipper:latest` (Debian trixie, Python 3.12.15, FFmpeg 7.1.5,
deno 2.9.7, OpenCV 4.14). Evidence screenshots were kept locally and are not part of this repository.

Legend: **PASS** = verified as described · **BLOCKED** = could not be run · **NOT VERIFIED** = not attempted.

## 1. Local functional acceptance

| Requirement | Status | Evidence |
|---|---|---|
| Local file → transcription + model-selected moments → playable captioned MP4s | **PASS (live)** | §3 run A: 16-min public-domain ISS interview uploaded through the browser UI; 3 clips |
| YouTube link imports and processes | **PASS (live)** | §3 run B: CC-BY lecture via the UI (yt-dlp + deno), 1 clip; real download/probe/audio also re-verified without a key |
| ≥ 3 meaningful, distinct clips from a rich source, never forced | **PASS** | Run A: 3 distinct Q&A answers; run C: 3 distinct lecture points; run B honestly returned 1 ("1 of the 5 requested clips met the quality bar") |
| Browser: import, status, playback/seeking, editing, re-rendering, downloads, history after restart | **PASS** | §4 Playwright suites (synthetic + live + restart) |
| Production build, configurable origin, no hard-coded localhost | **PASS** | `npm run build`; `grep -rE "localhost|127.0.0.1" frontend/dist` → none |
| Hosted mode: unauthenticated API/media/mutations rejected; login, logout, expiry, authorized use | **PASS (local HTTPS stack)** | §5 compose stack behind Caddy on `https://localhost:8443` |
| Closing the browser does not cancel a background job | **PASS** | §5 hosted suite test 3 (job finished import/probe/audio with no browser) |
| ffprobe + full decode; dimensions; duration; frames; audio edges; crop; caption sync | **PASS** | §6 (`tests/inspect_render.py`, frame-accurate measurements, live re-transcription check) |
| Focused tests: chunk offsets/overlap, invalid IDs/ranges, dedupe, cache invalidation, caption timing after trim, safe paths, subprocess args, cancel/restart, real FFmpeg integration | **PASS** | §2 backend suite (1295 tests) |
| Caption edits do not trigger transcription/selection; cancelled job never becomes successful | **PASS** | `test_pipeline_e2e.py::test_full_pipeline_then_caption_edit_rerender`, `test_jobs_queue.py`, `test_lifecycle_fixes.py`; live reselect reused the paid transcript (§3) |
| Review for defects, fix, re-run checks | **PASS** | §7: two adversarial review rounds (63 confirmed findings fixed), all suites re-run |

## 2. Automated tests (exact commands, observed results)

| Command | Result |
|---|---|
| `cd backend && .venv/bin/python -m pytest -q -p no:cacheprovider` | **1295 passed, 2 skipped** (skips: shellcheck not installed; network test is opt-in) — 6 min 41 s |
| same suite inside the image (`docker run … local-clipper:latest … pytest tests --deselect tests/test_cli.py --deselect tests/test_supervisor.py --deselect tests/test_deploy.py`) | **1209 passed, 2 skipped**; 4 failed only because the image deliberately omits repo files they read (`/app/.env.example`, `scripts/start.sh`); they pass on the host |
| `cd frontend && npm test` | **74 passed** |
| `cd frontend && npm run build` | type-checked build OK; JS ≈ 200 KB (≈ 72 KB gzip) |
| `CLIPPER_NETWORK_TESTS=1 … pytest tests/test_youtube.py -m network` | passed (yt-dlp metadata for `nKyWusznRgQ`, duration 602 s) |
| `docker compose --env-file … -f deploy/compose.yaml config` | valid (also checked with `CLIPPER_PORT=9000`) |
| `scripts/setup.sh` (end to end) | 13 PASS, 1 WARN (no key at that time), 0 FAIL |

Real FFmpeg is used for all media tests (probe, audio chunking, frames, framing with YuNet, captions, rendering,
pipeline end to end). Fakes are used only for OpenAI (deterministic provider behaviour, failure paths) and in the
seeded synthetic UI fixture.

## 3. Live OpenAI acceptance (real key, 2026-10-07)

Smoke test (`docs/validation/live_smoke.json`): `gpt-6.1-sol` and `whisper-1` available; `responses.parse` with a
strict schema + `input_image` → `completed` (request `req_<redacted>`, 357/40 tokens);
`whisper-1` word timestamps through the app adapter (request `req_<redacted>`). The sample
was digital silence and whisper returned "you" (`no_speech_prob` 0.94) → silence filter added (§7).

| Run | Source | Path | Result | Time | Usage |
|---|---|---|---|---|---|
| A | NASA Don Pettit ISS interview, 15:55, public domain | browser upload → Generate (`e2e/live.spec.ts`) | ready, **3 clips** (asked 5) | 2.2 min | 4 requests, 13.3k in / 1.3k out tokens, 962 s audio |
| B | CopyrightX Lecture 2.1, 10:02, CC BY (YouTube `nKyWusznRgQ`) | browser YouTube link → Generate | ready, **1 clip** | 59 s | 3 requests, 8.3k in / 0.7k out, 603 s audio |
| B′ | same project, "Find different clips" after the sentence-boundary fix | API reselect | ready, 1 clip; **0 new transcription calls** | 37 s | +1 candidates, +1 review |
| C | CopyrightX Lecture 3.1, 21:09, CC BY (speaker + slides) | API upload | ready, **3 clips** | ≈ 2.6 min | 7 requests, 23.4k in / 2.5k out, 1283 s audio |

Total live spend (list prices checked 2026-10-05, estimate, cached-token discounts not applied): ≈ **$0.46**
(whisper-1 2848 s ≈ $0.28; gpt-6.1-sol 55.9k in / 5.7k out ≈ $0.17; smoke/sync checks < $0.01).

Clips chosen (full text inspected; titles are the model's, checked against the transcript):

- A1 "Why freeze-dried ice cream is off the ISS menu" (48 s, score 88) — question hook → crumbs float, ice cream
  re-melts and sticks to the wall. A2 "Why an astronaut misses sinks and running water" (51 s, 83) — complete,
  opening "But the thing I miss most…" leans slightly on the earlier question; reviewer noted repetition.
  A3 "Why landing leaves Don Pettit sick for 24 hours" (46 s, 75). All face-crop. 20 hallucinated words dropped
  from a 185 s silent stretch.
- B "Why knowing copyright law is not enough" (36 s, 73). First run started mid-sentence ("crucial doctrines…",
  a unit split at a 1.5 s pause) → fixed (§7); the reselect starts at "As you saw in the last lecture, …".
  Only 2 candidates existed; review rejected the other as abstract/repetitive. Returning one is intended.
- C1 "How copyright expanded beyond a book's exact words" (39 s), C2 "Did opening a software box mean accepting a
  contract?" (29 s), C3 "How a circus poster helped ads win copyright protection" (39 s; slides → full frame).
  All start and end on sentence boundaries.

Evidence: screenshots of both results pages, clip frames and live processing status (kept locally, not in this
repository). Rendered files: `data/projects/<id>/clips/`.

## 4. Browser acceptance (Playwright, Chromium 153 headless)

| Suite | Command | Result |
|---|---|---|
| synthetic (history, play+seek with HTTP 206, MP4/SRT/ZIP downloads verified with ffprobe/unzip, ZIP eligibility, cancel pending render, hidden-caption restore, trim limits, edit trim/style/framing/caption text → real re-render, Settings, 360 px no horizontal scroll) | `E2E_BASE_URL=http://127.0.0.1:8766 npx playwright test e2e/synthetic.spec.ts` | **8 passed** |
| restart persistence (after `scripts/stop.sh` + `scripts/start.sh`) | `… npx playwright test e2e/restart.spec.ts` | **1 passed** |
| live upload + YouTube | `E2E_LIVE=1 E2E_UPLOAD=… E2E_YOUTUBE=… npx playwright test e2e/live.spec.ts` | **2 passed** (paid) |
| hosted (§5) | `E2E_BASE_URL=https://localhost:8443 E2E_PASSWORD=… E2E_UPLOAD=… npx playwright test e2e/hosted.spec.ts` | **5 passed** |

The synthetic instance served a project seeded by `backend/tests/seed_synthetic_project.py` (real FFmpeg, fake
providers, title prefixed "[SYNTHETIC FIXTURE]") from a scratch data directory.

## 5. Hosted-mode checks (local, not the real domain)

`docker build -f deploy/Dockerfile -t local-clipper:latest .` (1.34 GB) → `docker compose up` with a throwaway
`deploy/.env.hosted` (`CLIPPER_HOSTNAME=localhost`, public origin `https://localhost:8443`, session TTL 72 s,
`CLIPPER_PROXY_SUBNET` overridden because 172.30.0.0/16 collides on this machine) and a port override to 8443.
Observed: only the proxy publishes a port; API and worker publish none; container runs as uid 10001;
unauthenticated `/api/*` and `/media/*` → 401, mutation without session → 401, login from a foreign Origin → 403,
wrong Host at the app → 400 (Caddy answers unknown hosts with an empty 200 and never forwards), 6th wrong password
with spoofed `X-Forwarded-For` → 429, HSTS/CSP/nosniff/X-Frame-Options present, cookie host-only + Secure +
HttpOnly + SameSite=Lax, logout and 72 s expiry enforced server-side, `cli backup` produced a consistent backup
(integrity_check ok). Container logs contained no password, session secret, hash, cookie or host path.
The test stack, its volumes and credentials were removed afterwards.

**Not verified:** a real public hostname, public DNS, Let's Encrypt certificates, a real server's
resources, and `nginx`/systemd variants beyond `nginx -t` syntax checks in `tests/test_deploy.py`.

## 6. Media inspection

- `backend/tests/inspect_render.py` (ffprobe streams, full decode, audio edges, caption band changes, YuNet face
  positions, contact sheet). All live clips: H.264 yuv420p 1080×1920, AAC 48 kHz stereo, exact standard frame rates
  (e.g. 60000/1001), decode clean, faces centred (x 0.43–0.55).
- Caption sync, synthetic ground truth (tone bursts at known times): every word highlight switches within
  −10…+20 ms of its word start across 63 words (half a frame at 25 fps); cue starts within 0–20 ms of audio onsets.
- Caption sync, real speech: live clip A1's rendered audio re-transcribed with whisper-1
  (`req_<redacted>`): 113/114 words matched; median offset 0.000 s, mean |offset| 0.073 s,
  p90 0.160 s (within whisper's own timestamp jitter); no word truncated at either clip edge.
- Framing on real footage (screenshots kept locally): interview → face crop; slides → full frame on blurred
  background; mixed speaker/slide clip → per-shot (slide shown whole, speaker cropped).
- Opening/ending: 30 ms fade-in / 50 ms fade-out; clips start up to 0.25 s before the first word (padding into
  silence) unless the source starts there.
- Not inspected by listening: audio quality was checked with measurements only (no human listening).

## 7. Review and fixes

1. **Review round 1** (7 dimensions, each finding independently reproduced): 50 findings, **48 confirmed**
   (1 critical: resuming after a cancel during upload import could delete the uploaded source; 2 high:
   orphaned job stuck after container restart with PID reuse; full-range (yuvj) sources failed validation).
   All fixed with regression tests (6 parallel fix groups).
2. **Re-review of the fixes**: 14 findings, **13 confirmed** (no critical/high), all fixed: caption overlay time
   base, YouTube resume disk check, seam-word recovery edge cases, truncated review batches, proxied-local login
   lockout, transcription cache keyed by audio content (no re-billing on version bumps), start.sh recovery, …
3. **Live-run findings fixed**: whisper silence hallucinations (dropped only when inside measured silence AND
   `no_speech_prob ≥ 0.5`); clips starting mid-sentence (deterministic sentence completion + "…" continuation
   marks in the prompt); per-shot framing for mixed speaker/slide clips; stable time-based word ids.

Unresolved / partially resolved (explicit):
- Finding 33 (cosmetic): after a cancelled reselect, stage rows may show the newer run's state although the
  previous clips/options are kept.
- HDR: tone-mapped with `zscale` on FFmpeg 7.1 (container); local FFmpeg 9 lacks `zscale` and uses swscale's
  conversion (lighter result); a build with neither keeps the old behaviour with a warning.
- MP3 transcription chunks (non-default): FFmpeg 7.1 reports ~80 ms longer container duration (content aligned).
- A SIGTERM during an in-flight OpenAI request can outlast the 30 s grace period; the job is recovered as
  interrupted within about a minute and that one request is repeated on resume.
- Two review claims were rejected by verifiers (render request during cancelling; SIGTERM handler deadlock).

## 8. Known limitations (first release)

- Validated for English speech-led video only; action/gaming content not validated. Longest tested source 21 min;
  the 120-minute limit is configured, not load-tested.
- Face detection is not active-speaker detection; multi-person shots use the full frame.
- `whisper-1` is the only OpenAI model with word timestamps and is removed on **2027-02-26** (`docs/MODELS.md`).
- Low-resolution sources (e.g. 480p) are upscaled to 1080×1920 and look soft; no detail is added.
- Mixed speaker/slide clips switch layout at detected shot changes (±0.125 s); the reviewer's score is an
  editorial heuristic, not a prediction of views.

## 9. Test media provenance

- NASA astronaut Don Pettit ISS interview with KOIN-TV (public domain, `{{PD-USGov-NASA}}`), Wikimedia Commons 480p.
- William Fisher, CopyrightX Lecture 3.1 (CC BY; third-party excerpts excluded), Wikimedia Commons 720p.
- William Fisher, CopyrightX Lecture 2.1 (YouTube, "Creative Commons Attribution license (reuse allowed)").
- Face fixture: NASA portrait of Anne C. McClain (public domain), cropped (`backend/tests/fixtures/PROVENANCE.md`).
- Synthetic fixtures: tone bursts + face overlay generated by FFmpeg in tests (mechanics only).
Downloaded copies live in `tests-media/` (git-ignored; `scripts/fetch-test-media.sh` re-fetches them).
