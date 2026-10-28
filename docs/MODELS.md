# AI models used by Local Clipper

Facts below were verified on **2026-10-05** against developers.openai.com and the
`openai-python` v3.24.0 source (details and sources: `docs/RESEARCH_NOTES.md`). Model
availability, deprecations and prices change; re-check them before changing a default.

## What runs where

| Step (UI label) | Setting | Default | Why this model |
|---|---|---|---|
| Transcribe speech | `CLIPPER_TRANSCRIBE_MODEL` | `whisper-1` | The **only** OpenAI speech-to-text model that returns **word-level timestamps** (`response_format="verbose_json"`, `timestamp_granularities=["word","segment"]`). Word times drive caption timing and exact clip boundaries. |
| Find moments | `CLIPPER_SELECT_MODEL` | `gpt-6.1-sol` | Responses API + strict Structured Outputs (JSON schema), 1.05M-token context (long transcripts in large windows), 128k max output. |
| Review & rank | `CLIPPER_REVIEW_MODEL` | `gpt-6.1-sol` | Same, plus image input: low-detail frames from each candidate are reviewed together with its transcript. |

Reasoning effort for both text steps defaults to `low` (`CLIPPER_SELECT_REASONING_EFFORT`,
`CLIPPER_REVIEW_REASONING_EFFORT`). `gpt-6.1-sol` accepts `low`, `medium` (its default),
`high`, `xhigh` and `max`; it has **no** `none`/`minimal`. When an effort is sent the app never
sends `temperature`/`top_p`. An empty value omits the `reasoning` parameter entirely (for models
without reasoning support).

The models only ever see transcript text (wrapped as untrusted data), candidate unit IDs and
low-detail frames. They choose **existing transcript unit IDs**; every start/end time is computed
in code from the word timestamps, never taken from model output.

## Changing models

Edit `.env` (local) or `deploy/.env.hosted` (server) and restart:

```dotenv
CLIPPER_SELECT_MODEL=gpt-6-luna
CLIPPER_REVIEW_MODEL=gpt-6.1-sol
CLIPPER_SELECT_REASONING_EFFORT=low
```

Then run `scripts/doctor.sh` and analyse a short test video. Requirements for a replacement:

* Selection/review models must support the Responses API with strict Structured Outputs; the
  review model must also accept image input. Unsupported models fail with a plain-language
  error naming the model (no silent fallback).
* `gpt-6-luna` is the cheaper alternative (it also accepts effort `none`); expect weaker
  editorial judgement. A sensible split is a cheap model for *Find moments* and the default for
  *Review & rank*.
* Avoid deprecated models: `gpt-5-mini-2025-08-07` and `gpt-5-nano` (shutdown 2026-12-11),
  `gpt-4.1-nano` (2026-10-23), `gpt-5.1` and `gpt-5.4-nano` (2027-04-01), `o4-mini`.
* Results are cached per model and prompt version, so switching models re-runs only the AI steps
  (transcripts are reused unless the transcription model changes).
* The transcription model cannot be swapped by configuration alone: the adapter accepts only
  models with word timestamps (today: `whisper-1`) and refuses others instead of degrading
  captions.

## whisper-1 removal (2027-02-26) and migration plan

`whisper-1` (and the `gpt-4o-transcribe` family) are deprecated with **removal scheduled for
2027-02-26**. The announced successors (`gpt-transcribe`, `gpt-live-transcribe`) return **no
word timestamps**. After removal, analysis of new videos fails at *Transcribe speech* with an
error naming the model; existing projects, edits and renders keep working because transcripts
are stored locally.

Plan (to be completed before 2027-02-26, ideally by mid-January 2027):

1. Keep everything downstream unchanged: the pipeline only depends on the `Transcriber` protocol
   (`clipper/ai/interfaces.py`), which returns words with real start/end times.
2. Implement a new `Transcriber` behind that protocol, either
   * **text + forced alignment**: `gpt-transcribe` for the text, then a local forced aligner
     (CTC-based) to place each word on the audio; or
   * **local timestamping model**: a local speech model that produces word timestamps itself.
3. **Never spread words evenly** across a segment or invent timings. Words the aligner cannot
   place must be reported as unaligned (and handled like missing speech), not guessed.
4. Validate with the existing transcript checks (`validate_transcript`: monotonic,
   in-bounds, non-overlapping) plus a comparison against `whisper-1` output on the test media in
   `tests-media/` while `whisper-1` is still available (word-time error, caption sync by ear).
5. Bump the transcript cache version so old cached chunks are not mixed with new ones, update the
   default `CLIPPER_TRANSCRIBE_MODEL`, this document and `docs/RESEARCH_NOTES.md`.

## SDK

`openai==3.24.0`, pinned with hashes in `backend/requirements.lock`. It uses `httpx2`, which
trusts the **operating system certificate store** (the Docker image installs `ca-certificates`).
The client is created with `max_retries=0`; the app does its own bounded retries with backoff and
never logs request headers or the API key.

## Prices (list prices as of 2026-10-05; estimates only)

| Model | Input | Cached input | Output | Audio |
|---|---|---|---|---|
| `whisper-1` | – | – | – | $0.006 / minute |
| `gpt-transcribe` (future migration candidate) | – | – | – | $0.0045 / minute |
| `gpt-6.1-sol` | $2.00 / 1M tokens | $0.10 / 1M | $10.00 / 1M | – |
| `gpt-6-luna` | $0.10 / 1M tokens | $0.01 / 1M | $0.50 / 1M | – |

Reasoning tokens are billed as output tokens. Transcription of a 60-minute video costs about
$0.36 with `whisper-1`. Selection and review costs depend on transcript length, the number of
candidates and frames; the app records the real token usage of every request and shows a
clearly labelled estimate per project (`clipper/ai/pricing.py`, prices dated 2026-10-05).
Cached provider results are reused and not billed again.

## Live verification (2026-10-07)

Recorded with a real key (details and request ids in `VALIDATION.md` and `docs/validation/live_smoke.json`):

- `client.models.retrieve`: `gpt-6.1-sol` (owned_by system) and `whisper-1` (owned_by openai-internal) available.
- `gpt-6.1-sol` via `client.responses.parse` with a strict Pydantic schema, an `input_image` (`detail: "low"`) and
  `reasoning={"effort": "low"}`: status `completed`, parsed output, 357 input / 40 output tokens.
- `whisper-1` via the app's `OpenAITranscriber` (`verbose_json`, word + segment timestamps, FLAC upload): word
  timestamps returned. It also showed whisper's known hallucination on digital silence ("you",
  `no_speech_prob` 0.94, `avg_logprob` -0.51), which led to the silence filter in `merge_chunks`.
- Three full analyses (16-, 10- and 21-minute speech videos) completed end to end; see `VALIDATION.md`.
- SDK: openai 3.24.0 (Python 3.12.7).
