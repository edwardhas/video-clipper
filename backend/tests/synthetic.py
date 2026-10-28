"""Synthetic speech-like media and deterministic FAKE providers for tests and UI fixtures.

Everything here is a test harness: the fakes stand in for OpenAI and are never
reachable from the application. The audio is tone bursts with known timings
(one burst = one "word"), so word timings are ground truth by construction.
This validates mechanics (chunk offsets, caching, rendering, caption sync), not
editorial quality.
"""

from __future__ import annotations

from pathlib import Path

from clipper import db
from clipper.ai.interfaces import RawChunkResult, RawSegment, RawWord
from clipper.jobs import queue
from clipper.models import Candidate, CandidateScores, ReviewDecision
from clipper.procs import run

FIXTURES = Path(__file__).parent / "fixtures"

SOURCE_SECONDS = 48.0
BLOCK = 4.0            # one "sentence" per 4 s block: speech for 3.2 s then a 0.8 s pause
WORD_ON, WORD_PERIOD = 0.30, 0.45
VOCAB = ["clear", "ideas", "make", "short", "clips", "worth", "watching", "every", "single", "time"]


def ground_truth_words() -> list[tuple[str, float, float, bool]]:
    """(text, start, end, sentence_end) for every tone burst in the synthetic audio."""
    words = []
    t_block = 0.0
    k = 0
    while t_block + BLOCK <= SOURCE_SECONDS + 1e-6:
        t = t_block
        block_words = []
        while t + WORD_ON <= t_block + 3.2 + 1e-6:
            block_words.append((VOCAB[k % len(VOCAB)], round(t, 3), round(t + WORD_ON, 3)))
            k += 1
            t += WORD_PERIOD
        for i, (w, s, e) in enumerate(block_words):
            words.append((w, s, e, i == len(block_words) - 1))
        t_block += BLOCK
    return words


def make_source(path: Path, *, with_audio: bool = True, seconds: float = SOURCE_SECONDS) -> None:
    face = FIXTURES / "face_portrait_pd.jpg"
    gate = "if(lt(mod(t,4),3.2),if(lt(mod(mod(t,4),0.45),0.3),1,0),0)"
    args = ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-f", "lavfi",
            "-i", f"color=c=0x203040:size=640x360:rate=25:duration={seconds}",
            "-loop", "1", "-i", str(face)]
    if with_audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={seconds}"]
    fc = "[1:v]scale=150:-2[f];[0:v][f]overlay=x='245+60*sin(t/3)':y=60:shortest=1[v]"
    if with_audio:
        fc += f";[2:a]volume='{gate}':eval=frame[a]"
    args += ["-filter_complex", fc, "-map", "[v]"]
    if with_audio:
        args += ["-map", "[a]", "-c:a", "aac", "-b:a", "96k"]
    args += ["-t", str(seconds), "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path)]
    run(args, timeout=120)


class FakeTranscriber:
    provider = "fake"
    model = "whisper-1"

    def __init__(self, db_path: Path, project_id: str, *, sparse: bool = False, cancel_on_call: int | None = None,
                 job_ref: dict | None = None) -> None:
        self.db_path = db_path
        self.project_id = project_id
        self.calls: list[str] = []
        self.sparse = sparse
        self.cancel_on_call = cancel_on_call
        self.job_ref = job_ref if job_ref is not None else {}

    def _plan_for(self, audio_path: Path) -> tuple[float, float]:
        with db.session(self.db_path) as c:
            row = c.execute("SELECT artifact FROM stages WHERE project_id=? AND stage='audio'",
                            (self.project_id,)).fetchone()
        art = db.loads(row["artifact"])
        for ch in art["chunks"]:
            if ch["path"].endswith(audio_path.name):
                return ch["start"], ch["end"]
        raise AssertionError("unknown chunk")

    def transcribe_chunk(self, audio_path, *, language, cancel, usage=None, audio_seconds=None) -> RawChunkResult:
        self.calls.append(audio_path.name)
        if self.cancel_on_call is not None and len(self.calls) == self.cancel_on_call:
            with db.session(self.db_path) as c:
                queue.request_cancel(c, self.job_ref["id"])
        start, end = self._plan_for(Path(audio_path))
        gt = ground_truth_words()
        if self.sparse:
            gt = gt[:6]
        words, segs = [], []
        block_words: list[tuple[str, float, float, bool]] = []
        for w in gt:
            mid = (w[1] + w[2]) / 2
            if start <= mid < end:
                words.append(RawWord(text=w[0], start=w[1] - start, end=w[2] - start))
                block_words.append(w)
                if w[3]:
                    text = " " + " ".join(x[0] for x in block_words).capitalize() + "."
                    segs.append(RawSegment(text=text, start=block_words[0][1] - start, end=w[2] - start,
                                           no_speech_prob=0.01, avg_logprob=-0.2))
                    block_words = []
        if block_words:
            segs.append(RawSegment(text=" " + " ".join(x[0] for x in block_words), start=block_words[0][1] - start,
                                   end=block_words[-1][2] - start))
        return RawChunkResult(words=words, segments=segs, language="english", duration=end - start,
                              request_id=f"req_fake_{len(self.calls)}", raw={"fake": True})


class FakeSelector:
    provider = "fake"
    select_model = "fake-select"
    review_model = "fake-review"

    def __init__(self) -> None:
        self.find_calls = 0
        self.review_calls = 0
        self.caches: list = []          # ``cache=`` checkpoint stores the pipeline passed in

    def find_candidates(self, transcript, options, *, cancel, usage, cache=None):
        from clipper.ai.transcript import range_for_units
        self.find_calls += 1
        self.caches.append(cache)
        out = []
        for n, (a, b) in enumerate([(0, 2), (4, 6), (8, 10)]):
            if b >= len(transcript.units):
                break
            u0, u1 = transcript.units[a].id, transcript.units[b].id
            s, e = range_for_units(transcript, u0, u1)
            out.append(Candidate(id=f"c{n + 1:03d}", window_index=0, start_unit_id=u0, end_unit_id=u1,
                                 src_start=s, src_end=e, title=f"Moment {n + 1}", hook="Clear ideas",
                                 payoff="Worth watching", reason="Complete thought", topic="clips",
                                 scores=CandidateScores(hook=4, completeness=4, value=4, emotion=3),
                                 model_score=80 - n))
        return out

    def review(self, transcript, candidates, frames, options, *, cancel, usage, cache=None):
        self.review_calls += 1
        self.caches.append(cache)
        assert all(len(frames[c.id]) == 2 and all(Path(p).exists() for _, p in frames[c.id]) for c in candidates)
        return [ReviewDecision(candidate_id=c.id, keep=True, final_score=85 - i * 5, title=c.title,
                               reason="Strong, complete moment", opening=4, completeness=4, interest=4, visual=4,
                               duplicate_of=None, start_unit_id=c.start_unit_id, end_unit_id=c.end_unit_id)
                for i, c in enumerate(candidates)]
