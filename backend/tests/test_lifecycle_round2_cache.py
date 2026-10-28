"""Second review round: the per-chunk transcription cache and the pauses passed to the transcript merge.

Real FFmpeg media, FAKE transcriber (never OpenAI). Checks that

* a cached paid chunk is found by its audio content: an AUDIO_VERSION bump that leaves the
  analysis audio unchanged makes no new transcription calls, while audio whose timeline really
  changed (MPEG-TS with a late audio start, before/after the timeline fix) is transcribed again;
* results saved under the earlier source + time-range keys are reused after the upgrade only when
  their audio provably is the current audio, and are then re-saved under the content key;
* the audio stage stores the detected pauses and the transcribe stage hands them to merge_chunks,
  also for an audio artifact written before pauses were stored, without re-billing.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from clipper import db
from clipper.storage import atomic_path, stable_hash
from synthetic import FakeTranscriber, make_source
from test_lifecycle_fixes import OPTS, _analyze, _NoMoments, _project
from test_pipeline_e2e import _create_upload_project, _install_fakes, env  # noqa: F401 (fixture)

pytestmark = [
    pytest.mark.ffmpeg,
    pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed"),
]


# ----------------------------------------------------------------------------- helpers
def _source(tmp_path: Path, container: str) -> Path:
    """The synthetic talk as MP4, or as MPEG-TS whose audio starts 0.5 s after its video."""
    mp4 = tmp_path / "talk.mp4"
    if not mp4.exists():
        make_source(mp4)
    if container == "mp4":
        return mp4
    ts = tmp_path / "talk.ts"
    subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-y", "-i", str(mp4),
                    "-itsoffset", "0.5", "-i", str(mp4), "-map", "0:v", "-map", "1:a", "-c", "copy",
                    "-f", "mpegts", str(ts)], check=True, timeout=120)
    return ts


def _extract_before_timeline_fix(src: Path, out_wav: Path, *, settings, cancel=None) -> float:
    """extract_analysis_audio as it was at AUDIO_VERSION 1 (before the MPEG-TS/PS timeline fix)."""
    from clipper.media.audio import wav_duration
    from clipper.procs import run
    args = [settings.ffmpeg, "-hide_banner", "-nostdin", "-v", "error", "-y", "-i", str(src),
            "-map", "0:a:0", "-vn", "-sn", "-dn", "-map_metadata", "-1", "-af", "aresample=async=1:first_pts=0",
            "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", "-fflags", "+bitexact", "-f", "wav"]
    with atomic_path(out_wav) as tmp:
        run([*args, str(tmp)], cancel=cancel, timeout=120, capture_stdout=False)
        return wav_duration(tmp)


def _analyze_source(env, monkeypatch, source: Path) -> tuple[str, FakeTranscriber]:
    """A new project of ``source`` analysed up to the transcript (no moments → no render work)."""
    settings, storage, db_path, worker = env
    pid = _create_upload_project(settings, storage, db_path, options=OPTS,
                                 src_maker=lambda p: shutil.copy(source, p))
    fake_t = FakeTranscriber(db_path, pid)
    _install_fakes(monkeypatch, fake_t, _NoMoments())
    job = _analyze(worker, db_path, pid)
    assert job["state"] == "succeeded", job["error_message"]
    assert _project(db_path, pid)["status"] == "no_clips"
    return pid, fake_t


def _audio_artifact(db_path, pid: str) -> dict:
    with db.session(db_path) as c:
        row = c.execute("SELECT artifact FROM stages WHERE project_id=? AND stage='audio'", (pid,)).fetchone()
    return db.loads(row["artifact"])


def _chunk_cache(storage) -> Path:
    return storage.root / "cache" / "transcribe-chunk"


def _entries_by_chunk(storage) -> dict[int, Path]:
    """Cache files of a single analysed project, by chunk index (the fake transcribes chunks in order)."""
    out = {}
    for f in _chunk_cache(storage).iterdir():
        out[int(json.loads(f.read_text())["request_id"].rsplit("_", 1)[1]) - 1] = f
    return out


def _legacy_key(settings, sha: str, chunk: dict, *, versioned: bool) -> str:
    """The keys earlier releases saved chunk transcripts under (source + time range, ± AUDIO_VERSION 2)."""
    parts = ("tchunk", sha, 2) if versioned else ("tchunk", sha)
    return stable_hash(*parts, round(chunk["start"], 3), round(chunk["end"], 3), settings.transcribe_model,
                       OPTS.language, settings.transcribe_audio_format)


def _spy_merge(monkeypatch) -> list:
    import clipper.ai.transcript as transcript
    real = transcript.merge_chunks
    seen: list = []

    def spy(*a, **k):
        seen.append(k.get("silences"))
        return real(*a, **k)

    monkeypatch.setattr(transcript, "merge_chunks", spy)
    return seen


# ----------------------------------------------------------------------------- content-addressed cache
@pytest.mark.parametrize(("container", "new_calls"), [("mp4", 0), ("mpegts", 3)])
def test_only_audio_whose_timeline_changed_is_transcribed_again(env, monkeypatch, tmp_path, container, new_calls):
    import clipper.media.audio as audio
    source = _source(tmp_path, container)
    calls = []
    for extract, version in ((_extract_before_timeline_fix, 1), (audio.extract_analysis_audio, 2)):
        monkeypatch.setattr(audio, "extract_analysis_audio", extract)
        monkeypatch.setattr(audio, "AUDIO_VERSION", version)
        _, fake_t = _analyze_source(env, monkeypatch, source)
        calls.append(len(fake_t.calls))
    # MP4: the timeline fix left the analysis WAV byte-identical, so the version bump re-bills nothing.
    # MPEG-TS with late audio: the WAV gained the 0.5 s lead-in, so every chunk is new audio.
    assert calls == [3, new_calls]


def test_audio_version_bump_on_reselect_reextracts_but_never_rebills(env, monkeypatch):
    from test_lifecycle_fixes import _queue_reselect, _ready_project, _run_next
    settings, storage, db_path, worker = env
    pid, fake_t, _ = _ready_project(env, monkeypatch)
    assert len(fake_t.calls) == 3
    import clipper.media.audio as audio
    real = audio.extract_analysis_audio
    extracted: list[Path] = []
    monkeypatch.setattr(audio, "extract_analysis_audio", lambda *a, **k: extracted.append(a[1]) or real(*a, **k))
    monkeypatch.setattr(audio, "AUDIO_VERSION", audio.AUDIO_VERSION + 1)
    _queue_reselect(db_path, pid)
    assert _run_next(worker, db_path)["state"] == "succeeded"
    assert len(extracted) == 1                  # the audio stage key changed (local work) ...
    assert len(fake_t.calls) == 3               # ... but the identical audio was not transcribed again
    assert _project(db_path, pid)["status"] == "ready"


@pytest.mark.parametrize(("container", "versioned", "new_calls"), [
    ("mp4", True, 0), ("mp4", False, 0),        # unchanged audio: reused under either earlier layout
    ("mpegts", True, 0),                        # made by the current (AUDIO_VERSION 2) extraction
    ("mpegts", False, 3),                       # made from a pre-fix MPEG-TS WAV: different audio
])
def test_chunks_saved_under_earlier_key_layouts(env, monkeypatch, tmp_path, container, versioned, new_calls):
    settings, storage, db_path, worker = env
    source = _source(tmp_path, container)
    pid, _ = _analyze_source(env, monkeypatch, source)
    sha = _project(db_path, pid)["source_sha256"]
    chunks = _audio_artifact(db_path, pid)["chunks"]
    content_files = _entries_by_chunk(storage)
    for i, path in content_files.items():       # what the earlier release had saved
        os.replace(path, storage.cache_path("transcribe-chunk", _legacy_key(settings, sha, chunks[i],
                                                                             versioned=versioned)))

    _, fake_t = _analyze_source(env, monkeypatch, source)
    assert len(fake_t.calls) == new_calls
    if new_calls == 0:                          # migrated: found by content from now on
        assert all(p.exists() for p in content_files.values())


# ----------------------------------------------------------------------------- pauses for the merge
def test_audio_artifact_stores_pauses_and_merge_receives_them(env, monkeypatch, tmp_path):
    settings, storage, db_path, worker = env
    seen = _spy_merge(monkeypatch)
    pid, fake_t = _analyze_source(env, monkeypatch, _source(tmp_path, "mp4"))
    art = _audio_artifact(db_path, pid)
    # the synthetic talk pauses for 0.8 s after every 3.2 s of speech: [3.2, 4.0], [7.2, 8.0], ...
    pauses = [p for p in art["silences"] if p[1] - p[0] > 0.6]
    assert len(pauses) == 12
    for k, (a, b) in enumerate(pauses):
        assert abs(a - (4 * k + 3.2)) < 0.1 and abs(b - min(4 * k + 4.0, 48.0)) < 0.1
    assert all(len(c["pcm_sha256"]) == 64 for c in art["chunks"])
    assert seen == [[tuple(p) for p in art["silences"]]]


class _HallucinatingTranscriber(FakeTranscriber):
    """Like whisper-1 on digital silence: "you" in a segment with no_speech_prob 0.94 at 25 s."""

    def transcribe_chunk(self, audio_path, **k):
        from clipper.ai.interfaces import RawSegment, RawWord
        res = super().transcribe_chunk(audio_path, **k)
        start, end = self._plan_for(Path(audio_path))
        if start <= 25.15 < end:
            res.words = sorted([*res.words, RawWord("you", 25.0 - start, 25.3 - start)], key=lambda w: w.start)
            res.segments = sorted([*res.segments, RawSegment(" you", 24.0 - start, 26.0 - start, no_speech_prob=0.94,
                                                             avg_logprob=-0.51)], key=lambda x: x.start)
        return res


def test_word_returned_for_a_long_digital_silence_is_not_in_the_transcript(env, monkeypatch, tmp_path):
    import synthetic
    from clipper.models import Transcript
    settings, storage, db_path, worker = env
    silent = (20.0, 32.0)                       # three 4 s blocks muted: pause [19.2, 32.0]
    loud = tmp_path / "loud.mp4"
    make_source(loud)
    source = tmp_path / "gap.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-y", "-i", str(loud), "-c:v", "copy",
                    "-af", f"volume=enable='between(t,{silent[0]},{silent[1]})':volume=0", "-c:a", "aac",
                    "-b:a", "96k", str(source)], check=True, timeout=120)
    heard = [w for w in synthetic.ground_truth_words() if not silent[0] <= (w[1] + w[2]) / 2 <= silent[1]]
    monkeypatch.setattr(synthetic, "ground_truth_words", lambda: heard)
    pid = _create_upload_project(settings, storage, db_path, options=OPTS,
                                 src_maker=lambda p: shutil.copy(source, p))
    _install_fakes(monkeypatch, _HallucinatingTranscriber(db_path, pid), _NoMoments())
    job = _analyze(worker, db_path, pid)
    assert job["state"] == "succeeded", job["error_message"]
    t = Transcript.model_validate_json((storage.work_dir(pid) / "transcript.json").read_text())
    assert [w.raw.strip() for w in t.words] == [w[0] for w in heard]        # no "you", no real word lost
    assert t.fixes.get("dropped_in_silence", 0) >= 1


def test_older_audio_artifact_gets_pauses_from_the_wav_without_rebilling(env, monkeypatch, tmp_path):
    """Upgrade path: TRANSCRIPT_VERSION changed, the audio stage row predates pauses/content hashes and
    the chunk transcripts were saved under the earlier (AUDIO_VERSION 2) keys."""
    settings, storage, db_path, worker = env
    pid, fake_t = _analyze_source(env, monkeypatch, _source(tmp_path, "mp4"))
    sha = _project(db_path, pid)["source_sha256"]
    art = _audio_artifact(db_path, pid)
    stored_pauses = [tuple(p) for p in art["silences"]]
    for i, path in _entries_by_chunk(storage).items():
        os.replace(path, storage.cache_path("transcribe-chunk", _legacy_key(settings, sha, art["chunks"][i],
                                                                             versioned=True)))
    old = {"duration": art["duration"], "mean_volume_db": art["mean_volume_db"],
           "chunks": [{k: v for k, v in c.items() if k != "pcm_sha256"} for c in art["chunks"]]}
    with db.session(db_path) as c:
        c.execute("UPDATE stages SET artifact=? WHERE project_id=? AND stage='audio'", (db.dumps(old), pid))

    import clipper.media.audio as audio
    import clipper.models as models
    monkeypatch.setattr(models, "TRANSCRIPT_VERSION", models.TRANSCRIPT_VERSION + 1)
    real_detect, real_extract = audio.detect_silences, audio.extract_analysis_audio
    detected, extracted = [], []
    monkeypatch.setattr(audio, "detect_silences", lambda *a, **k: detected.append(a[0]) or real_detect(*a, **k))
    monkeypatch.setattr(audio, "extract_analysis_audio", lambda *a, **k: extracted.append(1) or real_extract(*a, **k))
    seen = _spy_merge(monkeypatch)

    job = _analyze(worker, db_path, pid, resume=True)
    assert job["state"] == "succeeded", job["error_message"]
    assert extracted == []                      # the older audio artifact was reused as it is
    assert detected == [storage.work_dir(pid) / "audio" / "analysis.wav"]
    assert seen == [stored_pauses]
    assert len(fake_t.calls) == 3               # only the first analysis paid; nothing re-billed
    assert len(list(_chunk_cache(storage).iterdir())) == 6     # + the same results under content keys


def test_older_audio_artifact_without_its_wav_is_prepared_again_locally(env, monkeypatch, tmp_path):
    settings, storage, db_path, worker = env
    pid, fake_t = _analyze_source(env, monkeypatch, _source(tmp_path, "mp4"))
    art = _audio_artifact(db_path, pid)
    old = {"duration": art["duration"], "mean_volume_db": art["mean_volume_db"],
           "chunks": [{k: v for k, v in c.items() if k != "pcm_sha256"} for c in art["chunks"]]}
    with db.session(db_path) as c:
        c.execute("UPDATE stages SET artifact=? WHERE project_id=? AND stage='audio'", (db.dumps(old), pid))
    (storage.work_dir(pid) / "audio" / "analysis.wav").unlink()
    import clipper.models as models
    monkeypatch.setattr(models, "TRANSCRIPT_VERSION", models.TRANSCRIPT_VERSION + 1)
    seen = _spy_merge(monkeypatch)

    job = _analyze(worker, db_path, pid, resume=True)
    assert job["state"] == "succeeded", job["error_message"]
    new = _audio_artifact(db_path, pid)
    assert new["silences"] == art["silences"] and new["chunks"] == art["chunks"]
    assert seen == [[tuple(p) for p in art["silences"]]]
    assert len(fake_t.calls) == 3               # identical audio: found by content, nothing re-billed
