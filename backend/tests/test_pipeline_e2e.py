"""End-to-end pipeline tests with real FFmpeg media and deterministic FAKE providers.

The fakes stand in for OpenAI only inside these tests (they are injected with
monkeypatch and are not reachable from the application). They validate the
mechanics — chunking/offsets, caching, publishing, cancellation, restart and
re-rendering — not editorial quality.
"""

from __future__ import annotations

import os
import shutil
import socket
from pathlib import Path

import pytest

from clipper import db
from clipper.jobs import queue
from clipper.models import ClipEdit, ProjectOptions
from clipper.storage import new_id
from synthetic import FakeSelector, FakeTranscriber, ground_truth_words, make_source

pytestmark = [
    pytest.mark.ffmpeg,
    pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed"),
]

@pytest.fixture
def env(settings, storage, db_path, monkeypatch):
    settings.transcribe_chunk_seconds = 20.0
    settings.transcribe_overlap_seconds = 3.0
    settings.min_free_disk_mb = 50
    from clipper.worker import Worker
    worker = Worker(settings)
    worker.setup()
    return settings, storage, db_path, worker


def _create_upload_project(settings, storage, db_path, src_maker=make_source, options=None) -> str:
    upload_id = new_id("u")
    udir = storage.upload_dir(upload_id)
    udir.mkdir(parents=True)
    src = udir / "talk.mp4"
    src_maker(src)
    pid = new_id("p")
    ts = db.now()
    opts = options or ProjectOptions(clip_count=3, min_seconds=8, max_seconds=16)
    with db.session(db_path) as c:
        c.execute("INSERT INTO uploads(id, filename, path, size_bytes, state, created_at) VALUES (?,?,?,?,?,?)",
                  (upload_id, "talk.mp4", storage.rel(src), src.stat().st_size, "complete", ts))
        c.execute("""INSERT INTO projects(id, title, source_kind, upload_id, options, status, created_at, updated_at)
                     VALUES (?, 'talk', 'upload', ?, ?, 'queued', ?, ?)""",
                  (pid, upload_id, opts.model_dump_json(), ts, ts))
    return pid


def _run_next(worker, db_path) -> dict:
    conn = db.connect(db_path)
    try:
        job = queue.claim_next(conn, worker.worker_id)
        assert job is not None
        worker.run_job(conn, dict(job))
        return dict(conn.execute("SELECT * FROM jobs WHERE id=?", (job["id"],)).fetchone())
    finally:
        conn.close()


def _install_fakes(monkeypatch, transcriber, selector):
    import clipper.ai.factory as factory
    monkeypatch.setattr(factory, "get_transcriber", lambda s: transcriber)
    monkeypatch.setattr(factory, "get_selector", lambda s: selector)


def test_full_pipeline_then_caption_edit_rerender(env, monkeypatch):
    settings, storage, db_path, worker = env
    pid = _create_upload_project(settings, storage, db_path)
    fake_t = FakeTranscriber(db_path, pid)
    fake_s = FakeSelector()
    _install_fakes(monkeypatch, fake_t, fake_s)
    with db.session(db_path) as c:
        queue.create_job(c, project_id=pid, kind="analyze")
    job = _run_next(worker, db_path)
    assert job["state"] == "succeeded", job["error_message"]

    with db.session(db_path) as c:
        project = c.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
        clips = c.execute("SELECT * FROM clips WHERE project_id=? ORDER BY rank", (pid,)).fetchall()
        st = {r["stage"]: r["state"] for r in c.execute("SELECT * FROM stages WHERE project_id=?", (pid,))}
    assert project["status"] == "ready"
    assert len(clips) == 3
    from clipper.models import ClipSelection
    for row in clips:   # each clip remembers the style it was first rendered with ("Restore original edit")
        assert ClipSelection.model_validate_json(row["selection"]).original_style == \
            ClipEdit.model_validate_json(row["edit"]).style
    assert all(v == "done" for v in st.values()) and len(st) == 8
    assert len(fake_t.calls) == 3          # 48 s at 20 s chunks → 3 chunks
    assert fake_s.find_calls == 1 and fake_s.review_calls == 1
    # the original upload was moved into the project (never copied/left behind)
    assert not storage.upload_dir(project["upload_id"]).exists()

    from clipper.models import Transcript
    t = Transcript.model_validate_json((storage.work_dir(pid) / "transcript.json").read_text())
    gt = ground_truth_words()
    assert len(t.words) == len(gt)         # overlap reconciliation: no duplicated or lost words
    for w, g in zip(t.words, gt):
        assert abs(w.start - g[1]) < 1e-6 and abs(w.end - g[2]) < 1e-6

    from clipper.render.renderer import validate_output
    for clip in clips:
        mp4 = storage.resolve(clip["render_path"])
        edit = ClipEdit.model_validate_json(clip["edit"])
        info = validate_output(mp4, expected_duration=edit.src_end - edit.src_start, settings=settings)
        assert (info["width"], info["height"]) == (1080, 1920)
        assert storage.resolve(clip["srt_path"]).read_text().startswith("1\n")

    # --- caption edit → re-render with zero provider calls
    from clipper.render.captions import apply_caption_edit
    from clipper.ai.transcript import words_between
    clip = clips[0]
    edit = ClipEdit.model_validate_json(clip["edit"])
    words = words_between(t, edit.src_start, edit.src_end)
    new_edit = apply_caption_edit(edit, words, words[0].id, words[0].id, "CLEAR")
    new_edit = new_edit.model_copy(update={"style": new_edit.style.model_copy(update={"color": "yellow"})})
    calls_before = (len(fake_t.calls), fake_s.find_calls, fake_s.review_calls)
    with db.session(db_path) as c:
        c.execute("UPDATE clips SET edit=?, render_state='stale' WHERE id=?", (new_edit.model_dump_json(), clip["id"]))
        queue.create_job(c, project_id=pid, kind="render", clip_id=clip["id"], priority=10)
    rjob = _run_next(worker, db_path)
    assert rjob["state"] == "succeeded", rjob["error_message"]
    assert (len(fake_t.calls), fake_s.find_calls, fake_s.review_calls) == calls_before
    with db.session(db_path) as c:
        after = c.execute("SELECT * FROM clips WHERE id=?", (clip["id"],)).fetchone()
    assert after["render_state"] == "ready" and after["render_hash"] != clip["render_hash"]
    assert storage.resolve(after["render_path"]).exists()
    assert not storage.resolve(clip["render_path"]).exists()      # old render cleaned up
    assert "CLEAR" in storage.resolve(after["srt_path"]).read_text()


def test_cancel_during_transcription_then_resume_reuses_paid_results(env, monkeypatch):
    settings, storage, db_path, worker = env
    pid = _create_upload_project(settings, storage, db_path)
    job_ref: dict = {}
    fake_t = FakeTranscriber(db_path, pid, cancel_on_call=2, job_ref=job_ref)
    fake_s = FakeSelector()
    _install_fakes(monkeypatch, fake_t, fake_s)
    with db.session(db_path) as c:
        job_ref["id"] = queue.create_job(c, project_id=pid, kind="analyze")
    job = _run_next(worker, db_path)
    assert job["state"] == "cancelled"
    with db.session(db_path) as c:
        assert c.execute("SELECT status FROM projects WHERE id=?", (pid,)).fetchone()["status"] == "cancelled"
        assert c.execute("SELECT COUNT(*) FROM clips WHERE project_id=?", (pid,)).fetchone()[0] == 0
    assert fake_t.calls == ["chunk_000.flac", "chunk_001.flac"]
    assert fake_s.find_calls == 0

    # resume: the two completed (paid) chunk results are reused; only chunk 3 is requested
    fake_t.cancel_on_call = None
    with db.session(db_path) as c:
        queue.create_job(c, project_id=pid, kind="analyze", params={"resume": True})
    job2 = _run_next(worker, db_path)
    assert job2["state"] == "succeeded", job2["error_message"]
    assert fake_t.calls == ["chunk_000.flac", "chunk_001.flac", "chunk_002.flac"]
    # the cancelled job stays cancelled forever
    with db.session(db_path) as c:
        assert c.execute("SELECT state FROM jobs WHERE id=?", (job_ref["id"],)).fetchone()["state"] == "cancelled"


def test_restart_recovery_marks_interrupted_and_resume_completes(env, monkeypatch):
    settings, storage, db_path, worker = env
    pid = _create_upload_project(settings, storage, db_path)
    fake_t, fake_s = FakeTranscriber(db_path, pid), FakeSelector()
    _install_fakes(monkeypatch, fake_t, fake_s)
    with db.session(db_path) as c:
        jid = queue.create_job(c, project_id=pid, kind="analyze")
        ts = db.now()
        c.execute("INSERT INTO workers(id, pid, hostname, started_at, heartbeat_at) VALUES (?,?,?,?,?)",
                  ("w_deadbeef00000000", 2**22 + 777, socket.gethostname(), ts - 900, ts - 900))
        queue.claim_next(c, "w_deadbeef00000000")       # a worker that then "crashed"
        c.execute("UPDATE projects SET status='running' WHERE id=?", (pid,))
    from clipper.worker import Worker
    restarted = Worker(settings)
    restarted.setup()                                     # restart → recovery
    with db.session(db_path) as c:
        assert c.execute("SELECT state FROM jobs WHERE id=?", (jid,)).fetchone()["state"] == "interrupted"
        assert c.execute("SELECT status FROM projects WHERE id=?", (pid,)).fetchone()["status"] == "interrupted"
        queue.create_job(c, project_id=pid, kind="analyze", params={"resume": True})
    job = _run_next(restarted, db_path)
    assert job["state"] == "succeeded", job["error_message"]


def test_insufficient_speech_is_explained_not_failed(env, monkeypatch):
    settings, storage, db_path, worker = env
    pid = _create_upload_project(settings, storage, db_path)
    fake_t, fake_s = FakeTranscriber(db_path, pid, sparse=True), FakeSelector()
    _install_fakes(monkeypatch, fake_t, fake_s)
    with db.session(db_path) as c:
        queue.create_job(c, project_id=pid, kind="analyze")
    job = _run_next(worker, db_path)
    assert job["state"] == "succeeded"
    with db.session(db_path) as c:
        p = c.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
        st = {r["stage"]: r["state"] for r in c.execute("SELECT * FROM stages WHERE project_id=?", (pid,))}
    assert p["status"] == "no_clips" and p["status_detail"]
    assert st["candidates"] == "skipped" and st["render"] == "skipped"
    assert fake_s.find_calls == 0


def test_video_without_audio_track(env, monkeypatch):
    settings, storage, db_path, worker = env
    pid = _create_upload_project(settings, storage, db_path,
                                 src_maker=lambda p: make_source(p, with_audio=False, seconds=12))
    fake_t, fake_s = FakeTranscriber(db_path, pid), FakeSelector()
    _install_fakes(monkeypatch, fake_t, fake_s)
    with db.session(db_path) as c:
        queue.create_job(c, project_id=pid, kind="analyze")
    job = _run_next(worker, db_path)
    assert job["state"] == "succeeded"
    with db.session(db_path) as c:
        p = c.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
    assert p["status"] == "no_clips" and "no audio" in p["status_detail"].lower()
    assert fake_t.calls == []


def test_corrupt_upload_fails_clearly(env, monkeypatch):
    settings, storage, db_path, worker = env
    pid = _create_upload_project(settings, storage, db_path,
                                 src_maker=lambda p: p.write_bytes(os.urandom(200_000)))
    _install_fakes(monkeypatch, FakeTranscriber(db_path, pid), FakeSelector())
    with db.session(db_path) as c:
        queue.create_job(c, project_id=pid, kind="analyze")
    job = _run_next(worker, db_path)
    assert job["state"] == "failed"
    assert job["error_message"] and "/" not in job["error_message"]
    with db.session(db_path) as c:
        p = c.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone()
        st = c.execute("SELECT state FROM stages WHERE project_id=? AND stage='probe'", (pid,)).fetchone()
    assert p["status"] == "failed" and st["state"] == "failed"
