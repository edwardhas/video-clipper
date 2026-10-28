"""Job lifecycle regressions: resume/cancel/restart edge cases with real FFmpeg and FAKE providers.

Each test reproduces a verified defect (cancel or kill at an awkward moment, concurrent
edits, re-selection, cleanup, orphaned jobs) and checks the invariants: the upload is
never lost, paid work is checkpointed before a cancel is honoured, a cancelled job never
changes what is published, and a clip is never 'ready' without the render it names.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import socket
import threading
import time
from pathlib import Path

import pytest

from clipper import db
from clipper.errors import CancelledError, InterruptedError_
from clipper.jobs import queue
from clipper.jobs.cache import FileJsonCache
from clipper.models import Candidate, ClipEdit, FramingPlan, ProjectOptions, ReviewDecision
from clipper.storage import new_id
from synthetic import FakeSelector, FakeTranscriber, make_source
from test_pipeline_e2e import _create_upload_project, _install_fakes, _run_next, env  # noqa: F401 (fixture)

pytestmark = [
    pytest.mark.ffmpeg,
    pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed"),
]

OPTS = ProjectOptions(clip_count=3, min_seconds=8, max_seconds=16)


# ----------------------------------------------------------------------------- helpers
def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _analyze(worker, db_path, pid: str, **params) -> dict:
    with db.session(db_path) as c:
        queue.create_job(c, project_id=pid, kind="analyze", params=params or None)
    return _run_next(worker, db_path)


def _clips(db_path, pid: str) -> list[dict]:
    with db.session(db_path) as c:
        return [dict(r) for r in c.execute("SELECT * FROM clips WHERE project_id=? ORDER BY rank", (pid,))]


def _clip(db_path, cid: str) -> dict | None:
    with db.session(db_path) as c:
        row = c.execute("SELECT * FROM clips WHERE id=?", (cid,)).fetchone()
    return dict(row) if row is not None else None


def _project(db_path, pid: str) -> dict:
    with db.session(db_path) as c:
        return dict(c.execute("SELECT * FROM projects WHERE id=?", (pid,)).fetchone())


def _queue_reselect(db_path, pid: str, options: ProjectOptions | None = None) -> str:
    """What POST /reselect writes (new options on the project, previous ones on the job)."""
    with db.session(db_path) as c:
        prev = c.execute("SELECT options FROM projects WHERE id=?", (pid,)).fetchone()["options"]
        if options is not None:
            c.execute("UPDATE projects SET options=? WHERE id=?", (options.model_dump_json(), pid))
        return queue.create_job(c, project_id=pid, kind="analyze",
                                params={"reselect": True, "previous_options": json.loads(prev)})


def _trim(db_path, clip: dict, seconds: float = 2.0) -> ClipEdit:
    edit = ClipEdit.model_validate_json(clip["edit"])
    new = edit.model_copy(update={"src_start": edit.src_start + seconds})
    with db.session(db_path) as c:
        c.execute("UPDATE clips SET edit=?, render_state='stale' WHERE id=?", (new.model_dump_json(), clip["id"]))
    return new


def _ready_project(env, monkeypatch, selector=None, **kw):
    settings, storage, db_path, worker = env
    pid = _create_upload_project(settings, storage, db_path, options=OPTS, **kw)
    fake_t, fake_s = FakeTranscriber(db_path, pid), selector or FakeSelector()
    _install_fakes(monkeypatch, fake_t, fake_s)
    job = _analyze(worker, db_path, pid)
    assert job["state"] == "succeeded", job["error_message"]
    return pid, fake_t, fake_s


# ----------------------------------------------------------------------------- import (finding 0)
def test_cancel_during_upload_import_then_resume_keeps_the_upload(env, monkeypatch):
    settings, storage, db_path, worker = env
    pid = _create_upload_project(settings, storage, db_path, options=OPTS)
    _install_fakes(monkeypatch, FakeTranscriber(db_path, pid), FakeSelector())
    import clipper.jobs.pipeline as pipeline
    real_hash = pipeline.sha256_file
    ref: dict = {}

    def hash_after_cancel(path, *a, **k):
        with db.session(db_path) as c:      # Cancel pressed while the moved upload is being hashed
            queue.request_cancel(c, ref["id"])
        return real_hash(path, *a, **k)

    monkeypatch.setattr(pipeline, "sha256_file", hash_after_cancel)
    with db.session(db_path) as c:
        ref["id"] = queue.create_job(c, project_id=pid, kind="analyze")
    assert _run_next(worker, db_path)["state"] == "cancelled"
    moved = storage.source_dir(pid) / "source.mp4"
    size = moved.stat().st_size                 # the upload now only exists here

    monkeypatch.setattr(pipeline, "sha256_file", real_hash)
    job = _analyze(worker, db_path, pid, resume=True)
    assert job["state"] == "succeeded", job["error_message"]
    assert moved.stat().st_size == size
    assert _project(db_path, pid)["source_path"] == storage.rel(moved)


def test_resume_after_kill_between_upload_move_and_row_update(env, monkeypatch):
    settings, storage, db_path, worker = env
    pid = _create_upload_project(settings, storage, db_path, options=OPTS)
    _install_fakes(monkeypatch, FakeTranscriber(db_path, pid), FakeSelector())
    with db.session(db_path) as c:
        up = c.execute("SELECT u.* FROM uploads u JOIN projects p ON p.upload_id=u.id WHERE p.id=?",
                       (pid,)).fetchone()
    # the worker was killed right after the atomic move, before uploads.path was repointed
    src_dir = storage.source_dir(pid)
    src_dir.mkdir(parents=True)
    os.replace(storage.resolve(up["path"]), src_dir / "source.mp4")
    job = _analyze(worker, db_path, pid)
    assert job["state"] == "succeeded", job["error_message"]
    with db.session(db_path) as c:
        row = c.execute("SELECT path, state FROM uploads WHERE id=?", (up["id"],)).fetchone()
    assert (row["path"], row["state"]) == (storage.rel(src_dir / "source.mp4"), "attached")
    assert not storage.upload_dir(up["id"]).exists()


# ----------------------------------------------------------------------------- import (finding 36)
def _youtube_project(db_path) -> str:
    pid = new_id("p")
    ts = db.now()
    with db.session(db_path) as c:
        c.execute("""INSERT INTO projects(id, title, source_kind, source_url, youtube_id, options, status,
                         created_at, updated_at)
                     VALUES (?, 't', 'youtube', 'https://www.youtube.com/watch?v=abcdefghijk', 'abcdefghijk', ?,
                             'queued', ?, ?)""", (pid, OPTS.model_dump_json(), ts, ts))
    return pid


def test_interrupted_youtube_download_keeps_partial_files_for_resume(env, monkeypatch, tmp_path):
    settings, storage, db_path, worker = env
    pid = _youtube_project(db_path)
    _install_fakes(monkeypatch, FakeTranscriber(db_path, pid), FakeSelector())
    video = tmp_path / "video.mp4"
    make_source(video)
    import clipper.media.youtube as youtube
    monkeypatch.setattr(youtube, "fetch_metadata", lambda url, **k: {"title": "Talk", "duration": 48.0,
                                                                     "filesize_approx": video.stat().st_size})
    seen: list[bool] = []

    def download(url, out_dir, **k):
        part = Path(out_dir) / ".part" / "abcdefghijk.mp4.part"
        seen.append(part.exists())
        if len(seen) == 1:                      # worker stopped 1.5 GB into the download
            part.parent.mkdir(parents=True, exist_ok=True)
            part.write_bytes(b"x" * 1000)
            raise InterruptedError_("Processing was interrupted because the worker stopped.")
        final = Path(out_dir) / "abcdefghijk.mp4"
        shutil.copy(video, final)
        return final

    monkeypatch.setattr(youtube, "download", download)
    assert _analyze(worker, db_path, pid)["state"] == "interrupted"
    assert (storage.source_dir(pid) / "download" / ".part" / "abcdefghijk.mp4.part").exists()

    job = _analyze(worker, db_path, pid, resume=True)
    assert job["state"] == "succeeded", job["error_message"]
    assert seen == [False, True]                # the resumed attempt found the partial data to continue
    assert not (storage.source_dir(pid) / "download").exists()


def test_cancelled_youtube_download_leaves_no_partial_files(env, monkeypatch):
    settings, storage, db_path, worker = env
    pid = _youtube_project(db_path)
    import clipper.media.youtube as youtube
    monkeypatch.setattr(youtube, "fetch_metadata", lambda url, **k: {"title": "Talk", "duration": 48.0})

    def download(url, out_dir, **k):
        (Path(out_dir) / ".part").mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / ".part" / "x.part").write_bytes(b"x")
        raise CancelledError("Cancelled.")

    monkeypatch.setattr(youtube, "download", download)
    assert _analyze(worker, db_path, pid)["state"] == "cancelled"
    assert not (storage.source_dir(pid) / "download").exists()


# ----------------------------------------------------------------------------- orphans (findings 1/13)
def _orphan(db_path, *, hostname: str, worker_pid: int, started_at: float, heartbeat: float,
            reselect_from: ProjectOptions | None = None) -> tuple[str, str]:
    """A project whose analyze job is 'running' for a worker that is gone."""
    pid = _youtube_project(db_path)
    with db.session(db_path) as c:
        params = {"reselect": True, "previous_options": reselect_from.model_dump(mode="json")} \
            if reselect_from is not None else None
        jid = queue.create_job(c, project_id=pid, kind="analyze", params=params)
        wid = new_id("w")
        c.execute("INSERT INTO workers(id, pid, hostname, started_at, heartbeat_at) VALUES (?,?,?,?,?)",
                  (wid, worker_pid, hostname, started_at, heartbeat))
        queue.claim_next(c, wid)
        c.execute("UPDATE jobs SET heartbeat_at=? WHERE id=?", (heartbeat, jid))
        c.execute("UPDATE projects SET status='running' WHERE id=?", (pid,))
    return pid, jid


def _job_state(db_path, jid: str) -> str:
    with db.session(db_path) as c:
        return c.execute("SELECT state FROM jobs WHERE id=?", (jid,)).fetchone()["state"]


def test_restarted_container_recovers_job_of_previous_worker_with_same_pid(settings, storage, db_path):
    # Same container restarted: same hostname, and the fresh PID namespace hands the new worker the
    # crashed worker's PID again. Its job heartbeat is only seconds old (killed during a provider call).
    from clipper.worker import Worker
    pid, jid = _orphan(db_path, hostname=socket.gethostname(), worker_pid=os.getpid(),
                       started_at=queue.PROCESS_STARTED_AT - 600, heartbeat=db.now() - 5)
    Worker(settings).setup()
    assert _job_state(db_path, jid) == "interrupted"
    assert _project(db_path, pid)["status"] == "interrupted"


def test_running_worker_reclaims_orphan_once_its_heartbeat_is_stale(settings, storage, db_path, monkeypatch):
    # Recreated container: new hostname, so the old worker can never be proven dead; only its heartbeat
    # ages. Recovery must therefore repeat while the new worker runs, not only once at startup.
    import clipper.worker as worker_mod
    monkeypatch.setattr(worker_mod, "HEARTBEAT_SECONDS", 0.02)
    monkeypatch.setattr(worker_mod, "RECOVER_SECONDS", 0.05)
    pid, jid = _orphan(db_path, hostname="old-container-id", worker_pid=1, started_at=db.now() - 600,
                       heartbeat=db.now())
    w = worker_mod.Worker(settings)
    w.setup()
    # our own running job with an old heartbeat is never reclaimed by our own recovery passes
    own_pid = _youtube_project(db_path)
    with db.session(db_path) as c:
        own = queue.create_job(c, project_id=own_pid, kind="analyze")
        queue.claim_next(c, w.worker_id)
        c.execute("UPDATE jobs SET heartbeat_at=? WHERE id=?", (db.now() - 600, own))
    hb = threading.Thread(target=w._heartbeat_loop, daemon=True)
    hb.start()
    try:
        time.sleep(0.3)                         # several recovery passes: fresh heartbeat → left alone
        assert _job_state(db_path, jid) == "running"
        with db.session(db_path) as c:          # ... until the orphan's heartbeat goes stale
            c.execute("UPDATE jobs SET heartbeat_at=? WHERE id=?",
                      (db.now() - queue.HEARTBEAT_STALE_SECONDS - 1, jid))
        deadline = time.monotonic() + 10
        while _job_state(db_path, jid) == "running" and time.monotonic() < deadline:
            time.sleep(0.02)
    finally:
        w.stop_event.set()
        hb.join(5)
    assert _job_state(db_path, jid) == "interrupted"
    assert _project(db_path, pid)["status"] == "interrupted"
    assert _job_state(db_path, own) == "running"


def test_recovered_reselect_restores_previous_options(settings, storage, db_path):
    from clipper.worker import Worker
    original = ProjectOptions(clip_count=5, min_seconds=20, max_seconds=60)
    pid, jid = _orphan(db_path, hostname=socket.gethostname(), worker_pid=2**22 + 4321,
                       started_at=db.now() - 600, heartbeat=db.now() - 600, reselect_from=original)
    with db.session(db_path) as c:
        c.execute("UPDATE projects SET options=? WHERE id=?",
                  (ProjectOptions(clip_count=2, min_seconds=10, max_seconds=30).model_dump_json(), pid))
        c.execute("""INSERT INTO clips(id, project_id, rank, title, reason, score, selection, edit, created_at,
                         updated_at) VALUES ('c_0123456789abcdef', ?, 1, 't', 'r', 1, '{}', '{}', 0, 0)""", (pid,))
    Worker(settings).setup()
    p = _project(db_path, pid)
    assert p["status"] == "ready" and "previous clips are kept" in p["status_detail"]
    assert ProjectOptions.model_validate_json(p["options"]) == original


# ----------------------------------------------------------------------------- paid checkpoints (finding 4)
class CheckpointingSelector(FakeSelector):
    """Behaves like the real selector: one paid request per window / review batch, each saved through
    ``cache`` before a cancel is honoured. ``cancel_after`` requests a cancel after that many paid calls."""

    def __init__(self, db_path: Path, job_ref: dict) -> None:
        super().__init__()
        self.db_path, self.job_ref = db_path, job_ref
        self.paid: list[str] = []
        self.cancel_after: int | None = None

    def _unit(self, name: str, cache, cancel, compute):
        key = hashlib.sha256(name.encode()).hexdigest()
        hit = cache.get(key) if cache is not None else None
        if hit is None:
            self.paid.append(name)
            hit = compute()
            if cache is not None:
                cache.put(key, hit)
            if self.cancel_after is not None and len(self.paid) == self.cancel_after:
                with db.session(self.db_path) as c:
                    queue.request_cancel(c, self.job_ref["id"])
        cancel.raise_if_cancelled()
        return hit

    def find_candidates(self, transcript, options, *, cancel, usage, cache=None):
        everything = super().find_candidates(transcript, options, cancel=cancel, usage=usage, cache=cache)
        out = []
        for i, cand in enumerate(everything):           # one "window" per candidate
            data = self._unit(f"window-{i}", cache, cancel, lambda c=cand: [c.model_dump(mode="json")])
            out += [Candidate.model_validate(x) for x in data]
        return out

    def review(self, transcript, candidates, frames, options, *, cancel, usage, cache=None):
        everything = super().review(transcript, candidates, frames, options, cancel=cancel, usage=usage,
                                    cache=cache)
        out = []
        for i, dec in enumerate(everything):            # one "batch" per candidate
            data = self._unit(f"batch-{i}", cache, cancel, lambda d=dec: [d.model_dump(mode="json")])
            out += [ReviewDecision.model_validate(x) for x in data]
        return out


def test_cancelled_selection_and_review_keep_their_paid_windows_and_batches(env, monkeypatch):
    settings, storage, db_path, worker = env
    pid = _create_upload_project(settings, storage, db_path, options=OPTS)
    job_ref: dict = {}
    selector = CheckpointingSelector(db_path, job_ref)
    _install_fakes(monkeypatch, FakeTranscriber(db_path, pid), selector)

    selector.cancel_after = 2                   # Cancel during window 3 of "Find moments"
    with db.session(db_path) as c:
        job_ref["id"] = queue.create_job(c, project_id=pid, kind="analyze")
    assert _run_next(worker, db_path)["state"] == "cancelled"
    assert selector.paid == ["window-0", "window-1"]

    selector.cancel_after = 4                   # Resume, then Cancel during review batch 2
    with db.session(db_path) as c:
        job_ref["id"] = queue.create_job(c, project_id=pid, kind="analyze", params={"resume": True})
    assert _run_next(worker, db_path)["state"] == "cancelled"
    assert selector.paid == ["window-0", "window-1", "window-2", "batch-0"]

    selector.cancel_after = None
    job = _analyze(worker, db_path, pid, resume=True)
    assert job["state"] == "succeeded", job["error_message"]
    # every paid unit was requested exactly once over the three jobs
    assert selector.paid == ["window-0", "window-1", "window-2", "batch-0", "batch-1", "batch-2"]
    assert len(_clips(db_path, pid)) == 3
    assert len(list((storage.root / "cache" / "candidates-window").iterdir())) == 3
    assert len(list((storage.root / "cache" / "review-batch").iterdir())) == 3

    # other options → another stage key → its own checkpoint scope (nothing reused across scopes)
    _queue_reselect(db_path, pid, ProjectOptions(clip_count=2, min_seconds=8, max_seconds=16))
    assert _run_next(worker, db_path)["state"] == "succeeded"
    assert selector.paid[6:9] == ["window-0", "window-1", "window-2"]


def test_file_json_cache_is_scoped_and_tolerates_bad_entries(storage):
    a = FileJsonCache(storage, "candidates-window", "stage-key-a")
    b = FileJsonCache(storage, "candidates-window", "stage-key-b")
    key = "0" * 64
    assert a.get(key) is None
    a.put(key, [{"id": "c001"}])
    assert a.get(key) == [{"id": "c001"}] and b.get(key) is None
    assert FileJsonCache(storage, "candidates-window", "stage-key-a").get(key) == [{"id": "c001"}]
    a._path(key).write_text("{not json", encoding="utf-8")       # torn/corrupt entry → re-requested
    assert a.get(key) is None
    with pytest.raises(ValueError):
        FileJsonCache(storage, "../escape", "k").get(key)


# ----------------------------------------------------------------------- transcription cache (finding 14; round 2)
class _NoMoments(FakeSelector):
    """Finds nothing, so a run ends after transcription (no review/render work)."""

    def find_candidates(self, *a, **k):
        return []


def test_chunk_transcripts_are_reused_across_audio_versions_when_the_audio_is_unchanged(env, monkeypatch, tmp_path):
    settings, storage, db_path, worker = env
    source = tmp_path / "talk.mp4"
    make_source(source)
    calls = []
    import clipper.media.audio as audio

    for bump in (0, 0, 1):
        monkeypatch.setattr(audio, "AUDIO_VERSION", audio.AUDIO_VERSION + bump)
        pid = _create_upload_project(settings, storage, db_path, options=OPTS,
                                     src_maker=lambda p: shutil.copy(source, p))
        fake_t = FakeTranscriber(db_path, pid)
        _install_fakes(monkeypatch, fake_t, _NoMoments())
        assert _analyze(worker, db_path, pid)["state"] == "succeeded"
        assert _project(db_path, pid)["status"] == "no_clips"
        calls.append(len(fake_t.calls))
    # same source + same audio → paid chunks reused. Review round 2 changed the last value from 3: chunks are
    # cached by audio content, so a new AUDIO_VERSION that leaves an MP4's WAV unchanged re-bills nothing
    # (audio whose timeline did change is covered in test_lifecycle_round2_cache.py).
    assert calls == [3, 0, 0]


# ----------------------------------------------------------------------------- re-selection (findings 11, 12, 31, 33, 37)
def test_reselect_keeps_face_tracking_of_a_trimmed_clip(env, monkeypatch):
    from clipper.framing.analyze import REASON_PENDING_CROP, REASON_PENDING_FIT
    settings, storage, db_path, worker = env
    pid, _, _ = _ready_project(env, monkeypatch)
    clip = _clips(db_path, pid)[0]
    edit = _trim(db_path, clip)
    with db.session(db_path) as c:
        queue.create_job(c, project_id=pid, kind="render", clip_id=clip["id"], priority=10)
    assert _run_next(worker, db_path)["state"] == "succeeded"
    rendered = _clip(db_path, clip["id"])
    tracked = FramingPlan.model_validate_json(rendered["framing"])
    assert abs(tracked.src_start - edit.src_start) < 1e-6

    _queue_reselect(db_path, pid)               # same options: same moments, same clip ids
    assert _run_next(worker, db_path)["state"] == "succeeded"
    after = _clip(db_path, clip["id"])
    plan = FramingPlan.model_validate_json(after["framing"])
    assert after["render_state"] == "ready"
    assert after["render_hash"] == rendered["render_hash"]          # the tracked render is kept as is
    assert storage.resolve(after["render_path"]).exists()
    assert abs(plan.src_start - edit.src_start) < 1e-6 and plan.reason not in (REASON_PENDING_CROP,
                                                                               REASON_PENDING_FIT)


def test_reselect_analyses_framing_for_a_trimmed_clip_never_rendered_since(env, monkeypatch):
    from clipper.framing.analyze import REASON_PENDING_CROP, REASON_PENDING_FIT
    settings, storage, db_path, worker = env
    pid, _, _ = _ready_project(env, monkeypatch)
    clip = _clips(db_path, pid)[0]
    edit = _trim(db_path, clip)                 # saved without re-rendering
    _queue_reselect(db_path, pid)
    assert _run_next(worker, db_path)["state"] == "succeeded"
    after = _clip(db_path, clip["id"])
    plan = FramingPlan.model_validate_json(after["framing"])
    assert after["render_state"] == "ready" and after["render_hash"] != clip["render_hash"]
    assert abs(plan.src_start - edit.src_start) < 1e-6
    assert plan.reason not in (REASON_PENDING_CROP, REASON_PENDING_FIT)


def _edit_during_render_stage(monkeypatch, db_path, target: dict, change) -> None:
    """Apply ``change(conn, clip_row)`` right after stage_render read ``target``'s edit."""
    import clipper.render.renderer as renderer
    real = renderer.render_hash
    done = {"v": False}
    start = ClipEdit.model_validate_json(target["edit"]).src_start

    def hook(**kw):
        h = real(**kw)
        if not done["v"] and abs(kw["edit"].src_start - start) < 1e-6:
            done["v"] = True
            with db.session(db_path) as c:
                with db.tx(c):
                    change(c, c.execute("SELECT * FROM clips WHERE id=?", (target["id"],)).fetchone())
        return h

    monkeypatch.setattr(renderer, "render_hash", hook)


def test_edit_saved_during_reselect_is_published_stale(env, monkeypatch):
    from clipper.api.routes_clips import save_edit
    settings, storage, db_path, worker = env
    pid, _, _ = _ready_project(env, monkeypatch)
    target = _clips(db_path, pid)[0]

    def recolour(c, row):                       # the owner saves (without re-rendering), as PATCH does
        e = ClipEdit.model_validate_json(row["edit"])
        save_edit(c, row, e.model_copy(update={"style": e.style.model_copy(update={"color": "yellow"})}))

    _edit_during_render_stage(monkeypatch, db_path, target, recolour)
    _queue_reselect(db_path, pid)
    assert _run_next(worker, db_path)["state"] == "succeeded"
    after = _clip(db_path, target["id"])
    assert ClipEdit.model_validate_json(after["edit"]).style.color == "yellow"
    assert after["render_state"] == "stale" and after["render_hash"] == target["render_hash"]
    others = [c for c in _clips(db_path, pid) if c["id"] != target["id"]]
    assert others and all(c["render_state"] == "ready" for c in others)


def test_render_requested_during_reselect_keeps_its_queued_state(env, monkeypatch):
    from clipper.api.routes_clips import save_edit
    settings, storage, db_path, worker = env
    pid, _, _ = _ready_project(env, monkeypatch)
    target = _clips(db_path, pid)[0]

    def save_and_render(c, row):                # "Save and re-render" while the reselect runs
        e = ClipEdit.model_validate_json(row["edit"])
        save_edit(c, row, e.model_copy(update={"captions_enabled": False}))
        queue.create_job(c, project_id=pid, kind="render", clip_id=row["id"], priority=10)
        c.execute("UPDATE clips SET render_state='queued' WHERE id=?", (row["id"],))

    _edit_during_render_stage(monkeypatch, db_path, target, save_and_render)
    _queue_reselect(db_path, pid)
    assert _run_next(worker, db_path)["state"] == "succeeded"
    assert _clip(db_path, target["id"])["render_state"] == "queued"
    assert _run_next(worker, db_path)["state"] == "succeeded"      # the render job publishes the new edit
    after = _clip(db_path, target["id"])
    assert after["render_state"] == "ready" and after["srt_path"] is None


def test_clip_whose_render_vanished_is_never_published_ready(env, monkeypatch):
    settings, storage, db_path, worker = env
    pid, _, _ = _ready_project(env, monkeypatch)
    first, last = _clips(db_path, pid)[0], _clips(db_path, pid)[-1]

    def remove_first_render(c, row):            # e.g. the clip's files were deleted meanwhile
        storage.resolve(first["render_path"]).unlink()

    _edit_during_render_stage(monkeypatch, db_path, last, remove_first_render)
    _queue_reselect(db_path, pid)
    assert _run_next(worker, db_path)["state"] == "succeeded"
    after = _clip(db_path, first["id"])
    assert after["render_state"] == "none" and after["render_path"] is None
    for c in _clips(db_path, pid):
        if c["render_state"] == "ready":
            assert storage.resolve(c["render_path"]).exists()


def test_new_clip_whose_render_vanished_is_not_published(env, monkeypatch):
    settings, storage, db_path, worker = env
    pid = _create_upload_project(settings, storage, db_path, options=OPTS)
    _install_fakes(monkeypatch, FakeTranscriber(db_path, pid), FakeSelector())
    import clipper.render.renderer as renderer
    real = renderer.make_thumbnail
    made: list[Path] = []

    def thumb(mp4, out, **k):
        made.append(Path(mp4))
        real(mp4, out, **k)
        if len(made) == 3:                      # the first clip's directory is removed before publishing
            shutil.rmtree(made[0].parent)

    monkeypatch.setattr(renderer, "make_thumbnail", thumb)
    assert _analyze(worker, db_path, pid)["state"] == "succeeded"
    clips = _clips(db_path, pid)
    assert len(clips) == 2 and all(storage.resolve(c["render_path"]).exists() for c in clips)


def test_cancelled_running_reselect_restores_previous_options(env, monkeypatch):
    settings, storage, db_path, worker = env
    pid, _, selector = _ready_project(env, monkeypatch)
    clips = _clips(db_path, pid)
    real_find = selector.find_candidates
    ref: dict = {}

    def find_then_cancel(*a, **k):              # Cancel pressed during "Find moments"
        out = real_find(*a, **k)
        with db.session(db_path) as c:
            queue.request_cancel(c, ref["id"])
        return out

    selector.find_candidates = find_then_cancel
    ref["id"] = _queue_reselect(db_path, pid, ProjectOptions(clip_count=2, min_seconds=10, max_seconds=30,
                                                             captions_enabled=False))
    assert _run_next(worker, db_path)["state"] == "cancelled"
    p = _project(db_path, pid)
    assert p["status"] == "ready" and "previous clips are kept" in p["status_detail"]
    assert ProjectOptions.model_validate_json(p["options"]) == OPTS
    assert [c["id"] for c in _clips(db_path, pid)] == [c["id"] for c in clips]


def test_cancelled_jobs_never_change_the_published_thumbnail(env, monkeypatch):
    settings, storage, db_path, worker = env
    pid, _, _ = _ready_project(env, monkeypatch)
    clip, second = _clips(db_path, pid)[:2]
    thumb, thumb2 = storage.resolve(clip["thumb_path"]), storage.resolve(second["thumb_path"])
    before, before2 = _sha(thumb), _sha(thumb2)
    _trim(db_path, clip)
    import clipper.render.renderer as renderer
    real = renderer.make_thumbnail

    def thumb_then_cancel(*a, **k):             # Cancel lands right after the still was grabbed
        real(*a, **k)
        with db.session(db_path) as c:
            c.execute("UPDATE jobs SET cancel_requested=1 WHERE state='running'")

    monkeypatch.setattr(renderer, "make_thumbnail", thumb_then_cancel)
    with db.session(db_path) as c:              # 1) a render job of the trimmed clip
        queue.create_job(c, project_id=pid, kind="render", clip_id=clip["id"], priority=10)
    assert _run_next(worker, db_path)["state"] == "cancelled"
    after = _clip(db_path, clip["id"])
    assert after["thumb_path"] == clip["thumb_path"] and after["render_hash"] == clip["render_hash"]
    assert _sha(thumb) == before
    _trim(db_path, second)                      # 2) a re-selection that re-renders a trimmed clip
    _queue_reselect(db_path, pid)
    assert _run_next(worker, db_path)["state"] == "cancelled"
    for old, now, digest, path in ((clip, _clip(db_path, clip["id"]), before, thumb),
                                   (second, _clip(db_path, second["id"]), before2, thumb2)):
        assert now["thumb_path"] == old["thumb_path"] and now["render_hash"] == old["render_hash"]
        assert _sha(path) == digest

    monkeypatch.setattr(renderer, "make_thumbnail", real)
    with db.session(db_path) as c:              # 3) a render that completes publishes its own thumbnail
        queue.create_job(c, project_id=pid, kind="render", clip_id=clip["id"], priority=10)
    assert _run_next(worker, db_path)["state"] == "succeeded"
    done = _clip(db_path, clip["id"])
    assert Path(done["thumb_path"]).name == f"{done['render_hash']}.jpg"
    assert storage.resolve(done["thumb_path"]).exists() and not thumb.exists()


# ----------------------------------------------------------------------------- cleanup (finding 34)
def test_reselect_after_cleanup_reuses_transcript_without_reextracting_audio(env, monkeypatch):
    from clipper.api.routes_projects import _cleanup_work_dir
    settings, storage, db_path, worker = env
    pid, fake_t, _ = _ready_project(env, monkeypatch)
    _cleanup_work_dir(storage, pid)
    import clipper.media.audio as audio
    real = audio.extract_analysis_audio
    extracted: list[Path] = []

    def spy(src, out, **k):
        extracted.append(Path(out))
        return real(src, out, **k)

    monkeypatch.setattr(audio, "extract_analysis_audio", spy)
    _queue_reselect(db_path, pid)
    assert _run_next(worker, db_path)["state"] == "succeeded"
    assert extracted == [] and not (storage.work_dir(pid) / "audio").exists()
    assert len(fake_t.calls) == 3

    # control: when the transcript cannot be reused, the audio really is prepared again
    settings.transcribe_model = "whisper-2-test"
    _queue_reselect(db_path, pid)
    assert _run_next(worker, db_path)["state"] == "succeeded"
    assert len(extracted) == 1 and (storage.work_dir(pid) / "audio").exists()
