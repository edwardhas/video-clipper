"""Durable job queue invariants: cancellation can never end in success; restart recovery."""

from __future__ import annotations

import os
import socket

import pytest

from clipper import db
from clipper.jobs import queue
from clipper.models import ProjectOptions


def _project(conn, pid="p_0123456789ab"):
    ts = db.now()
    conn.execute(
        """INSERT INTO projects(id, title, source_kind, source_url, youtube_id, options, status, created_at, updated_at)
           VALUES (?, 't', 'youtube', 'https://www.youtube.com/watch?v=abcdefghijk', 'abcdefghijk', ?, 'queued', ?, ?)""",
        (pid, ProjectOptions().model_dump_json(), ts, ts),
    )
    return pid


@pytest.fixture
def conn(db_path):
    c = db.connect(db_path)
    yield c
    c.close()


def _register_worker(conn, wid, pid=None, heartbeat=None):
    ts = db.now()
    conn.execute("INSERT INTO workers(id, pid, hostname, started_at, heartbeat_at) VALUES (?,?,?,?,?)",
                 (wid, pid or os.getpid(), socket.gethostname(), ts, heartbeat if heartbeat is not None else ts))


def test_claim_is_priority_then_fifo(conn):
    pid = _project(conn)
    a = queue.create_job(conn, project_id=pid, kind="analyze")
    r = queue.create_job(conn, project_id=pid, kind="render", clip_id="c_0123456789abcdef", priority=10)
    first = queue.claim_next(conn, "w_1")
    assert first["id"] == r and first["state"] == "running" and first["attempts"] == 1
    second = queue.claim_next(conn, "w_1")
    assert second["id"] == a
    assert queue.claim_next(conn, "w_1") is None


def test_cancel_queued_is_immediate_and_never_claimed(conn):
    pid = _project(conn)
    j = queue.create_job(conn, project_id=pid, kind="analyze")
    assert queue.request_cancel(conn, j) == "cancelled"
    assert queue.claim_next(conn, "w_1") is None
    assert conn.execute("SELECT state FROM jobs WHERE id=?", (j,)).fetchone()["state"] == "cancelled"


def test_cancelled_running_job_cannot_succeed_or_publish(conn):
    pid = _project(conn)
    j = queue.create_job(conn, project_id=pid, kind="analyze")
    queue.claim_next(conn, "w_1")
    assert queue.request_cancel(conn, j) == "cancelling"
    published = []
    with pytest.raises(queue.JobLost):
        queue.commit_success(conn, j, "w_1", publish=lambda c: published.append(1))
    assert published == []
    # the worker then finishes it: any requested cancel wins over the given state
    final = queue.finish(conn, j, "w_1", state="failed", error_code="x", error_message="boom")
    assert final == "cancelled"
    row = conn.execute("SELECT * FROM jobs WHERE id=?", (j,)).fetchone()
    assert row["state"] == "cancelled" and row["error_code"] == "cancelled"
    # a late success attempt after cancellation is still refused
    with pytest.raises(queue.JobLost):
        queue.commit_success(conn, j, "w_1")
    assert conn.execute("SELECT state FROM jobs WHERE id=?", (j,)).fetchone()["state"] == "cancelled"


def test_publish_runs_in_same_transaction_as_success(conn):
    pid = _project(conn)
    j = queue.create_job(conn, project_id=pid, kind="analyze")
    queue.claim_next(conn, "w_1")

    def publish(c):
        c.execute("UPDATE projects SET status='ready' WHERE id=?", (pid,))
        raise RuntimeError("publish failed midway")

    with pytest.raises(RuntimeError):
        queue.commit_success(conn, j, "w_1", publish)
    # rolled back: neither the publish nor the success transition happened
    assert conn.execute("SELECT status FROM projects WHERE id=?", (pid,)).fetchone()["status"] == "queued"
    assert conn.execute("SELECT state FROM jobs WHERE id=?", (j,)).fetchone()["state"] == "running"


def test_other_worker_cannot_commit(conn):
    pid = _project(conn)
    j = queue.create_job(conn, project_id=pid, kind="analyze")
    queue.claim_next(conn, "w_1")
    with pytest.raises(queue.JobLost):
        queue.commit_success(conn, j, "w_2")
    assert queue.finish(conn, j, "w_2", state="failed") == "running"


def test_recover_interrupted_marks_dead_worker_jobs(conn):
    pid = _project(conn)
    j = queue.create_job(conn, project_id=pid, kind="analyze")
    _register_worker(conn, "w_dead", pid=2**22 + 12345, heartbeat=db.now() - 600)  # no such pid
    queue.claim_next(conn, "w_dead")
    seen = []
    recovered = queue.recover_interrupted(conn, "w_new", on_interrupt=lambda c, row: seen.append(row["id"]))
    assert recovered == [j] and seen == [j]
    row = conn.execute("SELECT * FROM jobs WHERE id=?", (j,)).fetchone()
    assert row["state"] == "interrupted" and row["retryable"] == 1


def test_recover_keeps_live_worker_jobs(conn):
    pid = _project(conn)
    j = queue.create_job(conn, project_id=pid, kind="analyze")
    _register_worker(conn, "w_live")  # this test process: alive, fresh heartbeat
    queue.claim_next(conn, "w_live")
    queue.heartbeat(conn, j, "w_live")
    assert queue.recover_interrupted(conn, "w_new") == []
    assert conn.execute("SELECT state FROM jobs WHERE id=?", (j,)).fetchone()["state"] == "running"


def test_recover_previous_incarnation_with_our_own_pid(conn):
    """A restarted container's worker usually gets the crashed worker's PID again (fresh PID
    namespace, same hostname): probing that PID finds ourselves, yet that worker is gone."""
    pid = _project(conn)
    j = queue.create_job(conn, project_id=pid, kind="analyze")
    ts = db.now()
    conn.execute("INSERT INTO workers(id, pid, hostname, started_at, heartbeat_at) VALUES (?,?,?,?,?)",
                 ("w_old", os.getpid(), socket.gethostname(), queue.PROCESS_STARTED_AT - 600, ts - 5))
    queue.claim_next(conn, "w_old")                  # job heartbeat is fresh: not stale
    assert queue.recover_interrupted(conn, "w_new") == [j]
    assert conn.execute("SELECT state FROM jobs WHERE id=?", (j,)).fetchone()["state"] == "interrupted"


def test_recover_other_host_only_once_stale(conn):
    """A worker on another hostname (e.g. a recreated container) is never provably dead; its job is
    reclaimed once its heartbeat is older than HEARTBEAT_STALE_SECONDS (the worker re-checks periodically)."""
    pid = _project(conn)
    j = queue.create_job(conn, project_id=pid, kind="analyze")
    ts = db.now()
    conn.execute("INSERT INTO workers(id, pid, hostname, started_at, heartbeat_at) VALUES (?,?,?,?,?)",
                 ("w_gone", 1, "old-container-id", ts - 600, ts))
    queue.claim_next(conn, "w_gone")
    assert queue.recover_interrupted(conn, "w_new") == []
    conn.execute("UPDATE jobs SET heartbeat_at=? WHERE id=?", (ts - queue.HEARTBEAT_STALE_SECONDS - 1, j))
    assert queue.recover_interrupted(conn, "w_new") == [j]


def test_recover_cancel_requested_becomes_cancelled(conn):
    pid = _project(conn)
    j = queue.create_job(conn, project_id=pid, kind="analyze")
    _register_worker(conn, "w_dead", pid=2**22 + 12346, heartbeat=db.now() - 600)
    queue.claim_next(conn, "w_dead")
    queue.request_cancel(conn, j)
    queue.recover_interrupted(conn, "w_new")
    assert conn.execute("SELECT state FROM jobs WHERE id=?", (j,)).fetchone()["state"] == "cancelled"


def test_progress_only_updates_running_jobs(conn):
    pid = _project(conn)
    j = queue.create_job(conn, project_id=pid, kind="analyze")
    queue.set_progress(conn, j, stage="transcribe", current=1, total=3)
    assert conn.execute("SELECT stage FROM jobs WHERE id=?", (j,)).fetchone()["stage"] is None
    queue.claim_next(conn, "w_1")
    queue.set_progress(conn, j, stage="transcribe", detail="x", current=2, total=3)
    row = conn.execute("SELECT * FROM jobs WHERE id=?", (j,)).fetchone()
    assert (row["stage"], row["progress_current"], row["progress_total"]) == ("transcribe", 2, 3)
