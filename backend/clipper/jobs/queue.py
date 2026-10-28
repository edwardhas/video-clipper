"""Durable job queue on SQLite with guarded state transitions.

Invariants
----------
* A job reaches ``succeeded`` only via :func:`commit_success`, which runs the
  caller's result-publishing writes **inside the same IMMEDIATE transaction** as
  the check ``state='running' AND cancel_requested=0 AND worker_id=?``. If the
  job was cancelled (or reclaimed) meanwhile, nothing is published and
  ``JobLost`` is raised – so a cancelled job can never become successful.
* Cancel of a queued job is immediate; cancel of a running job sets
  ``cancel_requested`` and the worker kills its subprocess group.
* On worker start, and periodically while it runs, jobs left ``running`` by a
  dead or silent worker become ``interrupted`` (resumable). Resume/retry enqueue a
  *new* job; stage caches avoid repeating completed work.
"""

from __future__ import annotations

import os
import sqlite3
import time
from typing import Any, Callable

from .. import db
from ..storage import new_id

ACTIVE_STATES = ("queued", "running")
TERMINAL_STATES = ("succeeded", "failed", "cancelled", "interrupted")
HEARTBEAT_STALE_SECONDS = 45.0
# Workers register at startup, after this module is imported. A worker row that claims our own PID
# but was registered before this process existed belongs to an earlier process (see _worker_gone).
PROCESS_STARTED_AT = time.time()


class JobLost(Exception):
    """The job is no longer ours to finish (cancelled or reclaimed)."""


def create_job(conn: sqlite3.Connection, *, project_id: str, kind: str, clip_id: str | None = None,
               params: dict[str, Any] | None = None, priority: int = 0) -> str:
    job_id = new_id("j")
    conn.execute(
        """INSERT INTO jobs(id, project_id, clip_id, kind, state, priority, params, created_at)
           VALUES (?, ?, ?, ?, 'queued', ?, ?, ?)""",
        (job_id, project_id, clip_id, kind, priority, db.dumps(params or {}), db.now()),
    )
    return job_id


def active_job(conn: sqlite3.Connection, project_id: str, kind: str | None = None,
               clip_id: str | None = None) -> sqlite3.Row | None:
    sql = "SELECT * FROM jobs WHERE project_id=? AND state IN ('queued','running')"
    args: list[Any] = [project_id]
    if kind:
        sql += " AND kind=?"
        args.append(kind)
    if clip_id:
        sql += " AND clip_id=?"
        args.append(clip_id)
    sql += " ORDER BY created_at DESC LIMIT 1"
    return conn.execute(sql, args).fetchone()


def claim_next(conn: sqlite3.Connection, worker_id: str) -> sqlite3.Row | None:
    """Atomically move the highest-priority queued job to running for this worker."""
    ts = db.now()
    with db.tx(conn):
        row = conn.execute(
            """SELECT id FROM jobs WHERE state='queued' AND cancel_requested=0
               ORDER BY priority DESC, created_at ASC LIMIT 1"""
        ).fetchone()
        if row is None:
            return None
        conn.execute(
            """UPDATE jobs SET state='running', worker_id=?, heartbeat_at=?, started_at=?,
                   attempts=attempts+1, error_code=NULL, error_message=NULL, retryable=NULL
               WHERE id=? AND state='queued'""",
            (worker_id, ts, ts, row["id"]),
        )
        return conn.execute("SELECT * FROM jobs WHERE id=?", (row["id"],)).fetchone()


def heartbeat(conn: sqlite3.Connection, job_id: str, worker_id: str) -> None:
    conn.execute("UPDATE jobs SET heartbeat_at=? WHERE id=? AND worker_id=? AND state='running'",
                 (db.now(), job_id, worker_id))


def set_progress(conn: sqlite3.Connection, job_id: str, *, stage: str | None = None,
                 detail: str | None = None, current: int | None = None, total: int | None = None) -> None:
    """Record real progress (stage name + optional 'current of total' step counts)."""
    conn.execute(
        """UPDATE jobs SET stage=COALESCE(?, stage), stage_detail=?, progress_current=?, progress_total=?,
               heartbeat_at=? WHERE id=? AND state='running'""",
        (stage, detail, current, total, db.now(), job_id),
    )


def is_cancel_requested(conn: sqlite3.Connection, job_id: str) -> bool:
    row = conn.execute("SELECT cancel_requested, state FROM jobs WHERE id=?", (job_id,)).fetchone()
    return row is None or bool(row["cancel_requested"]) or row["state"] not in ACTIVE_STATES


def request_cancel(conn: sqlite3.Connection, job_id: str) -> str | None:
    """Cancel a job. Returns the resulting state (queued jobs cancel immediately)."""
    with db.tx(conn):
        row = conn.execute("SELECT state FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            return None
        if row["state"] == "queued":
            conn.execute(
                """UPDATE jobs SET state='cancelled', cancel_requested=1, finished_at=?,
                       error_code='cancelled', error_message='Cancelled by user.' WHERE id=?""",
                (db.now(), job_id),
            )
            return "cancelled"
        if row["state"] == "running":
            conn.execute("UPDATE jobs SET cancel_requested=1 WHERE id=?", (job_id,))
            return "cancelling"
        return row["state"]


def commit_success(conn: sqlite3.Connection, job_id: str, worker_id: str,
                   publish: Callable[[sqlite3.Connection], None] | None = None) -> None:
    """Publish results and mark succeeded atomically, unless cancelled/reclaimed.

    Raises ``JobLost`` (and publishes nothing) if the job is no longer running for
    this worker or cancellation was requested.
    """
    with db.tx(conn):
        row = conn.execute(
            "SELECT state, cancel_requested, worker_id FROM jobs WHERE id=?", (job_id,)
        ).fetchone()
        if row is None or row["state"] != "running" or row["cancel_requested"] or row["worker_id"] != worker_id:
            raise JobLost(job_id)
        if publish is not None:
            publish(conn)
        conn.execute(
            """UPDATE jobs SET state='succeeded', finished_at=?, stage='done', stage_detail=NULL
               WHERE id=? AND state='running' AND cancel_requested=0 AND worker_id=?""",
            (db.now(), job_id, worker_id),
        )


def finish(conn: sqlite3.Connection, job_id: str, worker_id: str, *, state: str,
           error_code: str | None = None, error_message: str | None = None,
           retryable: bool | None = None,
           on_finish: Callable[[sqlite3.Connection, str], None] | None = None) -> str | None:
    """Move a running job to failed/cancelled/interrupted. Returns the final state.

    A job with ``cancel_requested`` always ends ``cancelled`` regardless of the
    error that stopped it. ``on_finish(conn, final_state)`` runs in the same
    transaction (e.g. to update the project status).
    """
    assert state in ("failed", "cancelled", "interrupted")
    with db.tx(conn):
        row = conn.execute("SELECT state, cancel_requested, worker_id FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None or row["state"] != "running" or row["worker_id"] != worker_id:
            return row["state"] if row else None
        final = "cancelled" if row["cancel_requested"] else state
        if final == "cancelled":
            error_code, error_message, retryable = "cancelled", "Cancelled by user.", True
        conn.execute(
            """UPDATE jobs SET state=?, finished_at=?, error_code=?, error_message=?, retryable=?
               WHERE id=? AND state='running'""",
            (final, db.now(), error_code, error_message,
             None if retryable is None else int(retryable), job_id),
        )
        if on_finish is not None:
            on_finish(conn, final)
        return final


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - exists but not ours
        return True
    return True


def _worker_gone(w: sqlite3.Row | None) -> bool:
    """True when the worker row certainly belongs to a process that no longer runs.

    Only provable on the same host. A stored PID equal to our own PID that was registered
    before this process started is an earlier incarnation: a restarted container gets a fresh
    PID namespace, so the new worker typically receives the crashed worker's PID again and a
    plain liveness probe would find *itself*. Other hosts are left to the heartbeat check.
    """
    if w is None:
        return True
    if w["hostname"] != os.uname().nodename:
        return False
    pid = int(w["pid"])
    if pid == os.getpid() and (w["started_at"] or 0) < PROCESS_STARTED_AT:
        return True
    return not _pid_alive(pid)


def recover_interrupted(conn: sqlite3.Connection, current_worker_id: str,
                        on_interrupt: Callable[[sqlite3.Connection, sqlite3.Row], None] | None = None) -> list[str]:
    """Mark jobs orphaned by a dead/stale worker as interrupted. Returns their ids.

    Safe to call repeatedly (the worker does so every ~30 s): jobs of ``current_worker_id``
    are never touched, and a live worker refreshes its job heartbeat every few seconds,
    well inside ``HEARTBEAT_STALE_SECONDS``.
    """
    recovered: list[str] = []
    ts = db.now()
    with db.tx(conn):
        rows = conn.execute("SELECT * FROM jobs WHERE state='running'").fetchall()
        for job in rows:
            if job["worker_id"] == current_worker_id:
                continue
            w = conn.execute("SELECT pid, heartbeat_at, hostname, started_at FROM workers WHERE id=?",
                             (job["worker_id"],)).fetchone()
            stale = (job["heartbeat_at"] or 0) < ts - HEARTBEAT_STALE_SECONDS
            if not (stale or _worker_gone(w)):
                continue
            final = "cancelled" if job["cancel_requested"] else "interrupted"
            conn.execute(
                """UPDATE jobs SET state=?, finished_at=?, error_code=?, error_message=?, retryable=1
                   WHERE id=? AND state='running'""",
                (final, ts, final,
                 "Cancelled by user." if final == "cancelled"
                 else "Processing was interrupted (the app or worker stopped). You can resume it.",
                 job["id"]),
            )
            if on_interrupt is not None:
                on_interrupt(conn, job)
            recovered.append(job["id"])
    return recovered
