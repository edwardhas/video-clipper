"""Per-project pipeline stage bookkeeping (the ``stages`` table)."""

from __future__ import annotations

import sqlite3
from typing import Any

from .. import db

STAGES: list[tuple[str, str]] = [
    ("import", "Import source"),
    ("probe", "Inspect media"),
    ("audio", "Extract audio"),
    ("transcribe", "Transcribe speech"),
    ("candidates", "Find moments"),
    ("review", "Review & rank"),
    ("framing", "Frame clips"),
    ("render", "Render clips"),
]
STAGE_KEYS = [k for k, _ in STAGES]
STAGE_LABELS = dict(STAGES)


def get(conn: sqlite3.Connection, project_id: str, stage: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM stages WHERE project_id=? AND stage=?", (project_id, stage)).fetchone()


def done_artifact(conn: sqlite3.Connection, project_id: str, stage: str, cache_key: str) -> dict[str, Any] | None:
    """Artifact of a completed stage whose cache key still matches, else None."""
    row = get(conn, project_id, stage)
    if row is None or row["state"] != "done" or row["cache_key"] != cache_key:
        return None
    return db.loads(row["artifact"], {}) or {}


def mark_running(conn: sqlite3.Connection, project_id: str, stage: str, cache_key: str | None) -> None:
    conn.execute(
        """INSERT INTO stages(project_id, stage, state, cache_key, attempts, started_at)
           VALUES (?, ?, 'running', ?, 1, ?)
           ON CONFLICT(project_id, stage) DO UPDATE SET state='running', cache_key=excluded.cache_key,
               attempts=stages.attempts+1, started_at=excluded.started_at, finished_at=NULL,
               duration_s=NULL, detail=NULL""",
        (project_id, stage, cache_key, db.now()),
    )


def mark_done(conn: sqlite3.Connection, project_id: str, stage: str, cache_key: str | None,
              artifact: dict[str, Any] | None = None, detail: str | None = None) -> None:
    ts = db.now()
    row = get(conn, project_id, stage)
    started = row["started_at"] if row is not None and row["started_at"] else ts
    conn.execute(
        """INSERT INTO stages(project_id, stage, state, cache_key, artifact, detail, attempts, started_at,
               finished_at, duration_s)
           VALUES (?, ?, 'done', ?, ?, ?, 1, ?, ?, ?)
           ON CONFLICT(project_id, stage) DO UPDATE SET state='done', cache_key=excluded.cache_key,
               artifact=excluded.artifact, detail=excluded.detail, finished_at=excluded.finished_at,
               duration_s=excluded.duration_s""",
        (project_id, stage, cache_key, db.dumps(artifact or {}), detail, started, ts, max(0.0, ts - started)),
    )


def mark_state(conn: sqlite3.Connection, project_id: str, stage: str, state: str, detail: str | None = None) -> None:
    assert state in ("pending", "failed", "skipped")
    conn.execute(
        """INSERT INTO stages(project_id, stage, state, detail, finished_at) VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(project_id, stage) DO UPDATE SET state=excluded.state, detail=excluded.detail,
               finished_at=excluded.finished_at""",
        (project_id, stage, state, detail, db.now()),
    )


def fail_running(conn: sqlite3.Connection, project_id: str, detail: str | None) -> None:
    """Mark whichever stage is currently running as failed (used on job failure)."""
    conn.execute("UPDATE stages SET state='failed', detail=?, finished_at=? WHERE project_id=? AND state='running'",
                 (detail, db.now(), project_id))


def reset_running(conn: sqlite3.Connection, project_id: str) -> None:
    """Running stages of an interrupted/cancelled job go back to pending."""
    conn.execute("UPDATE stages SET state='pending', finished_at=? WHERE project_id=? AND state='running'",
                 (db.now(), project_id))


def skip_after(conn: sqlite3.Connection, project_id: str, stage: str, detail: str) -> None:
    idx = STAGE_KEYS.index(stage)
    for key in STAGE_KEYS[idx + 1:]:
        mark_state(conn, project_id, key, "skipped", detail)
