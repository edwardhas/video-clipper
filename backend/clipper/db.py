"""SQLite persistence (single host, WAL mode) shared by the API and the worker.

Use :func:`connect` for a short-lived connection (one per request / unit of work)
and :func:`tx` for an ``IMMEDIATE`` write transaction. JSON columns are stored as
TEXT; use :func:`dumps` / :func:`loads`.
"""

from __future__ import annotations

import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL              -- JSON
);

CREATE TABLE IF NOT EXISTS uploads (
    id          TEXT PRIMARY KEY,
    filename    TEXT NOT NULL,       -- sanitized original name (display only)
    path        TEXT NOT NULL,       -- relative to data_dir
    size_bytes  INTEGER NOT NULL,
    state       TEXT NOT NULL CHECK (state IN ('receiving','complete','attached','failed')),
    created_at  REAL NOT NULL,
    project_id  TEXT
);

CREATE TABLE IF NOT EXISTS projects (
    id             TEXT PRIMARY KEY,
    title          TEXT NOT NULL,
    source_kind    TEXT NOT NULL CHECK (source_kind IN ('youtube','upload')),
    source_url     TEXT,             -- normalized YouTube URL
    youtube_id     TEXT,
    upload_id      TEXT,
    source_path    TEXT,             -- relative path of imported media
    source_sha256  TEXT,
    media_info     TEXT,             -- JSON MediaInfo
    options        TEXT NOT NULL,    -- JSON ProjectOptions
    status         TEXT NOT NULL CHECK (status IN
                     ('queued','running','ready','no_clips','failed','cancelled','interrupted')),
    status_detail  TEXT,             -- user-facing explanation (errors / empty results)
    thumb_path     TEXT,
    created_at     REAL NOT NULL,
    updated_at     REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_projects_created ON projects(created_at DESC);

CREATE TABLE IF NOT EXISTS jobs (
    id                TEXT PRIMARY KEY,
    project_id        TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    clip_id           TEXT,
    kind              TEXT NOT NULL CHECK (kind IN ('analyze','render')),
    state             TEXT NOT NULL CHECK (state IN
                        ('queued','running','succeeded','failed','cancelled','interrupted')),
    priority          INTEGER NOT NULL DEFAULT 0,
    params            TEXT NOT NULL DEFAULT '{}',   -- JSON
    stage             TEXT,
    stage_detail      TEXT,
    progress_current  INTEGER,                      -- real step counts only (e.g. chunk 2 of 5)
    progress_total    INTEGER,
    attempts          INTEGER NOT NULL DEFAULT 0,
    cancel_requested  INTEGER NOT NULL DEFAULT 0,
    error_code        TEXT,
    error_message     TEXT,
    retryable         INTEGER,
    worker_id         TEXT,
    heartbeat_at      REAL,
    created_at        REAL NOT NULL,
    started_at        REAL,
    finished_at       REAL
);
CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state, priority DESC, created_at);
CREATE INDEX IF NOT EXISTS idx_jobs_project ON jobs(project_id, created_at DESC);

-- Per-project pipeline stage bookkeeping (resume skips 'done' stages whose
-- cache_key still matches).
CREATE TABLE IF NOT EXISTS stages (
    project_id   TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    stage        TEXT NOT NULL,
    state        TEXT NOT NULL CHECK (state IN ('pending','running','done','failed','skipped')),
    cache_key    TEXT,
    artifact     TEXT,              -- JSON: relative paths / small results
    detail       TEXT,              -- user-facing note
    attempts     INTEGER NOT NULL DEFAULT 0,
    started_at   REAL,
    finished_at  REAL,
    duration_s   REAL,
    PRIMARY KEY (project_id, stage)
);

CREATE TABLE IF NOT EXISTS clips (
    id               TEXT PRIMARY KEY,
    project_id       TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    rank             INTEGER NOT NULL,
    title            TEXT NOT NULL,
    reason           TEXT NOT NULL,
    score            REAL NOT NULL,          -- 0..100 editorial heuristic
    score_detail     TEXT,                   -- JSON sub-scores
    selection        TEXT NOT NULL,          -- JSON ClipSelection (AI's original choice)
    edit             TEXT NOT NULL,          -- JSON ClipEdit (current user edit)
    framing          TEXT,                   -- JSON FramingPlan for the current range (auto analysis)
    render_state     TEXT NOT NULL DEFAULT 'none'
                       CHECK (render_state IN ('none','queued','rendering','ready','failed','stale')),
    render_hash      TEXT,                   -- hash of the inputs of the current ready render
    render_path      TEXT,
    srt_path         TEXT,
    thumb_path       TEXT,
    render_info      TEXT,                   -- JSON validation summary
    render_error     TEXT,
    created_at       REAL NOT NULL,
    updated_at       REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_clips_project ON clips(project_id, rank);

CREATE TABLE IF NOT EXISTS api_usage (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id        TEXT,
    job_id            TEXT,
    stage             TEXT NOT NULL,
    provider          TEXT NOT NULL,
    model             TEXT NOT NULL,
    request_id        TEXT,
    status            TEXT NOT NULL,          -- ok | error | cached
    input_tokens      INTEGER,
    output_tokens     INTEGER,
    reasoning_tokens  INTEGER,
    cached_tokens     INTEGER,
    audio_seconds     REAL,
    duration_s        REAL,
    error_code        TEXT,
    created_at        REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_usage_project ON api_usage(project_id);

CREATE TABLE IF NOT EXISTS sessions (
    token_hash    TEXT PRIMARY KEY,           -- HMAC-SHA256(session_secret, token)
    csrf_token    TEXT NOT NULL,
    username      TEXT NOT NULL,
    created_at    REAL NOT NULL,
    last_seen_at  REAL NOT NULL,
    expires_at    REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS login_attempts (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ip       TEXT NOT NULL,
    at       REAL NOT NULL,
    success  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_login_attempts ON login_attempts(ip, at);

CREATE TABLE IF NOT EXISTS workers (
    id            TEXT PRIMARY KEY,
    pid           INTEGER NOT NULL,
    hostname      TEXT,
    started_at    REAL NOT NULL,
    heartbeat_at  REAL NOT NULL,
    info          TEXT                        -- JSON: encoder, versions
);
"""


def now() -> float:
    return time.time()


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=_default)


def _default(o: Any) -> Any:
    if hasattr(o, "model_dump"):
        return o.model_dump(mode="json")
    raise TypeError(f"not JSON serialisable: {type(o)!r}")


def loads(value: str | None, default: Any = None) -> Any:
    if value is None or value == "":
        return default
    return json.loads(value)


def connect(db_path: Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=30.0, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


@contextmanager
def tx(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """``BEGIN IMMEDIATE`` transaction: serialises writers, so check-then-write is atomic."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


def init_db(db_path: Path) -> None:
    conn = connect(db_path)
    try:
        conn.executescript(SCHEMA)
        row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        if row is None:
            conn.execute("INSERT INTO meta(key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
        elif int(row["value"]) > SCHEMA_VERSION:
            raise RuntimeError(
                f"Database schema version {row['value']} is newer than this application ({SCHEMA_VERSION})."
            )
    finally:
        conn.close()


def row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


@contextmanager
def session(db_path: Path) -> Iterator[sqlite3.Connection]:
    conn = connect(db_path)
    try:
        yield conn
    finally:
        conn.close()
