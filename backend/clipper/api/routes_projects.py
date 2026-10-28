"""Project endpoints: create/list/detail/delete and job controls.

Job controls never run work in the request: they only create or cancel rows in
the durable job queue, which the worker process picks up.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends

from .. import db
from ..config import Settings
from ..jobs import queue
from ..models import ProjectOptions
from ..storage import Storage, is_valid_id, new_id
from . import presenters
from .auth import require_auth, require_csrf
from .deps import (
    ApiError,
    conflict,
    get_conn,
    get_settings,
    get_storage,
    invalid,
    load_defaults,
    not_found,
    options_from_defaults,
    require_id,
)
from .schemas import ProjectCreateIn, ProjectOut, ProjectSummaryOut, ReselectIn, UploadSource, YoutubeSource

log = logging.getLogger("clipper.api")

KEEP_ON_CLEANUP = frozenset({"transcript.json", "candidates.json", "review.json"})
BUSY_MESSAGE = "This project is still processing. Cancel it or wait until it finishes."

router = APIRouter(prefix="/api/projects", tags=["projects"],
                   dependencies=[Depends(require_auth), Depends(require_csrf)])


# ----------------------------------------------------------------------------- helpers
def load_project(conn: sqlite3.Connection, project_id: str) -> sqlite3.Row:
    require_id(project_id, "This project")
    row = conn.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
    if row is None:
        raise not_found("This project")
    return row


def _project_out(conn: sqlite3.Connection, project_id: str) -> ProjectOut:
    return presenters.project_out(conn, load_project(conn, project_id))


def check_clip_limit(options: ProjectOptions, settings: Settings) -> None:
    if options.clip_count > settings.max_clips:
        raise invalid(f"At most {settings.max_clips} clips per video are allowed.")


def _title_from_filename(filename: str) -> str:
    stem = Path(filename).stem.replace("_", " ").strip(" .-")
    return (stem or "Uploaded video")[:200]


def _youtube_fields(source: YoutubeSource) -> dict[str, Any]:
    from ..media.youtube import parse_youtube_url   # lazy: owned by the media module
    video_id, canonical = parse_youtube_url(source.url)
    return {"source_kind": "youtube", "source_url": canonical, "youtube_id": video_id,
            "upload_id": None, "title": f"YouTube video {video_id}"}


def _claim_upload(conn: sqlite3.Connection, source: UploadSource, project_id: str) -> dict[str, Any]:
    """Check (inside the caller's transaction) that the upload is complete and unused; attach it."""
    upload_id = source.upload_id
    row = None
    if is_valid_id(upload_id):
        row = conn.execute("SELECT * FROM uploads WHERE id=?", (upload_id,)).fetchone()
    if row is None:
        raise ApiError(404, "That upload was not found. Please upload the file again.", "not_found")
    if row["state"] == "receiving":
        raise conflict("That file is still uploading. Wait until the upload finishes.")
    if row["state"] != "complete" or row["project_id"]:
        raise conflict("That upload has already been used or is no longer available. Upload the file again.")
    conn.execute("UPDATE uploads SET state='attached', project_id=? WHERE id=?", (project_id, upload_id))
    return {"source_kind": "upload", "source_url": None, "youtube_id": None, "upload_id": upload_id,
            "title": _title_from_filename(row["filename"])}


def _insert_project(conn: sqlite3.Connection, project_id: str, fields: dict[str, Any],
                    options: ProjectOptions) -> None:
    ts = db.now()
    conn.execute(
        """INSERT INTO projects(id, title, source_kind, source_url, youtube_id, upload_id, options, status,
               created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?)""",
        (project_id, fields["title"], fields["source_kind"], fields["source_url"], fields["youtube_id"],
         fields["upload_id"], options.model_dump_json(), ts, ts),
    )


def _require_idle(conn: sqlite3.Connection, project_id: str) -> None:
    if queue.active_job(conn, project_id) is not None:
        raise conflict(BUSY_MESSAGE)


def _after_queued_cancel(conn: sqlite3.Connection, job: sqlite3.Row) -> None:
    """Bookkeeping for a job that was cancelled before the worker picked it up."""
    ts = db.now()
    if job["kind"] == "render":
        conn.execute(
            """UPDATE clips SET render_state=CASE WHEN render_path IS NOT NULL THEN 'stale' ELSE 'none' END,
                   updated_at=? WHERE id=? AND render_state IN ('queued','rendering')""",
            (ts, job["clip_id"]))
        return
    from ..jobs.pipeline import restore_reselect_options
    params = db.loads(job["params"], {}) or {}
    has_clips = conn.execute("SELECT COUNT(*) FROM clips WHERE project_id=?", (job["project_id"],)).fetchone()[0]
    if params.get("reselect") and has_clips:
        status, detail = "ready", "Re-selection was cancelled; your previous clips are kept."
        restore_reselect_options(conn, job["project_id"], params)
    else:
        status, detail = "cancelled", "Cancelled."
    conn.execute("UPDATE projects SET status=?, status_detail=?, updated_at=? WHERE id=?",
                 (status, detail, ts, job["project_id"]))


def _cleanup_work_dir(storage: Storage, project_id: str) -> None:
    work = storage.work_dir(project_id)
    if not work.is_dir():
        return
    for child in work.iterdir():
        if child.name in KEEP_ON_CLEANUP and child.is_file() and not child.is_symlink():
            continue
        if child.is_dir() and not child.is_symlink():
            storage.delete_tree(child)
        else:
            storage.delete_file(child)


# ----------------------------------------------------------------------------- routes
@router.get("", response_model=list[ProjectSummaryOut])
def list_projects(conn: sqlite3.Connection = Depends(get_conn)) -> list[ProjectSummaryOut]:
    rows = conn.execute("SELECT * FROM projects ORDER BY created_at DESC, rowid DESC").fetchall()
    return [presenters.project_summary_out(conn, r) for r in rows]


@router.post("", response_model=ProjectOut, status_code=201)
def create_project(body: ProjectCreateIn,
                   conn: sqlite3.Connection = Depends(get_conn),
                   settings: Settings = Depends(get_settings)) -> ProjectOut:
    if body.options is None:      # saved defaults may predate a lower configured cap: clamp them
        options = options_from_defaults(load_defaults(conn, settings))
        options = options.model_copy(update={"clip_count": max(1, min(options.clip_count, settings.max_clips))})
    else:
        options = body.options
    check_clip_limit(options, settings)
    project_id = new_id("p")
    youtube = _youtube_fields(body.source) if isinstance(body.source, YoutubeSource) else None
    with db.tx(conn):
        fields = youtube if youtube is not None else _claim_upload(conn, body.source, project_id)
        _insert_project(conn, project_id, fields, options)
        queue.create_job(conn, project_id=project_id, kind="analyze", priority=0)
    return _project_out(conn, project_id)


@router.get("/{project_id}", response_model=ProjectOut)
def get_project(project_id: str, conn: sqlite3.Connection = Depends(get_conn)) -> ProjectOut:
    return _project_out(conn, project_id)


@router.delete("/{project_id}")
def delete_project(project_id: str,
                   conn: sqlite3.Connection = Depends(get_conn),
                   storage: Storage = Depends(get_storage)) -> dict[str, bool]:
    load_project(conn, project_id)
    with db.tx(conn):           # IMMEDIATE: the worker cannot claim a job of this project meanwhile
        _require_idle(conn, project_id)
        upload_ids = [r["id"] for r in conn.execute("SELECT id FROM uploads WHERE project_id=?", (project_id,))]
        conn.execute("DELETE FROM api_usage WHERE project_id=?", (project_id,))
        conn.execute("DELETE FROM uploads WHERE project_id=?", (project_id,))
        conn.execute("DELETE FROM projects WHERE id=?", (project_id,))   # cascades jobs/stages/clips
    for upload_id in upload_ids:
        storage.delete_tree(storage.upload_dir(upload_id))
    storage.delete_tree(storage.project_dir(project_id))
    return {"deleted": True}


@router.post("/{project_id}/cancel", response_model=ProjectOut)
def cancel_project(project_id: str, conn: sqlite3.Connection = Depends(get_conn)) -> ProjectOut:
    load_project(conn, project_id)
    jobs = presenters.active_jobs(conn, project_id)
    if not jobs:
        raise conflict("Nothing is running for this project.")
    for job in jobs:
        if queue.request_cancel(conn, job["id"]) == "cancelled":
            with db.tx(conn):
                _after_queued_cancel(conn, job)
    return _project_out(conn, project_id)


@router.post("/{project_id}/resume", response_model=ProjectOut)
def resume_project(project_id: str, conn: sqlite3.Connection = Depends(get_conn)) -> ProjectOut:
    load_project(conn, project_id)
    with db.tx(conn):
        _require_idle(conn, project_id)
        row = conn.execute("SELECT status FROM projects WHERE id=?", (project_id,)).fetchone()
        if row["status"] not in presenters.RESUMABLE_STATUSES:
            raise conflict("Only failed, cancelled or interrupted projects can be resumed.")
        queue.create_job(conn, project_id=project_id, kind="analyze", params={"resume": True}, priority=0)
        conn.execute("UPDATE projects SET status='queued', status_detail=NULL, updated_at=? WHERE id=?",
                     (db.now(), project_id))
    return _project_out(conn, project_id)


@router.post("/{project_id}/reselect", response_model=ProjectOut)
def reselect_project(project_id: str, body: ReselectIn,
                     conn: sqlite3.Connection = Depends(get_conn),
                     settings: Settings = Depends(get_settings)) -> ProjectOut:
    load_project(conn, project_id)
    check_clip_limit(body.options, settings)
    with db.tx(conn):
        _require_idle(conn, project_id)
        if presenters.stage_state(conn, project_id, "transcribe") != "done":
            raise conflict("Moments can be re-selected only after the transcript is finished.")
        # The run reads the new options from the project; the previous ones travel with the job so they
        # are put back if it ends without publishing and the previous clips are kept.
        previous = conn.execute("SELECT options FROM projects WHERE id=?", (project_id,)).fetchone()["options"]
        conn.execute("UPDATE projects SET options=?, status='queued', status_detail=NULL, updated_at=? WHERE id=?",
                     (body.options.model_dump_json(), db.now(), project_id))
        queue.create_job(conn, project_id=project_id, kind="analyze", priority=0,
                         params={"reselect": True, "previous_options": db.loads(previous, None)})
    return _project_out(conn, project_id)


@router.post("/{project_id}/cleanup", response_model=ProjectOut)
def cleanup_project(project_id: str,
                    conn: sqlite3.Connection = Depends(get_conn),
                    storage: Storage = Depends(get_storage)) -> ProjectOut:
    load_project(conn, project_id)
    _require_idle(conn, project_id)
    _cleanup_work_dir(storage, project_id)
    return _project_out(conn, project_id)
