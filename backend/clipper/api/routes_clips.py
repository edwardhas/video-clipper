"""Clip editor endpoints.

Edits only change the stored ``ClipEdit`` (and mark a ready render ``stale``); they
never trigger transcription or selection. Rendering is an explicit, separate
request that enqueues a local-only ``render`` job (priority 10).

Times: ``src_*`` are source seconds; cues returned to the editor are clip-relative.
"""

from __future__ import annotations

import logging
import math
import sqlite3
import threading
from typing import Any

from fastapi import APIRouter, Depends

from .. import db
from ..config import Settings
from ..jobs import queue
from ..models import (
    MAX_CLIP_SECONDS, MIN_CLIP_SECONDS, ClipEdit, ClipSelection, ProjectOptions, Transcript,
)
from ..storage import Storage
from . import presenters
from .auth import require_auth, require_csrf
from .deps import conflict, get_conn, get_settings, get_storage, invalid, load_defaults, not_found, require_id
from .schemas import CaptionEditIn, ClipDetailOut, ClipEditIn, TranscriptWordOut

log = logging.getLogger("clipper.api")

CONTEXT_SECONDS = 30.0
RENDER_PRIORITY = 10

router = APIRouter(prefix="/api/clips", tags=["clips"],
                   dependencies=[Depends(require_auth), Depends(require_csrf)])


# ----------------------------------------------------------------------------- transcript cache
class _TranscriptCache:
    """Parsed transcripts keyed by file identity (path, mtime, size); a handful kept."""

    def __init__(self, maxsize: int = 4) -> None:
        self._lock = threading.Lock()
        self._items: dict[str, tuple[tuple[int, int], Transcript]] = {}
        self._maxsize = maxsize

    def load(self, storage: Storage, project_id: str) -> Transcript | None:
        path = storage.work_dir(project_id) / "transcript.json"
        try:
            st = path.stat()
        except OSError:
            return None
        key, ident = str(path), (st.st_mtime_ns, st.st_size)
        with self._lock:
            hit = self._items.get(key)
            if hit is not None and hit[0] == ident:
                return hit[1]
        try:
            transcript = Transcript.model_validate_json(path.read_bytes())
        except (OSError, ValueError):
            log.warning("transcript for project %s could not be read", project_id)
            return None
        with self._lock:
            self._items.pop(key, None)
            self._items[key] = (ident, transcript)
            while len(self._items) > self._maxsize:
                self._items.pop(next(iter(self._items)))
        return transcript


transcripts = _TranscriptCache()


# ----------------------------------------------------------------------------- helpers
def load_clip(conn: sqlite3.Connection, clip_id: str) -> sqlite3.Row:
    require_id(clip_id, "This clip")
    row = conn.execute("SELECT * FROM clips WHERE id=?", (clip_id,)).fetchone()
    if row is None:
        raise not_found("This clip")
    return row


def load_project_row(conn: sqlite3.Connection, project_id: str) -> sqlite3.Row:
    row = conn.execute("SELECT * FROM projects WHERE id=?", (project_id,)).fetchone()
    if row is None:
        raise not_found("This clip's project")
    return row


def source_duration(project: sqlite3.Row) -> float | None:
    info = presenters.media_info(project)
    return info.duration if info else None





def _fmt_time(seconds: float) -> str:
    total = max(0, int(seconds))
    return f"{total // 3600}:{total % 3600 // 60:02d}:{total % 60:02d}" if total >= 3600 \
        else f"{total // 60}:{total % 60:02d}"


def validate_range(src_start: float, src_end: float, duration: float | None) -> None:
    if not (math.isfinite(src_start) and math.isfinite(src_end)):
        raise invalid("The clip start and end must be numbers.")
    if src_start < 0 or src_start >= src_end:
        raise invalid("The clip must start at or after 0:00 and end after it starts.")
    if duration is not None and src_end > duration + 1e-3:
        raise invalid(f"The clip must end before the video ends ({_fmt_time(duration)}).")
    length = src_end - src_start
    if length < MIN_CLIP_SECONDS - 1e-6 or length > MAX_CLIP_SECONDS + 1e-6:
        raise invalid(f"A clip must be between {MIN_CLIP_SECONDS:g} seconds and "
                      f"{MAX_CLIP_SECONDS / 60:g} minutes long (this one is {length:.1f} s).")


def save_edit(conn: sqlite3.Connection, clip: sqlite3.Row, edit: ClipEdit) -> None:
    """Store a changed edit; a ready render becomes ``stale`` (still playable) until re-rendered."""
    if ClipEdit.model_validate_json(clip["edit"]) == edit:
        return
    state = "stale" if clip["render_state"] == "ready" else clip["render_state"]
    conn.execute("UPDATE clips SET edit=?, render_state=?, updated_at=? WHERE id=?",
                 (edit.model_dump_json(), state, db.now(), clip["id"]))


def _require_no_analysis(conn: sqlite3.Connection, project_id: str) -> None:
    """Clips can't be deleted while an analysis/re-selection may still (re)publish them."""
    if queue.active_job(conn, project_id, kind="analyze") is not None:
        raise conflict("This project is still processing. Cancel it or wait until it finishes, "
                       "then delete the clip.")


def _updated_edit(edit: ClipEdit, changes: dict[str, Any]) -> ClipEdit:
    data = edit.model_dump()
    data.update(changes)
    return ClipEdit.model_validate(data)


def clip_detail(conn: sqlite3.Connection, storage: Storage, clip_id: str) -> ClipDetailOut:
    from ..render.captions import build_cues   # lazy: owned by the render module

    clip = load_clip(conn, clip_id)
    project = load_project_row(conn, clip["project_id"])
    job = queue.active_job(conn, clip["project_id"], kind="render", clip_id=clip_id)
    base = presenters.clip_out(clip, job)
    edit = base.edit
    info = presenters.media_info(project)
    transcript = transcripts.load(storage, project["id"])
    duration = info.duration if info else (transcript.duration if transcript else edit.src_end)
    context_start = max(0.0, edit.src_start - CONTEXT_SECONDS)
    context_end = min(duration, edit.src_end + CONTEXT_SECONDS)
    # The editor also lists captions the owner hid (empty text) so they can be restored one by one;
    # render paths keep the default and never show them.
    cues = build_cues(transcript.words, edit, include_hidden=True) if transcript else []
    words = [TranscriptWordOut(id=w.id, text=w.text, start=w.start, end=w.end)
             for w in (transcript.words if transcript else [])
             if w.end > context_start and w.start < context_end]
    return ClipDetailOut.model_validate({
        **base.model_dump(), "cues": [c.model_dump() for c in cues], "words": [w.model_dump() for w in words],
        "context_start": context_start, "context_end": context_end, "source_duration": duration,
        "source_width": info.width if info else 0, "source_height": info.height if info else 0,
    })


# ----------------------------------------------------------------------------- routes
@router.get("/{clip_id}", response_model=ClipDetailOut)
def get_clip(clip_id: str, conn: sqlite3.Connection = Depends(get_conn),
             storage: Storage = Depends(get_storage)) -> ClipDetailOut:
    return clip_detail(conn, storage, clip_id)


@router.patch("/{clip_id}", response_model=ClipDetailOut)
def patch_clip(clip_id: str, body: ClipEditIn,
               conn: sqlite3.Connection = Depends(get_conn),
               storage: Storage = Depends(get_storage)) -> ClipDetailOut:
    load_clip(conn, clip_id)
    with db.tx(conn):
        clip = load_clip(conn, clip_id)
        edit = ClipEdit.model_validate_json(clip["edit"])
        changes: dict[str, Any] = {}
        if body.src_start is not None or body.src_end is not None:
            start = edit.src_start if body.src_start is None else float(body.src_start)
            end = edit.src_end if body.src_end is None else float(body.src_end)
            validate_range(start, end, source_duration(load_project_row(conn, clip["project_id"])))
            changes.update(src_start=start, src_end=end)
        if body.captions_enabled is not None:
            changes["captions_enabled"] = body.captions_enabled
        if body.style is not None:
            changes["style"] = body.style.model_dump()
        if body.framing is not None:
            changes["framing"] = body.framing.model_dump()
        if changes:
            save_edit(conn, clip, _updated_edit(edit, changes))
    return clip_detail(conn, storage, clip_id)


@router.post("/{clip_id}/captions", response_model=ClipDetailOut)
def edit_captions(clip_id: str, body: CaptionEditIn,
                  conn: sqlite3.Connection = Depends(get_conn),
                  storage: Storage = Depends(get_storage)) -> ClipDetailOut:
    from ..ai.transcript import words_between
    from ..render.captions import apply_caption_edit   # lazy: owned by the render module

    clip = load_clip(conn, clip_id)
    transcript = transcripts.load(storage, clip["project_id"])
    if transcript is None:
        raise conflict("The transcript for this project is not available, so captions cannot be edited.")
    with db.tx(conn):
        clip = load_clip(conn, clip_id)
        edit = ClipEdit.model_validate_json(clip["edit"])
        span = words_between(transcript, edit.src_start, edit.src_end)
        ids = {w.id for w in span}
        if body.first_word_id not in ids or body.last_word_id not in ids:
            raise invalid("That caption is outside this clip. Reload the clip and try again.")
        save_edit(conn, clip, apply_caption_edit(edit, span, body.first_word_id, body.last_word_id, body.text))
    return clip_detail(conn, storage, clip_id)


@router.post("/{clip_id}/captions/reset", response_model=ClipDetailOut)
def reset_captions(clip_id: str, conn: sqlite3.Connection = Depends(get_conn),
                   storage: Storage = Depends(get_storage)) -> ClipDetailOut:
    load_clip(conn, clip_id)
    with db.tx(conn):
        clip = load_clip(conn, clip_id)
        edit = ClipEdit.model_validate_json(clip["edit"])
        save_edit(conn, clip, _updated_edit(edit, {"word_overrides": {}, "cue_overrides": []}))
    return clip_detail(conn, storage, clip_id)


@router.post("/{clip_id}/reset", response_model=ClipDetailOut)
def reset_clip(clip_id: str, conn: sqlite3.Connection = Depends(get_conn),
               settings: Settings = Depends(get_settings),
               storage: Storage = Depends(get_storage)) -> ClipDetailOut:
    load_clip(conn, clip_id)
    default_style = load_defaults(conn, settings).caption_style
    with db.tx(conn):
        clip = load_clip(conn, clip_id)
        project = load_project_row(conn, clip["project_id"])
        selection = ClipSelection.model_validate_json(clip["selection"])
        # The style the clip was first rendered with; older clips without it use today's defaults.
        style = selection.original_style or default_style
        options = ProjectOptions.model_validate_json(project["options"])
        save_edit(conn, clip, ClipEdit(src_start=selection.src_start, src_end=selection.src_end,
                                       captions_enabled=options.captions_enabled, style=style))
    return clip_detail(conn, storage, clip_id)


@router.post("/{clip_id}/render", response_model=ClipDetailOut)
def render(clip_id: str, conn: sqlite3.Connection = Depends(get_conn),
           storage: Storage = Depends(get_storage)) -> ClipDetailOut:
    load_clip(conn, clip_id)
    with db.tx(conn):
        clip = load_clip(conn, clip_id)
        project = load_project_row(conn, clip["project_id"])
        if queue.active_job(conn, project["id"], kind="render", clip_id=clip_id) is None:
            if not project["source_path"]:
                raise conflict("The source video for this clip is not available, so it cannot be rendered.")
            queue.create_job(conn, project_id=project["id"], kind="render", clip_id=clip_id,
                             priority=RENDER_PRIORITY)
            if clip["render_state"] != "ready":
                conn.execute("UPDATE clips SET render_state='queued', render_error=NULL, updated_at=? WHERE id=?",
                             (db.now(), clip_id))
    return clip_detail(conn, storage, clip_id)


@router.delete("/{clip_id}")
def delete_clip(clip_id: str, conn: sqlite3.Connection = Depends(get_conn),
                storage: Storage = Depends(get_storage)) -> dict[str, bool]:
    clip = load_clip(conn, clip_id)
    project_id = clip["project_id"]
    _require_no_analysis(conn, project_id)
    for job in conn.execute("SELECT id FROM jobs WHERE clip_id=? AND state IN ('queued','running')",
                            (clip_id,)).fetchall():
        if queue.request_cancel(conn, job["id"]) == "cancelling":
            raise conflict("This clip is being rendered. Rendering was asked to stop; "
                           "try deleting it again in a few seconds.")
    with db.tx(conn):
        # Checked again in the same IMMEDIATE transaction as the delete: an analysis publishes under
        # the same write lock, so it can never re-create this clip with its files already removed.
        _require_no_analysis(conn, project_id)
        conn.execute("DELETE FROM clips WHERE id=?", (clip_id,))
    storage.delete_tree(storage.clip_dir(project_id, clip_id))
    storage.delete_tree(storage.work_dir(project_id) / "render" / clip_id)
    return {"deleted": True}
