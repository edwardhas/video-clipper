"""Media files: rendered clips (Range-capable), captions, thumbnails, source frames, ZIP export.

Every file is located through a DB-stored relative path resolved by
``Storage.resolve`` (which refuses anything outside the data directory); ids in
the URL must look like app-generated ids. Missing files are a plain 404.
"""

from __future__ import annotations

import logging
import sqlite3
import zipfile
from pathlib import Path

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from ..config import Settings
from ..storage import Storage, new_id
from . import presenters
from .auth import require_auth
from .deps import ApiError, get_conn, get_settings, get_storage, invalid, not_found, require_id

log = logging.getLogger("clipper.api")

MAX_PREVIEW_FRAMES = 50
MAX_EXPORT_CLIPS = 50
PREVIEW_PREFIX = "preview_"
VERSIONED_CACHE = "private, max-age=3600"
UNVERSIONED_CACHE = "private, no-cache"
EXPORT_STATES = ("ready",)

router = APIRouter(prefix="/media", tags=["media"], dependencies=[Depends(require_auth)])


# ----------------------------------------------------------------------------- helpers
def _row(conn: sqlite3.Connection, table: str, row_id: str, what: str) -> sqlite3.Row:
    require_id(row_id, what)
    row = conn.execute(f"SELECT * FROM {table} WHERE id=?", (row_id,)).fetchone()  # table is a constant
    if row is None:
        raise not_found(what)
    return row


def stored_file(storage: Storage, rel_path: str | None, what: str = "This file") -> Path:
    """Absolute path of an existing file under ``projects/``, else 404 (never reveals the path)."""
    if not rel_path:
        raise not_found(what)
    try:
        path = storage.resolve(rel_path)
    except ValueError:
        raise not_found(what) from None
    if not path.is_relative_to(storage.root / "projects") or not path.is_file():
        raise not_found(what)
    return path


def clip_filename(title: str, rank: int, ext: str) -> str:
    return f"{presenters.slugify(title)}-{int(rank)}.{ext}"


def unique_name(name: str, taken: set[str]) -> str:
    stem, dot, ext = name.rpartition(".")
    candidate, n = name, 2
    while candidate in taken:
        candidate = f"{stem}-{n}{dot}{ext}"
        n += 1
    taken.add(candidate)
    return candidate


def _prune_previews(frames_dir: Path, keep: int = MAX_PREVIEW_FRAMES) -> None:
    files = []
    for f in frames_dir.glob(f"{PREVIEW_PREFIX}*.jpg"):
        try:
            files.append((f.stat().st_mtime, f))
        except OSError:
            continue
    files.sort()
    for _, f in files[:max(0, len(files) - keep)]:
        try:
            f.unlink()
        except OSError:
            pass


def _remove(path: Path) -> None:
    try:
        path.unlink()
    except OSError:
        pass


def parse_clip_ids(raw: str | None) -> list[str] | None:
    if raw is None or not raw.strip():
        return None
    ids = [p.strip() for p in raw.split(",") if p.strip()]
    if len(ids) > MAX_EXPORT_CLIPS:
        raise invalid(f"At most {MAX_EXPORT_CLIPS} clips can be exported at once.")
    for cid in ids:
        require_id(cid, "A selected clip")
    return list(dict.fromkeys(ids))


def build_export_zip(storage: Storage, clips: list[sqlite3.Row], include_srt: bool, out: Path) -> int:
    """Write the ZIP (videos stored uncompressed, captions deflated); returns the number of videos."""
    taken: set[str] = set()
    count = 0
    with zipfile.ZipFile(out, "w", allowZip64=True) as zf:
        for clip in clips:
            try:
                video = stored_file(storage, clip["render_path"])
            except ApiError:
                continue
            name = unique_name(clip_filename(clip["title"], clip["rank"], "mp4"), taken)
            zf.write(video, name, compress_type=zipfile.ZIP_STORED)
            count += 1
            if include_srt and clip["srt_path"]:
                try:
                    srt = stored_file(storage, clip["srt_path"])
                except ApiError:
                    continue
                srt_name = unique_name(name[:-4] + ".srt", taken)
                zf.write(srt, srt_name, compress_type=zipfile.ZIP_DEFLATED)
    return count


# ----------------------------------------------------------------------------- clip files
@router.api_route("/clips/{clip_id}/video.mp4", methods=["GET", "HEAD"])
def clip_video(clip_id: str, v: str | None = Query(default=None, max_length=128),
               conn: sqlite3.Connection = Depends(get_conn),
               storage: Storage = Depends(get_storage)) -> FileResponse:
    clip = _row(conn, "clips", clip_id, "This clip")
    path = stored_file(storage, clip["render_path"], "This clip's video")
    cache = VERSIONED_CACHE if v and clip["render_hash"] and v == clip["render_hash"] else UNVERSIONED_CACHE
    return FileResponse(path, media_type="video/mp4", headers={"Cache-Control": cache})


@router.api_route("/clips/{clip_id}/download.mp4", methods=["GET", "HEAD"])
def clip_download(clip_id: str, conn: sqlite3.Connection = Depends(get_conn),
                  storage: Storage = Depends(get_storage)) -> FileResponse:
    clip = _row(conn, "clips", clip_id, "This clip")
    path = stored_file(storage, clip["render_path"], "This clip's video")
    return FileResponse(path, media_type="video/mp4", filename=clip_filename(clip["title"], clip["rank"], "mp4"),
                        headers={"Cache-Control": UNVERSIONED_CACHE})


@router.api_route("/clips/{clip_id}/captions.srt", methods=["GET", "HEAD"])
def clip_captions(clip_id: str, conn: sqlite3.Connection = Depends(get_conn),
                  storage: Storage = Depends(get_storage)) -> FileResponse:
    clip = _row(conn, "clips", clip_id, "This clip")
    path = stored_file(storage, clip["srt_path"], "This clip's captions")
    return FileResponse(path, media_type="application/x-subrip",
                        filename=clip_filename(clip["title"], clip["rank"], "srt"),
                        headers={"Cache-Control": UNVERSIONED_CACHE})


@router.api_route("/clips/{clip_id}/thumb.jpg", methods=["GET", "HEAD"])
def clip_thumb(clip_id: str, conn: sqlite3.Connection = Depends(get_conn),
               storage: Storage = Depends(get_storage)) -> FileResponse:
    clip = _row(conn, "clips", clip_id, "This clip")
    path = stored_file(storage, clip["thumb_path"], "This thumbnail")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": UNVERSIONED_CACHE})


# ----------------------------------------------------------------------------- project files
@router.api_route("/projects/{project_id}/thumb.jpg", methods=["GET", "HEAD"])
def project_thumb(project_id: str, conn: sqlite3.Connection = Depends(get_conn),
                  storage: Storage = Depends(get_storage)) -> FileResponse:
    project = _row(conn, "projects", project_id, "This project")
    path = stored_file(storage, project["thumb_path"], "This thumbnail")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": UNVERSIONED_CACHE})


@router.get("/projects/{project_id}/frame.jpg")
def project_frame(project_id: str, t: float = Query(...),
                  conn: sqlite3.Connection = Depends(get_conn),
                  settings: Settings = Depends(get_settings),
                  storage: Storage = Depends(get_storage)) -> FileResponse:
    from ..media.frames import extract_frame   # lazy: owned by the media module

    project = _row(conn, "projects", project_id, "This project")
    info = presenters.media_info(project)
    if info is None:
        raise not_found("This project's video")
    if not (0.0 <= t <= info.duration):
        raise invalid("The requested time is outside the video.")
    src = stored_file(storage, project["source_path"], "This project's video")
    frames_dir = storage.work_dir(project_id) / "frames"
    out = frames_dir / f"{PREVIEW_PREFIX}{int(round(t * 1000))}.jpg"
    if not out.is_file():
        frames_dir.mkdir(parents=True, exist_ok=True)
        seek = min(t, max(0.0, info.duration - 0.1))      # the very last instant has no frame
        extract_frame(src, seek, out, max_width=640, settings=settings)
        _prune_previews(frames_dir)
    return FileResponse(out, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})


@router.get("/projects/{project_id}/export.zip")
def project_export(project_id: str, clips: str | None = Query(default=None, max_length=2000),
                   srt: int = Query(default=1, ge=0, le=1),
                   conn: sqlite3.Connection = Depends(get_conn),
                   settings: Settings = Depends(get_settings),
                   storage: Storage = Depends(get_storage)) -> FileResponse:
    project = _row(conn, "projects", project_id, "This project")
    wanted = parse_clip_ids(clips)
    rows = conn.execute(
        f"SELECT * FROM clips WHERE project_id=? AND render_state IN ({','.join('?' * len(EXPORT_STATES))}) "
        "ORDER BY rank, created_at", (project_id, *EXPORT_STATES)).fetchall()
    if wanted is not None:
        wanted_ids = set(wanted)
        rows = [r for r in rows if r["id"] in wanted_ids]
    if not rows:
        raise ApiError(404, "None of the selected clips has a finished render to export.", "not_found")
    needed = 0
    for r in rows:
        try:
            needed += stored_file(storage, r["render_path"]).stat().st_size
        except ApiError:
            continue
    storage.require_free(needed, settings.min_free_disk_mb, "this export")
    out = storage.tmp_dir() / f"export-{new_id('x')}.zip"
    try:
        count = build_export_zip(storage, rows, bool(srt), out)
    except BaseException:
        _remove(out)
        raise
    if count == 0:
        _remove(out)
        raise ApiError(404, "None of the selected clips has a finished render to export.", "not_found")
    name = f"{presenters.slugify(project['title'], default='clips')}-clips.zip"
    return FileResponse(out, media_type="application/zip", filename=name,
                        headers={"Cache-Control": "no-store"}, background=BackgroundTask(_remove, out))

