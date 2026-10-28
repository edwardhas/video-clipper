"""Turn database rows into API response models (``clipper.api.schemas``).

URLs are always relative (same origin). Nothing here returns filesystem paths.
"""

from __future__ import annotations

import re
import sqlite3
import unicodedata
from typing import Any

from .. import db
from ..ai import pricing
from ..models import ClipEdit, ClipSelection, MediaInfo, ProjectOptions
from .schemas import (
    ClipOut,
    ClipRenderOut,
    FramingSummary,
    JobOut,
    MediaSummary,
    ProjectActions,
    ProjectOut,
    ProjectSummaryOut,
    RenderInfoOut,
    StageOut,
    UsageOut,
)

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
RESUMABLE_STATUSES = ("failed", "cancelled", "interrupted")


# ----------------------------------------------------------------------------- helpers
def slugify(text: str, default: str = "clip", max_len: int = 60) -> str:
    """ASCII, lower-case, hyphen-separated filename stem."""
    norm = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "-", norm.lower()).strip("-")
    slug = slug[:max_len].rstrip("-")
    return slug or default


def _float_dict(raw: Any) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for k, v in raw.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            out[str(k)] = float(v)
    return out


def media_info(row: sqlite3.Row) -> MediaInfo | None:
    raw = row["media_info"]
    if not raw:
        return None
    try:
        return MediaInfo.model_validate_json(raw)
    except ValueError:
        return None


def active_jobs(conn: sqlite3.Connection, project_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM jobs WHERE project_id=? AND state IN ('queued','running') ORDER BY created_at DESC",
        (project_id,),
    ).fetchall()


def stage_state(conn: sqlite3.Connection, project_id: str, stage: str) -> str | None:
    row = conn.execute("SELECT state FROM stages WHERE project_id=? AND stage=?", (project_id, stage)).fetchone()
    return row["state"] if row else None


# ----------------------------------------------------------------------------- jobs/stages
def job_out(row: sqlite3.Row | None) -> JobOut | None:
    if row is None:
        return None
    return JobOut(
        id=row["id"], kind=row["kind"], state=row["state"], stage=row["stage"],
        stage_detail=row["stage_detail"], progress_current=row["progress_current"],
        progress_total=row["progress_total"], cancel_requested=bool(row["cancel_requested"]),
        error_code=row["error_code"], error_message=row["error_message"],
        retryable=None if row["retryable"] is None else bool(row["retryable"]),
        attempts=int(row["attempts"] or 0), created_at=row["created_at"],
        started_at=row["started_at"], finished_at=row["finished_at"],
    )


def stages_out(conn: sqlite3.Connection, project_id: str) -> list[StageOut]:
    rows = {r["stage"]: r for r in conn.execute("SELECT * FROM stages WHERE project_id=?", (project_id,))}
    out: list[StageOut] = []
    for key, label in STAGES:
        r = rows.get(key)
        out.append(StageOut(
            key=key, label=label, state=r["state"] if r else "pending",
            detail=r["detail"] if r else None,
            duration_s=r["duration_s"] if r else None,
        ))
    return out


# ----------------------------------------------------------------------------- clips
def render_info_out(raw: str | None) -> RenderInfoOut | None:
    info = db.loads(raw, None)
    if not isinstance(info, dict):
        return None
    try:
        return RenderInfoOut(
            width=int(info["width"]), height=int(info["height"]), duration=float(info["duration"]),
            size_bytes=int(info["size_bytes"]), video_codec=str(info["video_codec"]),
            audio_codec=None if info.get("audio_codec") is None else str(info["audio_codec"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def clip_render_out(row: sqlite3.Row) -> ClipRenderOut:
    cid = row["id"]
    h = row["render_hash"]
    has_video = bool(row["render_path"])
    version = f"?v={h}" if h else ""
    return ClipRenderOut(
        state=row["render_state"],
        error=row["render_error"],
        video_url=f"/media/clips/{cid}/video.mp4{version}" if has_video else None,
        download_url=f"/media/clips/{cid}/download.mp4" if has_video else None,
        srt_url=f"/media/clips/{cid}/captions.srt" if row["srt_path"] else None,
        thumb_url=f"/media/clips/{cid}/thumb.jpg{version}" if row["thumb_path"] else None,
        render_hash=h,
        info=render_info_out(row["render_info"]) if has_video else None,
        stale=row["render_state"] == "stale",
    )


def framing_summary(raw: str | None) -> FramingSummary | None:
    plan = db.loads(raw, None)
    if not isinstance(plan, dict):
        return None
    try:
        return FramingSummary(layout=plan["layout"], reason=str(plan.get("reason", "")),
                              detector=str(plan.get("detector", "")))
    except (KeyError, ValueError):
        return None


def clip_out(row: sqlite3.Row, render_job: sqlite3.Row | None) -> ClipOut:
    edit = ClipEdit.model_validate_json(row["edit"])
    sel = ClipSelection.model_validate_json(row["selection"])
    return ClipOut(
        id=row["id"], project_id=row["project_id"], rank=row["rank"], title=row["title"],
        reason=row["reason"], score=row["score"], score_detail=_float_dict(db.loads(row["score_detail"], {})),
        src_start=edit.src_start, src_end=edit.src_end, duration=round(edit.src_end - edit.src_start, 3),
        ai_src_start=sel.src_start, ai_src_end=sel.src_end, hook=sel.hook, topic=sel.topic,
        edit=edit, render=clip_render_out(row), framing=framing_summary(row["framing"]),
        job=job_out(render_job),
    )


def active_render_jobs(conn: sqlite3.Connection, project_id: str) -> dict[str, sqlite3.Row]:
    out: dict[str, sqlite3.Row] = {}
    for r in conn.execute(
        """SELECT * FROM jobs WHERE project_id=? AND kind='render' AND state IN ('queued','running')
           ORDER BY created_at ASC""", (project_id,)):
        if r["clip_id"]:
            out[r["clip_id"]] = r          # newest wins
    return out


def clips_out(conn: sqlite3.Connection, project_id: str) -> list[ClipOut]:
    jobs = active_render_jobs(conn, project_id)
    rows = conn.execute("SELECT * FROM clips WHERE project_id=? ORDER BY rank, created_at", (project_id,))
    return [clip_out(r, jobs.get(r["id"])) for r in rows]


# ----------------------------------------------------------------------------- projects
def usage_out(conn: sqlite3.Connection, project_id: str) -> UsageOut:
    rows = [dict(r) for r in conn.execute(
        "SELECT model, status, input_tokens, output_tokens, audio_seconds FROM api_usage WHERE project_id=?",
        (project_id,))]
    totals = pricing.summarize_usage(rows)
    return UsageOut(requests=totals.requests, input_tokens=totals.input_tokens,
                    output_tokens=totals.output_tokens, audio_seconds=round(totals.audio_seconds, 3),
                    estimated_cost_usd=totals.estimated_cost_usd, cost_note=totals.cost_note)


def project_actions(conn: sqlite3.Connection, row: sqlite3.Row) -> ProjectActions:
    active = active_jobs(conn, row["id"])
    idle = not active
    return ProjectActions(
        can_cancel=any(not j["cancel_requested"] for j in active),
        can_resume=idle and row["status"] in RESUMABLE_STATUSES,
        can_reselect=idle and stage_state(conn, row["id"], "transcribe") == "done",
        can_delete=idle,
    )


def _summary_fields(conn: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
    info = media_info(row)
    count = conn.execute("SELECT COUNT(*) FROM clips WHERE project_id=?", (row["id"],)).fetchone()[0]
    return {
        "id": row["id"], "title": row["title"], "source_kind": row["source_kind"],
        "source_url": row["source_url"], "status": row["status"], "status_detail": row["status_detail"],
        "created_at": row["created_at"], "updated_at": row["updated_at"],
        "duration": info.duration if info else None, "clip_count": int(count),
        "thumb_url": f"/media/projects/{row['id']}/thumb.jpg" if row["thumb_path"] else None,
    }


def project_summary_out(conn: sqlite3.Connection, row: sqlite3.Row) -> ProjectSummaryOut:
    return ProjectSummaryOut(**_summary_fields(conn, row))


def media_summary(row: sqlite3.Row) -> MediaSummary | None:
    info = media_info(row)
    if info is None:
        return None
    return MediaSummary(duration=info.duration, width=info.width, height=info.height, fps=info.fps,
                        has_audio=info.has_audio)


def project_out(conn: sqlite3.Connection, row: sqlite3.Row) -> ProjectOut:
    latest = conn.execute(
        "SELECT * FROM jobs WHERE project_id=? AND kind='analyze' ORDER BY created_at DESC, rowid DESC LIMIT 1",
        (row["id"],)).fetchone()
    return ProjectOut(
        **_summary_fields(conn, row),
        options=ProjectOptions.model_validate_json(row["options"]),
        media=media_summary(row),
        stages=stages_out(conn, row["id"]),
        job=job_out(latest),
        clips=clips_out(conn, row["id"]),
        usage=usage_out(conn, row["id"]),
        actions=project_actions(conn, row),
    )
