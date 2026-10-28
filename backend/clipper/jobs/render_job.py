"""Re-render one clip after the owner edited it (job kind ``render``).

Uses only local work: the stored transcript, local face analysis (re-run only when
the clip range changed) and FFmpeg. It never calls transcription or selection, so
caption/style/framing edits cost nothing in API usage.
"""

from __future__ import annotations

import logging
import shutil
from typing import Callable

from .. import db
from ..errors import MediaError
from ..models import ClipEdit, FramingPlan, MediaInfo, Transcript
from ..storage import read_json, write_json_atomic
from . import queue
from .context import JobContext

log = logging.getLogger(__name__)
MB = 1024 * 1024


def run_render(ctx: JobContext) -> None:
    from ..ai.transcript import words_between
    from ..framing.analyze import FRAMING_VERSION, analyze_framing, plan_for_edit
    from ..render.captions import build_cues, to_srt
    from ..render.renderer import make_thumbnail, render_clip, render_hash

    conn, st, s = ctx.conn, ctx.storage, ctx.settings
    clip_id = ctx.job["clip_id"]
    clip = conn.execute("SELECT * FROM clips WHERE id=?", (clip_id,)).fetchone()
    if clip is None:
        raise MediaError("This clip no longer exists.")
    project = conn.execute("SELECT * FROM projects WHERE id=?", (clip["project_id"],)).fetchone()
    if project is None or not project["source_path"] or not project["media_info"]:
        raise MediaError("The source video for this clip is no longer available.")
    conn.execute("UPDATE clips SET render_state='rendering', render_error=NULL WHERE id=?", (clip_id,))

    edit = ClipEdit.model_validate_json(clip["edit"])
    media = MediaInfo.model_validate_json(project["media_info"])
    src = st.resolve(project["source_path"])
    if not src.exists():
        raise MediaError("The source video for this clip is no longer available.")
    tpath = st.work_dir(project["id"]) / "transcript.json"
    transcript = Transcript.model_validate_json(tpath.read_text("utf-8"))

    auto = FramingPlan.model_validate_json(clip["framing"]) if clip["framing"] else None
    same_range = auto is not None and abs(auto.src_start - edit.src_start) <= 0.05 \
        and abs(auto.src_end - edit.src_end) <= 0.05
    needs_analysis = edit.framing.mode == "auto" or (edit.framing.mode == "crop" and edit.framing.crop_x is None)
    if needs_analysis and (not same_range or auto is None or auto.version != FRAMING_VERSION):
        ctx.progress(stage="framing", detail="Analysing faces for the new range")
        auto = analyze_framing(src, media, edit.src_start, edit.src_end, settings=s, cancel=ctx.cancel)
    ctx.cancel.raise_if_cancelled()
    plan = plan_for_edit(auto, edit.framing, media, edit.src_start, edit.src_end)
    cues = build_cues(words_between(transcript, edit.src_start, edit.src_end), edit) if edit.captions_enabled else []
    h = render_hash(source_sha256=project["source_sha256"], edit=edit, plan=plan, cues=cues,
                    encoder=ctx.encoder.name)

    cdir = st.clip_dir(project["id"], clip_id)
    out = cdir / f"{h}.mp4"
    info_path = cdir / f"{h}.json"
    ctx.progress(stage="render", detail="Rendering clip", current=1, total=1)
    if out.exists() and info_path.exists():
        info = read_json(info_path)
    else:
        st.require_free(int((edit.src_end - edit.src_start) * 2.5 * MB), s.min_free_disk_mb, "rendering this clip")
        rwork = st.work_dir(project["id"]) / "render" / clip_id
        info = render_clip(src=src, media=media, edit=edit, plan=plan, cues=cues, out_mp4=out, work_dir=rwork,
                           settings=s, cancel=ctx.cancel, encoder=ctx.encoder)
        write_json_atomic(info_path, info)
        shutil.rmtree(rwork, ignore_errors=True)
    ctx.cancel.raise_if_cancelled()
    srt_rel = None
    if cues:
        srt = cdir / f"{h}.srt"
        srt.write_text(to_srt(cues), encoding="utf-8")
        srt_rel = st.rel(srt)
    # Versioned like the render: if this job is cancelled or loses its commit, the published
    # thumbnail is untouched (the row points at the new one only inside commit_success).
    thumb = cdir / f"{h}.jpg"
    try:
        if not thumb.exists():
            make_thumbnail(out, thumb, t=min(1.0, (edit.src_end - edit.src_start) / 2), settings=s)
        thumb_rel = st.rel(thumb)
    except MediaError:
        thumb_rel = None
    rendered_edit_json = edit.model_dump_json()
    old_hash = clip["render_hash"]
    old_thumb = clip["thumb_path"]

    def publish(c) -> None:
        current = c.execute("SELECT edit FROM clips WHERE id=?", (clip_id,)).fetchone()
        if current is None:
            return
        # If the owner changed the edit while we were rendering, this render is already stale.
        changed = ClipEdit.model_validate_json(current["edit"]).model_dump_json() != rendered_edit_json
        c.execute(
            """UPDATE clips SET render_state=?, render_hash=?, render_path=?, srt_path=?, thumb_path=?,
                   render_info=?, render_error=NULL, framing=?, updated_at=? WHERE id=?""",
            ("stale" if changed else "ready", h, st.rel(out), srt_rel, thumb_rel, db.dumps(info),
             auto.model_dump_json() if auto is not None else None, db.now(), clip_id),
        )

    queue.commit_success(conn, ctx.job_id, ctx.worker_id, publish)
    old_files = [cdir / f"{old_hash}{suffix}" for suffix in (".mp4", ".srt", ".json", ".jpg")] \
        if old_hash and old_hash != h else []
    if old_thumb and old_thumb != thumb_rel:      # e.g. a legacy unversioned thumb.jpg
        try:
            old_files.append(st.resolve(old_thumb))
        except ValueError:
            pass
    for f in old_files:
        if f.parent != cdir:
            continue
        try:
            st.delete_file(f)
        except (ValueError, OSError):
            pass


def on_render_finished(ctx: JobContext, message: str | None) -> Callable:
    clip_id = ctx.job["clip_id"]

    def cb(c, final_state: str) -> None:
        row = c.execute("SELECT render_path FROM clips WHERE id=?", (clip_id,)).fetchone()
        if row is None:
            return
        if final_state == "failed":
            c.execute("UPDATE clips SET render_state='failed', render_error=?, updated_at=? WHERE id=?",
                      (message, db.now(), clip_id))
        else:
            c.execute("UPDATE clips SET render_state=?, render_error=NULL, updated_at=? WHERE id=?",
                      ("stale" if row["render_path"] else "none", db.now(), clip_id))

    return cb
