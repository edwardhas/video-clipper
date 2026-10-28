"""Analyze pipeline: import → probe → audio → transcribe → candidates → review → framing → render.

Each stage is skipped on resume when its ``stages`` row is ``done`` with a matching
cache key and its artifacts still exist. Paid provider results are additionally
cached content-addressed in ``data/cache`` so a retry/resume or another project of
the same source never repeats a completed API call.

Nothing user-visible is published until the end: clip rows, the project status and
the final stage row are written inside ``queue.commit_success`` – a cancelled job
can never publish results or become successful.
"""

from __future__ import annotations

import dataclasses
import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from .. import db
from ..errors import ClipperError, InputError, InterruptedError_, MediaError, NoSpeechError
from ..models import (
    CaptionStyle, Candidate, ClipEdit, ClipSelection, FramingPlan, MediaInfo, ProjectOptions,
    ReviewDecision, Transcript, TranscriptChunk,
)
from ..storage import fmt_bytes, read_json, sha256_file, stable_hash, write_json_atomic
from . import queue, stages
from .cache import FileJsonCache
from .context import JobContext

if TYPE_CHECKING:
    from ..media.audio import ChunkPlan

log = logging.getLogger(__name__)

MB = 1024 * 1024


@dataclass
class PreparedClip:
    clip_id: str
    rank: int
    selection: ClipSelection
    edit: ClipEdit
    framing: FramingPlan
    render_hash: str
    render_path: str          # relative to data dir
    srt_path: str | None
    thumb_path: str | None
    render_info: dict[str, Any]


@dataclass
class Outcome:
    status: str                              # "ready" | "no_clips"
    detail: str | None = None
    clips: list[PreparedClip] = field(default_factory=list)
    final_stage: str = "render"
    final_key: str | None = None
    final_artifact: dict[str, Any] = field(default_factory=dict)


def load_default_style(conn, settings) -> CaptionStyle:
    row = conn.execute("SELECT value FROM settings WHERE key='defaults'").fetchone()
    if row is not None:
        try:
            data = db.loads(row["value"], {}) or {}
            if isinstance(data.get("caption_style"), dict):
                return CaptionStyle.model_validate(data["caption_style"])
        except Exception:  # invalid stored defaults fall back to built-ins
            log.warning("ignoring invalid stored caption defaults")
    return CaptionStyle()


def clip_id_for(project_id: str, sel: ClipSelection) -> str:
    return "c_" + stable_hash("clip", project_id, round(sel.src_start, 3), round(sel.src_end, 3))[:16]


def raw_chunk_to_json(result) -> dict[str, Any]:
    return dataclasses.asdict(result)


def raw_chunk_from_json(data: dict[str, Any]):
    from ..ai.interfaces import RawChunkResult, RawSegment, RawWord
    return RawChunkResult(
        words=[RawWord(**w) for w in data.get("words", [])],
        segments=[RawSegment(**s) for s in data.get("segments", [])],
        language=data.get("language"),
        duration=data.get("duration"),
        request_id=data.get("request_id"),
        raw=data.get("raw") or {},
    )


class AnalyzeRun:
    def __init__(self, ctx: JobContext) -> None:
        self.ctx = ctx
        self.s = ctx.settings
        self.st = ctx.storage
        self.conn = ctx.conn
        self.pid = ctx.project_id
        row = self.conn.execute("SELECT * FROM projects WHERE id=?", (self.pid,)).fetchone()
        if row is None:
            raise MediaError("The project no longer exists.")
        self.project = dict(row)
        self.options = ProjectOptions.model_validate_json(self.project["options"])
        self.work = self.st.work_dir(self.pid)
        self.work.mkdir(parents=True, exist_ok=True)
        self.src: Path | None = None
        self.sha: str | None = None
        self.media: MediaInfo | None = None
        self._transcriber = None
        self._selector = None
        self.last_stage = "import"

    # ------------------------------------------------------------------ helpers
    def _stage(self, key: str, cache_key: str, compute: Callable[[], dict[str, Any]],
               valid: Callable[[dict[str, Any]], bool] = lambda a: True) -> dict[str, Any]:
        """Run ``compute`` unless the stage is done with the same key and ``valid(artifact)``."""
        self.ctx.cancel.raise_if_cancelled()
        self.last_stage = key
        art = stages.done_artifact(self.conn, self.pid, key, cache_key)
        if art is not None:
            try:
                if valid(art):
                    log.info("stage %s: reusing completed result", key)
                    return art
            except Exception:
                pass
        stages.mark_running(self.conn, self.pid, key, cache_key)
        self.ctx.progress(stage=key, detail=None)
        art = compute()
        self.ctx.cancel.raise_if_cancelled()
        stages.mark_done(self.conn, self.pid, key, cache_key, art)
        return art

    def _exists(self, *rels: str | None) -> bool:
        try:
            return all(r and self.st.resolve(r).exists() for r in rels)
        except ValueError:
            return False

    def transcriber(self):
        if self._transcriber is None:
            from ..ai.factory import get_transcriber
            self._transcriber = get_transcriber(self.s)
        return self._transcriber

    def selector(self):
        if self._selector is None:
            from ..ai.factory import get_selector
            self._selector = get_selector(self.s)
        return self._selector

    # ------------------------------------------------------------------ stages
    def stage_import(self) -> None:
        kind = self.project["source_kind"]
        ident = self.project["youtube_id"] if kind == "youtube" else self.project["upload_id"]
        key = stable_hash("import", kind, ident)

        def compute() -> dict[str, Any]:
            src_dir = self.st.source_dir(self.pid)
            src_dir.mkdir(parents=True, exist_ok=True)
            # An earlier attempt may have moved the upload here and then been cancelled or stopped
            # before this stage was marked done: that file is now the only copy, so it is kept.
            keep = self._placed_upload(src_dir) if kind != "youtube" else None
            # Clear any previous partial import.
            for old in src_dir.glob("source.*"):
                if old != keep:
                    old.unlink()
            if kind == "youtube":
                final = self._import_youtube(src_dir)
            else:
                final = self._import_upload(src_dir)
            self.ctx.progress(stage="import", detail="Checking file integrity")
            sha = sha256_file(final)
            rel = self.st.rel(final)
            self.conn.execute("UPDATE projects SET source_path=?, source_sha256=?, updated_at=? WHERE id=?",
                              (rel, sha, db.now(), self.pid))
            return {"source_path": rel, "sha256": sha}

        art = self._stage("import", key, compute, valid=lambda a: self._exists(a.get("source_path")))
        self.src = self.st.resolve(art["source_path"])
        self.sha = art["sha256"]

    def _import_youtube(self, src_dir: Path) -> Path:
        from ..media import youtube
        url = self.project["source_url"]
        tmp = src_dir / "download"
        # Not wiped up front: partial files left by an interrupted attempt let yt-dlp continue the download.
        keep_partial = False
        try:
            self.ctx.progress(stage="import", detail="Checking the YouTube video")
            meta = youtube.fetch_metadata(url, settings=self.s, cancel=self.ctx.cancel)
            title = (meta.get("title") or "").strip()
            if title:
                self.conn.execute("UPDATE projects SET title=?, updated_at=? WHERE id=?",
                                  (title[:200], db.now(), self.pid))
            duration = float(meta.get("duration") or 0)
            estimate = int(meta.get("filesize_approx") or duration * 1.2 * MB)
            # Bytes kept from an interrupted attempt are already on disk (no longer free) and are not
            # downloaded again, so they are not counted a second time.
            kept = self.st.usage_bytes(tmp) if tmp.is_dir() else 0
            self.st.require_free(max(0, int(estimate * 1.2) - kept), self.s.min_free_disk_mb,
                                 "downloading this video")
            tmp.mkdir(parents=True, exist_ok=True)

            def on_progress(done: int | None, total: int | None) -> None:
                if done is None:
                    return
                detail = f"Downloaded {fmt_bytes(done)}" + (f" of about {fmt_bytes(total)}" if total else "")
                self.ctx.progress(stage="import", detail=detail, throttle=1.0)

            max_bytes = int(self.s.max_upload_mb) * MB
            path = youtube.download(url, tmp, settings=self.s, cancel=self.ctx.cancel, progress_cb=on_progress,
                                    max_bytes=max_bytes)
            final = src_dir / f"source{path.suffix.lower() or '.mp4'}"
            os.replace(path, final)
        except InterruptedError_:
            keep_partial = True       # worker shutdown: the resumed job continues the download
            raise
        finally:
            # Otherwise never leave partial downloads behind (cancel, failure incl. the metadata or disk
            # check of a resumed attempt, or success).
            if not keep_partial:
                shutil.rmtree(tmp, ignore_errors=True)
        return final

    def _placed_upload(self, src_dir: Path) -> Path | None:
        """The upload's file if an earlier attempt of this stage already moved it into ``src_dir``."""
        up = self.conn.execute("SELECT path FROM uploads WHERE id=?", (self.project["upload_id"],)).fetchone()
        if up is None:
            return None
        try:
            upath = self.st.resolve(up["path"])
        except ValueError:
            return None
        src_dir = src_dir.resolve()
        if upath.parent == src_dir:
            return upath if upath.is_file() else None
        # Killed between the move and the row update: the row still names the old location.
        moved = src_dir / f"source{upath.suffix.lower()}"
        return moved if not upath.exists() and moved.is_file() else None

    def _import_upload(self, src_dir: Path) -> Path:
        up = self.conn.execute("SELECT * FROM uploads WHERE id=?", (self.project["upload_id"],)).fetchone()
        if up is None:
            raise InputError("The uploaded file is no longer available. Please upload it again.")
        final = self._placed_upload(src_dir)
        if final is None:
            upath = self.st.resolve(up["path"])
            if not upath.exists():
                raise InputError("The uploaded file is no longer available. Please upload it again.")
            final = src_dir / f"source{upath.suffix.lower()}"
            os.replace(upath, final)   # same filesystem (both under data_dir): atomic move, no copy
        self.conn.execute("UPDATE uploads SET path=?, state='attached', project_id=? WHERE id=?",
                          (self.st.rel(final), self.pid, up["id"]))
        try:
            self.st.delete_tree(self.st.upload_dir(up["id"]))
        except (ValueError, OSError):
            pass
        return final

    def stage_probe(self) -> None:
        from ..media.frames import extract_frame
        from ..media.probe import PROBE_VERSION, probe
        key = stable_hash("probe", self.sha, PROBE_VERSION, self.s.max_source_minutes)

        def compute() -> dict[str, Any]:
            assert self.src is not None
            media = probe(self.src, settings=self.s, cancel=self.ctx.cancel)
            if not media.has_video:
                raise InputError("This file has no video stream. Local Clipper needs a video with speech.")
            if media.duration > self.s.max_source_minutes * 60 + 1:
                raise InputError(
                    f"This video is {media.duration / 60:.0f} minutes long; the limit is "
                    f"{self.s.max_source_minutes:.0f} minutes (CLIPPER_MAX_SOURCE_MINUTES)."
                )
            if media.duration < 10:
                raise InputError("This video is shorter than 10 seconds — too short to find clips in.")
            thumb = self.st.source_dir(self.pid) / "thumb.jpg"
            try:
                extract_frame(self.src, min(media.duration * 0.1, 30.0), thumb, max_width=480,
                              settings=self.s, cancel=self.ctx.cancel)
                thumb_rel = self.st.rel(thumb)
            except MediaError:
                thumb_rel = None
            self.conn.execute("UPDATE projects SET media_info=?, thumb_path=?, updated_at=? WHERE id=?",
                              (media.model_dump_json(), thumb_rel, db.now(), self.pid))
            return {"media": media.model_dump(mode="json")}

        art = self._stage("probe", key, compute)
        self.media = MediaInfo.model_validate(art["media"])
        if not self.media.has_audio:
            raise NoSpeechError("This video has no audio track, so there is no speech to find clips in.")

    def stage_audio(self) -> tuple[str, dict[str, Any]]:
        from ..media import audio
        s = self.s
        key = stable_hash("audio", self.sha, audio.AUDIO_VERSION, s.transcribe_chunk_seconds,
                          s.transcribe_overlap_seconds, s.transcribe_audio_format)
        wav = self.work / "audio" / "analysis.wav"

        def complete(a: dict[str, Any]) -> bool:
            # Artifacts written before pauses and chunk content hashes were stored: stage_transcribe
            # derives both from analysis.wav, so that file must still be there.
            return "silences" in a and all(c.get("pcm_sha256") for c in a.get("chunks", []))

        def valid(a: dict[str, Any]) -> bool:
            if self._exists(*[c["path"] for c in a.get("chunks", [])]) and (complete(a) or wav.is_file()):
                return True
            # Chunks removed by "Clean up": still fine while the transcript made from them is reusable
            # (only stage_transcribe's compute reads them, and it then is not run).
            t = stages.done_artifact(self.conn, self.pid, "transcribe", self._transcribe_key(key))
            return t is not None and self._exists(t.get("transcript_path"))

        def compute() -> dict[str, Any]:
            assert self.src is not None and self.media is not None
            adir = wav.parent
            if adir.exists():
                shutil.rmtree(adir)
            adir.mkdir(parents=True)
            self.st.require_free(int(self.media.duration * 32000 * 1.8), s.min_free_disk_mb, "audio extraction")
            self.ctx.progress(stage="audio", detail="Extracting audio")
            duration = audio.extract_analysis_audio(self.src, wav, settings=s, cancel=self.ctx.cancel)
            vol = audio.mean_volume_db(wav, settings=s, cancel=self.ctx.cancel)
            if vol is not None and vol < -60.0:
                raise NoSpeechError("The audio track is silent, so there is no speech to find clips in.")
            self.ctx.progress(stage="audio", detail="Finding pauses for chunking")
            silences = audio.detect_silences(wav, settings=s, cancel=self.ctx.cancel)
            plans = audio.plan_chunks(duration, silences, target=s.transcribe_chunk_seconds,
                                      overlap=s.transcribe_overlap_seconds)
            ext = audio.chunk_extension(s)
            chunks = []
            for plan in plans:
                self.ctx.progress(stage="audio", detail="Preparing audio chunks", current=plan.index + 1,
                                  total=len(plans))
                out = adir / f"chunk_{plan.index:03d}{ext}"
                audio.encode_chunk(wav, out, plan.start, plan.end, settings=s, cancel=self.ctx.cancel)
                chunks.append({**dataclasses.asdict(plan), "path": self.st.rel(out),
                               "pcm_sha256": audio.pcm_sha256(wav, plan.start, plan.end)})
            # Pauses in source seconds: the transcript merge drops words a provider returned for silence.
            return {"duration": duration, "chunks": chunks, "mean_volume_db": vol,
                    "silences": [[a, b] for a, b in silences]}

        return key, self._stage("audio", key, compute, valid=valid)

    def _transcribe_key(self, audio_key: str) -> str:
        from ..models import TRANSCRIPT_VERSION
        return stable_hash("transcribe", audio_key, self.s.transcribe_model, self.options.language,
                           TRANSCRIPT_VERSION)

    def stage_transcribe(self, audio_key: str, audio_art: dict[str, Any]) -> tuple[str, Transcript]:
        from ..ai.transcript import insufficient_speech_reason, merge_chunks, validate_transcript
        from ..media import audio
        from ..media.audio import ChunkPlan
        s = self.s
        lang = self.options.language
        key = self._transcribe_key(audio_key)
        tpath = self.work / "transcript.json"

        def compute() -> dict[str, Any]:
            chunks = audio_art["chunks"]
            wav = self.work / "audio" / "analysis.wav"
            silences = audio_art.get("silences")
            if silences is None and wav.is_file():      # audio artifact from before pauses were stored
                silences = audio.detect_silences(wav, settings=s, cancel=self.ctx.cancel)
            results = []
            meta: list[TranscriptChunk] = []
            raw_dir = self.work / "transcribe_raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            for i, c in enumerate(chunks):
                self.ctx.cancel.raise_if_cancelled()
                plan = ChunkPlan(index=c["index"], start=c["start"], end=c["end"], keep_start=c["keep_start"],
                                 keep_end=c["keep_end"])
                content = c.get("pcm_sha256") or \
                    (audio.pcm_sha256(wav, plan.start, plan.end) if wav.is_file() else None)
                data, cpath = self._cached_chunk(plan, content)
                cached = data is not None
                if cached:
                    self.ctx.progress(stage="transcribe", detail="Using saved transcription", current=i + 1,
                                      total=len(chunks))
                else:
                    self.ctx.progress(stage="transcribe", detail="Transcribing with OpenAI", current=i + 1,
                                      total=len(chunks))
                    result = self.transcriber().transcribe_chunk(
                        self.st.resolve(c["path"]), language=lang, cancel=self.ctx.cancel,
                        usage=self.ctx.usage(), audio_seconds=plan.end - plan.start)
                    data = raw_chunk_to_json(result)
                    write_json_atomic(cpath, data)        # cache the paid result before honouring a cancel
                    self.ctx.cancel.raise_if_cancelled()
                raw_rel = self.st.rel(raw_dir / f"chunk_{plan.index:03d}.json")
                write_json_atomic(self.st.resolve(raw_rel), data.get("raw") or {})
                result = raw_chunk_from_json(data)
                results.append((plan, result))
                meta.append(TranscriptChunk(index=plan.index, start=plan.start, end=plan.end,
                                            keep_start=plan.keep_start, keep_end=plan.keep_end,
                                            model=s.transcribe_model, request_id=result.request_id,
                                            raw_path=raw_rel, cached=cached))
            transcript = merge_chunks(results, duration=float(audio_art["duration"]),
                                      provider=self.transcriber_provider(), model=s.transcribe_model,
                                      language=lang, chunk_meta=meta,
                                      silences=[(a, b) for a, b in silences] if silences is not None else None)
            validate_transcript(transcript)
            write_json_atomic(tpath, transcript.model_dump(mode="json"))
            return {"transcript_path": self.st.rel(tpath), "words": len(transcript.words)}

        art = self._stage("transcribe", key, compute, valid=lambda a: self._exists(a.get("transcript_path")))
        transcript = Transcript.model_validate_json(self.st.resolve(art["transcript_path"]).read_text("utf-8"))
        assert self.media is not None
        reason = insufficient_speech_reason(transcript, self.media.duration)
        if reason:
            raise NoSpeechError(reason)
        return key, transcript

    def _cached_chunk(self, plan: ChunkPlan, content: str | None) -> tuple[dict[str, Any] | None, Path]:
        """A saved transcription of this chunk's audio (or None) and the cache file a new one is saved to.

        Keyed by the chunk's audio content (``pcm_sha256``) + model, language and upload format: a paid
        result is reused exactly when the same audio would be sent again, whatever AUDIO_VERSION made it,
        and audio whose timeline changed misses. Results saved under the earlier source + time-range keys
        are reused (and saved again under the content key) only while their audio provably is this audio:
        made by the same AUDIO_VERSION, or by version 1, whose WAVs equal version 2's except for
        MPEG-TS/PS sources (verified byte-identical for MP4/MOV and Matroska/WebM).
        """
        from ..media.audio import AUDIO_VERSION
        s = self.s
        tail = (s.transcribe_model, self.options.language, s.transcribe_audio_format)
        span = (round(plan.start, 3), round(plan.end, 3))
        versioned = self.st.cache_path("transcribe-chunk", stable_hash("tchunk", self.sha, AUDIO_VERSION, *span,
                                                                       *tail))
        target = self.st.cache_path("transcribe-chunk", stable_hash("tchunk2", content, *tail)) \
            if content else versioned
        found = [target, versioned]
        container = (self.media.container if self.media is not None else "").split(",")[0]
        if AUDIO_VERSION == 2 and container in ("mov", "matroska"):
            found.append(self.st.cache_path("transcribe-chunk", stable_hash("tchunk", self.sha, *span, *tail)))
        for path in dict.fromkeys(found):
            if path.exists():
                data = read_json(path)
                if path != target:
                    write_json_atomic(target, data)       # from now on found by content
                return data, target
        return None, target

    def transcriber_provider(self) -> str:
        return getattr(self._transcriber, "provider", None) or "openai"

    def stage_candidates(self, tkey: str, transcript: Transcript) -> tuple[str, list[Candidate]]:
        from ..ai.prompts import PROMPT_VERSION
        s, o = self.s, self.options
        key = stable_hash("candidates", tkey, s.select_model, s.select_reasoning_effort, PROMPT_VERSION,
                          o.min_seconds, o.max_seconds, o.clip_count)
        cpath = self.st.cache_path("candidates", key)
        wpath = self.work / "candidates.json"

        def compute() -> dict[str, Any]:
            if cpath.exists():
                data = read_json(cpath)
            else:
                self.ctx.progress(stage="candidates", detail="Reading the transcript for strong moments")
                # Each finished window is checkpointed (scoped to this stage key) before a cancel is
                # honoured, so a Resume/Retry only pays for the windows that never completed.
                cands = self.selector().find_candidates(transcript, o, cancel=self.ctx.cancel,
                                                        usage=self.ctx.usage(),
                                                        cache=FileJsonCache(self.st, "candidates-window", key))
                data = [c.model_dump(mode="json") for c in cands]
                write_json_atomic(cpath, data)
                self.ctx.cancel.raise_if_cancelled()
            write_json_atomic(wpath, data)
            return {"path": self.st.rel(wpath), "count": len(data)}

        art = self._stage("candidates", key, compute, valid=lambda a: self._exists(a.get("path")))
        cands = [Candidate.model_validate(c) for c in read_json(self.st.resolve(art["path"]))]
        if not cands:
            raise _NoClips(
                "No complete, standalone moments were found in this video. The first release works best with "
                "speech-led videos (interviews, podcasts, commentary, talks) where ideas are explained in "
                "full sentences.", "candidates")
        return key, cands

    def stage_review(self, ckey: str, transcript: Transcript, cands: list[Candidate]) -> tuple[str, list[ClipSelection]]:
        from ..ai.prompts import REVIEW_PROMPT_VERSION
        from ..ai.selection import apply_review, shortlist
        from ..media.frames import extract_frames
        s, o = self.s, self.options
        key = stable_hash("review", ckey, s.review_model, s.review_reasoning_effort, REVIEW_PROMPT_VERSION)
        cpath = self.st.cache_path("review", key)
        wpath = self.work / "review.json"

        def compute() -> dict[str, Any]:
            short = shortlist(cands, o)
            if cpath.exists():
                decisions = [ReviewDecision.model_validate(d) for d in read_json(cpath)]
            else:
                frames: dict[str, list[tuple[float, Path]]] = {}
                fdir = self.work / "frames"
                for i, c in enumerate(short):
                    self.ctx.progress(stage="review", detail="Sampling frames for review", current=i + 1,
                                      total=len(short))
                    dur = c.src_end - c.src_start
                    times = [c.src_start + dur * 0.3, c.src_start + dur * 0.7]
                    assert self.src is not None
                    paths = extract_frames(self.src, times, fdir / c.id, max_width=512, settings=s,
                                           cancel=self.ctx.cancel)
                    frames[c.id] = list(zip(times, paths))
                self.ctx.progress(stage="review", detail="Reviewing and ranking candidates")
                # Same for each review batch (scoped to the review stage key).
                decisions = self.selector().review(transcript, short, frames, o, cancel=self.ctx.cancel,
                                                   usage=self.ctx.usage(),
                                                   cache=FileJsonCache(self.st, "review-batch", key))
                write_json_atomic(cpath, [d.model_dump(mode="json") for d in decisions])
                self.ctx.cancel.raise_if_cancelled()
            selections = apply_review(short, decisions, transcript, o, select_model=s.select_model,
                                      review_model=s.review_model)
            write_json_atomic(wpath, [x.model_dump(mode="json") for x in selections])
            return {"path": self.st.rel(wpath), "count": len(selections), "shortlisted": len(short)}

        art = self._stage("review", key, compute, valid=lambda a: self._exists(a.get("path")))
        sels = [ClipSelection.model_validate(x) for x in read_json(self.st.resolve(art["path"]))]
        if not sels:
            raise _NoClips(
                "Some possible moments were found, but none passed the quality review (complete idea, strong "
                "opening, clear payoff). Try a different video or a wider duration range.", "review")
        return key, sels

    def stage_framing(self, rkey: str, sels: list[ClipSelection]) -> tuple[str, list[FramingPlan]]:
        from ..framing.analyze import FRAMING_VERSION, analyze_framing
        key = stable_hash("framing", rkey, self.sha, FRAMING_VERSION,
                          [(round(x.src_start, 3), round(x.src_end, 3)) for x in sels])

        def compute() -> dict[str, Any]:
            plans = []
            for i, sel in enumerate(sels):
                self.ctx.progress(stage="framing", detail="Analysing faces and shots", current=i + 1,
                                  total=len(sels))
                assert self.src is not None and self.media is not None
                plan = analyze_framing(self.src, self.media, sel.src_start, sel.src_end, settings=self.s,
                                       cancel=self.ctx.cancel)
                plans.append(plan.model_dump(mode="json"))
            return {"plans": plans}

        art = self._stage("framing", key, compute)
        return key, [FramingPlan.model_validate(p) for p in art["plans"]]

    def stage_render(self, fkey: str, transcript: Transcript, sels: list[ClipSelection],
                     plans: list[FramingPlan]) -> Outcome:
        from ..ai.transcript import words_between
        from ..framing.analyze import plan_for_edit
        from ..render.captions import build_cues, to_srt
        from ..render.renderer import make_thumbnail, render_clip, render_hash
        s, o = self.s, self.options
        style = load_default_style(self.conn, s)
        key = stable_hash("render", fkey, style.model_dump(), o.captions_enabled, self.ctx.encoder.name)
        stages.mark_running(self.conn, self.pid, "render", key)
        prepared: list[PreparedClip] = []
        assert self.src is not None and self.media is not None and self.sha is not None
        for i, (sel, auto) in enumerate(zip(sels, plans)):
            self.ctx.cancel.raise_if_cancelled()
            self.ctx.progress(stage="render", detail="Rendering vertical clips", current=i + 1, total=len(sels))
            clip_id = clip_id_for(self.pid, sel)
            edit = ClipEdit(src_start=sel.src_start, src_end=sel.src_end, captions_enabled=o.captions_enabled,
                            style=style)
            framing = auto
            existing = self.conn.execute("SELECT edit, framing FROM clips WHERE id=?", (clip_id,)).fetchone()
            if existing is not None:
                edit = ClipEdit.model_validate_json(existing["edit"])   # keep the owner's edits on re-runs
                framing = self._framing_for_edit(auto, existing["framing"], edit, step=(i + 1, len(sels)))
            plan = plan_for_edit(framing, edit.framing, self.media, edit.src_start, edit.src_end)
            cues = build_cues(words_between(transcript, edit.src_start, edit.src_end), edit) \
                if edit.captions_enabled else []
            h = render_hash(source_sha256=self.sha, edit=edit, plan=plan, cues=cues, encoder=self.ctx.encoder.name)
            cdir = self.st.clip_dir(self.pid, clip_id)
            out = cdir / f"{h}.mp4"
            info_path = cdir / f"{h}.json"
            if out.exists() and info_path.exists():
                info = read_json(info_path)
            else:
                dur = edit.src_end - edit.src_start
                self.st.require_free(int(dur * 2.5 * MB), s.min_free_disk_mb, "rendering clips")
                rwork = self.work / "render" / clip_id
                info = render_clip(src=self.src, media=self.media, edit=edit, plan=plan, cues=cues, out_mp4=out,
                                   work_dir=rwork, settings=s, cancel=self.ctx.cancel, encoder=self.ctx.encoder)
                write_json_atomic(info_path, info)
                shutil.rmtree(rwork, ignore_errors=True)
            srt_rel = None
            if cues:
                srt = cdir / f"{h}.srt"
                if not srt.exists():
                    srt.write_text(to_srt(cues), encoding="utf-8")
                srt_rel = self.st.rel(srt)
            # Versioned like the render: a run that is cancelled or loses its commit never changes the
            # thumbnail of the published render (the row points at it only after commit_success).
            thumb = cdir / f"{h}.jpg"
            if thumb.exists():
                thumb_rel = self.st.rel(thumb)
            else:
                try:
                    make_thumbnail(out, thumb, t=min(1.0, (edit.src_end - edit.src_start) / 2), settings=s)
                    thumb_rel = self.st.rel(thumb)
                except MediaError:
                    thumb_rel = None
            prepared.append(PreparedClip(clip_id=clip_id, rank=i + 1, selection=sel, edit=edit, framing=framing,
                                         render_hash=h, render_path=self.st.rel(out), srt_path=srt_rel,
                                         thumb_path=thumb_rel, render_info=info))
        detail = None
        if len(prepared) < o.clip_count:
            detail = (f"{len(prepared)} of the {o.clip_count} requested clips met the quality bar — this source "
                      "did not support more without forcing weak or overlapping moments.")
        return Outcome(status="ready", detail=detail, clips=prepared, final_stage="render", final_key=key,
                       final_artifact={"clips": [p.clip_id for p in prepared]})

    def _framing_for_edit(self, auto: FramingPlan, stored_json: str | None, edit: ClipEdit, *,
                          step: tuple[int, int]) -> FramingPlan:
        """Automatic framing for an existing clip's (possibly trimmed) edit range on a re-run.

        ``auto`` was analysed for the AI's selection range; keyframes of another range are never
        applied. When the owner trimmed the clip and its framing is automatic, the plan the render
        job stored for the edited range is reused, otherwise that range is analysed now (local work),
        so a re-selection never replaces a face-tracked render with a "re-analysis pending" crop.
        """
        # _same_range is the exact reuse test plan_for_edit applies, so both always agree.
        from ..framing.analyze import FRAMING_VERSION, _same_range, analyze_framing
        assert self.src is not None and self.media is not None
        needs_analysis = edit.framing.mode == "auto" or (edit.framing.mode == "crop" and edit.framing.crop_x is None)
        if not needs_analysis or _same_range(auto, self.media, edit.src_start, edit.src_end):
            return auto
        try:
            stored = FramingPlan.model_validate_json(stored_json) if stored_json else None
        except ValueError:
            stored = None
        if stored is not None and stored.version == FRAMING_VERSION \
                and _same_range(stored, self.media, edit.src_start, edit.src_end):
            return stored
        self.ctx.progress(stage="render", detail="Analysing faces for an edited clip", current=step[0],
                          total=step[1])
        return analyze_framing(self.src, self.media, edit.src_start, edit.src_end, settings=self.s,
                               cancel=self.ctx.cancel)

    # ------------------------------------------------------------------ main
    def run(self) -> Outcome:
        try:
            self.stage_import()
            self.stage_probe()
            akey, aart = self.stage_audio()
            tkey, transcript = self.stage_transcribe(akey, aart)
            ckey, cands = self.stage_candidates(tkey, transcript)
            rkey, sels = self.stage_review(ckey, transcript, cands)
            fkey, plans = self.stage_framing(rkey, sels)
            return self.stage_render(fkey, transcript, sels, plans)
        except NoSpeechError as exc:
            return Outcome(status="no_clips", detail=exc.user_message, final_stage=self.last_stage)
        except _NoClips as exc:
            return Outcome(status="no_clips", detail=exc.user_message, final_stage=exc.stage)


class _NoClips(ClipperError):
    code = "no_clips"

    def __init__(self, message: str, stage: str) -> None:
        super().__init__(message)
        self.stage = stage


def run_analyze(ctx: JobContext) -> None:
    conn = ctx.conn
    pid = ctx.project_id
    conn.execute("UPDATE projects SET status='running', status_detail=NULL, updated_at=? WHERE id=?",
                 (db.now(), pid))
    run = AnalyzeRun(ctx)
    outcome = run.run()
    removed_dirs: list[Path] = []

    def publish(c) -> None:
        ts = db.now()
        if outcome.status == "no_clips":
            # A no-clip outcome replaces earlier results only for a fresh analysis, not a re-selection
            # that found nothing (keep the previous clips visible in that case).
            has_clips = c.execute("SELECT COUNT(*) FROM clips WHERE project_id=?", (pid,)).fetchone()[0]
            status = "ready" if (ctx.params.get("reselect") and has_clips) else "no_clips"
            detail = outcome.detail if status == "no_clips" else \
                f"Re-selection found no new clips; your previous clips are kept. {outcome.detail}"
            c.execute("UPDATE projects SET status=?, status_detail=?, updated_at=? WHERE id=?",
                      (status, detail, ts, pid))
            if status == "ready":       # the kept clips were made with the previous options
                restore_reselect_options(c, pid, ctx.params)
            stage_row = c.execute("SELECT state FROM stages WHERE project_id=? AND stage=?",
                                  (pid, outcome.final_stage)).fetchone()
            if stage_row is not None and stage_row["state"] == "running":
                stages.mark_done(c, pid, outcome.final_stage, None, {}, outcome.detail)
            stages.skip_after(c, pid, outcome.final_stage, "Skipped: " + (outcome.detail or "no clips"))
            return
        keep = {p.clip_id for p in outcome.clips}
        for row in c.execute("SELECT id FROM clips WHERE project_id=?", (pid,)).fetchall():
            if row["id"] not in keep:
                c.execute("DELETE FROM clips WHERE id=?", (row["id"],))
                removed_dirs.append(ctx.storage.clip_dir(pid, row["id"]))
        for p in outcome.clips:
            sel = p.selection
            exists = c.execute("SELECT edit, render_state, selection FROM clips WHERE id=?", (p.clip_id,)).fetchone()
            # A clip is never published 'ready' without its video (e.g. its files were deleted meanwhile).
            rendered = _file_exists(ctx.storage, p.render_path)
            if exists is None:
                if not rendered:
                    log.warning("clip %s: rendered file vanished before publishing; not published", p.clip_id)
                    continue
                c.execute(
                    """INSERT INTO clips(id, project_id, rank, title, reason, score, score_detail, selection, edit,
                           framing, render_state, render_hash, render_path, srt_path, thumb_path, render_info,
                           created_at, updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,'ready',?,?,?,?,?,?,?)""",
                    (p.clip_id, pid, p.rank, sel.title, sel.reason, sel.score, db.dumps(sel.sub_scores),
                     sel.model_copy(update={"original_style": p.edit.style}).model_dump_json(),
                     p.edit.model_dump_json(), p.framing.model_dump_json(), p.render_hash,
                     p.render_path, p.srt_path, p.thumb_path, db.dumps(p.render_info), ts, ts),
                )
            else:
                # The owner may have edited the clip after stage_render read its edit: that render is
                # then already outdated (as in render_job.publish). A render job queued for the newer
                # edit keeps its state and publishes its own result.
                changed = ClipEdit.model_validate_json(exists["edit"]).model_dump_json() != p.edit.model_dump_json()
                if not rendered:
                    state = "none"
                elif not changed:
                    state = "ready"
                else:
                    state = exists["render_state"] if exists["render_state"] in ("queued", "rendering") else "stale"
                r = (p.render_hash, p.render_path, p.srt_path, p.thumb_path, db.dumps(p.render_info)) if rendered \
                    else (None, None, None, None, None)
                c.execute(
                    """UPDATE clips SET rank=?, title=?, reason=?, score=?, score_detail=?, selection=?, framing=?,
                           render_state=?, render_hash=?, render_path=?, srt_path=?, thumb_path=?,
                           render_info=?, render_error=NULL, updated_at=? WHERE id=?""",
                    (p.rank, sel.title, sel.reason, sel.score, db.dumps(sel.sub_scores),
                     sel.model_copy(update={"original_style": _stored_original_style(exists["selection"])})
                     .model_dump_json(),
                     p.framing.model_dump_json(), state, *r, ts, p.clip_id),
                )
        c.execute("UPDATE projects SET status='ready', status_detail=?, updated_at=? WHERE id=?",
                  (outcome.detail, ts, pid))
        stages.mark_done(c, pid, "render", outcome.final_key, outcome.final_artifact)

    queue.commit_success(conn, ctx.job_id, ctx.worker_id, publish)
    for d in removed_dirs:
        try:
            ctx.storage.delete_tree(d)
        except (ValueError, OSError):
            log.warning("could not remove old clip directory")
    _prune_unpublished_clip_dirs(ctx)


def _stored_original_style(selection_json: str | None) -> CaptionStyle | None:
    """The first-render caption style kept from an earlier publish of the same clip (re-selection)."""
    try:
        return ClipSelection.model_validate_json(selection_json).original_style if selection_json else None
    except Exception:
        return None


def _file_exists(storage, rel: str | None) -> bool:
    try:
        return bool(rel) and storage.resolve(rel).is_file()
    except ValueError:
        return False


def restore_reselect_options(conn, project_id: str, params: dict[str, Any]) -> None:
    """Put back the options the kept clips were made with after a re-selection ended unpublished.

    ``POST /reselect`` stores the new options on the project (the run reads them from there) and
    the previous ones in the job's ``previous_options``. Used by every "previous clips are kept"
    path: cancelled while queued, cancelled/failed/interrupted while running, orphan recovery, and a
    re-selection that found no new clips.
    """
    prev = params.get("previous_options")
    if not isinstance(prev, dict):
        return
    try:
        options = ProjectOptions.model_validate(prev)
    except ValueError:
        log.warning("project %s: stored previous options are invalid; keeping the current ones", project_id)
        return
    conn.execute("UPDATE projects SET options=? WHERE id=?", (options.model_dump_json(), project_id))


def _prune_unpublished_clip_dirs(ctx: JobContext) -> None:
    """Remove clip directories that are not referenced by a clip row (e.g. from a cancelled run)."""
    clips_root = ctx.storage.project_dir(ctx.project_id) / "clips"
    if not clips_root.exists():
        return
    known = {r["id"] for r in ctx.conn.execute("SELECT id FROM clips WHERE project_id=?", (ctx.project_id,))}
    for d in clips_root.iterdir():
        if d.is_dir() and d.name not in known:
            try:
                ctx.storage.delete_tree(d)
            except (ValueError, OSError):
                pass
        elif d.is_dir():
            # remove render files (and thumbnails, incl. a legacy unversioned thumb.jpg) that are no longer current
            row = ctx.conn.execute("SELECT render_hash, render_path, srt_path, thumb_path FROM clips WHERE id=?",
                                   (d.name,)).fetchone()
            current = row["render_hash"] if row else None
            referenced = {Path(r).name for r in (row["render_path"], row["srt_path"], row["thumb_path"]) if r} \
                if row else set()
            for f in d.iterdir():
                if f.name in referenced or (current and f.name.startswith(current)):
                    continue
                if f.suffix in (".mp4", ".srt", ".json", ".jpg"):
                    try:
                        ctx.storage.delete_file(f)
                    except (ValueError, OSError):
                        pass


def on_analyze_finished(ctx: JobContext, message: str | None) -> Callable:
    """Callback for queue.finish: update project status/stages in the same transaction."""
    pid = ctx.project_id

    def cb(c, final_state: str) -> None:
        has_clips = c.execute("SELECT COUNT(*) FROM clips WHERE project_id=?", (pid,)).fetchone()[0]
        if final_state == "failed":
            stages.fail_running(c, pid, message)
        else:
            stages.reset_running(c, pid)
        if has_clips and ctx.params.get("reselect"):
            note = {"failed": "Re-selection failed", "cancelled": "Re-selection was cancelled",
                    "interrupted": "Re-selection was interrupted"}.get(final_state, "Re-selection stopped")
            detail = f"{note}; your previous clips are kept." + (f" {message}" if message and final_state == "failed" else "")
            c.execute("UPDATE projects SET status='ready', status_detail=?, updated_at=? WHERE id=?",
                      (detail, db.now(), pid))
            restore_reselect_options(c, pid, ctx.params)
            return
        detail = message if final_state == "failed" else (
            "Cancelled." if final_state == "cancelled" else
            "Processing was interrupted (the app or worker stopped). Resume to continue where it left off.")
        c.execute("UPDATE projects SET status=?, status_detail=?, updated_at=? WHERE id=?",
                  (final_state, detail, db.now(), pid))

    return cb
