"""Clip editor endpoints: detail (cues + context words), range edits, captions, reset, render, delete."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from clipper import db
from clipper.api.schemas import ClipDetailOut
from clipper.jobs import queue
from clipper.models import ClipEdit, ClipSelection
from clipper.storage import Storage
from test_api_support import (
    app_storage,
    conn,
    jobs_for,
    local_client,
    local_settings,
    make_transcript,
    mutation_headers,
    seed_clip,
    seed_project,
    write_clip_file,
    write_transcript,
)

__all__ = ["app_storage", "conn", "local_client", "local_settings"]


@pytest.fixture
def clip(conn, app_storage: Storage) -> tuple[str, str]:
    """A ready clip [100, 130) s of a 300 s project with a transcript (word every 0.5 s)."""
    pid = seed_project(conn, app_storage, duration=300.0)
    write_transcript(app_storage, pid, make_transcript(300.0))
    cid = seed_clip(conn, pid, src_start=100.0, src_end=130.0, render_state="ready", render_hash="h1")
    rel = write_clip_file(app_storage, pid, cid, "h1.mp4", b"old render")
    conn.execute("UPDATE clips SET render_path=? WHERE id=?", (rel, cid))
    return pid, cid


def _edit(conn, cid: str) -> ClipEdit:
    return ClipEdit.model_validate_json(conn.execute("SELECT edit FROM clips WHERE id=?", (cid,)).fetchone()[0])


def _state(conn, cid: str) -> str:
    return conn.execute("SELECT render_state FROM clips WHERE id=?", (cid,)).fetchone()[0]


def _patch(client: TestClient, cid: str, body: dict):
    return client.patch(f"/api/clips/{cid}", json=body, headers=mutation_headers(client))


def _post(client: TestClient, path: str, body: dict | None = None):
    return client.post(path, json=body, headers=mutation_headers(client)) if body is not None \
        else client.post(path, headers=mutation_headers(client))


# ----------------------------------------------------------------------------- detail
def test_detail_has_cues_and_context_words(local_client: TestClient, clip) -> None:
    _, cid = clip
    out = ClipDetailOut.model_validate(local_client.get(f"/api/clips/{cid}").json())
    assert out.src_start == 100.0 and out.src_end == 130.0 and out.duration == 30.0
    assert out.ai_src_start == 100.0 and out.render.state == "ready" and not out.render.stale
    assert out.render.video_url == f"/media/clips/{cid}/video.mp4?v=h1"
    assert out.context_start == 70.0 and out.context_end == 160.0
    assert out.source_duration == 300.0 and (out.source_width, out.source_height) == (1280, 720)
    assert out.words[0].start >= 69.5 and out.words[-1].start < 160.0
    assert all(w.end > 70.0 and w.start < 160.0 for w in out.words)
    assert out.cues, "captions are built from the stored transcript"
    assert out.cues[0].rel_start == pytest.approx(0.0) and out.cues[0].first_word_id == "w000201"
    assert all(0.0 <= c.rel_start <= c.rel_end <= 30.0 for c in out.cues)
    assert out.job is None


def test_detail_without_transcript(local_client: TestClient, conn, app_storage: Storage) -> None:
    pid = seed_project(conn, app_storage)
    cid = seed_clip(conn, pid)
    out = local_client.get(f"/api/clips/{cid}").json()
    assert out["cues"] == [] and out["words"] == []


@pytest.mark.parametrize("bad", ["c_0123456789abcdef", "nope", "c_..", "..%2F..%2Fx"])
def test_unknown_clip(local_client: TestClient, bad: str) -> None:
    assert local_client.get(f"/api/clips/{bad}").status_code == 404


# ----------------------------------------------------------------------------- PATCH
def test_patch_range_marks_stale_without_analysis(local_client: TestClient, conn, clip) -> None:
    pid, cid = clip
    before_jobs = len(jobs_for(conn, pid))
    r = _patch(local_client, cid, {"src_start": 98.0, "src_end": 140.5})
    assert r.status_code == 200, r.text
    out = ClipDetailOut.model_validate(r.json())
    assert out.src_start == 98.0 and out.src_end == 140.5 and out.duration == 42.5
    assert out.render.state == "stale" and out.render.stale and out.render.video_url   # old render still playable
    assert len(jobs_for(conn, pid)) == before_jobs                                    # nothing enqueued
    assert _edit(conn, cid).src_end == 140.5
    assert out.ai_src_start == 100.0                                                   # AI choice untouched


@pytest.mark.parametrize("body", [
    {"src_start": -1.0},
    {"src_end": 300.5},
    {"src_start": 120.0, "src_end": 122.0},          # 2 s < 3 s minimum
    {"src_start": 10.0, "src_end": 200.0},           # 190 s > 180 s maximum
    {"src_start": 131.0},                            # start after end
    {"src_start": 100.0, "src_end": 100.0},
])
def test_patch_invalid_range(local_client: TestClient, conn, clip, body: dict) -> None:
    _, cid = clip
    r = _patch(local_client, cid, body)
    assert r.status_code == 422 and r.json()["code"] == "invalid_input", body
    assert _edit(conn, cid).src_start == 100.0 and _state(conn, cid) == "ready"


def test_patch_extremes_are_allowed(local_client: TestClient, clip) -> None:
    _, cid = clip
    assert _patch(local_client, cid, {"src_start": 0.0, "src_end": 3.0}).status_code == 200
    assert _patch(local_client, cid, {"src_start": 120.0, "src_end": 300.0}).status_code == 200


def test_patch_style_framing_captions(local_client: TestClient, conn, clip) -> None:
    _, cid = clip
    body = {"style": {"size": "L", "color": "boxed", "position": "top", "highlight": False, "uppercase": True},
            "framing": {"mode": "crop", "crop_x": 0.3}, "captions_enabled": False}
    out = _patch(local_client, cid, body).json()
    assert out["edit"]["style"]["color"] == "boxed" and out["edit"]["framing"]["crop_x"] == 0.3
    assert out["edit"]["captions_enabled"] is False and out["render"]["state"] == "stale"
    assert _patch(local_client, cid, {"framing": {"mode": "crop", "crop_x": 1.5}}).status_code == 422
    assert _patch(local_client, cid, {"src_start": 1, "bogus": 2}).status_code == 422


def test_patch_without_change_keeps_ready(local_client: TestClient, conn, clip) -> None:
    _, cid = clip
    assert _patch(local_client, cid, {"src_start": 100.0}).status_code == 200
    assert _patch(local_client, cid, {}).status_code == 200
    assert _state(conn, cid) == "ready"


def test_patch_never_rendered_stays_none(local_client: TestClient, conn, app_storage: Storage) -> None:
    pid = seed_project(conn, app_storage)
    cid = seed_clip(conn, pid)
    _patch(local_client, cid, {"src_end": 135.0})
    assert _state(conn, cid) == "none"


# ----------------------------------------------------------------------------- captions
def test_caption_edit_same_token_count(local_client: TestClient, conn, clip) -> None:
    _, cid = clip
    r = _post(local_client, f"/api/clips/{cid}/captions",
              {"first_word_id": "w000201", "last_word_id": "w000202", "text": "Hello, world!"})
    assert r.status_code == 200, r.text
    edit = _edit(conn, cid)
    assert edit.word_overrides == {"w000201": "Hello,", "w000202": "world!"} and edit.cue_overrides == []
    assert _state(conn, cid) == "stale"
    first_cue = ClipDetailOut.model_validate(r.json()).cues[0]
    assert first_cue.text.startswith("Hello, world!")


def test_caption_edit_different_token_count(local_client: TestClient, conn, clip) -> None:
    _, cid = clip
    r = _post(local_client, f"/api/clips/{cid}/captions",
              {"first_word_id": "w000203", "last_word_id": "w000205", "text": "One phrase"})
    assert r.status_code == 200
    edit = _edit(conn, cid)
    assert len(edit.cue_overrides) == 1 and edit.cue_overrides[0].text == "One phrase"
    cue = next(c for c in ClipDetailOut.model_validate(r.json()).cues if c.first_word_id == "w000203")
    assert cue.text == "One phrase" and cue.highlight is False


@pytest.mark.parametrize("first,last", [("w000001", "w000002"), ("w000300", "w000301"), ("w999999", "w000201")])
def test_caption_edit_outside_clip(local_client: TestClient, conn, clip, first: str, last: str) -> None:
    _, cid = clip
    r = _post(local_client, f"/api/clips/{cid}/captions", {"first_word_id": first, "last_word_id": last, "text": "x"})
    assert r.status_code == 422
    assert _edit(conn, cid).word_overrides == {} and _state(conn, cid) == "ready"


def test_caption_reset_and_clip_reset(local_client: TestClient, conn, clip) -> None:
    _, cid = clip
    _post(local_client, f"/api/clips/{cid}/captions",
          {"first_word_id": "w000201", "last_word_id": "w000201", "text": "Hi"})
    _post(local_client, f"/api/clips/{cid}/captions",
          {"first_word_id": "w000210", "last_word_id": "w000212", "text": "two words"})
    r = _post(local_client, f"/api/clips/{cid}/captions/reset")
    assert r.status_code == 200
    edit = _edit(conn, cid)
    assert edit.word_overrides == {} and edit.cue_overrides == []

    defaults = {"clip_count": 5, "min_seconds": 20, "max_seconds": 60, "captions_enabled": True,
                "caption_style": {"size": "S", "color": "yellow", "position": "middle", "highlight": True,
                                  "uppercase": False}}
    local_client.put("/api/settings", json=defaults, headers=mutation_headers(local_client))
    _patch(local_client, cid, {"src_start": 90.0, "src_end": 150.0, "framing": {"mode": "fit"}})
    out = _post(local_client, f"/api/clips/{cid}/reset").json()
    assert (out["src_start"], out["src_end"]) == (100.0, 130.0)
    assert out["edit"]["framing"]["mode"] == "auto" and out["edit"]["style"]["color"] == "yellow"


def test_hidden_caption_stays_listed_in_editor(local_client: TestClient, conn, clip) -> None:
    """A caption hidden with empty text is still returned (text '', edited) so it can be restored."""
    _, cid = clip
    first = ClipDetailOut.model_validate(local_client.get(f"/api/clips/{cid}").json()).cues[0]
    r = _post(local_client, f"/api/clips/{cid}/captions",
              {"first_word_id": first.first_word_id, "last_word_id": first.last_word_id, "text": ""})
    assert r.status_code == 200, r.text
    cues = ClipDetailOut.model_validate(r.json()).cues
    hidden = [c for c in cues if c.text == ""]
    assert len(hidden) == 1 and hidden[0].edited
    assert (hidden[0].first_word_id, hidden[0].last_word_id) == (first.first_word_id, first.last_word_id)
    # restoring it through the same editor path brings the words back
    r = _post(local_client, f"/api/clips/{cid}/captions",
              {"first_word_id": first.first_word_id, "last_word_id": first.last_word_id, "text": first.text})
    assert r.status_code == 200 and _edit(conn, cid).word_overrides == {}


# ----------------------------------------------------------------------------- render
def test_render_enqueues_once(local_client: TestClient, conn, app_storage: Storage) -> None:
    pid = seed_project(conn, app_storage, source_file=None)
    conn.execute("UPDATE projects SET source_path='projects/x/source/source.mp4' WHERE id=?", (pid,))
    cid = seed_clip(conn, pid, render_state="stale")
    r = _post(local_client, f"/api/clips/{cid}/render")
    assert r.status_code == 200
    out = ClipDetailOut.model_validate(r.json())
    assert out.render.state == "queued" and out.job is not None and out.job.kind == "render"
    jobs = [j for j in jobs_for(conn, pid) if j["kind"] == "render"]
    assert len(jobs) == 1 and jobs[0]["priority"] == 10 and jobs[0]["clip_id"] == cid
    assert all(j["kind"] != "analyze" for j in jobs_for(conn, pid))
    again = ClipDetailOut.model_validate(_post(local_client, f"/api/clips/{cid}/render").json())
    assert again.job.id == out.job.id and len(jobs_for(conn, pid)) == 1


def test_render_of_ready_clip_keeps_state(local_client: TestClient, conn, clip) -> None:
    pid, cid = clip
    out = _post(local_client, f"/api/clips/{cid}/render").json()
    assert out["render"]["state"] == "ready" and out["job"]["state"] == "queued"
    assert json.loads(jobs_for(conn, pid)[0]["params"]) == {}


def test_render_requires_csrf(local_client: TestClient, clip) -> None:
    _, cid = clip
    r = local_client.post(f"/api/clips/{cid}/render", headers={"Origin": "http://127.0.0.1:8765"})
    assert r.status_code == 403


# ----------------------------------------------------------------------------- delete
def test_delete_clip(local_client: TestClient, conn, app_storage: Storage, clip) -> None:
    pid, cid = clip
    queue.create_job(conn, project_id=pid, kind="render", clip_id=cid, priority=10)
    r = local_client.delete(f"/api/clips/{cid}", headers=mutation_headers(local_client))
    assert r.status_code == 200 and r.json() == {"deleted": True}
    assert conn.execute("SELECT COUNT(*) FROM clips WHERE id=?", (cid,)).fetchone()[0] == 0
    assert not app_storage.clip_dir(pid, cid).exists()
    assert jobs_for(conn, pid)[0]["state"] == "cancelled"
    assert local_client.get(f"/api/clips/{cid}").status_code == 404


def test_delete_clip_while_rendering(local_client: TestClient, conn, app_storage: Storage, clip) -> None:
    pid, cid = clip
    job = queue.create_job(conn, project_id=pid, kind="render", clip_id=cid, priority=10)
    conn.execute("UPDATE jobs SET state='running', worker_id='w_x', started_at=? WHERE id=?", (db.now(), job))
    r = local_client.delete(f"/api/clips/{cid}", headers=mutation_headers(local_client))
    assert r.status_code == 409
    assert app_storage.clip_dir(pid, cid).exists()
    assert conn.execute("SELECT cancel_requested FROM jobs WHERE id=?", (job,)).fetchone()[0] == 1


@pytest.mark.parametrize("job_state", ["queued", "running"])
def test_delete_clip_refused_while_analysis_active(local_client: TestClient, conn, app_storage: Storage, clip,
                                                   job_state: str) -> None:
    """An analysis/re-selection may re-publish this clip id, so deleting must wait (else it comes back
    'ready' with its video already removed)."""
    pid, cid = clip
    job = queue.create_job(conn, project_id=pid, kind="analyze", params={"reselect": True})
    if job_state == "running":
        conn.execute("UPDATE jobs SET state='running', worker_id='w_x', started_at=? WHERE id=?", (db.now(), job))
    r = local_client.delete(f"/api/clips/{cid}", headers=mutation_headers(local_client))
    assert r.status_code == 409 and "still processing" in r.json()["detail"]
    assert conn.execute("SELECT COUNT(*) FROM clips WHERE id=?", (cid,)).fetchone()[0] == 1
    assert app_storage.clip_dir(pid, cid).exists()
    row = conn.execute("SELECT state, cancel_requested FROM jobs WHERE id=?", (job,)).fetchone()
    assert (row["state"], row["cancel_requested"]) == (job_state, 0)
    conn.execute("UPDATE jobs SET state='cancelled' WHERE id=?", (job,))
    assert local_client.delete(f"/api/clips/{cid}", headers=mutation_headers(local_client)).status_code == 200


def test_clip_reset_restores_the_style_it_was_first_rendered_with(local_client: TestClient, conn, clip) -> None:
    """'Restore original edit' brings back the clip's own first-render style, not today's defaults."""
    from clipper.models import CaptionStyle

    _, cid = clip
    row = conn.execute("SELECT selection FROM clips WHERE id=?", (cid,)).fetchone()
    sel = ClipSelection.model_validate_json(row["selection"])
    original = CaptionStyle(size="L", color="boxed", position="top")
    conn.execute("UPDATE clips SET selection=? WHERE id=?",
                 (sel.model_copy(update={"original_style": original}).model_dump_json(), cid))
    defaults = {"clip_count": 5, "min_seconds": 20, "max_seconds": 60, "captions_enabled": True,
                "caption_style": {"size": "S", "color": "yellow", "position": "middle", "highlight": True,
                                  "uppercase": False}}
    local_client.put("/api/settings", json=defaults, headers=mutation_headers(local_client))
    _patch(local_client, cid, {"style": {"size": "M", "color": "white", "position": "bottom", "highlight": False,
                                         "uppercase": True}})
    out = _post(local_client, f"/api/clips/{cid}/reset").json()
    assert out["edit"]["style"] == original.model_dump()
