"""Project endpoints: create (YouTube/upload), list/detail shapes, job controls, cleanup and deletion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from clipper import db
from clipper.api.schemas import ProjectOut, ProjectSummaryOut
from clipper.errors import InputError
from clipper.jobs import queue
from clipper.models import ProjectOptions
from clipper.storage import Storage
from test_api_support import (
    app_storage,
    conn,
    jobs_for,
    local_client,
    local_settings,
    mutation_headers,
    seed_clip,
    seed_project,
)

__all__ = ["app_storage", "conn", "local_client", "local_settings"]

YT_URL = "https://www.youtube.com/watch?v=nKyWusznRgQ"
STAGE_KEYS = ["import", "probe", "audio", "transcribe", "candidates", "review", "framing", "render"]


@pytest.fixture
def fake_parser(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    import clipper.media.youtube as youtube

    calls: list[str] = []

    def parse(url: str) -> tuple[str, str]:
        calls.append(url)
        if "bad" in url:
            raise InputError("Paste a YouTube video link.")
        return "abcdefghijk", "https://www.youtube.com/watch?v=abcdefghijk"

    monkeypatch.setattr(youtube, "parse_youtube_url", parse)
    return calls


def _create(client: TestClient, url: str = YT_URL, options: dict | None = None):
    body: dict = {"source": {"kind": "youtube", "url": url}}
    if options is not None:
        body["options"] = options
    return client.post("/api/projects", json=body, headers=mutation_headers(client))


def _post(client: TestClient, path: str, body: dict | None = None):
    return client.post(path, json=body, headers=mutation_headers(client)) if body is not None \
        else client.post(path, headers=mutation_headers(client))


# ----------------------------------------------------------------------------- create
def test_create_youtube_project(local_client: TestClient, conn, fake_parser: list[str]) -> None:
    r = _create(local_client)
    assert r.status_code == 201, r.text
    out = ProjectOut.model_validate(r.json())
    assert fake_parser == [YT_URL]
    assert out.title == "YouTube video abcdefghijk" and out.status == "queued"
    assert out.source_url == "https://www.youtube.com/watch?v=abcdefghijk"
    assert [s.key for s in out.stages] == STAGE_KEYS and {s.state for s in out.stages} == {"pending"}
    assert out.stages[5].label == "Review & rank"
    assert out.job is not None and out.job.kind == "analyze" and out.job.state == "queued"
    assert out.actions.can_cancel and not out.actions.can_resume and not out.actions.can_delete
    assert out.usage.requests == 0 and out.clips == []
    job = jobs_for(conn, out.id)[0]
    assert job["priority"] == 0 and json.loads(job["params"]) == {}


def test_invalid_youtube_url(local_client: TestClient, conn, fake_parser: list[str]) -> None:
    r = _create(local_client, url="https://bad.example/video")
    assert r.status_code == 422 and r.json() == {"detail": "Paste a YouTube video link.", "code": "invalid_input"}
    assert conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0] == 0


def test_invalid_url_with_real_parser(local_client: TestClient) -> None:
    r = _create(local_client, url="https://vimeo.com/12345")
    assert r.status_code == 422 and r.json()["code"] == "invalid_input"


def test_options_default_to_saved_defaults(local_client: TestClient, fake_parser) -> None:
    saved = {"clip_count": 2, "min_seconds": 30.0, "max_seconds": 90.0, "captions_enabled": False}
    assert local_client.put("/api/settings", json=saved, headers=mutation_headers(local_client)).status_code == 200
    options = _create(local_client).json()["options"]
    assert options == {**saved, "language": "en"}
    explicit = _create(local_client, options={"clip_count": 7, "min_seconds": 20, "max_seconds": 60}).json()
    assert explicit["options"]["clip_count"] == 7


def test_clip_limit_enforced(local_client: TestClient, local_settings, fake_parser) -> None:
    local_settings.max_clips = 3
    r = _create(local_client, options={"clip_count": 5, "min_seconds": 20, "max_seconds": 60})
    assert r.status_code == 422 and "3" in r.json()["detail"]
    r = _create(local_client, options={"clip_count": 5, "min_seconds": 50, "max_seconds": 52})
    assert r.status_code == 422
    r = _create(local_client)                          # saved/config default of 5 is clamped to the cap
    assert r.status_code == 201 and r.json()["options"]["clip_count"] == 3


# ----------------------------------------------------------------------------- list/detail
def test_list_newest_first_and_detail(local_client: TestClient, conn, app_storage: Storage, fake_parser) -> None:
    first = _create(local_client).json()["id"]
    second = seed_project(conn, app_storage, title="Second")
    cid = seed_clip(conn, second, rank=1)
    conn.execute("UPDATE projects SET created_at=created_at+10 WHERE id=?", (second,))
    listed = [ProjectSummaryOut.model_validate(p) for p in local_client.get("/api/projects").json()]
    assert [p.id for p in listed] == [second, first]
    assert listed[0].clip_count == 1 and listed[0].duration == 300.0
    detail = ProjectOut.model_validate(local_client.get(f"/api/projects/{second}").json())
    assert detail.media is not None and detail.media.width == 1280
    assert [c.id for c in detail.clips] == [cid] and detail.clips[0].render.state == "none"
    assert detail.actions.can_delete and not detail.actions.can_cancel


@pytest.mark.parametrize("bad_id", ["p_0123456789abcdef", "nope", "..%2F..%2Fetc", "p_XYZ"])
def test_unknown_project_is_404(local_client: TestClient, bad_id: str) -> None:
    r = local_client.get(f"/api/projects/{bad_id}")
    assert r.status_code == 404


def test_stage_rows_and_usage_estimate(local_client: TestClient, conn, app_storage: Storage) -> None:
    pid = seed_project(conn, app_storage)
    conn.execute("INSERT INTO stages(project_id, stage, state, detail, duration_s) VALUES (?, 'import', 'done', "
                 "'Downloaded', 3.5)", (pid,))
    conn.execute("INSERT INTO stages(project_id, stage, state) VALUES (?, 'probe', 'running')", (pid,))
    rows = [
        ("transcribe", "whisper-1", "ok", None, None, 600.0),
        ("transcribe", "whisper-1", "cached", None, None, 600.0),
        ("candidates", "gpt-6.1-sol", "ok", 1_000_000, 100_000, None),
    ]
    for stage, model, status, inp, out, audio in rows:
        conn.execute("INSERT INTO api_usage(project_id, stage, provider, model, status, input_tokens, output_tokens, "
                     "audio_seconds, created_at) VALUES (?, ?, 'openai', ?, ?, ?, ?, ?, ?)",
                     (pid, stage, model, status, inp, out, audio, db.now()))
    body = ProjectOut.model_validate(local_client.get(f"/api/projects/{pid}").json())
    assert body.stages[0].state == "done" and body.stages[0].duration_s == 3.5
    assert body.stages[1].state == "running" and body.stages[2].state == "pending"
    assert body.usage.requests == 2 and body.usage.audio_seconds == 600.0
    assert body.usage.estimated_cost_usd == pytest.approx(0.06 + 2.0 + 1.0)
    assert "2026-10-05" in body.usage.cost_note and "cached-token" in body.usage.cost_note
    conn.execute("INSERT INTO api_usage(project_id, stage, provider, model, status, input_tokens, output_tokens, "
                 "created_at) VALUES (?, 'review', 'openai', 'gpt-9-unknown', 'ok', 10, 10, ?)", (pid, db.now()))
    body = local_client.get(f"/api/projects/{pid}").json()
    assert body["usage"]["estimated_cost_usd"] is None and "gpt-9-unknown" in body["usage"]["cost_note"]


# ----------------------------------------------------------------------------- job controls
def test_cancel_queued_job(local_client: TestClient, conn, fake_parser) -> None:
    pid = _create(local_client).json()["id"]
    r = _post(local_client, f"/api/projects/{pid}/cancel")
    assert r.status_code == 200
    out = ProjectOut.model_validate(r.json())
    assert out.status == "cancelled" and out.job.state == "cancelled"
    assert out.actions.can_resume and out.actions.can_delete
    assert _post(local_client, f"/api/projects/{pid}/cancel").status_code == 409


def test_cancel_running_job_and_queued_render(local_client: TestClient, conn, app_storage: Storage) -> None:
    pid = seed_project(conn, app_storage, status="running")
    cid = seed_clip(conn, pid, render_state="queued")
    analyze = queue.create_job(conn, project_id=pid, kind="analyze")
    render = queue.create_job(conn, project_id=pid, kind="render", clip_id=cid, priority=10)
    conn.execute("UPDATE jobs SET state='running', worker_id='w_x' WHERE id=?", (analyze,))
    out = ProjectOut.model_validate(_post(local_client, f"/api/projects/{pid}/cancel").json())
    states = {j["id"]: (j["state"], j["cancel_requested"]) for j in jobs_for(conn, pid)}
    assert states[analyze] == ("running", 1) and states[render] == ("cancelled", 1)
    assert out.status == "running" and not out.actions.can_cancel
    assert conn.execute("SELECT render_state FROM clips WHERE id=?", (cid,)).fetchone()[0] == "none"


def test_cancel_queued_reselect_keeps_clips(local_client: TestClient, conn, app_storage: Storage) -> None:
    pid = seed_project(conn, app_storage, status="queued")
    seed_clip(conn, pid)
    queue.create_job(conn, project_id=pid, kind="analyze", params={"reselect": True})
    out = _post(local_client, f"/api/projects/{pid}/cancel").json()
    assert out["status"] == "ready" and "previous clips are kept" in out["status_detail"]


def test_cancelled_reselect_restores_previous_options(local_client: TestClient, conn, app_storage: Storage) -> None:
    """A re-selection cancelled before it ran keeps the old clips, so it keeps their options too."""
    original = ProjectOptions(clip_count=5, min_seconds=20, max_seconds=60, captions_enabled=True)
    pid = seed_project(conn, app_storage, options=original)
    seed_clip(conn, pid)
    conn.execute("INSERT INTO stages(project_id, stage, state) VALUES (?, 'transcribe', 'done')", (pid,))
    new = {"clip_count": 3, "min_seconds": 10, "max_seconds": 30, "captions_enabled": False}
    r = _post(local_client, f"/api/projects/{pid}/reselect", {"options": new})
    assert r.status_code == 200 and r.json()["options"]["clip_count"] == 3
    out = _post(local_client, f"/api/projects/{pid}/cancel").json()
    assert out["status"] == "ready" and "previous clips are kept" in out["status_detail"]
    assert ProjectOptions.model_validate(out["options"]) == original
    # resetting a kept clip now uses the options it was made with (captions stay on)
    cid = out["clips"][0]["id"]
    assert _post(local_client, f"/api/clips/{cid}/reset").json()["edit"]["captions_enabled"] is True


def test_cancelled_reselect_without_clips_keeps_new_options(local_client: TestClient, conn,
                                                             app_storage: Storage) -> None:
    """Without previous clips the project becomes 'cancelled' and a Resume continues with the new options."""
    pid = seed_project(conn, app_storage, options=ProjectOptions(clip_count=5))
    conn.execute("INSERT INTO stages(project_id, stage, state) VALUES (?, 'transcribe', 'done')", (pid,))
    _post(local_client, f"/api/projects/{pid}/reselect", {"options": {"clip_count": 2, "min_seconds": 20,
                                                                      "max_seconds": 60}})
    out = _post(local_client, f"/api/projects/{pid}/cancel").json()
    assert out["status"] == "cancelled" and out["options"]["clip_count"] == 2


@pytest.mark.parametrize("status,allowed", [("failed", True), ("cancelled", True), ("interrupted", True),
                                            ("ready", False), ("no_clips", False)])
def test_resume(local_client: TestClient, conn, app_storage: Storage, status: str, allowed: bool) -> None:
    pid = seed_project(conn, app_storage, status=status)
    r = _post(local_client, f"/api/projects/{pid}/resume")
    if not allowed:
        assert r.status_code == 409 and jobs_for(conn, pid) == []
        return
    assert r.status_code == 200 and r.json()["status"] == "queued"
    jobs = jobs_for(conn, pid)
    assert len(jobs) == 1 and json.loads(jobs[0]["params"]) == {"resume": True} and jobs[0]["kind"] == "analyze"
    assert _post(local_client, f"/api/projects/{pid}/resume").status_code == 409     # now active


def test_reselect(local_client: TestClient, conn, app_storage: Storage) -> None:
    pid = seed_project(conn, app_storage)
    options = {"options": {"clip_count": 3, "min_seconds": 30, "max_seconds": 90, "captions_enabled": True}}
    assert _post(local_client, f"/api/projects/{pid}/reselect", options).status_code == 409   # no transcript
    conn.execute("INSERT INTO stages(project_id, stage, state) VALUES (?, 'transcribe', 'done')", (pid,))
    assert ProjectOut.model_validate(local_client.get(f"/api/projects/{pid}").json()).actions.can_reselect
    r = _post(local_client, f"/api/projects/{pid}/reselect", options)
    assert r.status_code == 200, r.text
    assert r.json()["options"]["clip_count"] == 3 and r.json()["status"] == "queued"
    jobs = jobs_for(conn, pid)
    # The job also carries the previous options, restored if the re-selection ends without publishing.
    assert len(jobs) == 1 and json.loads(jobs[0]["params"]) == {
        "reselect": True, "previous_options": json.loads(ProjectOptions().model_dump_json())}
    assert _post(local_client, f"/api/projects/{pid}/reselect", options).status_code == 409   # job active
    bad = {"options": {"clip_count": 3, "min_seconds": 60, "max_seconds": 20}}
    assert _post(local_client, f"/api/projects/{pid}/reselect", bad).status_code == 422


def test_cleanup_keeps_small_results(local_client: TestClient, conn, app_storage: Storage) -> None:
    pid = seed_project(conn, app_storage)
    work = app_storage.work_dir(pid)
    for rel in ("audio/analysis.wav", "audio/chunk_000.flac", "frames/c001_0.jpg", "transcribe_raw/chunk_000.json",
                "transcript.json", "candidates.json", "review.json", "scratch.tmp"):
        (work / rel).parent.mkdir(parents=True, exist_ok=True)
        (work / rel).write_bytes(b"x")
    r = _post(local_client, f"/api/projects/{pid}/cleanup")
    assert r.status_code == 200
    assert sorted(p.name for p in work.iterdir()) == ["candidates.json", "review.json", "transcript.json"]
    queue.create_job(conn, project_id=pid, kind="analyze")
    assert _post(local_client, f"/api/projects/{pid}/cleanup").status_code == 409


def test_delete_project(local_client: TestClient, conn, app_storage: Storage, tmp_path: Path, fake_parser) -> None:
    outside = tmp_path / "keep-me.txt"
    outside.write_text("user file")
    pid = _create(local_client).json()["id"]
    cid = seed_clip(conn, pid)
    (app_storage.clip_dir(pid, cid)).mkdir(parents=True)
    (app_storage.clip_dir(pid, cid) / "x.mp4").write_bytes(b"x")
    conn.execute("INSERT INTO api_usage(project_id, stage, provider, model, status, created_at) "
                 "VALUES (?, 'transcribe', 'openai', 'whisper-1', 'ok', ?)", (pid, db.now()))
    r = local_client.delete(f"/api/projects/{pid}", headers=mutation_headers(local_client))
    assert r.status_code == 409                                  # analyze job still queued
    assert app_storage.project_dir(pid).exists()
    _post(local_client, f"/api/projects/{pid}/cancel")
    r = local_client.delete(f"/api/projects/{pid}", headers=mutation_headers(local_client))
    assert r.status_code == 200 and r.json() == {"deleted": True}
    assert not app_storage.project_dir(pid).exists()
    for table, column in (("projects", "id"), ("jobs", "project_id"), ("clips", "project_id"),
                          ("api_usage", "project_id")):
        assert conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {column}=?", (pid,)).fetchone()[0] == 0
    assert outside.read_text() == "user file"
    assert local_client.get(f"/api/projects/{pid}").status_code == 404
    assert local_client.delete(f"/api/projects/{pid}", headers=mutation_headers(local_client)).status_code == 404
