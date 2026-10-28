"""Local mode: auth disabled on loopback only, CSRF still enforced, frontend serving, errors, system, settings."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from clipper import db
from clipper.api import auth, routes_clips, routes_media, routes_projects, routes_system, routes_uploads
from clipper.api.app import create_app
from clipper.api.schemas import DefaultsModel, SystemOut
from clipper.config import ConfigError
from test_api_support import (
    LOCAL_ORIGIN,
    local_client,
    local_settings,
    make_local_settings,
    mutation_headers,
)

__all__ = ["local_client", "local_settings"]

SETTINGS_BODY = {"clip_count": 4, "min_seconds": 15.0, "max_seconds": 45.0, "captions_enabled": False,
                 "caption_style": {"size": "L", "color": "yellow", "position": "middle", "highlight": False,
                                   "uppercase": True}}


# ----------------------------------------------------------------------------- auth disabled
def test_session_reports_local_token(local_client: TestClient, local_settings) -> None:
    body = local_client.get("/api/auth/session").json()
    assert body["authenticated"] is True and body["auth_required"] is False and body["mode"] == "local"
    conn = db.connect(local_settings.db_path)
    try:
        stored = conn.execute("SELECT value FROM meta WHERE key='local_csrf'").fetchone()["value"]
    finally:
        conn.close()
    assert body["csrf_token"] == stored and len(stored) >= 32
    # The token survives an app restart (same database).
    with TestClient(create_app(local_settings), base_url=LOCAL_ORIGIN) as again:
        assert again.get("/api/auth/session").json()["csrf_token"] == stored


def test_auth_can_only_be_disabled_on_loopback(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        create_app(make_local_settings(tmp_path, host="0.0.0.0", auth="disabled"))
    # "auto" on a non-loopback bind turns auth on, which then needs secrets.
    with pytest.raises(ConfigError):
        create_app(make_local_settings(tmp_path, host="0.0.0.0", auth="auto"))
    assert make_local_settings(tmp_path, host="127.0.0.1", auth="auto").auth_enabled is False


def test_get_is_open_but_post_needs_csrf_and_origin(local_client: TestClient) -> None:
    assert local_client.get("/api/projects").status_code == 200
    r = local_client.put("/api/settings", json=SETTINGS_BODY, headers={"Origin": LOCAL_ORIGIN})
    assert r.status_code == 403 and r.json()["code"] == "csrf_failed"
    headers = mutation_headers(local_client)
    assert local_client.put("/api/settings", json=SETTINGS_BODY, headers=headers).status_code == 200
    r = local_client.put("/api/settings", json=SETTINGS_BODY,
                         headers=dict(headers, Origin="http://localhost:5173"))
    assert r.status_code == 403 and r.json()["code"] == "bad_origin"
    assert local_client.put("/api/settings", json=SETTINGS_BODY,
                            headers=dict(headers, Origin="http://localhost:8765")).status_code == 200


def test_every_mutating_route_enforces_csrf(local_client: TestClient) -> None:
    exempt = {"/api/auth/login", "/api/auth/logout"}
    routers = (auth.router, routes_system.public_router, routes_system.router, routes_uploads.router,
               routes_projects.router, routes_clips.router, routes_media.router)
    checked = 0
    for route in (r for router in routers for r in router.routes):
        if not isinstance(route, APIRoute) or route.path in exempt:
            continue
        for method in route.methods - {"GET", "HEAD", "OPTIONS"}:
            path = re.sub(r"\{project_id\}", "p_0123456789abcdef", route.path)
            path = re.sub(r"\{clip_id\}", "c_0123456789abcdef", path)
            r = local_client.request(method, path, headers={"Origin": LOCAL_ORIGIN})
            assert r.status_code == 403 and r.json()["code"] == "csrf_failed", (method, route.path)
            checked += 1
    assert checked >= 14


def test_loopback_host_aliases_allowed(local_client: TestClient) -> None:
    for host in ("127.0.0.1:8765", "localhost:8765", "[::1]:8765"):
        assert local_client.get("/api/health", headers={"Host": host}).status_code == 200, host
    assert local_client.get("/api/health", headers={"Host": "attacker.example:8765"}).status_code == 400


# ----------------------------------------------------------------------------- errors
def test_error_bodies_are_plain_json(local_client: TestClient) -> None:
    r = local_client.get("/api/projects/p_0123456789abcdef")
    assert r.status_code == 404 and set(r.json()) == {"detail", "code"} and r.json()["code"] == "not_found"
    r = local_client.get("/api/does-not-exist")
    assert r.status_code == 404 and r.json()["code"] == "not_found"
    r = local_client.post("/api/projects", json={"source": {"kind": "youtube"}},
                          headers=mutation_headers(local_client))
    assert r.status_code == 422
    assert isinstance(r.json()["detail"], str) and r.json()["code"] == "invalid_input"


def test_unexpected_errors_are_generic(local_client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_a, **_k):
        raise RuntimeError("/Users/secret/path and sk-live-key")

    monkeypatch.setattr(routes_system, "worker_status", boom)
    r = local_client.get("/api/system")
    assert r.status_code == 500
    assert r.json() == {"detail": "Something went wrong on the server. Please try again.", "code": "internal_error"}
    assert "secret" not in r.text and "Traceback" not in r.text
    assert r.headers["x-content-type-options"] == "nosniff"


def test_large_json_bodies_are_refused(local_client: TestClient) -> None:
    headers = dict(mutation_headers(local_client), **{"Content-Type": "application/json"})
    r = local_client.put("/api/settings", content=b" " * (1024 * 1024 + 10), headers=headers)
    assert r.status_code == 413


# ----------------------------------------------------------------------------- frontend
def test_frontend_not_built_page(local_client: TestClient) -> None:
    r = local_client.get("/projects/abc")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "scripts/setup.sh" in r.text
    assert local_client.get("/media/nothing").status_code == 404
    assert local_client.get("/api").status_code == 404


def test_built_frontend_spa_fallback(tmp_path: Path) -> None:
    dist = tmp_path / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<!doctype html><div id=app></div>", encoding="utf-8")
    (dist / "assets" / "app-123.js").write_text("console.log(1)", encoding="utf-8")
    (dist / "favicon.svg").write_text("<svg/>", encoding="utf-8")
    (dist / ".env").write_text("TOKEN=hidden", encoding="utf-8")
    (tmp_path / "secret.txt").write_text("top secret", encoding="utf-8")
    settings = make_local_settings(tmp_path, frontend_dist=dist)
    with TestClient(create_app(settings), base_url=LOCAL_ORIGIN) as client:
        assert "id=app" in client.get("/").text
        assert "id=app" in client.get("/projects/p_1/clips/c_2").text
        js = client.get("/assets/app-123.js")
        assert js.status_code == 200 and "javascript" in js.headers["content-type"]
        assert client.get("/favicon.svg").text == "<svg/>"
        assert client.get("/assets/missing.js").status_code == 404
        assert "hidden" not in client.get("/.env").text
        assert "top secret" not in client.get("/..%2fsecret.txt").text
        assert "top secret" not in client.get("/assets/..%2f..%2fsecret.txt").text
        api = client.get("/api/unknown")
        assert api.status_code == 404 and api.json()["code"] == "not_found"
        assert client.get("/media/clips/x/video.mp4").status_code == 404


# ----------------------------------------------------------------------------- settings
def test_settings_defaults_roundtrip(local_client: TestClient, local_settings) -> None:
    first = DefaultsModel.model_validate(local_client.get("/api/settings").json())
    assert first.clip_count == local_settings.default_clip_count
    assert first.min_seconds == local_settings.default_min_seconds
    headers = mutation_headers(local_client)
    r = local_client.put("/api/settings", json=SETTINGS_BODY, headers=headers)
    assert r.status_code == 200
    assert local_client.get("/api/settings").json() == DefaultsModel.model_validate(SETTINGS_BODY).model_dump()


@pytest.mark.parametrize("change", [
    {"min_seconds": 40.0, "max_seconds": 42.0},       # range too narrow
    {"clip_count": 11},                               # above schema cap
    {"max_seconds": 500.0},
    {"unknown": 1},
])
def test_settings_validation(local_client: TestClient, change: dict) -> None:
    r = local_client.put("/api/settings", json={**SETTINGS_BODY, **change}, headers=mutation_headers(local_client))
    assert r.status_code == 422, change


def test_settings_respect_configured_clip_cap(tmp_path: Path) -> None:
    with TestClient(create_app(make_local_settings(tmp_path, max_clips=3)), base_url=LOCAL_ORIGIN) as client:
        r = client.put("/api/settings", json=SETTINGS_BODY, headers=mutation_headers(client))
        assert r.status_code == 422 and "3" in r.json()["detail"]


# ----------------------------------------------------------------------------- system
def test_system_info(local_client: TestClient, local_settings, monkeypatch: pytest.MonkeyPatch) -> None:
    import clipper.framing.faces as faces
    import clipper.media.probe as probe
    import clipper.media.youtube as youtube

    routes_system._cache.clear()
    monkeypatch.setattr(probe, "tool_version", lambda binary: "ffmpeg version 9.0.1 Copyright (c) the devs")
    monkeypatch.setattr(youtube, "ytdlp_version", lambda: "2026.08.19")
    monkeypatch.setattr(youtube, "js_runtime_info", lambda settings: "deno 2.8.2")
    monkeypatch.setattr(faces, "detector_available", lambda *a, **k: True)
    conn = db.connect(local_settings.db_path)
    try:
        t = db.now()
        conn.execute("INSERT INTO workers(id, pid, hostname, started_at, heartbeat_at, info) VALUES (?,?,?,?,?,?)",
                     ("w_aaaaaaaaaaaaaaaa", 1, "h", t - 100, t - 100, db.dumps({"encoder": "libx264"})))
        conn.execute("INSERT INTO workers(id, pid, hostname, started_at, heartbeat_at, info) VALUES (?,?,?,?,?,?)",
                     ("w_bbbbbbbbbbbbbbbb", 2, "h", t - 10, t - 2,
                      db.dumps({"encoder": "h264_videotoolbox"})))
    finally:
        conn.close()
    try:
        body = local_client.get("/api/system").json()
    finally:
        routes_system._cache.clear()
    out = SystemOut.model_validate(body)
    assert out.mode == "local" and out.storage.location_label == "This computer"
    assert out.storage.data_dir_display == str(local_settings.data_dir.resolve())
    assert out.storage.reserve_bytes == local_settings.min_free_disk_mb * 1024 * 1024
    assert out.storage.free_bytes > 0 and out.storage.used_bytes > 0
    assert out.openai.configured is False and out.openai.transcribe_model == "whisper-1"
    sent = " ".join(out.openai.data_sent).lower()
    assert "audio" in sent and "transcript" in sent and "frames" in sent and "not uploaded" in sent
    assert "2027-02-26" in out.openai.transcription_note
    assert out.tools.ffmpeg_version == "9.0.1" and out.tools.js_runtime == "deno 2.8.2"
    assert out.tools.encoder == "h264_videotoolbox" and out.tools.face_detector == "YuNet (OpenCV)"
    assert out.worker.online is True and out.worker.last_heartbeat_age_s is not None
    assert out.limits.max_clips == local_settings.max_clips


def test_system_worker_offline_and_tools_missing(local_client: TestClient, local_settings,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    import clipper.media.youtube as youtube

    def broken(*_a, **_k):
        raise RuntimeError("module not ready")

    routes_system._cache.clear()
    monkeypatch.setattr(youtube, "js_runtime_info", broken)
    conn = db.connect(local_settings.db_path)
    try:
        conn.execute("INSERT INTO workers(id, pid, hostname, started_at, heartbeat_at, info) VALUES (?,?,?,?,?,?)",
                     ("w_cccccccccccccccc", 1, "h", db.now() - 500, db.now() - 120, None))
    finally:
        conn.close()
    try:
        body = local_client.get("/api/system").json()
    finally:
        routes_system._cache.clear()
    assert body["worker"]["online"] is False and body["tools"]["js_runtime"] is None
    assert body["tools"]["encoder"] is None


def test_openai_configured_flag_never_leaks_key(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key-123456")
    settings = make_local_settings(tmp_path)
    with TestClient(create_app(settings), base_url=LOCAL_ORIGIN) as client:
        r = client.get("/api/system")
        assert r.json()["openai"]["configured"] is True
        assert "sk-test" not in r.text
