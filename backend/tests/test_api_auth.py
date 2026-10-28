"""Hosted-mode authentication, sessions, CSRF/Origin/Host checks and security headers."""

from __future__ import annotations

from http.cookies import SimpleCookie
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from clipper import db
from clipper.api import auth
from clipper.api.app import create_app
from clipper.api.security import CONTENT_SECURITY_POLICY
from clipper.config import ConfigError
from test_api_support import (
    HOSTED_ORIGIN,
    OWNER,
    PASSWORD,
    hosted_client,
    hosted_settings,
    login,
    make_hosted_settings,
    seed_clip,
    seed_project,
)

__all__ = ["hosted_client", "hosted_settings"]   # fixtures used by this module

FAKE_P = "p_0123456789abcdef"
FAKE_C = "c_0123456789abcdef"

PROTECTED_ROUTES: list[tuple[str, str]] = [
    ("GET", "/api/system"),
    ("GET", "/api/settings"),
    ("PUT", "/api/settings"),
    ("POST", "/api/uploads"),
    ("GET", "/api/projects"),
    ("POST", "/api/projects"),
    ("GET", f"/api/projects/{FAKE_P}"),
    ("DELETE", f"/api/projects/{FAKE_P}"),
    ("POST", f"/api/projects/{FAKE_P}/cancel"),
    ("POST", f"/api/projects/{FAKE_P}/resume"),
    ("POST", f"/api/projects/{FAKE_P}/reselect"),
    ("POST", f"/api/projects/{FAKE_P}/cleanup"),
    ("GET", f"/api/clips/{FAKE_C}"),
    ("PATCH", f"/api/clips/{FAKE_C}"),
    ("POST", f"/api/clips/{FAKE_C}/captions"),
    ("POST", f"/api/clips/{FAKE_C}/captions/reset"),
    ("POST", f"/api/clips/{FAKE_C}/reset"),
    ("POST", f"/api/clips/{FAKE_C}/render"),
    ("DELETE", f"/api/clips/{FAKE_C}"),
    ("GET", f"/media/clips/{FAKE_C}/video.mp4"),
    ("GET", f"/media/clips/{FAKE_C}/download.mp4"),
    ("GET", f"/media/clips/{FAKE_C}/captions.srt"),
    ("GET", f"/media/clips/{FAKE_C}/thumb.jpg"),
    ("GET", f"/media/projects/{FAKE_P}/thumb.jpg"),
    ("GET", f"/media/projects/{FAKE_P}/frame.jpg?t=1"),
    ("GET", f"/media/projects/{FAKE_P}/export.zip"),
]


def _cookie_attrs(set_cookie: str) -> tuple[SimpleCookie, list[str]]:
    jar = SimpleCookie()
    jar.load(set_cookie)
    attrs = [p.strip().split("=")[0].lower() for p in set_cookie.split(";")[1:]]
    return jar, attrs


def _logged_in(client: TestClient) -> dict[str, str]:
    r = login(client)
    assert r.status_code == 200, r.text
    return {"Origin": HOSTED_ORIGIN, "X-CSRF-Token": r.json()["csrf_token"]}


# ----------------------------------------------------------------------------- startup
def test_startup_rejects_missing_or_weak_secrets(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        create_app(make_hosted_settings(tmp_path, owner_password_hash=None))
    with pytest.raises(ConfigError):
        create_app(make_hosted_settings(tmp_path, session_secret="short"))
    with pytest.raises(ConfigError):
        create_app(make_hosted_settings(tmp_path, public_origin="http://clipper.example.test"))


# ----------------------------------------------------------------------------- unauthenticated
def test_health_and_session_are_public(hosted_client: TestClient) -> None:
    assert hosted_client.get("/api/health").json() == {"status": "ok"}
    body = hosted_client.get("/api/auth/session").json()
    assert body["authenticated"] is False and body["auth_required"] is True
    assert body["csrf_token"] is None and body["mode"] == "hosted"


@pytest.mark.parametrize("method,path", PROTECTED_ROUTES)
def test_protected_routes_require_a_session(hosted_client: TestClient, method: str, path: str) -> None:
    r = hosted_client.request(method, path, headers={"Origin": HOSTED_ORIGIN})
    assert r.status_code == 401, (method, path, r.text)
    assert r.json() == {"detail": "Please sign in to continue.", "code": "auth_required"}


# ----------------------------------------------------------------------------- login
def test_wrong_password_then_throttled(hosted_client: TestClient) -> None:
    for _ in range(3):                                  # login_max_failures = 3
        r = login(hosted_client, password="wrong")
        assert r.status_code == 401
        assert r.json()["code"] == "invalid_credentials"
        assert "set-cookie" not in r.headers
    r = login(hosted_client)                            # even the right password is refused now
    assert r.status_code == 429
    assert int(r.headers["Retry-After"]) > 0
    assert r.json()["code"] == "rate_limited"


def test_wrong_username_is_rejected(hosted_client: TestClient) -> None:
    r = hosted_client.post("/api/auth/login", json={"username": "admin", "password": PASSWORD},
                           headers={"Origin": HOSTED_ORIGIN})
    assert r.status_code == 401


def test_global_failure_cap(hosted_client: TestClient, hosted_settings) -> None:
    conn = db.connect(hosted_settings.db_path)
    try:
        for i in range(auth.GLOBAL_FAILURE_CAP):
            auth.record_failure(conn, f"10.0.{i // 250}.{i % 250}")
    finally:
        conn.close()
    r = login(hosted_client)
    assert r.status_code == 429 and "Retry-After" in r.headers


def test_successful_login_clears_failures_and_sets_cookie(hosted_client: TestClient, hosted_settings) -> None:
    assert login(hosted_client, password="wrong").status_code == 401
    r = login(hosted_client)
    assert r.status_code == 200
    body = r.json()
    assert body["authenticated"] is True and body["username"] == OWNER and body["csrf_token"]
    jar, attrs = _cookie_attrs(r.headers["set-cookie"])
    morsel = jar[auth.SESSION_COOKIE]
    assert len(morsel.value) >= 40
    assert "httponly" in attrs and "secure" in attrs and "domain" not in attrs
    assert morsel["samesite"] == "Lax" and morsel["path"] == "/"
    assert int(morsel["max-age"]) == int(hosted_settings.session_ttl_hours * 3600)
    conn = db.connect(hosted_settings.db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM login_attempts").fetchone()[0] == 0
        stored = conn.execute("SELECT token_hash FROM sessions").fetchall()
        assert len(stored) == 1 and stored[0]["token_hash"] != morsel.value   # only the HMAC is stored
    finally:
        conn.close()


def test_login_requires_allowed_origin(hosted_client: TestClient) -> None:
    r = login(hosted_client, origin="https://evil.example")
    assert r.status_code == 403 and r.json()["code"] == "bad_origin"


def test_relogin_issues_a_fresh_token(hosted_client: TestClient) -> None:
    first = login(hosted_client).cookies.get(auth.SESSION_COOKIE)
    second = login(hosted_client).cookies.get(auth.SESSION_COOKIE)
    assert first and second and first != second
    hosted_client.cookies.clear()
    hosted_client.cookies.set(auth.SESSION_COOKIE, first)
    assert hosted_client.get("/api/projects").status_code == 401


# ----------------------------------------------------------------------------- authorised use
def test_authorised_use_and_logout(hosted_client: TestClient) -> None:
    headers = _logged_in(hosted_client)
    token = hosted_client.cookies.get(auth.SESSION_COOKIE)
    assert hosted_client.get("/api/projects").json() == []
    r = hosted_client.put("/api/settings", json={"clip_count": 3, "min_seconds": 15, "max_seconds": 45,
                                                 "captions_enabled": True}, headers=headers)
    assert r.status_code == 200 and r.json()["clip_count"] == 3
    r = hosted_client.post("/api/auth/logout", headers=headers)
    assert r.status_code == 200 and r.json()["authenticated"] is False
    assert auth.SESSION_COOKIE in r.headers["set-cookie"]
    hosted_client.cookies.set(auth.SESSION_COOKIE, token)          # replaying the old cookie fails
    assert hosted_client.get("/api/projects").status_code == 401


def test_mutation_requires_csrf_token(hosted_client: TestClient) -> None:
    _logged_in(hosted_client)
    body = {"clip_count": 3, "min_seconds": 15, "max_seconds": 45, "captions_enabled": True}
    r = hosted_client.put("/api/settings", json=body, headers={"Origin": HOSTED_ORIGIN})
    assert r.status_code == 403 and r.json()["code"] == "csrf_failed"
    r = hosted_client.put("/api/settings", json=body, headers={"Origin": HOSTED_ORIGIN, "X-CSRF-Token": "nope"})
    assert r.status_code == 403


def test_origin_and_referer_checks(hosted_client: TestClient) -> None:
    headers = _logged_in(hosted_client)
    body = {"clip_count": 3, "min_seconds": 15, "max_seconds": 45, "captions_enabled": True}
    bad = dict(headers, Origin="https://clipper.example.test.evil.example")
    r = hosted_client.put("/api/settings", json=body, headers=bad)
    assert r.status_code == 403 and r.json()["code"] == "bad_origin"
    no_origin = {"X-CSRF-Token": headers["X-CSRF-Token"]}
    assert hosted_client.put("/api/settings", json=body, headers=no_origin).status_code == 403
    referer = dict(no_origin, Referer=f"{HOSTED_ORIGIN}/settings")
    assert hosted_client.put("/api/settings", json=body, headers=referer).status_code == 200
    bad_referer = dict(no_origin, Referer="https://evil.example/settings")
    assert hosted_client.put("/api/settings", json=body, headers=bad_referer).status_code == 403


def test_disallowed_host_is_rejected(hosted_client: TestClient) -> None:
    r = hosted_client.get("/api/health", headers={"Host": "evil.example"})
    assert r.status_code == 400 and r.json()["code"] == "bad_host"
    r = hosted_client.get("/api/health", headers={"Host": "127.0.0.1:8765"})   # loopback not allowed when hosted
    assert r.status_code == 400


def test_cross_site_fetch_is_denied(hosted_client: TestClient) -> None:
    _logged_in(hosted_client)
    for path in ("/api/projects", f"/media/clips/{FAKE_C}/video.mp4"):
        r = hosted_client.get(path, headers={"Sec-Fetch-Site": "cross-site"})
        assert r.status_code == 403 and r.json()["code"] == "cross_site"
    assert hosted_client.get("/api/projects", headers={"Sec-Fetch-Site": "same-origin"}).status_code == 200


def test_security_headers(hosted_client: TestClient) -> None:
    r = hosted_client.get("/api/health")
    assert r.headers["content-security-policy"] == CONTENT_SECURITY_POLICY
    assert r.headers["x-content-type-options"] == "nosniff"
    assert r.headers["referrer-policy"] == "same-origin"
    assert r.headers["x-frame-options"] == "DENY"
    assert r.headers["cross-origin-opener-policy"] == "same-origin"
    assert r.headers["cache-control"] == "no-store"
    assert "access-control-allow-origin" not in r.headers
    page = hosted_client.get("/")
    assert page.headers["content-security-policy"] == CONTENT_SECURITY_POLICY
    assert page.headers.get("cache-control") != "no-store"
    pre = hosted_client.options("/api/projects", headers={"Origin": "https://evil.example",
                                                          "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in pre.headers


# ----------------------------------------------------------------------------- expiry
def test_absolute_session_lifetime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = make_hosted_settings(tmp_path, session_ttl_hours=1.0, session_idle_minutes=30.0)
    clock = {"t": 1_800_000_000.0}
    monkeypatch.setattr(auth, "now", lambda: clock["t"])
    with TestClient(create_app(settings), base_url=HOSTED_ORIGIN) as client:
        assert login(client).status_code == 200
        for minutes in (20, 40, 59):                     # activity keeps the idle timer fresh
            clock["t"] = 1_800_000_000.0 + minutes * 60
            assert client.get("/api/projects").status_code == 200, minutes
        clock["t"] = 1_800_000_000.0 + 61 * 60           # past the absolute lifetime
        assert client.get("/api/projects").status_code == 401
        assert client.get("/api/auth/session").json()["authenticated"] is False


def test_idle_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = make_hosted_settings(tmp_path, session_ttl_hours=12.0, session_idle_minutes=30.0)
    clock = {"t": 1_800_000_000.0}
    monkeypatch.setattr(auth, "now", lambda: clock["t"])
    with TestClient(create_app(settings), base_url=HOSTED_ORIGIN) as client:
        assert login(client).status_code == 200
        clock["t"] += 29 * 60
        assert client.get("/api/projects").status_code == 200
        clock["t"] += 29 * 60                            # slid: 58 min after login, 29 min idle
        assert client.get("/api/projects").status_code == 200
        clock["t"] += 31 * 60
        assert client.get("/api/projects").status_code == 401


def test_last_seen_writes_are_throttled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = make_hosted_settings(tmp_path)
    clock = {"t": 1_800_000_000.0}
    monkeypatch.setattr(auth, "now", lambda: clock["t"])
    with TestClient(create_app(settings), base_url=HOSTED_ORIGIN) as client:
        login(client)
        conn = db.connect(settings.db_path)
        try:
            clock["t"] += 30
            client.get("/api/projects")
            assert conn.execute("SELECT last_seen_at FROM sessions").fetchone()[0] == 1_800_000_000.0
            clock["t"] += 31
            client.get("/api/projects")
            assert conn.execute("SELECT last_seen_at FROM sessions").fetchone()[0] == clock["t"]
        finally:
            conn.close()


# ----------------------------------------------------------------------------- privacy
def test_responses_never_contain_the_data_dir(hosted_client: TestClient, hosted_settings) -> None:
    headers = _logged_in(hosted_client)
    from clipper.storage import Storage
    storage = Storage(hosted_settings.data_dir)
    conn = db.connect(hosted_settings.db_path)
    try:
        pid = seed_project(conn, storage)
        cid = seed_clip(conn, pid)
    finally:
        conn.close()
    secret_dir = str(hosted_settings.data_dir.resolve())
    responses = [
        hosted_client.get("/api/system"),
        hosted_client.get("/api/projects"),
        hosted_client.get(f"/api/projects/{pid}"),
        hosted_client.get(f"/api/clips/{cid}"),
        hosted_client.get(f"/media/clips/{cid}/video.mp4"),
        hosted_client.get(f"/media/projects/{pid}/frame.jpg?t=1"),
        hosted_client.post("/api/projects", json={"source": {"kind": "upload", "upload_id": "u_0123456789abcdef"}},
                           headers=headers),
        hosted_client.post("/api/uploads", content=b"x", headers=dict(headers, **{"X-Filename": "a.exe"})),
    ]
    system = responses[0].json()
    assert system["storage"]["location_label"] == "Server data volume"
    for r in responses:
        assert secret_dir not in r.text, r.url
        assert str(hosted_settings.data_dir) not in r.text, r.url
