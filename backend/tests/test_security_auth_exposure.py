"""auth=auto/disabled in local mode: sign-in may only be off while every address the app is told
to accept (bind host, public origin, allowed hosts, allowed origins) is loopback."""

from __future__ import annotations

from pathlib import Path

import pytest
from argon2 import PasswordHasher
from fastapi.testclient import TestClient

from clipper.api.app import create_app
from clipper.config import ConfigError
from test_api_support import SESSION_SECRET, make_local_settings

EXTERNAL_ORIGIN = "https://clips.example.com"
_HASH = PasswordHasher(time_cost=1, memory_cost=8192, parallelism=1).hash("owner password 1")


@pytest.mark.parametrize("exposure", [
    {"public_origin": EXTERNAL_ORIGIN},
    {"public_origin": "http://192.168.1.20:8765"},
    {"allowed_hosts": "clips.example.com"},
    {"allowed_hosts": "localhost, 192.168.1.20"},
    {"allowed_hosts": "[2001:db8::1]"},
    {"allowed_origins": "http://localhost:5173,https://clips.example.com"},
    {"host": "0.0.0.0"},
])
def test_auto_turns_sign_in_on_for_any_non_loopback_name(tmp_path: Path, exposure: dict) -> None:
    s = make_local_settings(tmp_path, auth="auto", **exposure)
    assert s.auth_enabled is True
    assert s.exposure_reasons
    with pytest.raises(ConfigError) as ei:          # sign-in on, but no secrets configured
        s.validate_for_startup()
    assert "CLIPPER_OWNER_PASSWORD_HASH" in str(ei.value)


@pytest.mark.parametrize("exposure,named", [
    ({"public_origin": EXTERNAL_ORIGIN}, "CLIPPER_PUBLIC_ORIGIN"),
    ({"allowed_hosts": "192.168.1.20"}, "CLIPPER_ALLOWED_HOSTS"),
    ({"allowed_origins": "http://192.168.1.20:8765"}, "CLIPPER_ALLOWED_ORIGINS"),
    ({"host": "0.0.0.0"}, "CLIPPER_HOST"),
])
def test_disabled_is_refused_when_reachable_from_elsewhere(tmp_path: Path, exposure: dict, named: str) -> None:
    s = make_local_settings(tmp_path, auth="disabled", **exposure)
    assert s.auth_enabled is False
    with pytest.raises(ConfigError) as ei:
        s.validate_for_startup()
    assert "CLIPPER_AUTH=disabled" in str(ei.value) and named in str(ei.value)


@pytest.mark.parametrize("loopback", [
    {},
    {"public_origin": "http://127.0.0.1:8765"},
    {"public_origin": "http://localhost:8765/"},
    {"public_origin": "http://[::1]:8765"},
    {"allowed_origins": "http://localhost:5173, http://127.0.0.1:5173"},
    {"allowed_hosts": "localhost,127.0.0.1,[::1],::1,LOCALHOST:5173"},
    {"host": "::1"},
])
def test_loopback_names_keep_local_convenience(tmp_path: Path, loopback: dict) -> None:
    auto = make_local_settings(tmp_path, auth="auto", **loopback)
    assert auto.auth_enabled is False and auto.exposure_reasons == []
    auto.validate_for_startup()
    make_local_settings(tmp_path, auth="disabled", **loopback).validate_for_startup()


def test_proxied_public_origin_requires_sign_in_end_to_end(tmp_path: Path) -> None:
    """The reviewer's scenario: backend on 127.0.0.1 behind a same-machine HTTPS proxy."""
    with pytest.raises(ConfigError):
        create_app(make_local_settings(tmp_path, auth="auto", public_origin=EXTERNAL_ORIGIN))
    s = make_local_settings(tmp_path, auth="auto", public_origin=EXTERNAL_ORIGIN,
                           owner_password_hash=_HASH, session_secret=SESSION_SECRET)
    with TestClient(create_app(s), base_url=EXTERNAL_ORIGIN) as client:
        body = client.get("/api/auth/session").json()
        assert body["authenticated"] is False and body["auth_required"] is True and body["csrf_token"] is None
        assert client.get("/api/projects").status_code == 401
        r = client.post("/api/projects", json={}, headers={"Origin": EXTERNAL_ORIGIN})
        assert r.status_code == 401
