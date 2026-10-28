"""Local mode behind a same-host reverse proxy or tunnel (sign-in on because of a non-loopback name,
backend bound to loopback): login throttling must tell internet clients apart when
CLIPPER_TRUSTED_PROXIES names the proxy, so one client cannot lock the owner out, and
check-config/doctor must warn when it is left empty."""

from __future__ import annotations

from pathlib import Path

import pytest
from argon2 import PasswordHasher
from fastapi.testclient import TestClient
from starlette.requests import Request

from clipper import cli, db
from clipper.api.app import create_app
from clipper.api.deps import client_ip
from clipper.cli import WARN, check_config
from test_api_support import OWNER, PASSWORD, SESSION_SECRET, make_hosted_settings, make_local_settings

TUNNEL_ORIGIN = "https://clips.example.test"
ATTACKER = "203.0.113.9"
OWNER_IP = "198.51.100.7"
_HASH = PasswordHasher(time_cost=1, memory_cost=8192, parallelism=1).hash(PASSWORD)


def proxied_local(tmp_path: Path, **overrides):
    values = dict(auth="auto", public_origin=TUNNEL_ORIGIN, owner_username=OWNER, owner_password_hash=_HASH,
                  session_secret=SESSION_SECRET, login_max_failures=5, login_window_seconds=900)
    values.update(overrides)
    return make_local_settings(tmp_path, **values)


def _login(client: TestClient, password: str, xff: str | None):
    headers = {"Origin": TUNNEL_ORIGIN}
    if xff is not None:
        headers["X-Forwarded-For"] = xff
    return client.post("/api/auth/login", json={"username": OWNER, "password": password}, headers=headers)


def _attempt_ips(settings) -> set[str]:
    conn = db.connect(settings.db_path)
    try:
        return {r[0] for r in conn.execute("SELECT ip FROM login_attempts")}
    finally:
        conn.close()


def _request(peer: str, xff: str | None = None) -> Request:
    headers = [(b"x-forwarded-for", xff.encode())] if xff is not None else []
    return Request({"type": "http", "method": "POST", "path": "/api/auth/login", "headers": headers,
                    "client": (peer, 40000), "query_string": b""})


# ----------------------------------------------------------------------------- throttling
def test_local_proxied_attacker_cannot_lock_out_the_owner(tmp_path: Path) -> None:
    """Same-host proxy connects from 127.0.0.1; with CLIPPER_TRUSTED_PROXIES=127.0.0.1 each forwarded
    client gets its own bucket (before the fix every client shared 127.0.0.1's and the owner got 429)."""
    settings = proxied_local(tmp_path, trusted_proxies="127.0.0.1")
    assert settings.mode == "local" and settings.auth_enabled
    with TestClient(create_app(settings), base_url=TUNNEL_ORIGIN, client=("127.0.0.1", 40000)) as client:
        codes = [_login(client, "wrong", ATTACKER).status_code for _ in range(5)]
        assert codes == [401] * 5
        assert _login(client, "wrong", ATTACKER).status_code == 429       # the attacker is throttled
        assert _login(client, PASSWORD, OWNER_IP).status_code == 200      # the owner is not
        # The owner's success clears only the owner's bucket, not the attacker's.
        assert _login(client, PASSWORD, ATTACKER).status_code == 429
    assert _attempt_ips(settings) == {ATTACKER}


def test_local_proxied_spoofed_left_hops_do_not_escape_the_throttle(tmp_path: Path) -> None:
    """A client-supplied X-Forwarded-For prefix is ignored: the proxy-appended (rightmost) hop counts."""
    settings = proxied_local(tmp_path, trusted_proxies="127.0.0.1", login_max_failures=3)
    with TestClient(create_app(settings), base_url=TUNNEL_ORIGIN, client=("127.0.0.1", 40000)) as client:
        codes = [_login(client, "wrong", f"10.9.9.{i}, {ATTACKER}").status_code for i in range(4)]
    assert codes == [401, 401, 401, 429]
    assert _attempt_ips(settings) == {ATTACKER}


def test_local_without_trusted_proxies_ignores_forwarded_for(tmp_path: Path) -> None:
    """Default (empty) CLIPPER_TRUSTED_PROXIES: X-Forwarded-For is never used to pick the bucket."""
    settings = proxied_local(tmp_path, login_max_failures=3)
    with TestClient(create_app(settings), base_url=TUNNEL_ORIGIN, client=("127.0.0.1", 40000)) as client:
        codes = [_login(client, "wrong", f"203.0.113.{i}").status_code for i in range(4)]
    assert codes == [401, 401, 401, 429]
    assert _attempt_ips(settings) == {"127.0.0.1"}


def test_client_ip_honours_trusted_proxies_in_local_mode(tmp_path: Path) -> None:
    local = make_local_settings(tmp_path, trusted_proxies="127.0.0.1, ::1")
    assert client_ip(_request("127.0.0.1", f"{OWNER_IP}"), local) == OWNER_IP
    assert client_ip(_request("::1", f"1.1.1.1, {OWNER_IP}"), local) == OWNER_IP
    assert client_ip(_request("127.0.0.1"), local) == "127.0.0.1"                 # direct visit, no header
    assert client_ip(_request("127.0.0.1", "127.0.0.1"), local) == "127.0.0.1"    # only trusted hops
    assert client_ip(_request("192.0.2.5", OWNER_IP), local) == "192.0.2.5"       # untrusted peer: header ignored
    untrusted = make_local_settings(tmp_path)
    assert client_ip(_request("127.0.0.1", OWNER_IP), untrusted) == "127.0.0.1"
    hosted = make_hosted_settings(tmp_path, trusted_proxies="10.0.0.2")              # hosted unchanged
    assert client_ip(_request("10.0.0.2", f"5.5.5.5, {OWNER_IP}"), hosted) == OWNER_IP
    assert client_ip(_request("10.0.0.3", OWNER_IP), hosted) == "10.0.0.3"


# ----------------------------------------------------------------------------- check-config / doctor
def _proxy_warnings(settings) -> list:
    return [c for c in check_config(settings) if c.status == WARN and c.name == "Proxy headers"]


def test_check_config_warns_for_proxied_local_setup_without_trusted_proxies(tmp_path: Path) -> None:
    warnings = _proxy_warnings(proxied_local(tmp_path))
    assert len(warnings) == 1
    detail = warnings[0].detail
    assert "127.0.0.1" in detail and "CLIPPER_TRUSTED_PROXIES" in detail and "CLIPPER_PUBLIC_ORIGIN" in detail


@pytest.mark.parametrize("overrides", [
    {"trusted_proxies": "127.0.0.1"},                                   # configured: nothing to warn
    {"public_origin": "", "auth": "disabled"},                          # loopback only, sign-in off
    {"public_origin": "", "auth": "required"},                          # sign-in on, but loopback only
    {"public_origin": "", "host": "0.0.0.0", "auth": "required"},       # direct exposure: real peer IPs
])
def test_check_config_no_proxy_warning_otherwise(tmp_path: Path, overrides: dict) -> None:
    assert _proxy_warnings(proxied_local(tmp_path, **overrides)) == []


def test_check_config_warns_for_required_auth_with_tunnel_name(tmp_path: Path) -> None:
    assert len(_proxy_warnings(proxied_local(tmp_path, auth="required", public_origin="",
                                             allowed_hosts="clips.example.test"))) == 1


def test_hosted_proxy_warning_kept(tmp_path: Path) -> None:
    assert len(_proxy_warnings(make_hosted_settings(tmp_path))) == 1
    assert _proxy_warnings(make_hosted_settings(tmp_path, trusted_proxies="10.0.0.2")) == []


def test_check_config_command_prints_the_warning(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    assert cli.main(["check-config"], settings=proxied_local(tmp_path)) == 0      # a warning, not a failure
    out = capsys.readouterr().out
    assert "Configuration OK." in out
    assert "CLIPPER_TRUSTED_PROXIES" in out and "shares one login throttle" in out
    assert cli.main(["check-config"], settings=proxied_local(tmp_path, trusted_proxies="127.0.0.1")) == 0
    assert "shares one login throttle" not in capsys.readouterr().out


# ----------------------------------------------------------------------------- launch + docs
def test_local_mode_keeps_uvicorn_proxy_headers_off(tmp_path: Path) -> None:
    """client_ip parses X-Forwarded-For from the raw peer; uvicorn must not rewrite request.client."""
    from clipper import supervisor

    cmd = supervisor.api_command(proxied_local(tmp_path, trusted_proxies="127.0.0.1"), python="py")
    assert "--no-proxy-headers" in cmd and "--proxy-headers" not in cmd and "--forwarded-allow-ips" not in cmd


def test_env_example_documents_trusted_proxies_for_local_proxies() -> None:
    from clipper.config import REPO_ROOT

    text = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    block = text[:text.index("# CLIPPER_TRUSTED_PROXIES=")]
    section = block[block.rindex("# ------"):]
    assert "Hosted mode only" not in section and "127.0.0.1" in section and "X-Forwarded-For" in section
    guide = (REPO_ROOT / "DEPLOYMENT.md").read_text(encoding="utf-8")
    assert "### Local mode behind a proxy or tunnel" in guide
