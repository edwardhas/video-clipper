"""Consistency tests for the hosted deployment package (deploy/*, DEPLOYMENT.md).

Static checks always run. Checks that use the real tools (docker compose, caddy, nginx) run only
when Docker is reachable and the image is already present locally: they never pull images and
never touch a running stack (`compose config` on a temporary copy, `docker run --rm --network none`).
"""

from __future__ import annotations

import functools
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from clipper.config import REPO_ROOT, Settings

DEPLOY = REPO_ROOT / "deploy"
CADDYFILE = DEPLOY / "Caddyfile"
COMPOSE = DEPLOY / "compose.yaml"
HOSTED_EXAMPLE = DEPLOY / "env.hosted.example"
DEPLOYMENT_MD = REPO_ROOT / "DEPLOYMENT.md"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _example_value(name: str) -> str:
    match = re.search(rf"^{name}=(.*)$", _read(HOSTED_EXAMPLE), re.M)
    assert match, name
    return match.group(1).strip()


@functools.cache
def _docker_ok() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "info", "--format", "{{.ServerVersion}}"], capture_output=True,
                              timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _require_image(image: str) -> None:
    if not _docker_ok():
        pytest.skip("Docker is not reachable")
    if subprocess.run(["docker", "image", "inspect", image], capture_output=True, timeout=30).returncode != 0:
        pytest.skip(f"image {image} is not present locally (tests never pull)")


# ============================================================================ Caddy upstream port
def test_caddy_upstream_follows_clipper_port() -> None:
    """Caddy used to dial a fixed api:8765 while the api listens on CLIPPER_PORT (502 when changed)."""
    caddyfile = _read(CADDYFILE)
    upstreams = re.findall(r"^\s*reverse_proxy\s+(\S+)", caddyfile, re.M)
    assert upstreams == ["api:{$CLIPPER_PORT:8765}"]
    compose = _read(COMPOSE)
    # The proxy gets the value Caddy substitutes, and the api gets the very same substituted value.
    assert compose.count("CLIPPER_PORT: ${CLIPPER_PORT:-8765}") == 2
    assert '- "${CLIPPER_PORT:-8765}"' in compose
    assert _example_value("CLIPPER_PORT") == "8765" == str(Settings.model_fields["port"].default)


def _compose_config(tmp_path: Path, env_lines: list[str]) -> dict:
    work = tmp_path / "deploy"
    work.mkdir(exist_ok=True)
    shutil.copy2(COMPOSE, work / "compose.yaml")
    shutil.copy2(CADDYFILE, work / "Caddyfile")
    (work / ".env.hosted").write_text("\n".join(env_lines) + "\n", encoding="utf-8")
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("CLIPPER_", "COMPOSE_")) and k != "OPENAI_API_KEY"}
    result = subprocess.run(["docker", "compose", "--env-file", str(work / ".env.hosted"), "-f",
                             str(work / "compose.yaml"), "config", "--format", "json"],
                            capture_output=True, text=True, env=env, timeout=60)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_compose_gives_api_and_proxy_the_same_port(tmp_path: Path) -> None:
    if not _docker_ok():
        pytest.skip("Docker is not reachable")
    custom = _compose_config(tmp_path, ["CLIPPER_HOSTNAME=clipper.example.org", "CLIPPER_PORT=9000"])["services"]
    assert custom["api"]["environment"]["CLIPPER_PORT"] == "9000"
    assert custom["proxy"]["environment"]["CLIPPER_PORT"] == "9000"
    assert [str(p) for p in custom["api"]["expose"]] == ["9000"]
    default = _compose_config(tmp_path, ["CLIPPER_HOSTNAME=clipper.example.org"])["services"]
    assert default["api"]["environment"]["CLIPPER_PORT"] == "8765"
    assert default["proxy"]["environment"]["CLIPPER_PORT"] == "8765"


@pytest.mark.parametrize("port", ["9000", None])
def test_caddyfile_dials_the_configured_api_port(port: str | None) -> None:
    _require_image("caddy:2")
    env_args = ["-e", "CLIPPER_HOSTNAME=clipper.example.org"] + (["-e", f"CLIPPER_PORT={port}"] if port else [])
    result = subprocess.run(["docker", "run", "--rm", "--network", "none", *env_args,
                             "-v", f"{CADDYFILE}:/etc/caddy/Caddyfile:ro", "caddy:2",
                             "caddy", "adapt", "--config", "/etc/caddy/Caddyfile"],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert sorted(set(re.findall(r'"dial":"([^"]+)"', result.stdout))) == [f"api:{port or 8765}"]


# ============================================================================ DEPLOYMENT.md
def _nginx_block() -> str:
    blocks = re.findall(r"```nginx\n(.*?)```", _read(DEPLOYMENT_MD), re.S)
    assert len(blocks) == 1
    return blocks[0]


def test_documented_nginx_config_avoids_http2_on_directive() -> None:
    """`http2 on;` needs nginx >= 1.25.1; Ubuntu 24.04 ships 1.24 and Debian 12 ships 1.22."""
    code = [line.split("#", 1)[0].strip() for line in _nginx_block().splitlines()]
    assert not any(re.fullmatch(r"http2\s+on;", line) for line in code)
    assert "listen 443 ssl http2;" in code and "listen [::]:443 ssl http2;" in code


@pytest.mark.parametrize("image", ["nginx:1.22", "nginx:1.24", "nginx:1.26"])
def test_documented_nginx_config_passes_nginx_t(image: str, tmp_path: Path) -> None:
    _require_image(image)
    if shutil.which("openssl") is None:
        pytest.skip("openssl not installed")
    certs = tmp_path / "certs"
    certs.mkdir()
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
                    "-subj", "/CN=clipper.example.com", "-keyout", str(certs / "privkey.pem"),
                    "-out", str(certs / "fullchain.pem")], check=True, capture_output=True, timeout=60)
    conf = tmp_path / "clipper.conf"
    conf.write_text(_nginx_block(), encoding="utf-8")
    result = subprocess.run(["docker", "run", "--rm", "--network", "none",
                             "-v", f"{conf}:/etc/nginx/conf.d/default.conf:ro",
                             "-v", f"{certs}:/etc/letsencrypt/live/clipper.example.com:ro",
                             image, "nginx", "-t"], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr


def test_deployment_guide_lists_every_hosted_setting_and_venv_package() -> None:
    """Every variable in deploy/env.hosted.example (incl. CLIPPER_PROXY_SUBNET/IP) is documented,
    and the native install names python3.12-venv (setup.sh cannot create the venv without it)."""
    names = set(re.findall(r"^#?\s*((?:CLIPPER_[A-Z0-9_]+)|OPENAI_API_KEY)=", _read(HOSTED_EXAMPLE), re.M))
    guide = _read(DEPLOYMENT_MD)
    assert {"CLIPPER_PROXY_SUBNET", "CLIPPER_PROXY_IP", "CLIPPER_PORT"} <= names
    assert {n for n in names if f"| `{n}` |" not in guide} == set()
    native = guide.split("## 6.", 1)[1].split("## 7.", 1)[0]
    assert "python3.12-venv" in native
