"""scripts/start.sh --detach after a crashed supervisor (re-review finding 12).

A supervisor that is SIGKILLed / OOM-killed leaves its api and worker running (each leads its own
process group); the api keeps answering /api/health and supervisor.pid still names the dead
supervisor. start.sh --detach must stop those leftovers (they are listed in this run dir's
children.pids) and start cleanly, while a copy that answers but is not recorded here (e.g. started
with another CLIPPER_DATA_DIR) must still be refused, and an unrelated process is never signalled.
"""

from __future__ import annotations

import http.server
import importlib.util
import os
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path

import pytest

from clipper import supervisor as sup
from clipper.config import REPO_ROOT, Settings

SCRIPTS = REPO_ROOT / "scripts"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _env(data_dir: Path, port: int) -> dict[str, str]:
    """Environment that can never reach the owner's real app (own port and data folder, no key)."""
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("CLIPPER_") and k not in ("OPENAI_API_KEY", "PYTHONPATH")}
    env.update(CLIPPER_MODE="local", CLIPPER_HOST="127.0.0.1", CLIPPER_PORT=str(port), CLIPPER_AUTH="disabled",
               CLIPPER_HW_ENCODE="off", CLIPPER_DATA_DIR=str(data_dir), OPENAI_API_KEY="",
               CLIPPER_FRONTEND_DIST=str(data_dir.parent / "no-dist"))
    return env


def _settings(data_dir: Path, port: int) -> Settings:
    return Settings(_env_file=None, data_dir=data_dir, mode="local", host="127.0.0.1", port=port,
                    auth="disabled", hw_encode="off")


def _script(name: str, *args: str, env: dict[str, str], timeout: float = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", str(SCRIPTS / name), *args], env=env, capture_output=True, text=True,
                          timeout=timeout)


def _gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return state == "" or state.startswith("Z")


def _wait_for(predicate, timeout: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.1)
    return predicate()


def _dead_pid() -> int:
    proc = subprocess.Popen(["true"])
    proc.wait()
    return proc.pid


def _kill_group(pid: int) -> None:
    for kill in (lambda: os.killpg(pid, signal.SIGKILL), lambda: os.kill(pid, signal.SIGKILL)):
        try:
            kill()
            return
        except (ProcessLookupError, PermissionError):
            continue


def _real_modules_available() -> bool:
    try:
        return all(importlib.util.find_spec(m) is not None for m in ("clipper.api.app", "clipper.worker"))
    except ImportError:
        return False


real_stack = pytest.mark.skipif(not _real_modules_available(), reason="API/worker modules not available")


@pytest.mark.timeout(240)
@real_stack
def test_start_detach_recovers_after_supervisor_was_killed(tmp_path: Path) -> None:
    port, data_dir = _free_port(), tmp_path / "data dir"
    env, s = _env(data_dir, port), _settings(data_dir, port)
    old_children: dict[str, int] = {}
    try:
        first = _script("start.sh", "--detach", "--no-browser", env=env, timeout=90)
        assert first.returncode == 0, first.stdout + first.stderr
        old_sup = sup.read_pid(sup.pid_file(s))
        old_children = sup.read_children(sup.children_file(s))
        assert old_sup and set(old_children) == {"api", "worker"}

        os.kill(old_sup, signal.SIGKILL)  # a crash / OOM kill: the children survive
        assert _wait_for(lambda: _gone(old_sup))
        assert sup.app_healthy(s), "the orphaned api should still answer"
        assert sup.read_pid(sup.pid_file(s)) == old_sup  # stale PID file

        again = _script("start.sh", "--detach", "--no-browser", env=env, timeout=120)
        out = again.stdout + again.stderr
        assert again.returncode == 0, out  # was 1: "Another copy ... CLIPPER_DATA_DIR ..."
        assert "CLIPPER_DATA_DIR" not in out and "already answering" not in out
        assert f"Local Clipper is running at http://127.0.0.1:{port}" in again.stdout
        new_sup = sup.read_pid(sup.pid_file(s))
        assert new_sup and new_sup != old_sup and not _gone(new_sup)
        for name, pid in old_children.items():
            assert _wait_for(lambda pid=pid: _gone(pid), timeout=10), f"leftover {name} ({pid}) survived"
        new_children = sup.read_children(sup.children_file(s))
        assert set(new_children) == {"api", "worker"}
        assert not set(new_children.values()) & set(old_children.values())
        assert sup.app_healthy(s)
    finally:
        stop = _script("stop.sh", env=env, timeout=90)
        for pid in old_children.values():
            if not _gone(pid):
                _kill_group(pid)
    assert stop.returncode == 0, stop.stdout + stop.stderr
    assert not sup.app_healthy(s)


@pytest.mark.timeout(240)
@real_stack
def test_start_detach_ignores_stale_run_files_when_nothing_answers(tmp_path: Path) -> None:
    """A PID file of a dead supervisor (e.g. after a reboot) and a children list of dead processes
    never block a start."""
    port, data_dir = _free_port(), tmp_path / "data"
    env, s = _env(data_dir, port), _settings(data_dir, port)
    run = sup.pid_file(s).parent
    run.mkdir(parents=True)
    stale = _dead_pid()
    sup.pid_file(s).write_text(f"{stale}\n", encoding="utf-8")
    sup.children_file(s).write_text(f"api {_dead_pid()}\nworker {_dead_pid()}\n", encoding="utf-8")
    try:
        started = _script("start.sh", "--detach", "--no-browser", env=env, timeout=120)
        assert started.returncode == 0, started.stdout + started.stderr
        new_sup = sup.read_pid(sup.pid_file(s))
        assert new_sup and new_sup != stale and not _gone(new_sup)
    finally:
        stop = _script("stop.sh", env=env, timeout=90)
    assert stop.returncode == 0, stop.stdout + stop.stderr


class _Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        body = b'{"status":"ok"}'
        self.send_response(200 if self.path == "/api/health" else 404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args) -> None:
        pass


def test_start_detach_still_refuses_a_copy_not_recorded_here(tmp_path: Path) -> None:
    """Something answers on the port, the PID file is stale and children.pids lists only a dead
    process and an unrelated live one: start.sh refuses (as before) and signals nobody."""
    port, data_dir = _free_port(), tmp_path / "data"
    env, s = _env(data_dir, port), _settings(data_dir, port)
    srv = http.server.HTTPServer(("127.0.0.1", port), _Health)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    bystander = subprocess.Popen(["sleep", "120"])
    try:
        run = sup.pid_file(s).parent
        run.mkdir(parents=True)
        stale = _dead_pid()
        sup.pid_file(s).write_text(f"{stale}\n", encoding="utf-8")
        sup.children_file(s).write_text(f"api {_dead_pid()}\nworker {bystander.pid}\n", encoding="utf-8")

        start = _script("start.sh", "--detach", "--no-browser", env=env, timeout=60)
        assert start.returncode != 0, start.stdout + start.stderr
        assert "already answering" in start.stderr and "is running at" not in start.stdout
        assert bystander.poll() is None, "an unrelated process listed in children.pids was signalled"
        assert sup.read_pid(sup.pid_file(s)) == stale  # no new supervisor was launched
    finally:
        bystander.kill()
        bystander.wait()
        srv.shutdown()
        srv.server_close()
