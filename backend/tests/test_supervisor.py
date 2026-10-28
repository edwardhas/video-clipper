"""Tests for clipper.supervisor and scripts/start.sh / scripts/stop.sh.

Child processes are small injected Python programs (a health server and a worker that
records SIGTERM), so these tests do not depend on the real API/worker modules, except
the final real-stack test, which is skipped while those modules cannot be imported.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest
from pydantic import ValidationError

from clipper import supervisor as sup
from clipper.config import BACKEND_ROOT, REPO_ROOT, Settings

SCRIPTS = REPO_ROOT / "scripts"
PY = sys.executable

DUMMY_API = r"""
import http.server, signal, sys
port = int(sys.argv[1])

class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        ok = self.path == "/api/health"
        body = b'{"status":"ok"}' if ok else b'{}'
        self.send_response(200 if ok else 404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    def log_message(self, *args):
        pass

def stop(*_):
    raise SystemExit(0)

signal.signal(signal.SIGTERM, stop)
srv = http.server.HTTPServer(("127.0.0.1", port), H)
print("dummy api ready", flush=True)
srv.serve_forever()
"""

# Records SIGTERM like the real worker marking its job interrupted, and leaves a grandchild in
# its process group that ignores SIGTERM (only the supervisor's group SIGKILL removes it).
DUMMY_WORKER = r"""
import pathlib, signal, subprocess, sys, time
marker, grandchild_file = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
gc = subprocess.Popen([sys.executable, "-c",
    "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(300)"])
grandchild_file.write_text(str(gc.pid))
def stop(*_):
    marker.write_text("interrupted")
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
print("dummy worker ready", flush=True)
while True:
    time.sleep(0.05)
"""

STUBBORN = r"""
import signal, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
print("stubborn ready", flush=True)
while True:
    time.sleep(0.05)
"""

# Runs a Supervisor with injected children as its own process (real signal handling).
# The trailing "clipper.supervisor" argument makes `ps` identify it like the real one.
DRIVER = r"""
import json, sys
from clipper import supervisor as sup
from clipper.config import Settings
cfg = json.loads(sys.argv[1])
settings = Settings(_env_file=None, data_dir=cfg["data_dir"], port=cfg["port"], host="127.0.0.1",
                    mode="local", auth="disabled")
specs = [sup.ChildSpec(c["name"], c["argv"], stop_timeout=c["stop_timeout"]) for c in cfg["specs"]]
sys.exit(sup.Supervisor(settings, specs, out=sys.stdout, detached=True).run())
"""


# ----------------------------------------------------------------------------- helpers
def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def make_settings(tmp_path: Path, **overrides) -> Settings:
    values = dict(data_dir=tmp_path / "data", mode="local", host="127.0.0.1", port=free_port(),
                  auth="disabled", hw_encode="off")
    values.update(overrides)
    return Settings(_env_file=None, **values)


def clean_env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("CLIPPER_") and k not in ("OPENAI_API_KEY", "PYTHONPATH")}
    env.update(extra)
    return env


def wait_for(predicate, timeout: float = 15.0, interval: float = 0.05) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def gone(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    # A zombie still answers kill(0); ps shows state Z until it is reaped.
    state = subprocess.run(["ps", "-o", "stat=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()
    return state == "" or state.startswith("Z")


def reap_in_background(proc: subprocess.Popen) -> None:
    threading.Thread(target=proc.wait, daemon=True).start()


def dummy_specs(tmp_path: Path, port: int, *, worker_stop: float = 10.0) -> list[sup.ChildSpec]:
    return [
        sup.ChildSpec("api", [PY, "-c", DUMMY_API, str(port)], stop_timeout=5.0),
        sup.ChildSpec("worker", [PY, "-c", DUMMY_WORKER, str(tmp_path / "marker"), str(tmp_path / "grandchild")],
                      stop_timeout=worker_stop),
    ]


class ThreadedSupervisor:
    """Run Supervisor.run in a thread (no signal handlers) and capture its output."""

    def __init__(self, supervisor: sup.Supervisor) -> None:
        self.supervisor = supervisor
        self.result: int | None = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        self.result = self.supervisor.run(install_signals=False)

    def start(self) -> "ThreadedSupervisor":
        self.thread.start()
        return self

    def output(self) -> str:
        return self.supervisor.out.getvalue()  # type: ignore[attr-defined]

    def stop(self, timeout: float = 30.0) -> int | None:
        self.supervisor.request_stop()
        self.thread.join(timeout)
        return self.result


# ============================================================================ unit tests
def test_api_command_local_disables_proxy_headers(tmp_path: Path) -> None:
    cmd = sup.api_command(make_settings(tmp_path, port=9000), python="py")
    assert cmd[:5] == ["py", "-m", "uvicorn", "clipper.api.app:create_app", "--factory"]
    assert cmd[cmd.index("--host") + 1] == "127.0.0.1" and cmd[cmd.index("--port") + 1] == "9000"
    assert "--no-access-log" in cmd and "--no-proxy-headers" in cmd and "--proxy-headers" not in cmd


def test_api_command_hosted_trusts_only_configured_proxies(tmp_path: Path) -> None:
    hosted = make_settings(tmp_path, mode="hosted", host="0.0.0.0", trusted_proxies=" 172.30.57.10, 10.0.0.0/8 ",
                           session_secret="z" * 40, openai_api_key="sk-never-in-argv")
    cmd = sup.api_command(hosted)
    assert cmd[cmd.index("--forwarded-allow-ips") + 1] == "172.30.57.10,10.0.0.0/8"
    assert "--proxy-headers" in cmd
    assert not any("sk-never" in a or "z" * 40 in a for a in cmd)
    no_proxies = sup.api_command(make_settings(tmp_path, mode="hosted", host="0.0.0.0"))
    assert "--no-proxy-headers" in no_proxies and "--forwarded-allow-ips" not in no_proxies
    local_with_proxies = sup.api_command(make_settings(tmp_path, trusted_proxies="10.0.0.1"))
    assert "--no-proxy-headers" in local_with_proxies


def test_health_target_maps_wildcard_bind_and_public_host(tmp_path: Path) -> None:
    s = make_settings(tmp_path, mode="hosted", host="0.0.0.0", port=8765, public_origin="https://clipper.example.org/")
    assert sup.health_target(s) == ("http://127.0.0.1:8765/api/health", "clipper.example.org")
    s = make_settings(tmp_path, host="::", port=9001)
    assert sup.health_target(s) == ("http://[::1]:9001/api/health", "[::1]:9001")
    s = make_settings(tmp_path, port=9002)
    assert sup.health_target(s) == ("http://127.0.0.1:9002/api/health", "127.0.0.1:9002")


def test_port_in_use_detects_listener(tmp_path: Path) -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        assert sup.port_in_use("127.0.0.1", port)
    assert not sup.port_in_use("127.0.0.1", free_port())


def test_describe_settings_error_names_variables_not_values() -> None:
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None, port="not-a-port-123", mode="sideways")
    text = sup.describe_settings_error(excinfo.value)
    assert "CLIPPER_PORT" in text and "CLIPPER_MODE" in text
    assert "not-a-port-123" not in text and "sideways" not in text


def test_rotating_log_rotates_by_size(tmp_path: Path) -> None:
    log = sup.RotatingLog(tmp_path / "x.log", max_bytes=100, backups=2)
    for i in range(12):
        log.write(f"line {i:02d} ".encode() * 3 + b"\n")
    log.close()
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == ["x.log", "x.log.1", "x.log.2"]
    assert all(p.stat().st_size <= 100 for p in tmp_path.iterdir())
    assert "line 11" in (tmp_path / "x.log").read_text()


def test_run_file_helpers(tmp_path: Path) -> None:
    f = tmp_path / "children.pids"
    f.write_text("api 123\nworker abc\n\ngarbage\nworker 456\nx 1\n")
    assert sup.read_children(f) == {"api": 123, "worker": 456}
    (tmp_path / "pid").write_text("  4242\n")
    assert sup.read_pid(tmp_path / "pid") == 4242
    (tmp_path / "pid").write_text("not a pid")
    assert sup.read_pid(tmp_path / "pid") is None
    assert sup.read_pid(tmp_path / "missing") is None


def test_running_supervisor_pid_ignores_reused_pid(tmp_path: Path) -> None:
    s = make_settings(tmp_path)
    sup.run_dir(s).mkdir(parents=True)
    other = subprocess.Popen([PY, "-c", "import time; time.sleep(30)"])
    try:
        sup.pid_file(s).write_text(str(other.pid))
        assert sup.running_supervisor_pid(s) is None              # alive, but not a supervisor
        assert sup.running_supervisor_pid(s, verify_command=False) == other.pid
    finally:
        other.kill()
        other.wait()


# ============================================================================ in-process lifecycle
def test_supervisor_starts_reports_health_and_stops_children(tmp_path: Path) -> None:
    s = make_settings(tmp_path)
    runner = ThreadedSupervisor(sup.Supervisor(s, dummy_specs(tmp_path, s.port), out=io.StringIO())).start()
    assert wait_for(lambda: "Local Clipper is running at" in runner.output()), runner.output()
    out = runner.output()
    assert f"running at http://127.0.0.1:{s.port}" in out and "scripts/stop.sh" in out
    assert sup.read_pid(sup.pid_file(s)) == os.getpid()
    children = sup.read_children(sup.children_file(s))
    assert set(children) == {"api", "worker"}
    assert sup.check_health(*sup.health_target(s))
    assert wait_for(lambda: (tmp_path / "grandchild").exists())
    grandchild = int((tmp_path / "grandchild").read_text())
    assert wait_for(lambda: "dummy worker ready" in (s.data_dir / "logs" / "worker.log").read_text())

    assert runner.stop() == 0
    assert (tmp_path / "marker").read_text() == "interrupted"     # worker got SIGTERM first
    for pid in [*children.values(), grandchild]:
        assert wait_for(lambda: gone(pid), timeout=5), f"process {pid} survived"
    assert not sup.pid_file(s).exists() and not sup.children_file(s).exists()
    assert "dummy api ready" in (s.data_dir / "logs" / "api.log").read_text()


def test_stubborn_child_is_killed_after_its_timeout(tmp_path: Path) -> None:
    s = make_settings(tmp_path)
    specs = [sup.ChildSpec("api", [PY, "-c", DUMMY_API, str(s.port)], stop_timeout=5.0),
             sup.ChildSpec("worker", [PY, "-c", STUBBORN], stop_timeout=1.0)]
    runner = ThreadedSupervisor(sup.Supervisor(s, specs, out=io.StringIO())).start()
    assert wait_for(lambda: "running at" in runner.output())
    worker_pid = sup.read_children(sup.children_file(s))["worker"]
    started = time.monotonic()
    assert runner.stop() == 0
    assert time.monotonic() - started < 8
    assert "did not stop within 1 s; killing it" in runner.output()
    assert wait_for(lambda: gone(worker_pid), timeout=5)


def test_crashing_child_is_restarted_then_supervisor_gives_up(tmp_path: Path) -> None:
    s = make_settings(tmp_path)
    specs = [sup.ChildSpec("api", [PY, "-c", DUMMY_API, str(s.port)], stop_timeout=5.0),
             sup.ChildSpec("worker", [PY, "-c", "import sys; print('worker boom', flush=True); sys.exit(1)"])]
    runner = ThreadedSupervisor(sup.Supervisor(s, specs, out=io.StringIO(), restart_delay=0.05)).start()
    runner.thread.join(30)
    out = runner.output()
    assert runner.result == 1, out
    assert out.count("exited unexpectedly (exit code 1); restarting") == 3
    assert "crashed 4 times within 5 minutes; giving up" in out
    assert (s.data_dir / "logs" / "worker.log").read_text().count("worker boom") == 4
    assert not sup.check_health(*sup.health_target(s))          # api was stopped too


def test_restart_window_forgets_old_crashes(tmp_path: Path) -> None:
    s = make_settings(tmp_path)
    specs = [sup.ChildSpec("api", [PY, "-c", DUMMY_API, str(s.port)], stop_timeout=5.0),
             sup.ChildSpec("worker", [PY, "-c", "import time, sys; time.sleep(0.3); sys.exit(1)"])]
    supervisor = sup.Supervisor(s, specs, out=io.StringIO(), restart_delay=0.0, restart_window=0.2)
    runner = ThreadedSupervisor(supervisor).start()
    assert wait_for(lambda: runner.output().count("restarting") >= 5, timeout=20), runner.output()
    assert runner.stop() == 0
    assert "giving up" not in runner.output()


def test_config_exit_code_stops_without_restart(tmp_path: Path) -> None:
    s = make_settings(tmp_path)
    specs = [sup.ChildSpec("api", [PY, "-c", DUMMY_API, str(s.port)], stop_timeout=5.0),
             sup.ChildSpec("worker", [PY, "-c", "import time, sys; time.sleep(0.5); sys.exit(2)"])]
    runner = ThreadedSupervisor(sup.Supervisor(s, specs, out=io.StringIO())).start()
    runner.thread.join(20)
    assert runner.result == 2
    assert "configuration problem" in runner.output() and "restarting" not in runner.output()


def test_startup_failure_reports_log_tail(tmp_path: Path) -> None:
    s = make_settings(tmp_path)
    specs = [sup.ChildSpec("api", [PY, "-c", "import sys; print('Address already in use', flush=True); sys.exit(1)"]),
             sup.ChildSpec("worker", [PY, "-c", STUBBORN], stop_timeout=1.0)]
    runner = ThreadedSupervisor(sup.Supervisor(s, specs, out=io.StringIO())).start()
    runner.thread.join(20)
    out = runner.output()
    assert runner.result == 1
    assert "could not start" in out and "exited during startup (exit code 1)" in out
    assert "Address already in use" in out
    assert not sup.pid_file(s).exists()


def test_health_timeout_is_reported(tmp_path: Path) -> None:
    s = make_settings(tmp_path)
    specs = [sup.ChildSpec("api", [PY, "-c", STUBBORN], stop_timeout=1.0)]
    runner = ThreadedSupervisor(sup.Supervisor(s, specs, out=io.StringIO(), health_timeout=1.0)).start()
    runner.thread.join(20)
    assert runner.result == 1
    assert "did not answer within 1 s" in runner.output()


# ============================================================================ real signals (subprocess)
def start_driver(tmp_path: Path, port: int, data_dir: Path) -> subprocess.Popen[str]:
    cfg = {"data_dir": str(data_dir), "port": port,
           "specs": [{"name": c.name, "argv": c.argv, "stop_timeout": c.stop_timeout}
                     for c in dummy_specs(tmp_path, port)]}
    return subprocess.Popen([PY, "-c", DRIVER, json.dumps(cfg), "clipper.supervisor"],
                            env=clean_env(PYTHONPATH=str(BACKEND_ROOT)), cwd=tmp_path,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def read_until(proc: subprocess.Popen[str], needle: str, timeout: float = 20.0) -> str:
    lines: list[str] = []
    found = threading.Event()

    def pump() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line)
            if needle in line:
                found.set()
                return

    threading.Thread(target=pump, daemon=True).start()
    found.wait(timeout)
    assert found.is_set(), "".join(lines)
    return "".join(lines)


@pytest.mark.timeout(90)
def test_sigterm_stops_everything_without_orphans(tmp_path: Path) -> None:
    port, data_dir = free_port(), tmp_path / "data"
    proc = start_driver(tmp_path, port, data_dir)
    try:
        read_until(proc, "Local Clipper is running at")
        children = sup.read_children(data_dir / "run" / "children.pids")
        grandchild = int((tmp_path / "grandchild").read_text())
        assert sup.read_pid(data_dir / "run" / "supervisor.pid") == proc.pid
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=30) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
    assert (tmp_path / "marker").read_text() == "interrupted"
    for pid in [*children.values(), grandchild]:
        assert wait_for(lambda: gone(pid), timeout=5), f"orphan {pid}"
    assert not (data_dir / "run" / "supervisor.pid").exists()


@pytest.mark.timeout(90)
def test_stop_script_stops_running_supervisor(tmp_path: Path) -> None:
    port, data_dir = free_port(), tmp_path / "data"
    proc = start_driver(tmp_path, port, data_dir)
    env = clean_env(CLIPPER_DATA_DIR=str(data_dir), CLIPPER_PORT=str(port))
    try:
        read_until(proc, "Local Clipper is running at")
        children = sup.read_children(data_dir / "run" / "children.pids")
        reap_in_background(proc)  # as init would for a detached supervisor (no zombie for stop.sh)
        result = subprocess.run(["bash", str(SCRIPTS / "stop.sh")], env=env, capture_output=True, text=True,
                                timeout=60)
        assert result.returncode == 0, result.stderr
        assert f"Stopping Local Clipper (process {proc.pid})" in result.stdout
        assert "Local Clipper stopped." in result.stdout
        assert proc.wait(timeout=10) == 0
    finally:
        if proc.poll() is None:
            proc.kill()
    for pid in children.values():
        assert wait_for(lambda: gone(pid), timeout=5)
    again = subprocess.run(["bash", str(SCRIPTS / "stop.sh")], env=env, capture_output=True, text=True, timeout=60)
    assert again.returncode == 0 and "not running" in again.stdout


def test_stop_script_never_signals_unrelated_process(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    (data_dir / "run").mkdir(parents=True)
    bystander = subprocess.Popen([PY, "-c", "import time; time.sleep(60)"])
    try:
        (data_dir / "run" / "supervisor.pid").write_text(f"{bystander.pid}\n")
        (data_dir / "run" / "children.pids").write_text(f"worker {bystander.pid}\n")
        env = clean_env(CLIPPER_DATA_DIR=str(data_dir), CLIPPER_PORT=str(free_port()))
        result = subprocess.run(["bash", str(SCRIPTS / "stop.sh")], env=env, capture_output=True, text=True,
                                timeout=60)
        assert result.returncode == 0, result.stderr
        assert "not running" in result.stdout
        assert bystander.poll() is None                       # still alive
        assert not (data_dir / "run" / "supervisor.pid").exists()
    finally:
        bystander.kill()
        bystander.wait()


def test_stop_script_reaps_orphaned_children(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    (data_dir / "run").mkdir(parents=True)
    orphan = subprocess.Popen([PY, "-c", "import time; time.sleep(60)", "clipper.worker"], start_new_session=True)
    reap_in_background(orphan)
    try:
        (data_dir / "run" / "children.pids").write_text(f"worker {orphan.pid}\n")
        env = clean_env(CLIPPER_DATA_DIR=str(data_dir), CLIPPER_PORT=str(free_port()))
        result = subprocess.run(["bash", str(SCRIPTS / "stop.sh")], env=env, capture_output=True, text=True,
                                timeout=60)
        assert result.returncode == 0, result.stderr
        assert f"Stopping leftover worker process ({orphan.pid})" in result.stdout
        assert orphan.wait(timeout=10) is not None
        assert not (data_dir / "run" / "children.pids").exists()
    finally:
        if orphan.poll() is None:
            orphan.kill()


# ============================================================================ main() entry point
def run_main(env: dict[str, str], *args: str, timeout: float = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run([PY, "-m", "clipper.supervisor", "--no-browser", *args], env=env, cwd=BACKEND_ROOT,
                          capture_output=True, text=True, timeout=timeout)


def test_main_config_error_exits_2_without_starting_children(tmp_path: Path) -> None:
    env = clean_env(PYTHONPATH=str(BACKEND_ROOT), CLIPPER_MODE="hosted", CLIPPER_DATA_DIR=str(tmp_path / "data"),
                    CLIPPER_PORT=str(free_port()), CLIPPER_SESSION_SECRET="short-secret-do-not-print",
                    OPENAI_API_KEY="sk-test-never-print")
    result = run_main(env)
    assert result.returncode == 2
    assert "cannot start because of its configuration" in result.stderr
    assert "CLIPPER_OWNER_PASSWORD_HASH" in result.stderr and "CLIPPER_PUBLIC_ORIGIN" in result.stderr
    assert "short-secret-do-not-print" not in result.stderr + result.stdout
    assert "sk-test-never-print" not in result.stderr + result.stdout
    assert not (tmp_path / "data" / "run" / "supervisor.pid").exists()

    env["CLIPPER_PORT"] = "eighty"
    result = run_main(env)
    assert result.returncode == 2 and "CLIPPER_PORT" in result.stderr and "eighty" not in result.stderr


def test_main_refuses_busy_port(tmp_path: Path) -> None:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        env = clean_env(PYTHONPATH=str(BACKEND_ROOT), CLIPPER_DATA_DIR=str(tmp_path / "data"),
                        CLIPPER_PORT=str(port), CLIPPER_AUTH="disabled")
        result = run_main(env)
    assert result.returncode == 1
    assert f"Port {port} on 127.0.0.1 is already used by another program" in result.stderr


# ============================================================================ real stack via scripts
def _real_modules_available() -> bool:
    try:
        return all(importlib.util.find_spec(m) is not None for m in ("clipper.api.app", "clipper.worker"))
    except ImportError:
        return False


@pytest.mark.timeout(180)
@pytest.mark.skipif(not _real_modules_available(), reason="API/worker modules not available yet")
def test_start_detach_and_stop_real_stack(tmp_path: Path) -> None:
    port, data_dir = free_port(), tmp_path / "data"
    env = clean_env(CLIPPER_DATA_DIR=str(data_dir), CLIPPER_PORT=str(port), CLIPPER_HOST="127.0.0.1",
                    CLIPPER_MODE="local", CLIPPER_AUTH="disabled", CLIPPER_HW_ENCODE="off",
                    CLIPPER_FRONTEND_DIST=str(tmp_path / "no-dist"))
    try:
        started = subprocess.run(["bash", str(SCRIPTS / "start.sh"), "--detach", "--no-browser"], env=env,
                                 capture_output=True, text=True, timeout=90)
        log = (data_dir / "logs" / "supervisor.log")
        assert started.returncode == 0, started.stdout + started.stderr + (log.read_text() if log.exists() else "")
        assert f"Local Clipper is running at http://127.0.0.1:{port}" in started.stdout
        s = make_settings(tmp_path, port=port)
        assert sup.check_health(*sup.health_target(s))
        sup_pid = sup.read_pid(sup.pid_file(s))
        children = sup.read_children(sup.children_file(s))
        assert sup_pid and set(children) == {"api", "worker"}
        # Detached: the supervisor leads its own session, away from the caller's terminal.
        assert os.getsid(sup_pid) == sup_pid
        again = subprocess.run(["bash", str(SCRIPTS / "start.sh"), "--detach", "--no-browser"], env=env,
                               capture_output=True, text=True, timeout=60)
        assert again.returncode == 0 and "already running" in again.stdout
    finally:
        stopped = subprocess.run(["bash", str(SCRIPTS / "stop.sh")], env=env, capture_output=True, text=True,
                                 timeout=90)
    assert stopped.returncode == 0, stopped.stderr
    assert "Local Clipper stopped." in stopped.stdout
    for pid in [sup_pid, *children.values()]:
        assert wait_for(lambda: gone(pid), timeout=10), f"process {pid} survived"
    assert not sup.pid_file(s).exists()
    assert "worker" in (data_dir / "logs" / "worker.log").read_text()
