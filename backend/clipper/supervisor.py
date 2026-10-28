"""One-command launcher: runs the API and the worker as supervised child processes.

Usage::

    python -m clipper.supervisor [--no-browser] [--detached]

* Loads and validates the configuration (friendly message + exit code 2 on errors),
  prepares the data directory and database.
* Starts the API (uvicorn) and the worker, each in its own process group, with
  output written to ``data/logs/{api,worker}.log`` (size-rotated) and echoed to the
  terminal when there is one.
* Waits for ``GET /api/health``, prints the address and how to stop.
* Forwards SIGINT/SIGTERM/SIGHUP: children get SIGTERM, up to 20 s to finish (the
  worker marks its current job ``interrupted``), then SIGKILL for their whole group.
* Restarts a crashed child at most 3 times within 5 minutes, then gives up.

Long-lived children are started with ``subprocess.Popen`` (argument lists, never a
shell) because :func:`clipper.procs.run` blocks until the program exits; short helper
commands (``ps``) still go through :func:`clipper.procs.run`.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Callable, Iterable, Sequence
from urllib.parse import urlsplit

from pydantic import ValidationError

from . import db
from .config import BACKEND_ROOT, ConfigError, Settings
from .errors import ClipperError
from .procs import run
from .storage import Storage

HEALTH_TIMEOUT_SECONDS = 30.0
WORKER_STOP_SECONDS = 20.0
API_STOP_SECONDS = 15.0
RESTART_LIMIT = 3
RESTART_WINDOW_SECONDS = 300.0
RESTART_DELAY_SECONDS = 1.0
LOG_MAX_BYTES = 10 * 1024 * 1024
LOG_BACKUPS = 3
EXIT_CONFIG = 2
_UVICORN_LOG_LEVELS = {"critical", "error", "warning", "info", "debug", "trace"}


# ----------------------------------------------------------------------------- settings
def describe_settings_error(exc: ValidationError) -> str:
    """Readable summary of invalid environment values (never echoes the values)."""
    lines: list[str] = []
    for err in exc.errors(include_input=False, include_url=False):
        loc = [str(p) for p in err.get("loc", ())]
        field = loc[0] if loc else "?"
        var = field if field.isupper() else f"CLIPPER_{field.upper()}"
        lines.append(f"{var}: {err.get('msg', 'invalid value')}")
    return "Invalid configuration value(s):\n  - " + "\n  - ".join(lines or ["unknown problem"])


def load_settings() -> Settings:
    """Fresh :class:`Settings` from the environment / ``.env``; raises ConfigError on bad values."""
    try:
        return Settings()
    except ValidationError as exc:
        raise ConfigError(describe_settings_error(exc)) from None


def config_help(message: str) -> str:
    return (f"Local Clipper cannot start because of its configuration.\n{message}\n\n"
            "Edit the .env file (see .env.example for every option), then run scripts/doctor.sh.")


# ----------------------------------------------------------------------------- run files
def run_dir(settings: Settings) -> Path:
    return Path(settings.data_dir) / "run"


def log_dir(settings: Settings) -> Path:
    return Path(settings.data_dir) / "logs"


def pid_file(settings: Settings) -> Path:
    return run_dir(settings) / "supervisor.pid"


def children_file(settings: Settings) -> Path:
    return run_dir(settings) / "children.pids"


def read_pid(path: Path) -> int | None:
    try:
        text = Path(path).read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError):
        return None
    return int(text) if text.isdigit() and int(text) > 1 else None


def read_children(path: Path) -> dict[str, int]:
    """``name pid`` lines written by the supervisor (invalid lines are ignored)."""
    result: dict[str, int] = {}
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError):
        return result
    for line in lines:
        parts = line.split()
        if len(parts) == 2 and parts[1].isdigit() and int(parts[1]) > 1:
            result[parts[0]] = int(parts[1])
    return result


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def process_command(pid: int) -> str | None:
    """Command line of ``pid`` via ``ps`` (None when unknown or not running)."""
    try:
        result = run(["ps", "-o", "command=", "-p", str(pid)], timeout=10, check=False,
                     error_message="process lookup failed")
    except (ClipperError, OSError, ValueError):
        return None
    if result.returncode != 0:
        return None
    text = result.stdout.decode("utf-8", "replace").strip()
    return text or None


def running_supervisor_pid(settings: Settings, *, verify_command: bool = True) -> int | None:
    """PID from ``data/run/supervisor.pid`` when that process is alive (and, by default,
    really is a supervisor - guards against PID reuse after a crash or reboot)."""
    pid = read_pid(pid_file(settings))
    if pid is None or pid == os.getpid() or not pid_alive(pid):
        return None
    if verify_command:
        cmd = process_command(pid)
        if cmd is not None and "clipper.supervisor" not in cmd:
            return None
    return pid


def orphaned_children(settings: Settings) -> dict[str, int]:
    """Children listed by a supervisor that is gone but whose processes still run."""
    if running_supervisor_pid(settings) is not None:
        return {}
    found: dict[str, int] = {}
    for name, pid in read_children(children_file(settings)).items():
        if pid == os.getpid() or not pid_alive(pid):
            continue
        cmd = process_command(pid)
        if cmd is None or "clipper" in cmd:
            found[name] = pid
    return found


def _signal_group(pgid: int, sig: int) -> None:
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def reap_orphans(settings: Settings, *, timeout: float = WORKER_STOP_SECONDS,
                 say: Callable[[str], None] | None = None) -> list[int]:
    """Stop children left behind by a supervisor that was killed. Returns their pids."""
    orphans = orphaned_children(settings)
    if not orphans:
        return []
    if say:
        say("Stopping processes left over from a previous run: " + ", ".join(sorted(orphans)))
    for pid in orphans.values():
        _signal_group(pid, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and any(pid_alive(p) for p in orphans.values()):
        time.sleep(0.2)
    for pid in orphans.values():
        _signal_group(pid, signal.SIGKILL)
    try:
        children_file(settings).unlink()
    except FileNotFoundError:
        pass
    return list(orphans.values())


# ----------------------------------------------------------------------------- health / port
def _connect_host(host: str) -> str:
    """Address to reach a server bound to ``host`` from this machine (URL form)."""
    h = (host or "").strip()
    if h in ("", "0.0.0.0", "*"):
        return "127.0.0.1"
    if h in ("::", "[::]"):
        return "[::1]"
    if ":" in h and not h.startswith("["):
        return f"[{h}]"
    return h


def health_target(settings: Settings) -> tuple[str, str]:
    """(URL, Host header) for the health probe.

    The API only accepts Host headers on its allow-list (DNS-rebinding defence), so
    the probe connects locally but presents the public origin's host.
    """
    connect = _connect_host(settings.host)
    url = f"http://{connect}:{settings.port}/api/health"
    # Without an explicit public origin the local address is on the allow-list (and, unlike
    # Settings.effective_public_origin, is correctly bracketed for IPv6).
    host_header = (urlsplit(settings.public_origin).netloc if settings.public_origin else "") \
        or f"{connect}:{settings.port}"
    return url, host_header


_NO_PROXY_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def check_health(url: str, host_header: str, *, timeout: float = 2.0) -> bool:
    req = urllib.request.Request(url, headers={"Host": host_header, "Accept": "application/json"})
    try:
        with _NO_PROXY_OPENER.open(req, timeout=timeout) as resp:
            if resp.status != 200:
                return False
            body = json.loads(resp.read(4096) or b"{}")
    except (urllib.error.URLError, http.client.HTTPException, OSError, ValueError):
        return False
    return isinstance(body, dict) and body.get("status") == "ok"


def app_healthy(settings: Settings, *, timeout: float = 2.0) -> bool:
    url, host = health_target(settings)
    return check_health(url, host, timeout=timeout)


def port_in_use(host: str, port: int) -> bool:
    """True when something listens on ``host:port`` or the address cannot be bound."""
    connect = _connect_host(host).strip("[]")
    try:
        with socket.create_connection((connect, port), timeout=0.5):
            return True
    except OSError:
        pass
    bind_host = (host or "").strip("[]") or "0.0.0.0"
    family = socket.AF_INET6 if ":" in bind_host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((bind_host, port))
        except OSError:
            return True
    return False


# ----------------------------------------------------------------------------- commands
def _proxy_list(value: str) -> str:
    return ",".join(p.strip() for p in value.split(",") if p.strip())


def api_command(settings: Settings, *, python: str | None = None) -> list[str]:
    """uvicorn command line for the API (proxy headers only when trusted proxies are set
    in hosted mode; uvicorn would otherwise trust 127.0.0.1 by default). Local mode always
    passes --no-proxy-headers: request.client stays the raw peer and api.deps.client_ip reads
    X-Forwarded-For itself, only from CLIPPER_TRUSTED_PROXIES."""
    level = settings.log_level.strip().lower()
    cmd = [
        python or sys.executable, "-m", "uvicorn", "clipper.api.app:create_app", "--factory",
        "--host", settings.host, "--port", str(settings.port),
        "--no-access-log", "--no-server-header", "--timeout-graceful-shutdown", "10",
        "--log-level", level if level in _UVICORN_LOG_LEVELS else "info",
    ]
    proxies = _proxy_list(settings.trusted_proxies)
    if settings.mode == "hosted" and proxies:
        cmd += ["--proxy-headers", "--forwarded-allow-ips", proxies]
    else:
        cmd += ["--no-proxy-headers"]
    return cmd


def worker_command(*, python: str | None = None) -> list[str]:
    return [python or sys.executable, "-m", "clipper.worker"]


@dataclass
class ChildSpec:
    name: str
    argv: list[str]
    stop_timeout: float = WORKER_STOP_SECONDS


def default_specs(settings: Settings) -> list[ChildSpec]:
    return [
        ChildSpec("api", api_command(settings), stop_timeout=API_STOP_SECONDS),
        ChildSpec("worker", worker_command(), stop_timeout=WORKER_STOP_SECONDS),
    ]


def child_env(settings: Settings, base: dict[str, str] | None = None) -> dict[str, str]:
    """Environment for children: importable package, unbuffered output, same data dir."""
    env = dict(os.environ if base is None else base)
    paths = [str(BACKEND_ROOT)] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(paths))
    env["PYTHONUNBUFFERED"] = "1"
    env["CLIPPER_DATA_DIR"] = str(Path(settings.data_dir).resolve())
    return env


# ----------------------------------------------------------------------------- logs
class RotatingLog:
    """Append-only log file rotated by size (``x.log`` -> ``x.log.1`` ...)."""

    def __init__(self, path: Path, *, max_bytes: int = LOG_MAX_BYTES, backups: int = LOG_BACKUPS) -> None:
        self.path = Path(path)
        self.max_bytes = max_bytes
        self.backups = max(1, backups)
        self._lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: IO[bytes] | None = open(self.path, "ab")
        self._size = self.path.stat().st_size

    def write(self, data: bytes) -> None:
        with self._lock:
            if self._fh is None:
                return
            if self._size and self._size + len(data) > self.max_bytes:
                self._rotate()
            self._fh.write(data)
            self._fh.flush()
            self._size += len(data)

    def _rotate(self) -> None:
        assert self._fh is not None
        self._fh.close()
        for i in range(self.backups - 1, 0, -1):
            older = self.path.with_name(f"{self.path.name}.{i}")
            if older.exists():
                os.replace(older, self.path.with_name(f"{self.path.name}.{i + 1}"))
        os.replace(self.path, self.path.with_name(f"{self.path.name}.1"))
        self._fh = open(self.path, "ab")
        self._size = 0

    def close(self) -> None:
        with self._lock:
            if self._fh is not None:
                self._fh.close()
                self._fh = None


def tail_lines(path: Path, n: int = 15) -> list[str]:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            fh.seek(max(0, fh.tell() - 16384))
            data = fh.read().decode("utf-8", "replace")
    except OSError:
        return []
    return data.splitlines()[-n:]


# ----------------------------------------------------------------------------- children
class Child:
    def __init__(self, spec: ChildSpec, log: RotatingLog, *, echo: IO[str] | None,
                 echo_lock: threading.Lock) -> None:
        self.spec = spec
        self.log = log
        self.echo = echo
        self.echo_lock = echo_lock
        self.proc: subprocess.Popen[bytes] | None = None
        self.reader: threading.Thread | None = None
        self.restarts: deque[float] = deque()

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def pid(self) -> int | None:
        return self.proc.pid if self.proc is not None else None

    def start(self, env: dict[str, str], cwd: Path) -> None:
        self.proc = subprocess.Popen(
            self.spec.argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=str(cwd), env=env, start_new_session=True, shell=False,
        )
        self.reader = threading.Thread(target=self._pump, args=(self.proc,), name=f"log-{self.name}", daemon=True)
        self.reader.start()

    def _pump(self, proc: subprocess.Popen[bytes]) -> None:
        assert proc.stdout is not None
        prefix = f"[{self.name}] "
        for raw in iter(proc.stdout.readline, b""):
            self.log.write(raw)
            if self.echo is not None:
                with self.echo_lock:
                    try:
                        self.echo.write(prefix + raw.decode("utf-8", "replace").rstrip("\n") + "\n")
                        self.echo.flush()
                    except (OSError, ValueError):
                        self.echo = None
        proc.stdout.close()

    def poll(self) -> int | None:
        return self.proc.poll() if self.proc is not None else 0

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def signal_group(self, sig: int) -> None:
        if self.proc is not None:
            _signal_group(self.proc.pid, sig)  # own session: pgid == pid

    def reap(self) -> None:
        if self.proc is None:
            return
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:  # pragma: no cover - SIGKILL already sent
            pass
        if self.reader is not None:
            self.reader.join(timeout=5)


# ----------------------------------------------------------------------------- supervisor
class Supervisor:
    """Runs ``specs`` until stopped. The child named ``health_child`` must serve /api/health."""

    def __init__(self, settings: Settings, specs: Sequence[ChildSpec], *, out: IO[str] | None = None,
                 echo: bool = False, open_browser: bool = False, detached: bool = False,
                 health_timeout: float = HEALTH_TIMEOUT_SECONDS, health_child: str = "api",
                 restart_limit: int = RESTART_LIMIT, restart_window: float = RESTART_WINDOW_SECONDS,
                 restart_delay: float = RESTART_DELAY_SECONDS, cwd: Path = BACKEND_ROOT,
                 env: dict[str, str] | None = None) -> None:
        self.settings = settings
        self.out = out or sys.stdout
        self.echo = echo
        self.open_browser = open_browser
        self.detached = detached
        self.health_timeout = health_timeout
        self.health_child = health_child
        self.restart_limit = restart_limit
        self.restart_window = restart_window
        self.restart_delay = restart_delay
        self.cwd = cwd
        self.env = env if env is not None else child_env(settings)
        self._out_lock = threading.Lock()
        self._announced = False
        logs = log_dir(settings)
        self.children = [
            Child(spec, RotatingLog(logs / f"{spec.name}.log"), echo=self.out if echo else None,
                  echo_lock=self._out_lock)
            for spec in specs
        ]
        # Plain attributes only: set from signal handlers, polled by the main loop.
        self._stop_requested = False
        self._force = False
        self.exit_code = 0

    # -- output -------------------------------------------------------------------
    def say(self, message: str) -> None:
        with self._out_lock:
            try:
                self.out.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")
                self.out.flush()
            except (OSError, ValueError):
                pass

    # -- control ------------------------------------------------------------------
    def request_stop(self, force: bool = False) -> None:
        if self._stop_requested or force:
            self._force = True
        self._stop_requested = True

    def _on_signal(self, signum: int, _frame: object) -> None:
        self.request_stop()

    def install_signal_handlers(self) -> None:
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, self._on_signal)

    def _sleep(self, seconds: float) -> None:
        deadline = time.monotonic() + seconds
        while not self._stop_requested and time.monotonic() < deadline:
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))

    # -- run files ------------------------------------------------------------------
    def _write_run_files(self) -> None:
        rd = run_dir(self.settings)
        rd.mkdir(parents=True, exist_ok=True)
        pid_file(self.settings).write_text(f"{os.getpid()}\n", encoding="utf-8")
        self._write_children()

    def _write_children(self) -> None:
        lines = [f"{c.name} {c.pid}" for c in self.children if c.pid is not None]
        tmp = children_file(self.settings).with_suffix(".tmp")
        tmp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        os.replace(tmp, children_file(self.settings))

    def _remove_run_files(self) -> None:
        pf = pid_file(self.settings)
        if read_pid(pf) == os.getpid():
            pf.unlink(missing_ok=True)
        children_file(self.settings).unlink(missing_ok=True)

    # -- lifecycle --------------------------------------------------------------------
    def run(self, *, install_signals: bool = True) -> int:
        if install_signals:
            self.install_signal_handlers()
        try:
            self._write_run_files()
            for child in self.children:
                child.start(self.env, self.cwd)
            self._write_children()
            if not self._wait_until_healthy():
                if not self._stop_requested:
                    self.exit_code = 1
                return self.exit_code
            self._announce()
            self._monitor()
            return self.exit_code
        finally:
            self.stop_children()
            self._remove_run_files()
            for child in self.children:
                child.log.close()
            if self._announced:
                self.say("Local Clipper stopped.")

    def _log_path(self, child: Child) -> Path:
        return log_dir(self.settings) / f"{child.name}.log"

    def _find(self, name: str) -> Child | None:
        return next((c for c in self.children if c.name == name), None)

    def _wait_until_healthy(self) -> bool:
        target = self._find(self.health_child)
        if target is None:
            return True
        url, host = health_target(self.settings)
        deadline = time.monotonic() + self.health_timeout
        while not self._stop_requested:
            if check_health(url, host, timeout=2.0):
                return True
            rc = target.poll()
            if rc is not None:
                self._startup_failed(target, f"The {target.name} process exited during startup (exit code {rc}).")
                return False
            if time.monotonic() >= deadline:
                self._startup_failed(target, f"The web server did not answer within {int(self.health_timeout)} s.")
                return False
            self._sleep(0.25)
        return False

    def _startup_failed(self, child: Child, reason: str) -> None:
        log_path = self._log_path(child)
        self.say(f"Local Clipper could not start. {reason}")
        lines = tail_lines(log_path)
        if lines:
            self.say(f"Last lines of {log_path}:")
            with self._out_lock:
                for line in lines:
                    self.out.write(f"    {line}\n")
                self.out.flush()

    def _announce(self) -> None:
        url = self.settings.effective_public_origin
        stop_hint = "Stop with: scripts/stop.sh" if self.detached else "Press Ctrl+C to stop (or run scripts/stop.sh)."
        with self._out_lock:
            self.out.write(f"\nLocal Clipper is running at {url}\n{stop_hint}\n"
                           f"Logs: {log_dir(self.settings)}\n\n")
            self.out.flush()
        self._announced = True
        if self.open_browser and self.settings.mode == "local":
            try:
                webbrowser.open(url)
            except Exception:  # pragma: no cover - no browser available
                pass

    def _monitor(self) -> None:
        while not self._stop_requested:
            for child in self.children:
                rc = child.poll()
                if rc is not None and not self._stop_requested:
                    if not self._handle_exit(child, rc):
                        self._stop_requested = True
                        break
            self._sleep(0.25)
        if self.exit_code == 0:
            self.say("Stopping Local Clipper…")

    def _handle_exit(self, child: Child, rc: int) -> bool:
        """Restart ``child`` if allowed. Returns False when the supervisor should stop."""
        child.reap()
        child.signal_group(signal.SIGKILL)  # leftovers in the crashed child's group
        if rc == EXIT_CONFIG:
            self.say(f"The {child.name} process stopped because of a configuration problem "
                     f"(see {self._log_path(child)}). Stopping.")
            self.exit_code = EXIT_CONFIG
            return False
        now = time.monotonic()
        while child.restarts and now - child.restarts[0] > self.restart_window:
            child.restarts.popleft()
        minutes = max(1, round(self.restart_window / 60))
        if len(child.restarts) >= self.restart_limit:
            self.say(f"The {child.name} process crashed {len(child.restarts) + 1} times within {minutes} minutes; "
                     f"giving up. See {self._log_path(child)}.")
            self.exit_code = 1
            return False
        child.restarts.append(now)
        self.say(f"The {child.name} process exited unexpectedly (exit code {rc}); restarting "
                 f"(restart {len(child.restarts)} of {self.restart_limit} allowed per {minutes} minutes).")
        self._sleep(self.restart_delay)
        if self._stop_requested:
            return True
        child.start(self.env, self.cwd)
        self._write_children()
        return True

    def stop_children(self) -> None:
        running = [c for c in self.children if c.alive()]
        for child in running:
            child.signal_group(signal.SIGTERM)
        started = time.monotonic()
        killed: set[str] = set()
        while any(c.alive() for c in running) and not self._force:
            elapsed = time.monotonic() - started
            for child in running:
                if child.alive() and elapsed >= child.spec.stop_timeout and child.name not in killed:
                    self.say(f"The {child.name} process did not stop within {int(child.spec.stop_timeout)} s; "
                             "killing it.")
                    child.signal_group(signal.SIGKILL)
                    killed.add(child.name)
            time.sleep(0.1)
        for child in self.children:
            # SIGKILL whatever is left in each child's process group (stragglers, grandchildren).
            child.signal_group(signal.SIGKILL)
            child.reap()


# ----------------------------------------------------------------------------- entry point
def _parse_args(argv: Iterable[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m clipper.supervisor",
                                     description="Run the Local Clipper web server and worker together.")
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser window")
    parser.add_argument("--detached", action="store_true",
                        help="run in a new session without echoing logs (used by scripts/start.sh --detach)")
    return parser.parse_args(list(argv) if argv is not None else None)


def prepare(settings: Settings) -> None:
    """Validate configuration and create the data layout + database (raises ConfigError)."""
    settings.validate_for_startup()
    Storage(settings.data_dir).ensure()
    log_dir(settings).mkdir(parents=True, exist_ok=True)
    run_dir(settings).mkdir(parents=True, exist_ok=True)
    try:
        db.init_db(settings.db_path)
    except RuntimeError as exc:  # newer schema than this version understands
        raise ConfigError(f"{exc} Install the newer version again or restore a matching backup.") from None


def main(argv: Iterable[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.detached:
        try:
            os.setsid()  # leave the launching terminal's process group
        except OSError:
            pass
    try:
        settings = load_settings()
        prepare(settings)
    except ConfigError as exc:
        print(config_help(str(exc)), file=sys.stderr)
        return EXIT_CONFIG
    except OSError as exc:
        print(f"Local Clipper cannot prepare its data folder ({exc.strerror or type(exc).__name__}). "
              "Check CLIPPER_DATA_DIR and its permissions, then run scripts/doctor.sh.", file=sys.stderr)
        return 1

    existing = running_supervisor_pid(settings)
    if existing is not None:
        print(f"Local Clipper is already running (process {existing}) at {settings.effective_public_origin}.\n"
              "Stop it with scripts/stop.sh first.", file=sys.stderr)
        return 1
    reap_orphans(settings, say=lambda m: print(m, file=sys.stderr))
    if port_in_use(settings.host, settings.port):
        if app_healthy(settings):
            print(f"Another copy of Local Clipper is already answering at {settings.effective_public_origin}.\n"
                  "Stop it first (scripts/stop.sh, or close the terminal running it).", file=sys.stderr)
        else:
            print(f"Port {settings.port} on {settings.host} is already used by another program.\n"
                  "Set CLIPPER_PORT in .env to a free port (and update CLIPPER_PUBLIC_ORIGIN if you set it).",
                  file=sys.stderr)
        return 1

    supervisor = Supervisor(
        settings, default_specs(settings),
        echo=(not args.detached) and sys.stdout.isatty(),
        open_browser=not args.no_browser,
        detached=args.detached,
    )
    return supervisor.run()


if __name__ == "__main__":
    sys.exit(main())
