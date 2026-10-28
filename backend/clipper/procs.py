"""Subprocess execution with argument arrays, process groups, timeouts and cancellation.

Every external tool (ffmpeg, ffprobe, yt-dlp) is launched through :func:`run`, which

* takes an argument *list* (never a shell string; ``shell=False`` always),
* starts the child in its own session/process group so cancellation can kill the
  whole tree (e.g. yt-dlp -> ffmpeg),
* polls a :class:`CancelToken` while the child runs and kills the group when set,
* enforces an optional timeout,
* keeps only a bounded tail of stderr for error messages.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from .errors import CancelledError, InterruptedError_, MediaError, MissingDependencyError


class CancelToken:
    """Cooperative cancellation shared between the job runner and running work.

    ``check`` is a callable returning True when cancellation was requested (the
    worker wires it to the job row's ``cancel_requested`` flag). ``shutdown`` is set
    when the worker process itself is stopping.
    """

    def __init__(self, check: Callable[[], bool] | None = None) -> None:
        self._check = check
        self._cancelled = threading.Event()
        self.shutdown = threading.Event()
        self._last_poll = 0.0

    def cancel(self) -> None:
        self._cancelled.set()

    @property
    def cancelled(self) -> bool:
        """Cheap check for tight polling loops (the external check runs at most every 0.5 s)."""
        return self._poll(min_interval=0.5)

    def check_now(self) -> bool:
        """Fresh check, ignoring the polling interval (use at checkpoints)."""
        return self._poll(min_interval=0.0)

    def _poll(self, *, min_interval: float) -> bool:
        if self._cancelled.is_set():
            return True
        now = time.monotonic()
        if self._check is not None and now - self._last_poll >= min_interval:
            self._last_poll = now
            try:
                if self._check():
                    self._cancelled.set()
            except Exception:  # pragma: no cover - db hiccup should not crash work
                pass
        return self._cancelled.is_set()

    def raise_if_cancelled(self) -> None:
        """Checkpoint between steps: always consults the external flag afresh."""
        if self.shutdown.is_set():
            raise InterruptedError_("The worker stopped before this step finished.")
        if self.check_now():
            raise CancelledError("Cancelled by user.")


NEVER_CANCELLED = CancelToken()


@dataclass
class ProcResult:
    args: list[str]
    returncode: int
    stdout: bytes
    stderr_tail: str
    duration_s: float


@dataclass
class _Tail:
    lines: deque = field(default_factory=lambda: deque(maxlen=60))

    def add(self, line: str) -> None:
        self.lines.append(line.rstrip("\n"))

    def text(self) -> str:
        return "\n".join(self.lines)


def _kill_group(proc: subprocess.Popen, grace: float = 3.0) -> None:
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            break
        time.sleep(0.05)
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:  # pragma: no cover
        pass


def run(
    args: Sequence[str | os.PathLike],
    *,
    cancel: CancelToken | None = None,
    timeout: float | None = None,
    capture_stdout: bool = True,
    stdout_line_cb: Callable[[str], None] | None = None,
    stdout_bytes_cb: Callable[[bytes], None] | None = None,
    stderr_line_cb: Callable[[str], None] | None = None,
    cwd: str | os.PathLike | None = None,
    env: dict[str, str] | None = None,
    check: bool = True,
    error_cls: type[Exception] = MediaError,
    error_message: str = "A media processing step failed.",
) -> ProcResult:
    """Run ``args`` (a list; no shell) and return its result.

    * If ``stdout_line_cb`` is set, stdout is streamed line by line to the callback
      and ``ProcResult.stdout`` is empty.
    * If ``stdout_bytes_cb`` is set, raw stdout chunks (≤ 1 MiB) are streamed to the
      callback (for binary pipes such as rawvideo) and ``ProcResult.stdout`` is empty.
    * Raises ``CancelledError``/``InterruptedError_`` when the token fires (the
      process group is killed first), ``error_cls(error_message)`` on non-zero exit
      when ``check`` is True, and ``MissingDependencyError`` if the binary is absent.
    """
    argv = [os.fspath(a) for a in args]
    if not argv or not isinstance(argv[0], str):
        raise ValueError("args must be a non-empty list")
    token = cancel or NEVER_CANCELLED
    token.raise_if_cancelled()
    started = time.monotonic()
    try:
        proc = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE if (capture_stdout or stdout_line_cb or stdout_bytes_cb) else subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=env,
            start_new_session=True,  # own process group -> killpg on cancel
            shell=False,
        )
    except FileNotFoundError as exc:
        raise MissingDependencyError(
            f"Required program '{Path(argv[0]).name}' was not found. Run scripts/doctor.sh for setup help."
        ) from exc

    tail = _Tail()
    out_chunks: list[bytes] = []

    def _read_stderr() -> None:
        assert proc.stderr is not None
        for raw in iter(proc.stderr.readline, b""):
            line = raw.decode("utf-8", "replace")
            tail.add(line)
            if stderr_line_cb:
                try:
                    stderr_line_cb(line)
                except Exception:  # pragma: no cover
                    pass

    def _read_stdout() -> None:
        assert proc.stdout is not None
        if stdout_line_cb:
            for raw in iter(proc.stdout.readline, b""):
                try:
                    stdout_line_cb(raw.decode("utf-8", "replace"))
                except Exception:  # pragma: no cover
                    pass
        elif stdout_bytes_cb:
            while True:
                chunk = proc.stdout.read(1 << 20)
                if not chunk:
                    break
                stdout_bytes_cb(chunk)
        else:
            while True:
                chunk = proc.stdout.read(65536)
                if not chunk:
                    break
                out_chunks.append(chunk)

    threads = [threading.Thread(target=_read_stderr, daemon=True)]
    if proc.stdout is not None:
        threads.append(threading.Thread(target=_read_stdout, daemon=True))
    for t in threads:
        t.start()

    killed_reason: str | None = None
    while True:
        if proc.poll() is not None:
            break
        if token.shutdown.is_set():
            killed_reason = "shutdown"
        elif token.cancelled:
            killed_reason = "cancel"
        elif timeout is not None and time.monotonic() - started > timeout:
            killed_reason = "timeout"
        if killed_reason:
            _kill_group(proc)
            break
        time.sleep(0.1)

    for t in threads:
        t.join(timeout=5)
    duration = time.monotonic() - started
    rc = proc.returncode if proc.returncode is not None else -9

    if killed_reason == "shutdown":
        raise InterruptedError_("The worker stopped before this step finished.")
    if killed_reason == "cancel":
        raise CancelledError("Cancelled by user.")
    if killed_reason == "timeout":
        raise error_cls(f"{error_message} (timed out after {int(timeout or 0)} s)")

    result = ProcResult(args=argv, returncode=rc, stdout=b"".join(out_chunks), stderr_tail=tail.text(),
                        duration_s=duration)
    if check and rc != 0:
        exc = error_cls(error_message)
        if hasattr(exc, "detail"):
            exc.detail = f"{Path(argv[0]).name} exited {rc}: {result.stderr_tail[-2000:]}"
        raise exc
    return result
