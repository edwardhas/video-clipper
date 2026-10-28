"""Subprocess runner (argument arrays, process-group cancellation, timeouts) and safe storage paths."""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import pytest

from clipper.errors import CancelledError, DiskSpaceError, InterruptedError_, MediaError, MissingDependencyError
from clipper.procs import CancelToken, run
from clipper.storage import Storage, atomic_path, is_valid_id, new_id, safe_filename, stable_hash


# ----------------------------------------------------------------------------- procs
def test_args_are_not_shell_interpreted(tmp_path: Path):
    marker = tmp_path / "pwned"
    evil = f"; touch {marker}"
    res = run([sys.executable, "-c", "import sys; print(sys.argv[1])", evil])
    assert res.stdout.decode().strip() == evil
    assert not marker.exists()


def test_missing_binary_raises_friendly_error():
    with pytest.raises(MissingDependencyError) as ei:
        run(["definitely-not-a-real-binary-xyz", "--version"])
    assert "definitely-not-a-real-binary-xyz" in ei.value.user_message


def test_nonzero_exit_raises_with_detail():
    with pytest.raises(MediaError) as ei:
        run([sys.executable, "-c", "import sys; sys.stderr.write('bad thing\\n'); sys.exit(3)"],
            error_message="Step failed.")
    assert ei.value.user_message == "Step failed."
    assert "bad thing" in (ei.value.detail or "")


def test_timeout_kills_process():
    t0 = time.monotonic()
    with pytest.raises(MediaError):
        run([sys.executable, "-c", "import time; time.sleep(30)"], timeout=0.5)
    assert time.monotonic() - t0 < 10


def _child_pid_script(pidfile: Path) -> list[str]:
    # parent spawns a grandchild sleeper and writes its pid; both must die on cancel
    code = (
        "import subprocess, sys, time\n"
        f"p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n"
        f"open({str(pidfile)!r}, 'w').write(str(p.pid))\n"
        "time.sleep(60)\n"
    )
    return [sys.executable, "-c", code]


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    # zombie processes still "exist"; check via waitpid where possible
    return True


def test_cancel_kills_whole_process_group(tmp_path: Path):
    pidfile = tmp_path / "grandchild.pid"
    token = CancelToken()

    def cancel_soon():
        for _ in range(100):
            if pidfile.exists() and pidfile.read_text():
                break
            time.sleep(0.05)
        token.cancel()

    threading.Thread(target=cancel_soon, daemon=True).start()
    with pytest.raises(CancelledError):
        run(_child_pid_script(pidfile), cancel=token, timeout=30)
    gpid = int(pidfile.read_text())
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and _alive(gpid):
        time.sleep(0.1)
    assert not _alive(gpid), "grandchild survived cancellation"


def test_cancel_check_callable_is_polled():
    flag = {"v": False}
    token = CancelToken(check=lambda: flag["v"])
    threading.Timer(0.3, lambda: flag.update(v=True)).start()
    with pytest.raises(CancelledError):
        run([sys.executable, "-c", "import time; time.sleep(30)"], cancel=token)


def test_shutdown_raises_interrupted():
    token = CancelToken()
    threading.Timer(0.3, token.shutdown.set).start()
    with pytest.raises(InterruptedError_):
        run([sys.executable, "-c", "import time; time.sleep(30)"], cancel=token)


def test_already_cancelled_does_not_start():
    token = CancelToken()
    token.cancel()
    with pytest.raises(CancelledError):
        run([sys.executable, "-c", "print(1)"], cancel=token)


def test_stdout_bytes_and_lines():
    chunks: list[bytes] = []
    run([sys.executable, "-c", "import sys; sys.stdout.buffer.write(b'\\x00\\x01' * 1000)"],
        stdout_bytes_cb=chunks.append)
    assert b"".join(chunks) == b"\x00\x01" * 1000
    lines: list[str] = []
    run([sys.executable, "-c", "print('a'); print('b')"], stdout_line_cb=lines.append)
    assert [l.strip() for l in lines] == ["a", "b"]


# ----------------------------------------------------------------------------- storage
def test_ids_and_filenames():
    pid = new_id("p")
    assert is_valid_id(pid)
    for bad in ("", "p_", "../p_0123456789ab", "p_0123456789AB", "P_0123456789ab", "p_0123/56789ab"):
        assert not is_valid_id(bad)
    assert safe_filename("../../etc/passwd") == "passwd"
    assert safe_filename("my video (final).mp4") == "my_video_final_.mp4"
    assert safe_filename("..") == "file"
    assert "/" not in safe_filename("a/b\\c.mp4")
    assert len(safe_filename("x" * 300 + ".mp4")) <= 80 and safe_filename("x" * 300 + ".mp4").endswith(".mp4")


def test_resolve_refuses_traversal(storage: Storage):
    with pytest.raises(ValueError):
        storage.resolve("../outside.txt")
    with pytest.raises(ValueError):
        storage.resolve("/etc/passwd")
    with pytest.raises(ValueError):
        storage.resolve("projects/../../x")
    with pytest.raises(ValueError):
        storage.resolve("")
    with pytest.raises(ValueError):
        storage.project_dir("../p_0123456789ab")
    ok = storage.resolve("projects/p_0123456789ab/source/source.mp4")
    assert ok.is_relative_to(storage.root)


def test_resolve_refuses_symlink_escape(storage: Storage, tmp_path: Path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("x")
    link = storage.root / "projects" / "p_0123456789ab"
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(outside)
    with pytest.raises(ValueError):
        storage.resolve("projects/p_0123456789ab/secret.txt")


def test_delete_tree_only_app_managed(storage: Storage, tmp_path: Path):
    user_file = tmp_path / "user_original.mp4"
    user_file.write_bytes(b"x")
    with pytest.raises(ValueError):
        storage.delete_tree(user_file)
    with pytest.raises(ValueError):
        storage.delete_tree(storage.root)
    with pytest.raises(ValueError):
        storage.delete_tree(storage.root / "projects")
    d = storage.project_dir("p_0123456789ab")
    d.mkdir(parents=True)
    (d / "a.txt").write_text("x")
    storage.delete_tree(d)
    assert not d.exists() and user_file.exists()


def test_atomic_path_leaves_no_partial_file(tmp_path: Path):
    final = tmp_path / "out.mp4"
    with pytest.raises(RuntimeError):
        with atomic_path(final) as tmp:
            tmp.write_bytes(b"partial")
            raise RuntimeError("boom")
    assert not final.exists()
    assert list(tmp_path.iterdir()) == []
    with atomic_path(final) as tmp:
        tmp.write_bytes(b"done")
    assert final.read_bytes() == b"done"


def test_require_free(storage: Storage):
    free = storage.free_bytes()
    storage.require_free(0, 0, "x")
    with pytest.raises(DiskSpaceError):
        storage.require_free(free + 10**12, 0, "a huge render")


def test_stable_hash_is_order_insensitive_for_dict_keys():
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})
    assert stable_hash("x", 1) != stable_hash("x", 2)
