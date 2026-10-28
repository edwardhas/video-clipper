"""Tests for clipper.cli (doctor, hash-password, check-config, health, paths, backup/restore)
and the thin scripts/*.sh wrappers around it."""

from __future__ import annotations

import http.server
import io
import ipaddress
import json
import os
import re
import shutil
import socket
import sqlite3
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

import pytest
from argon2 import PasswordHasher

from clipper import cli, db
from clipper.config import REPO_ROOT, Settings
from clipper.storage import Storage, new_id, sha256_file

from conftest import requires_ffmpeg

SCRIPTS = REPO_ROOT / "scripts"
VENV_PY = REPO_ROOT / "backend" / ".venv" / "bin" / "python"
SECRET_KEY = "sk-test-DO-NOT-PRINT-4242"


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


@pytest.fixture
def s(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return make_settings(tmp_path)


def clean_env(**extra: str) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("CLIPPER_") and k not in ("OPENAI_API_KEY", "PYTHONPATH")}
    env.update(extra)
    return env


def seed(settings: Settings) -> dict[str, str]:
    """A small but realistic data folder: one project with source/work/clip files."""
    storage = Storage(settings.data_dir)
    storage.ensure()
    db.init_db(settings.db_path)
    pid, cid, jid = new_id("p"), new_id("c"), new_id("j")
    t = db.now()
    with db.session(settings.db_path) as conn:
        conn.execute(
            "INSERT INTO projects(id, title, source_kind, options, status, created_at, updated_at) "
            "VALUES (?, ?, 'upload', '{}', 'ready', ?, ?)", (pid, "Original project", t, t))
        conn.execute("INSERT INTO jobs(id, project_id, kind, state, created_at) VALUES (?, ?, 'analyze', "
                     "'succeeded', ?)", (jid, pid, t))
        conn.execute(
            "INSERT INTO clips(id, project_id, rank, title, reason, score, selection, edit, created_at, updated_at) "
            "VALUES (?, ?, 1, 'Clip', 'Because', 80, '{}', '{}', ?, ?)", (cid, pid, t, t))
    proj = storage.project_dir(pid)
    (proj / "source").mkdir(parents=True)
    (proj / "source" / "source.mp4").write_bytes(b"\x00\x01source-media" * 1000)
    (proj / "source" / ".source.abc123.tmp.mp4").write_bytes(b"partial write")  # atomic temp: skipped
    (proj / "work").mkdir()
    (proj / "work" / "transcript.json").write_text('{"words": []}', encoding="utf-8")
    (proj / "clips" / cid).mkdir(parents=True)
    (proj / "clips" / cid / "abc.mp4").write_bytes(b"rendered" * 500)
    up = storage.upload_dir(new_id("u"))
    up.mkdir(parents=True)
    (up / "pending.mp4").write_bytes(b"upload")
    cache = settings.data_dir / "cache" / "transcribe-chunk"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / ("ab" * 16 + ".json")).write_text("{}", encoding="utf-8")
    logs = settings.data_dir / "logs"
    logs.mkdir(exist_ok=True)
    (logs / "api.log").write_text("log line\n", encoding="utf-8")
    return {"project": pid, "clip": cid}


def project_titles(settings: Settings) -> list[str]:
    with db.session(settings.db_path) as conn:
        return [r["title"] for r in conn.execute("SELECT title FROM projects ORDER BY title")]


def mutate(settings: Settings, ids: dict[str, str]) -> None:
    """Change the live data after a backup (the restore must undo this)."""
    shutil.rmtree(settings.data_dir / "projects" / ids["project"])
    t = db.now()
    with db.session(settings.db_path) as conn:
        conn.execute("DELETE FROM projects WHERE id=?", (ids["project"],))
        conn.execute("INSERT INTO projects(id, title, source_kind, options, status, created_at, updated_at) "
                     "VALUES (?, 'Newer project', 'upload', '{}', 'ready', ?, ?)", (new_id("p"), t, t))
    (settings.data_dir / "projects" / "p_0123456789ab").mkdir(parents=True)


class _HealthHandler(http.server.BaseHTTPRequestHandler):
    expected_host = ""
    seen_hosts: list[str] = []

    def do_GET(self) -> None:  # noqa: N802
        host = self.headers.get("Host", "")
        type(self).seen_hosts.append(host)
        ok = self.path == "/api/health" and host == type(self).expected_host
        body = b'{"status":"ok"}' if ok else b'{"detail":"bad host"}'
        self.send_response(200 if ok else 400)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args) -> None:
        pass


@pytest.fixture
def health_server():
    servers = []

    def start(port: int, expected_host: str):
        handler = type("H", (_HealthHandler,), {"expected_host": expected_host, "seen_hosts": []})
        srv = http.server.HTTPServer(("127.0.0.1", port), handler)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        return handler

    yield start
    for srv in servers:
        srv.shutdown()
        srv.server_close()


def run_script(name: str, *args: str, env: dict[str, str], cwd: Path | None = None, input: str | None = None,
               timeout: float = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["bash", str(SCRIPTS / name), *args], env=env, cwd=cwd, input=input,
                          capture_output=True, text=True, timeout=timeout)


def run_script_at(path: Path, *args: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    """Run a script copied outside the repository (from that copy's root)."""
    return subprocess.run(["bash", str(path), *args], env=env, cwd=path.parent.parent,
                          capture_output=True, text=True, timeout=300)


# ============================================================================ hash-password
def test_hash_password_stdin_produces_verifiable_argon2_hash(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("correct horse battery staple\n"))
    assert cli.main(["hash-password", "--stdin"]) == 0
    out, err = capsys.readouterr()
    digest = out.strip()
    assert digest.startswith("$argon2id$")
    assert len(out.strip().splitlines()) == 1
    assert PasswordHasher().verify(digest, "correct horse battery staple")
    assert "correct horse" not in out + err


def test_hash_password_interactive_asks_twice(monkeypatch, capsys) -> None:
    answers = iter(["a long enough secret", "a long enough secret"])
    monkeypatch.setattr(cli.getpass, "getpass", lambda prompt="": next(answers))
    assert cli.main(["hash-password"]) == 0
    out, err = capsys.readouterr()
    assert PasswordHasher().verify(out.strip(), "a long enough secret")
    assert "single quotes" in err


def test_hash_password_mismatch_prints_nothing_to_stdout(monkeypatch, capsys) -> None:
    answers = iter(["first password!", "second password"])
    monkeypatch.setattr(cli.getpass, "getpass", lambda prompt="": next(answers))
    assert cli.main(["hash-password"]) == 1
    out, err = capsys.readouterr()
    assert out == ""
    assert "do not match" in err


def test_hash_password_rejects_short_and_empty(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "stdin", io.StringIO("short\n"))
    assert cli.main(["hash-password", "--stdin"]) == 1
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    assert cli.main(["hash-password", "--stdin"]) == 1
    out, err = capsys.readouterr()
    assert out == ""
    assert "at least" in err and "No password" in err


def test_hash_password_script_and_settings_accept_hash(tmp_path: Path) -> None:
    result = run_script("hash-password.sh", "--stdin", env=clean_env(), input="script password 123\n")
    assert result.returncode == 0, result.stderr
    digest = result.stdout.strip()
    assert PasswordHasher().verify(digest, "script password 123")
    assert "script password" not in result.stdout + result.stderr
    # The documented .env form (single quotes keep the $ signs) round-trips through Settings.
    env_file = tmp_path / "test.env"
    env_file.write_text(f"CLIPPER_AUTH=required\nCLIPPER_OWNER_PASSWORD_HASH='{digest}'\n"
                        f"CLIPPER_SESSION_SECRET={'x' * 40}\n", encoding="utf-8")
    settings = Settings(_env_file=str(env_file), data_dir=tmp_path / "data")
    assert settings.owner_password_hash is not None
    assert settings.owner_password_hash.get_secret_value() == digest
    settings.validate_for_startup()


# ============================================================================ check-config / paths
def test_check_config_ok_hides_secrets(tmp_path: Path, capsys) -> None:
    settings = make_settings(tmp_path, auth="required", session_secret="s" * 40,
                             owner_password_hash=PasswordHasher().hash("owner password 1"),
                             openai_api_key=SECRET_KEY)
    assert cli.main(["check-config"], settings=settings) == 0
    out = capsys.readouterr().out
    assert "Configuration OK." in out
    assert "Owner password hash" in out and "set" in out
    assert SECRET_KEY not in out and "s" * 40 not in out and "$argon2" not in out


def test_check_config_invalid_hosted_exits_2(tmp_path: Path, capsys) -> None:
    settings = make_settings(tmp_path, mode="hosted", auth="auto", session_secret="tooshort-secret-value",
                             public_origin="http://insecure.example")
    assert cli.main(["check-config"], settings=settings) == 2
    err = capsys.readouterr().err
    assert "CLIPPER_OWNER_PASSWORD_HASH" in err and "CLIPPER_SESSION_SECRET" in err and "https://" in err
    assert "tooshort-secret-value" not in err


def test_paths_prints_key_values(s: Settings, capsys) -> None:
    assert cli.main(["paths"], settings=s) == 0
    values = dict(line.split("=", 1) for line in capsys.readouterr().out.splitlines())
    assert Path(values["data_dir"]) == Path(s.data_dir).resolve()
    assert values["pid_file"].endswith("run/supervisor.pid")
    assert values["url"] == f"http://127.0.0.1:{s.port}"


# ============================================================================ doctor
FFMPEG9_FILTERS = """Filters:
  T.. = Timeline support
  .S. = Slice threading
  ------
 TS overlay           VV->V      Overlay a video source on top of the input.
 T. afade             A->A       Fade in/out input audio.
 .. concat            N->N       Concatenate audio and video streams.
"""
FFMPEG51_FILTERS = """Filters:
  T.. = Timeline support
  .S. = Slice threading
  ..C = Command support
  ------
 TSC gblur             V->V       Apply Gaussian Blur filter.
 ..C scale             V->V       Scale the input video size and/or convert the image format.
"""
DEMUXERS = """File formats:
 D. = Demuxing supported
 .E = Muxing supported
 --
 D  concat          Virtual concatenation script
 D  mov,mp4,m4a,3gp,3g2,mj2 QuickTime / MOV
 D d lavfi           Libavfilter virtual input device
"""
ENCODERS = """Encoders:
 V..... = Video
 ------
 V....D libx264              libx264 H.264 / AVC / MPEG-4 AVC / MPEG-4 part 10 (codec h264)
 A....D aac                  AAC (Advanced Audio Coding)
"""


def test_parse_ffmpeg_list_handles_formats() -> None:
    assert cli.parse_ffmpeg_list(FFMPEG9_FILTERS) == {"overlay", "afade", "concat"}
    assert cli.parse_ffmpeg_list(FFMPEG51_FILTERS) == {"gblur", "scale"}
    assert {"concat", "mov", "mp4", "lavfi"} <= cli.parse_ffmpeg_list(DEMUXERS)
    assert cli.parse_ffmpeg_list(ENCODERS) == {"libx264", "aac"}
    assert cli.parse_ffmpeg_list("no separator here\n overlay x") == set()


def test_missing_ffmpeg_features_lists_each_gap() -> None:
    caps = {"filters": {"overlay", "crop", "scale", "gblur", "silencedetect", "volumedetect"},
            "demuxers": {"concat"}, "decoders": set(), "encoders": {"aac"}}
    assert cli.missing_ffmpeg_features(caps) == ["filter 'afade'", "decoder 'png'", "encoder 'libx264'"]


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_real_ffmpeg_has_required_features() -> None:
    caps = cli.ffmpeg_capabilities(shutil.which("ffmpeg"))
    assert cli.missing_ffmpeg_features(caps) == []


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_doctor_reports_lines_and_never_prints_key(tmp_path: Path, monkeypatch, capsys) -> None:
    for key in [k for k in os.environ if k.startswith("CLIPPER_")]:
        monkeypatch.delenv(key)
    monkeypatch.setenv("CLIPPER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CLIPPER_PORT", str(free_port()))
    monkeypatch.setenv("CLIPPER_HW_ENCODE", "off")
    monkeypatch.setenv("CLIPPER_AUTH", "disabled")
    monkeypatch.setenv("OPENAI_API_KEY", SECRET_KEY)
    rc = cli.main(["doctor"])
    out = capsys.readouterr().out
    assert SECRET_KEY not in out
    assert re.search(r"PASS\s+Python\s+3\.12", out)
    assert re.search(r"PASS\s+FFmpeg\s+ffmpeg version", out)
    assert re.search(r"PASS\s+OpenAI API key\s+configured \(value not shown\)", out)
    assert re.search(r"PASS\s+Data folder", out) and re.search(r"PASS\s+Port", out)
    assert re.search(r"INFO\s+Hardware encoder", out)
    assert "passed," in out
    assert rc == (1 if re.search(r"^\s+FAIL\s", out, re.M) else 0)


def test_doctor_fails_without_ffmpeg_and_on_bad_values(tmp_path: Path, monkeypatch, capsys) -> None:
    for key in [k for k in os.environ if k.startswith("CLIPPER_")]:
        monkeypatch.delenv(key)
    # An empty variable, not delenv: Settings() also reads the repository .env, which may now hold
    # a real key (environment variables take precedence over the .env file).
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("CLIPPER_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CLIPPER_PORT", str(free_port()))
    monkeypatch.setenv("CLIPPER_FFMPEG", str(tmp_path / "no-such-ffmpeg"))
    assert cli.main(["doctor"]) == 1
    out = capsys.readouterr().out
    assert re.search(r"FAIL\s+FFmpeg\s+'.*no-such-ffmpeg' was not found", out)
    assert re.search(r"WARN\s+OpenAI API key\s+not configured", out)

    monkeypatch.setenv("CLIPPER_MAX_CLIPS", "lots-of-clips")
    assert cli.main(["doctor"]) == 1
    out = capsys.readouterr().out
    assert re.search(r"FAIL\s+Configuration\s+Invalid configuration", out)
    assert "CLIPPER_MAX_CLIPS" in out and "lots-of-clips" not in out


def test_doctor_database_check_detects_newer_schema(s: Settings) -> None:
    seed(s)
    assert cli.check_database(s).status == cli.PASS
    with db.session(s.db_path) as conn:
        conn.execute("UPDATE meta SET value='99' WHERE key='schema_version'")
    check = cli.check_database(s)
    assert check.status == cli.FAIL and "newer" in check.detail


def test_doctor_assets_checks_pass() -> None:
    assert cli.check_yunet().status == cli.PASS
    assert cli.check_fonts().status == cli.PASS


def test_format_checks_summary() -> None:
    text = cli.format_checks([cli.Check(cli.PASS, "A", "fine"), cli.Check(cli.FAIL, "Longer name", "bad\nnext")])
    assert "  PASS  A            fine" in text
    assert "next" in text.splitlines()[2]
    assert text.endswith("1 passed, 0 warning(s), 1 failed.")


# ============================================================================ health
def test_health_uses_public_origin_host_header(tmp_path: Path, health_server, capsys) -> None:
    port = free_port()
    settings = make_settings(tmp_path, mode="hosted", host="0.0.0.0", port=port,
                             public_origin="https://clipper.example.org")
    handler = health_server(port, expected_host="clipper.example.org")
    assert cli.main(["health"], settings=settings) == 0
    assert handler.seen_hosts == ["clipper.example.org"]
    wrong = make_settings(tmp_path, mode="hosted", host="0.0.0.0", port=port,
                          public_origin="https://other.example.org")
    assert cli.main(["health", "--quiet"], settings=wrong) == 1


def test_health_gives_up_when_pid_exits(s: Settings) -> None:
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    started = time.monotonic()
    assert cli.main(["health", "--wait", "20", "--pid", str(proc.pid), "--quiet"], settings=s) == 3
    assert time.monotonic() - started < 5


def test_health_pid_must_be_alive_even_when_another_copy_answers(tmp_path: Path, health_server) -> None:
    """`health --pid` (start.sh) must not report success because some other copy answers on
    the port after the process it waits for has exited."""
    port = free_port()
    settings = make_settings(tmp_path, port=port)
    health_server(port, expected_host=f"127.0.0.1:{port}")
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    started = time.monotonic()
    assert cli.main(["health", "--wait", "20", "--pid", str(proc.pid), "--quiet"], settings=settings) == 3
    assert time.monotonic() - started < 5
    assert cli.main(["health", "--pid", str(os.getpid()), "--quiet"], settings=settings) == 0
    assert cli.main(["health", "--quiet"], settings=settings) == 0


def test_health_worker_heartbeat(s: Settings) -> None:
    seed(s)
    assert cli.main(["health", "--worker", "--quiet"], settings=s) == 1
    with db.session(s.db_path) as conn:
        conn.execute("INSERT INTO workers(id, pid, hostname, started_at, heartbeat_at) VALUES (?,?,?,?,?)",
                     (new_id("w"), os.getpid(), socket.gethostname(), db.now(), db.now()))
    assert cli.main(["health", "--worker", "--quiet"], settings=s) == 0


# ============================================================================ backup / restore
def test_backup_restore_round_trip(s: Settings, tmp_path: Path) -> None:
    ids = seed(s)
    out = tmp_path / "backups"
    backup = cli.create_backup(s, out, log=lambda m: None)
    assert backup.is_dir() and backup.name.startswith("clipper-backup-")
    manifest = json.loads((backup / "manifest.json").read_text())
    assert manifest["format"] == "local-clipper-backup"
    assert manifest["database"]["sha256"] == sha256_file(backup / "clipper.db")
    assert manifest["database"]["integrity_check"] == "ok"
    assert manifest["schema_version"] == db.SCHEMA_VERSION
    assert sorted(p.name for p in backup.iterdir()) == ["clipper.db", "manifest.json", "projects"]
    assert f"projects/{ids['project']}/source/source.mp4" in manifest["files"]
    assert not any(".tmp" in f for f in manifest["files"])  # in-progress atomic writes are skipped
    assert not list(out.glob("*.partial"))

    mutate(s, ids)
    assert project_titles(s) == ["Newer project"]
    assert cli.main(["restore", "--from", str(backup), "--yes"], settings=s) == 0

    assert project_titles(s) == ["Original project"]
    src = s.data_dir / "projects" / ids["project"] / "source" / "source.mp4"
    assert src.read_bytes() == b"\x00\x01source-media" * 1000
    assert not (s.data_dir / "projects" / "p_0123456789ab").exists()
    assert (s.data_dir / "logs" / "api.log").read_text() == "log line\n"   # logs stay in place
    assert any((s.data_dir / "cache" / "transcribe-chunk").iterdir())     # cache kept (not in backup)
    asides = list(tmp_path.glob("data.bak-*"))
    assert len(asides) == 1
    aside = asides[0]
    assert (aside / "projects" / "p_0123456789ab").is_dir()             # nothing was deleted
    assert (aside / "uploads").is_dir()
    with sqlite3.connect(aside / "clipper.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0] == 1
    with db.session(s.db_path) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert not list(s.data_dir.glob(".restore-*"))


def test_backup_archive_with_uploads_and_cache(s: Settings, tmp_path: Path) -> None:
    ids = seed(s)
    archive = cli.create_backup(s, tmp_path / "out", include_uploads=True, include_cache=True,
                                archive=True, log=lambda m: None)
    assert archive.name.endswith(".tar.gz") and tarfile.is_tarfile(archive)
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [archive.name]
    mutate(s, ids)
    shutil.rmtree(s.data_dir / "uploads")
    shutil.rmtree(s.data_dir / "cache")
    cli.restore_backup(s, archive, log=lambda m: None)
    assert project_titles(s) == ["Original project"]
    assert len(list((s.data_dir / "uploads").glob("*/pending.mp4"))) == 1
    assert (s.data_dir / "cache" / "transcribe-chunk" / ("ab" * 16 + ".json")).is_file()


def test_backup_includes_committed_wal_data(s: Settings, tmp_path: Path) -> None:
    seed(s)
    writer = db.connect(s.db_path)
    try:
        t = db.now()
        writer.execute("INSERT INTO projects(id, title, source_kind, options, status, created_at, updated_at) "
                       "VALUES (?, 'Only in WAL', 'upload', '{}', 'ready', ?, ?)", (new_id("p"), t, t))
        assert Path(str(s.db_path) + "-wal").stat().st_size > 0
        backup = cli.create_backup(s, tmp_path / "out", log=lambda m: None)
    finally:
        writer.close()
    assert not (backup / "clipper.db-wal").exists()
    with sqlite3.connect(backup / "clipper.db") as conn:
        titles = {r[0] for r in conn.execute("SELECT title FROM projects")}
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    assert titles == {"Original project", "Only in WAL"}


def test_backup_refuses_running_job_unless_forced(s: Settings, tmp_path: Path, capsys) -> None:
    ids = seed(s)
    with db.session(s.db_path) as conn:
        conn.execute("INSERT INTO jobs(id, project_id, kind, state, created_at) VALUES (?, ?, 'render', "
                     "'running', ?)", (new_id("j"), ids["project"], db.now()))
    assert cli.main(["backup", "--out", str(tmp_path / "out")], settings=s) == 3
    assert "running" in capsys.readouterr().err
    assert not (tmp_path / "out").exists() or not any((tmp_path / "out").iterdir())
    assert cli.main(["backup", "--out", str(tmp_path / "out"), "--force"], settings=s) == 0
    backup = next((tmp_path / "out").iterdir())
    assert json.loads((backup / "manifest.json").read_text())["forced"] is True


def test_backup_refuses_target_inside_data_dir_and_missing_db(s: Settings, capsys) -> None:
    assert cli.main(["backup", "--out", str(s.data_dir / "backups")], settings=s) == 1  # no db yet
    seed(s)
    assert cli.main(["backup", "--out", str(s.data_dir / "backups")], settings=s) == 3
    assert "outside the data folder" in capsys.readouterr().err


def test_restore_refuses_when_pid_file_points_at_live_process(s: Settings, tmp_path: Path, capsys) -> None:
    ids = seed(s)
    backup = cli.create_backup(s, tmp_path / "out", log=lambda m: None)
    mutate(s, ids)
    live = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        run_dir = s.data_dir / "run"
        run_dir.mkdir(exist_ok=True)
        (run_dir / "supervisor.pid").write_text(f"{live.pid}\n")
        assert cli.main(["restore", "--from", str(backup), "--yes"], settings=s) == 3
        err = capsys.readouterr().err
        assert "Restore refused" in err and str(live.pid) in err
        assert project_titles(s) == ["Newer project"]           # untouched
        assert not list(tmp_path.glob("data.bak-*"))
        assert not list(s.data_dir.glob(".restore-*"))
    finally:
        live.kill()
        live.wait()


def test_restore_refuses_when_app_answers_or_worker_alive(s: Settings, tmp_path: Path, health_server) -> None:
    seed(s)
    backup = cli.create_backup(s, tmp_path / "out", log=lambda m: None)
    with db.session(s.db_path) as conn:
        conn.execute("INSERT INTO workers(id, pid, hostname, started_at, heartbeat_at) VALUES (?,?,?,?,?)",
                     (new_id("w"), 999999, "elsewhere", db.now(), db.now()))
    with pytest.raises(cli.CliError, match="worker"):
        cli.restore_backup(s, backup, log=lambda m: None)
    health_server(s.port, expected_host=f"127.0.0.1:{s.port}")
    with pytest.raises(cli.CliError, match="answering"):
        cli.restore_backup(s, backup, log=lambda m: None)


def test_restore_rejects_damaged_backups(s: Settings, tmp_path: Path) -> None:
    seed(s)
    backup = cli.create_backup(s, tmp_path / "out", log=lambda m: None)
    before = project_titles(s)

    # 1) a listed file is missing
    victim = next(p for p in (backup / "projects").rglob("*.mp4"))
    saved = victim.read_bytes()
    victim.unlink()
    with pytest.raises(cli.CliError, match="incomplete"):
        cli.restore_backup(s, backup, log=lambda m: None)
    victim.write_bytes(saved)

    # 2) database checksum mismatch
    original_db = (backup / "clipper.db").read_bytes()
    with sqlite3.connect(backup / "clipper.db") as conn:
        conn.execute("UPDATE projects SET title='tampered'")
    with pytest.raises(cli.CliError, match="checksum"):
        cli.restore_backup(s, backup, log=lambda m: None)

    # 3) newer schema (checksum updated so only the schema check fails)
    with sqlite3.connect(backup / "clipper.db") as conn:
        conn.execute("UPDATE meta SET value='999' WHERE key='schema_version'")
    manifest = json.loads((backup / "manifest.json").read_text())
    manifest["database"]["sha256"] = sha256_file(backup / "clipper.db")
    (backup / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(cli.CliError, match="newer"):
        cli.restore_backup(s, backup, log=lambda m: None)

    # 4) not a backup at all
    not_a_backup = tmp_path / "random-folder"
    (not_a_backup / "a").mkdir(parents=True)
    (not_a_backup / "b").mkdir()
    with pytest.raises(cli.CliError, match="not a Local Clipper backup"):
        cli.restore_backup(s, not_a_backup, log=lambda m: None)

    assert project_titles(s) == before
    assert not list(tmp_path.glob("data.bak-*")) and not list(s.data_dir.glob(".restore-*"))
    (backup / "clipper.db").write_bytes(original_db)


def test_restore_rolls_back_when_finalising_fails(s: Settings, tmp_path: Path, monkeypatch) -> None:
    ids = seed(s)
    backup = cli.create_backup(s, tmp_path / "out", log=lambda m: None)
    mutate(s, ids)

    def boom(path):
        raise RuntimeError("simulated failure after the swap")

    monkeypatch.setattr(cli.db, "init_db", boom)
    with pytest.raises(RuntimeError, match="simulated"):
        cli.restore_backup(s, backup, log=lambda m: None)
    monkeypatch.undo()
    assert project_titles(s) == ["Newer project"]
    assert (s.data_dir / "projects" / "p_0123456789ab").is_dir()
    assert not list(tmp_path.glob("data.bak-*")) and not list(s.data_dir.glob(".restore-*"))


def test_restore_requires_confirmation_without_tty(s: Settings, tmp_path: Path, monkeypatch, capsys) -> None:
    seed(s)
    backup = cli.create_backup(s, tmp_path / "out", log=lambda m: None)
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    assert cli.main(["restore", "--from", str(backup)], settings=s) == 3
    assert "--yes" in capsys.readouterr().err


def test_backup_and_restore_scripts_with_relative_paths(s: Settings, tmp_path: Path) -> None:
    ids = seed(s)
    env = clean_env(CLIPPER_DATA_DIR=str(s.data_dir), CLIPPER_PORT=str(s.port), CLIPPER_AUTH="disabled")
    work = tmp_path / "cwd"
    work.mkdir()
    result = run_script("backup.sh", "--out", "my-backups", "--archive", env=env, cwd=work)
    assert result.returncode == 0, result.stderr
    archives = list((work / "my-backups").glob("clipper-backup-*.tar.gz"))
    assert len(archives) == 1
    mutate(s, ids)
    rel = os.path.relpath(archives[0], work)
    result = run_script("restore.sh", "--from", rel, "--yes", env=env, cwd=work)
    assert result.returncode == 0, result.stderr
    assert "moved to" in result.stdout
    assert project_titles(s) == ["Original project"]


# ============================================================================ config examples
def _env_names(text: str) -> set[str]:
    return set(re.findall(r"^#?\s*((?:CLIPPER_[A-Z0-9_]+)|OPENAI_API_KEY)=", text, re.M))


def test_env_example_documents_every_setting(tmp_path: Path) -> None:
    text = (REPO_ROOT / ".env.example").read_text(encoding="utf-8")
    expected = {("OPENAI_API_KEY" if name == "openai_api_key" else f"CLIPPER_{name.upper()}")
                for name in Settings.model_fields}
    assert expected - _env_names(text) == set()
    settings = Settings(_env_file=str(REPO_ROOT / ".env.example"))
    assert settings.mode == "local" and settings.host == "127.0.0.1" and settings.port == 8765
    assert not settings.openai_configured
    settings.validate_for_startup()


def test_hosted_env_example_refuses_placeholders(tmp_path: Path) -> None:
    example = (REPO_ROOT / "deploy" / "env.hosted.example").read_text(encoding="utf-8")
    settings = Settings(_env_file=str(REPO_ROOT / "deploy" / "env.hosted.example"))
    assert settings.mode == "hosted" and str(settings.data_dir) == "/data"
    assert settings.public_origin == "https://clipper.example.com"
    assert settings.trusted_proxies == "172.30.57.10"
    with pytest.raises(Exception) as excinfo:
        settings.validate_for_startup()
    assert "CLIPPER_OWNER_PASSWORD_HASH" in str(excinfo.value) and "CLIPPER_SESSION_SECRET" in str(excinfo.value)

    digest = PasswordHasher().hash("hosted owner password")
    filled = example.replace("'REPLACE_WITH_ARGON2_HASH'", f"'{digest}'").replace(
        "CLIPPER_SESSION_SECRET=\n", f"CLIPPER_SESSION_SECRET={'k' * 48}\n")
    path = tmp_path / ".env.hosted"
    path.write_text(filled, encoding="utf-8")
    ok = Settings(_env_file=str(path))
    ok.validate_for_startup()
    assert ok.owner_password_hash.get_secret_value() == digest
    # The proxy address is overridable (CLIPPER_PROXY_IP/SUBNET) since the subnet-collision fix,
    # so check the compose defaults rather than a literal address; they must match the example.
    compose = (REPO_ROOT / "deploy" / "compose.yaml").read_text(encoding="utf-8")
    proxy_ip = re.search(r"ipv4_address: \$\{CLIPPER_PROXY_IP:-([0-9.]+)\}", compose)
    subnet = re.search(r"subnet: \$\{CLIPPER_PROXY_SUBNET:-([0-9./]+)\}", compose)
    assert proxy_ip and subnet
    assert proxy_ip.group(1) == settings.trusted_proxies
    assert ipaddress.ip_address(proxy_ip.group(1)) in ipaddress.ip_network(subnet.group(1))
    assert f"# CLIPPER_PROXY_IP={proxy_ip.group(1)}\n" in example
    assert f"# CLIPPER_PROXY_SUBNET={subnet.group(1)}\n" in example


# ============================================================================ scripts
SCRIPT_NAMES = ["lib.sh", "setup.sh", "start.sh", "stop.sh", "doctor.sh", "hash-password.sh", "backup.sh",
                "restore.sh", "fetch-test-media.sh"]


@pytest.mark.parametrize("name", SCRIPT_NAMES)
def test_scripts_are_valid_bash(name: str) -> None:
    path = SCRIPTS / name
    assert path.is_file()
    for shell in ("bash", "/bin/bash"):
        if shutil.which(shell):
            result = subprocess.run([shell, "-n", str(path)], capture_output=True, text=True)
            assert result.returncode == 0, result.stderr
    text = path.read_text(encoding="utf-8")
    code = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
    assert not any(re.search(r"\bsudo\b", line) for line in code)
    if name != "lib.sh":
        assert text.startswith("#!/usr/bin/env bash\n") and "set -euo pipefail" in text
        assert os.access(path, os.X_OK)


@pytest.mark.skipif(shutil.which("shellcheck") is None, reason="shellcheck not installed")
def test_scripts_pass_shellcheck() -> None:
    result = subprocess.run(["shellcheck", "-x", "-P", str(SCRIPTS), *[str(SCRIPTS / n) for n in SCRIPT_NAMES]],
                            capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 0, result.stdout


def test_doctor_script_runs(tmp_path: Path) -> None:
    env = clean_env(CLIPPER_DATA_DIR=str(tmp_path / "data"), CLIPPER_PORT=str(free_port()),
                    CLIPPER_HW_ENCODE="off", OPENAI_API_KEY=SECRET_KEY)
    result = run_script("doctor.sh", env=env, cwd=tmp_path)
    assert "Local Clipper doctor" in result.stdout
    assert SECRET_KEY not in result.stdout + result.stderr
    assert result.returncode in (0, 1)


@pytest.mark.parametrize("shell", ["bash", "/bin/bash"])
def test_scripts_work_with_exported_cdpath(shell: str) -> None:
    """With CDPATH exported (".:~" in many profiles), `cd scripts/..` prints the directory; lib.sh
    used to capture that into CLIPPER_ROOT (twice, newline separated) and then claimed that
    backend/.venv was missing. The documented form `scripts/x.sh` from the repo root triggers it."""
    if not shutil.which(shell):
        pytest.skip(f"{shell} not available")
    env = clean_env(CDPATH=f".:{Path.home()}")
    result = subprocess.run([shell, "scripts/backup.sh", "--help"], cwd=REPO_ROOT, env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert "--out" in result.stdout and "missing" not in result.stderr


def _isolated_app_env(tmp_path: Path, port: int) -> dict[str, str]:
    """Environment for the start/stop scripts that can never reach the owner's real app."""
    return clean_env(CLIPPER_MODE="local", CLIPPER_HOST="127.0.0.1", CLIPPER_PORT=str(port),
                     CLIPPER_AUTH="disabled", CLIPPER_HW_ENCODE="off", CLIPPER_DATA_DIR=str(tmp_path / "data"),
                     OPENAI_API_KEY="")


def test_start_and_stop_scripts_do_not_mistake_another_copy(tmp_path: Path, health_server) -> None:
    """A copy of the app answers on the port but the PID file of the current settings does not
    point at it (e.g. CLIPPER_DATA_DIR was changed while it ran). stop.sh used to say "not
    running" (exit 0) and start.sh --detach used to report the other copy as its own success."""
    port = free_port()
    handler = health_server(port, expected_host=f"127.0.0.1:{port}")
    env = _isolated_app_env(tmp_path, port)

    stop = run_script("stop.sh", env=env, cwd=tmp_path)
    assert stop.returncode == 1, stop.stdout + stop.stderr
    assert "still answering" in stop.stderr and "not running" not in stop.stdout

    start = run_script("start.sh", "--detach", "--no-browser", env=env, cwd=tmp_path)
    assert start.returncode != 0, start.stdout + start.stderr
    assert "is running at" not in start.stdout
    assert "already answering" in start.stderr
    assert not (tmp_path / "data" / "run" / "supervisor.pid").exists()
    assert handler.seen_hosts  # the probes really reached the other copy


def test_start_detach_and_stop_scripts_round_trip(tmp_path: Path) -> None:
    """start.sh --detach succeeds only for the supervisor it launched (its PID file), and
    stop.sh stops it; afterwards stop.sh is idempotent."""
    port = free_port()
    env = _isolated_app_env(tmp_path, port)
    pid_path = tmp_path / "data" / "run" / "supervisor.pid"
    try:
        start = run_script("start.sh", "--detach", "--no-browser", env=env, cwd=tmp_path, timeout=90)
        assert start.returncode == 0, start.stdout + start.stderr
        assert f"is running at http://127.0.0.1:{port}" in start.stdout
        pid = int(pid_path.read_text(encoding="utf-8").strip())
        assert f"process {pid}" in start.stdout
        again = run_script("start.sh", "--detach", "--no-browser", env=env, cwd=tmp_path)
        assert again.returncode == 0 and "already running" in again.stdout
    finally:
        stop = run_script("stop.sh", env=env, cwd=tmp_path, timeout=90)
    assert stop.returncode == 0, stop.stdout + stop.stderr
    assert "Local Clipper stopped." in stop.stdout
    assert not pid_path.exists()
    assert cli.main(["health", "--quiet"], settings=make_settings(tmp_path, port=port)) == 1
    idle = run_script("stop.sh", env=env, cwd=tmp_path)
    assert idle.returncode == 0 and "not running" in idle.stdout


def test_setup_recreates_half_created_venv(tmp_path: Path) -> None:
    """Debian/Ubuntu without python3.12-venv leaves bin/python but no pip. setup.sh used to reuse
    that venv forever ("No module named pip"); now it recreates it, and reuses a working one."""
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    for name in ("lib.sh", "setup.sh"):
        shutil.copy2(SCRIPTS / name, root / "scripts" / name)
    venv = root / "backend" / ".venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)], check=True, timeout=120)
    marker = venv / "leftover.txt"
    marker.write_text("from the broken run", encoding="utf-8")
    assert subprocess.run([str(venv / "bin" / "python"), "-m", "pip", "--version"],
                          capture_output=True).returncode != 0
    # No package index: the run stops at "install uv" after the venv step, without network access.
    env = clean_env(CLIPPER_PYTHON=sys.executable, PIP_NO_INDEX="1", PIP_DISABLE_PIP_VERSION_CHECK="1")

    first = run_script_at(root / "scripts" / "setup.sh", "--prod", "--skip-frontend", env=env)
    assert "Using the existing backend/.venv" not in first.stdout, first.stdout + first.stderr
    assert "recreating it" in first.stdout
    assert not marker.exists()
    assert subprocess.run([str(venv / "bin" / "python"), "-m", "pip", "--version"],
                          capture_output=True).returncode == 0

    second = run_script_at(root / "scripts" / "setup.sh", "--prod", "--skip-frontend", env=env)
    assert "Using the existing backend/.venv" in second.stdout, second.stdout + second.stderr



def test_tar_restore_filter_drops_links(tmp_path):
    """Restoring a .tar.gz must never recreate symlinks/hard links inside the data folder."""
    import tarfile as _tarfile

    from clipper.cli import _data_without_links

    link = _tarfile.TarInfo("projects/p_0123456789ab/work")
    link.type = _tarfile.SYMTYPE
    link.linkname = "../../.."
    hard = _tarfile.TarInfo("clipper.db.lnk")
    hard.type = _tarfile.LNKTYPE
    hard.linkname = "clipper.db"
    regular = _tarfile.TarInfo("projects/p_0123456789ab/source/source.mp4")
    regular.size = 0
    assert _data_without_links(link, str(tmp_path)) is None
    assert _data_without_links(hard, str(tmp_path)) is None
    assert _data_without_links(regular, str(tmp_path)) is not None


def test_doctor_reports_why_sign_in_is_on():
    from clipper.cli import _sign_in_state
    from clipper.config import Settings

    assert _sign_in_state(Settings(_env_file=None)) == "off (loopback only)"
    on = _sign_in_state(Settings(_env_file=None, public_origin="https://clipper.example.com"))
    assert on.startswith("on (") and "CLIPPER_PUBLIC_ORIGIN" in on
