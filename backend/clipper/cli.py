"""Command-line tools: ``python -m clipper.cli <command>`` (wrapped by scripts/*.sh).

Commands
--------
doctor          environment checks as PASS/WARN/FAIL/INFO lines; exit 1 on any FAIL
hash-password   argon2 hash for CLIPPER_OWNER_PASSWORD_HASH (asks twice, or ``--stdin``)
check-config    validate the configuration and print a summary (secrets shown as set/not set)
health          probe ``GET /api/health`` (``--worker``: recent worker heartbeat instead)
paths           data/log/run paths and the URL as ``key=value`` lines (used by scripts)
serve-api       replace this process with the API server (container entry point)
backup          consistent copy of the database + projects (+ uploads / cache)
restore         restore a backup; the current data is moved aside, never deleted

Exit codes: 0 ok, 1 failure, 2 configuration error, 3 refused for safety.
Nothing here prints API keys, password hashes or the session secret.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import platform
import re
import shutil
import socket
import sqlite3
import sys
import tarfile
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable

from . import __version__, db
from .config import ASSETS_DIR, ConfigError, Settings, _is_loopback
from .errors import ClipperError
from .procs import run
from .storage import Storage, fmt_bytes, sha256_file
from .supervisor import (
    api_command,
    app_healthy,
    check_health,
    child_env,
    children_file,
    config_help,
    health_target,
    load_settings,
    log_dir,
    orphaned_children,
    pid_alive,
    pid_file,
    port_in_use,
    prepare,
    read_pid,
    run_dir,
)

EXIT_OK, EXIT_FAIL, EXIT_CONFIG, EXIT_REFUSED = 0, 1, 2, 3
PASS, WARN, FAIL, INFO = "PASS", "WARN", "FAIL", "INFO"

REQUIRED_FFMPEG = {
    "filters": ("overlay", "crop", "scale", "gblur", "silencedetect", "volumedetect", "afade"),
    "demuxers": ("concat",),
    "decoders": ("png",),
    "encoders": ("libx264", "aac"),
}
YUNET_FILE = ASSETS_DIR / "models" / "face_detection_yunet_2023mar.onnx"
YUNET_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
FONTS_DIR = ASSETS_DIR / "fonts"
FONT_FILES = ("Montserrat-ExtraBold.ttf", "Montserrat-Black.ttf")
MIN_DENO = (2, 3, 0)
MIN_PASSWORD_LENGTH = 10
MAX_PASSWORD_LENGTH = 1024
WORKER_ALIVE_SECONDS = 30.0
MB = 1024 * 1024

BACKUP_FORMAT = "local-clipper-backup"
BACKUP_FORMAT_VERSION = 1
DB_FILE = "clipper.db"
DB_SIDECARS = ("clipper.db-wal", "clipper.db-shm", "clipper.db-journal")
OPTIONAL_TREES = ("uploads", "cache")


class CliError(Exception):
    def __init__(self, message: str, exit_code: int = EXIT_FAIL) -> None:
        super().__init__(message)
        self.message = message
        self.exit_code = exit_code


Log = Callable[[str], None]


def _print(message: str) -> None:
    print(message, flush=True)


def _timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def _unique(path: Path) -> Path:
    if not path.exists():
        return path
    for i in range(2, 1000):
        candidate = path.with_name(f"{path.name}-{i}")
        if not candidate.exists():
            return candidate
    raise CliError("Could not find a free name in the target folder.")


def _free_backup_name(out_dir: Path, base: str) -> str:
    """``base`` or ``base-N`` such that no folder, archive or partial file uses it yet."""
    for i in range(1, 1000):
        name = base if i == 1 else f"{base}-{i}"
        if not any((out_dir / f"{name}{ext}").exists() for ext in ("", ".tar.gz", ".partial", ".tar.gz.partial")):
            return name
    raise CliError("Could not find a free name for the backup.")


# ============================================================================ doctor
@dataclass
class Check:
    status: str
    name: str
    detail: str


def _which(binary: str) -> str | None:
    return shutil.which(os.path.expanduser(binary))


def _version_line(argv: list[str]) -> str | None:
    try:
        result = run(argv, timeout=20, check=False, error_message="version check failed")
    except (ClipperError, OSError, ValueError):
        return None
    if result.returncode != 0:
        return None
    for line in result.stdout.decode("utf-8", "replace").splitlines():
        if line.strip():
            return line.strip()
    return None


def parse_ffmpeg_list(text: str) -> set[str]:
    """Names from ``ffmpeg -filters/-demuxers/-decoders/-encoders`` output (after the ``---`` line)."""
    names: set[str] = set()
    started = False
    for line in text.splitlines():
        stripped = line.strip()
        if not started:
            started = bool(stripped) and set(stripped) == {"-"}
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        idx = 2 if len(parts) > 2 and parts[1] in ("d", "E", ".") else 1  # optional device/mux flag column
        names.update(n for n in parts[idx].split(",") if n)
    return names


def ffmpeg_capabilities(ffmpeg: str) -> dict[str, set[str]]:
    caps: dict[str, set[str]] = {}
    for kind in REQUIRED_FFMPEG:
        result = run([ffmpeg, "-hide_banner", f"-{kind}"], timeout=30, check=False,
                     error_message="ffmpeg capability listing failed")
        caps[kind] = parse_ffmpeg_list(result.stdout.decode("utf-8", "replace"))
    return caps


def missing_ffmpeg_features(caps: dict[str, set[str]]) -> list[str]:
    return [f"{kind[:-1]} '{name}'" for kind, names in REQUIRED_FFMPEG.items()
            for name in names if name not in caps.get(kind, set())]


def check_python() -> Check:
    v = sys.version_info
    version = f"{v.major}.{v.minor}.{v.micro}"
    if (3, 12) <= (v.major, v.minor) < (3, 14):
        return Check(PASS, "Python", version)
    return Check(FAIL, "Python", f"{version}: Python 3.12 is required (run scripts/setup.sh)")


def check_ffmpeg_tools(settings: Settings) -> tuple[list[Check], str | None]:
    checks: list[Check] = []
    ffmpeg_path: str | None = None
    for label, var, binary in (("FFmpeg", "CLIPPER_FFMPEG", settings.ffmpeg),
                               ("FFprobe", "CLIPPER_FFPROBE", settings.ffprobe)):
        found = _which(binary)
        if not found:
            checks.append(Check(FAIL, label, f"'{binary}' was not found. Install FFmpeg (macOS: brew install "
                                             f"ffmpeg; Debian/Ubuntu: apt install ffmpeg) or set {var}."))
            continue
        version = _version_line([found, "-version"])
        if version is None:
            checks.append(Check(FAIL, label, f"'{binary}' was found but did not run"))
            continue
        checks.append(Check(PASS, label, version))
        if label == "FFmpeg":
            ffmpeg_path = found
    return checks, ffmpeg_path


def check_ffmpeg_features(caps: dict[str, set[str]]) -> Check:
    missing = missing_ffmpeg_features(caps)
    if missing:
        return Check(FAIL, "FFmpeg features", "missing " + ", ".join(missing) + ": install a full FFmpeg build")
    names = [n for group in REQUIRED_FFMPEG.values() for n in group]
    return Check(PASS, "FFmpeg features", ", ".join(names[:7]) + ", concat demuxer, png decoder, libx264, aac")


def check_hw_encoder(settings: Settings, encoders: set[str] | None) -> Check:
    name = "Hardware encoder"
    if platform.system() != "Darwin":
        return Check(INFO, name, "VideoToolbox is macOS-only; CPU encoding (libx264) is used")
    if encoders is not None and "h264_videotoolbox" not in encoders:
        return Check(INFO, name, "h264_videotoolbox is not in this FFmpeg build; CPU encoding (libx264) is used")
    if settings.hw_encode == "off":
        return Check(INFO, name, "disabled by CLIPPER_HW_ENCODE=off; CPU encoding (libx264) is used")
    from .media.encoders import choose_encoder

    choice = choose_encoder(settings)
    if choice.hardware:
        return Check(INFO, name, "h264_videotoolbox passed a real encode/decode test and will be used")
    return Check(INFO, name, "h264_videotoolbox failed the encode test; CPU encoding (libx264) is used")


def required_ejs_version() -> str | None:
    for req in metadata.requires("yt-dlp") or []:
        match = re.match(r"^\s*yt-dlp-ejs\s*==\s*([0-9][\w.]*)", req)
        if match:
            return match.group(1)
    return None


def check_ytdlp() -> Check:
    try:
        ytdlp = metadata.version("yt-dlp")
    except metadata.PackageNotFoundError:
        return Check(FAIL, "yt-dlp", "not installed (run scripts/setup.sh)")
    try:
        ejs: str | None = metadata.version("yt-dlp-ejs")
    except metadata.PackageNotFoundError:
        ejs = None
    wanted = required_ejs_version()
    if ejs is None:
        return Check(FAIL, "yt-dlp", f"yt-dlp {ytdlp} is installed without yt-dlp-ejs; YouTube imports will "
                                     "fail (run scripts/setup.sh)")
    if wanted and ejs != wanted:
        return Check(FAIL, "yt-dlp", f"yt-dlp {ytdlp} needs yt-dlp-ejs {wanted} but {ejs} is installed "
                                     "(run scripts/setup.sh)")
    return Check(PASS, "yt-dlp", f"yt-dlp {ytdlp} with yt-dlp-ejs {ejs}")


def check_js_runtime(settings: Settings) -> Check:
    runtime, _, path = settings.ytdlp_js_runtime.partition(":")
    runtime = runtime.strip().lower() or "deno"
    if runtime != "deno":
        return Check(INFO, "JS runtime", f"'{runtime}' configured by CLIPPER_YTDLP_JS_RUNTIME (version not checked)")
    hint = "YouTube imports need deno 2.3 or newer (macOS: brew install deno; see https://deno.com). " \
           "Uploaded files still work."
    found = _which(path.strip() or "deno")
    if not found:
        return Check(WARN, "JS runtime", f"deno was not found. {hint}")
    line = _version_line([found, "--version"]) or ""
    match = re.search(r"deno (\d+)\.(\d+)\.(\d+)", line)
    if not match:
        return Check(WARN, "JS runtime", f"could not read the deno version. {hint}")
    version = tuple(int(x) for x in match.groups())
    text = ".".join(match.groups())
    if version < MIN_DENO:
        return Check(WARN, "JS runtime", f"deno {text} is too old. {hint}")
    return Check(PASS, "JS runtime", f"deno {text}")


def check_yunet() -> Check:
    if not YUNET_FILE.is_file():
        return Check(WARN, "Face model", f"{YUNET_FILE.name} is missing: framing falls back to a centre crop")
    if sha256_file(YUNET_FILE) != YUNET_SHA256:
        return Check(FAIL, "Face model", f"{YUNET_FILE.name} has an unexpected checksum (damaged or replaced); "
                                         "restore it from the repository")
    return Check(PASS, "Face model", f"{YUNET_FILE.name} (checksum OK)")


def check_fonts() -> Check:
    missing = [f for f in (*FONT_FILES, "OFL.txt") if not (FONTS_DIR / f).is_file()]
    if missing:
        return Check(FAIL, "Caption fonts", "missing " + ", ".join(missing) + " in clipper/assets/fonts")
    from PIL import ImageFont

    for name in FONT_FILES:
        try:
            ImageFont.truetype(str(FONTS_DIR / name), 48)
        except OSError:
            return Check(FAIL, "Caption fonts", f"{name} cannot be loaded (damaged file)")
    return Check(PASS, "Caption fonts", "Montserrat ExtraBold + Black (SIL OFL 1.1)")


def _existing_ancestor(path: Path) -> Path:
    p = Path(path).absolute()
    while not p.exists() and p != p.parent:
        p = p.parent
    return p


def check_data_dir(settings: Settings) -> Check:
    data_dir = Path(settings.data_dir)
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=data_dir, prefix=".doctor-")
        os.close(fd)
        os.unlink(tmp)
    except OSError as exc:
        return Check(FAIL, "Data folder", f"{data_dir} is not writable ({exc.strerror or type(exc).__name__})")
    return Check(PASS, "Data folder", f"{data_dir} is writable")


def check_disk(settings: Settings) -> Check:
    free = shutil.disk_usage(_existing_ancestor(Path(settings.data_dir))).free
    reserve = settings.min_free_disk_mb * MB
    if free < reserve:
        return Check(FAIL, "Free disk", f"{fmt_bytes(free)} free, below CLIPPER_MIN_FREE_DISK_MB "
                                        f"({settings.min_free_disk_mb} MB). Free some space.")
    if free < reserve + 10 * 1024 * MB:
        return Check(WARN, "Free disk", f"{fmt_bytes(free)} free: long videos and renders can need several GB")
    return Check(PASS, "Free disk", f"{fmt_bytes(free)} free (reserve {settings.min_free_disk_mb} MB)")


def check_database(settings: Settings) -> Check:
    path = Path(settings.db_path)
    if not path.exists():
        return Check(INFO, "Database", "not created yet (created on first start)")
    conn = sqlite3.connect(str(path), timeout=10)
    try:
        row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        version = int(row[0]) if row else 0
        quick = conn.execute("PRAGMA quick_check").fetchone()[0]
    except sqlite3.DatabaseError as exc:
        return Check(FAIL, "Database", f"cannot be read ({exc}); restore a backup (scripts/restore.sh)")
    finally:
        conn.close()
    if version > db.SCHEMA_VERSION:
        return Check(FAIL, "Database", f"schema v{version} is newer than this app (v{db.SCHEMA_VERSION}): "
                                       "install the matching version or restore an older backup")
    if quick != "ok":
        return Check(FAIL, "Database", "integrity check failed; restore a backup (scripts/restore.sh)")
    return Check(PASS, "Database", f"schema v{version}, quick_check ok")


def check_openai(settings: Settings) -> Check:
    if settings.openai_configured:
        return Check(PASS, "OpenAI API key", "configured (value not shown)")
    return Check(WARN, "OpenAI API key", "not configured: add OPENAI_API_KEY to .env to analyse videos "
                                         "(the app starts without it)")


def _data_without_links(member: tarfile.TarInfo, dest: str) -> tarfile.TarInfo | None:
    """tarfile "data" filter that also drops symlinks and hard links (a backup never contains any;
    a restored link could later make deletions or media reads escape the data folder)."""
    if member.issym() or member.islnk():
        return None
    return tarfile.data_filter(member, dest)


def _sign_in_state(settings: Settings) -> str:
    if settings.auth_enabled:
        reasons = getattr(settings, "exposure_reasons", None) or []
        if settings.mode == "hosted":
            return "on (hosted mode)"
        if settings.auth == "required":
            return "on (CLIPPER_AUTH=required)"
        return "on (" + ", ".join(reasons) + ")" if reasons else "on"
    if settings.auth == "disabled":
        return "off (CLIPPER_AUTH=disabled, loopback only)"
    return "off (loopback only)"


def check_config(settings: Settings) -> list[Check]:
    try:
        settings.validate_for_startup()
    except ConfigError as exc:
        return [Check(FAIL, "Configuration", str(exc))]
    auth = _sign_in_state(settings)
    checks = [Check(PASS, "Configuration", f"mode {settings.mode}, sign-in {auth}, {settings.effective_public_origin}")]
    if settings.mode == "hosted" and not settings.trusted_proxies.strip():
        checks.append(Check(WARN, "Proxy headers", "CLIPPER_TRUSTED_PROXIES is empty: every client appears with "
                                                   "the proxy's address (shared login throttling)"))
    elif (settings.auth_enabled and not settings.trusted_proxies.strip() and _is_loopback(settings.host)
          and settings.exposure_reasons):
        # Local mode, loopback bind, sign-in on for a non-loopback name: that name can only arrive
        # through a proxy or tunnel on this computer, which connects from loopback for every client.
        loop = "::1" if ":" in settings.host else "127.0.0.1"
        checks.append(Check(WARN, "Proxy headers", (
            f"sign-in is on for {', '.join(settings.exposure_reasons)} but CLIPPER_TRUSTED_PROXIES is empty: "
            f"through a proxy or tunnel on this computer every client appears as {loop} and shares one "
            f"login throttle, so anyone can lock the owner out. Set CLIPPER_TRUSTED_PROXIES={loop} "
            "(the proxy must pass the client's address in X-Forwarded-For).")))
    return checks


def check_frontend(settings: Settings) -> Check:
    if (Path(settings.frontend_dist) / "index.html").is_file():
        return Check(PASS, "Web interface", "frontend/dist is built")
    return Check(WARN, "Web interface", "not built: run scripts/setup.sh (a placeholder page is shown until then)")


def check_port(settings: Settings) -> Check:
    where = f"{settings.host}:{settings.port}"
    if not port_in_use(settings.host, settings.port):
        return Check(PASS, "Port", f"{where} is free")
    if app_healthy(settings):
        return Check(PASS, "Port", f"{where} is used by a running Local Clipper")
    return Check(FAIL, "Port", f"{where} is used by another program: set CLIPPER_PORT in .env")


def _guarded(name: str, fn: Callable[[], Check | list[Check]]) -> list[Check]:
    try:
        result = fn()
    except Exception as exc:  # one broken check must not hide the others
        return [Check(FAIL, name, f"check failed unexpectedly ({type(exc).__name__})")]
    return result if isinstance(result, list) else [result]


def run_doctor(settings: Settings, *, load_error: str | None = None) -> list[Check]:
    checks: list[Check] = [check_python()]
    tools, ffmpeg_path = check_ffmpeg_tools(settings)
    checks += tools
    encoders: set[str] | None = None
    if ffmpeg_path:
        caps: dict[str, set[str]] = {}

        def features() -> Check:
            caps.update(ffmpeg_capabilities(ffmpeg_path))
            return check_ffmpeg_features(caps)

        checks += _guarded("FFmpeg features", features)
        encoders = caps.get("encoders")
        checks += _guarded("Hardware encoder", lambda: check_hw_encoder(settings, encoders))
    checks += _guarded("yt-dlp", check_ytdlp)
    checks += _guarded("JS runtime", lambda: check_js_runtime(settings))
    checks += _guarded("Face model", check_yunet)
    checks += _guarded("Caption fonts", check_fonts)
    checks += _guarded("Data folder", lambda: check_data_dir(settings))
    checks += _guarded("Free disk", lambda: check_disk(settings))
    checks += _guarded("Database", lambda: check_database(settings))
    checks += _guarded("OpenAI API key", lambda: check_openai(settings))
    if load_error:
        checks.append(Check(FAIL, "Configuration", load_error))
    else:
        checks += _guarded("Configuration", lambda: check_config(settings))
    checks += _guarded("Web interface", lambda: check_frontend(settings))
    checks += _guarded("Port", lambda: check_port(settings))
    return checks


def format_checks(checks: Iterable[Check]) -> str:
    items = list(checks)
    width = max((len(c.name) for c in items), default=10)
    lines: list[str] = []
    for c in items:
        first, *rest = c.detail.splitlines() or [""]
        lines.append(f"  {c.status:<4}  {c.name:<{width}}  {first}")
        lines += [" " * (width + 10) + r.strip() for r in rest if r.strip()]
    counts = {s: sum(1 for c in items if c.status == s) for s in (PASS, WARN, FAIL)}
    lines.append("")
    lines.append(f"{counts[PASS]} passed, {counts[WARN]} warning(s), {counts[FAIL]} failed.")
    return "\n".join(lines)


def cmd_doctor(_args: argparse.Namespace, settings: Settings | None) -> int:
    load_error = None
    if settings is None:
        try:
            settings = load_settings()
        except ConfigError as exc:
            load_error = str(exc)
            settings = Settings.model_construct()  # defaults, so the other checks still run
    _print("Local Clipper doctor\n")
    checks = run_doctor(settings, load_error=load_error)
    _print(format_checks(checks))
    return EXIT_FAIL if any(c.status == FAIL for c in checks) else EXIT_OK


# ============================================================================ hash-password
def _read_password(use_stdin: bool) -> str:
    if use_stdin:
        line = sys.stdin.readline()
        if not line:
            raise CliError("No password was given on standard input.")
        return line.rstrip("\r\n")
    first = getpass.getpass("New owner password: ")
    second = getpass.getpass("Repeat the password: ")
    if first != second:
        raise CliError("The two passwords do not match. Nothing was changed.")
    return first


def hash_password(password: str) -> str:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise CliError(f"The password must have at least {MIN_PASSWORD_LENGTH} characters.")
    if len(password) > MAX_PASSWORD_LENGTH:
        raise CliError(f"The password must have at most {MAX_PASSWORD_LENGTH} characters.")
    from argon2 import PasswordHasher

    return PasswordHasher().hash(password)


def cmd_hash_password(args: argparse.Namespace, _settings: Settings | None) -> int:
    digest = hash_password(_read_password(args.stdin))
    print(digest, flush=True)
    if not args.stdin:
        print("\nPut this line in .env (keep the single quotes; the hash contains $ signs):\n"
              "CLIPPER_OWNER_PASSWORD_HASH='<the hash above>'", file=sys.stderr)
    return EXIT_OK


# ============================================================================ check-config / paths / health
def _settings_or_exit(settings: Settings | None) -> Settings:
    if settings is not None:
        return settings
    try:
        return load_settings()
    except ConfigError as exc:
        raise CliError(config_help(str(exc)), EXIT_CONFIG) from None


def _set(value: object) -> str:
    secret = getattr(value, "get_secret_value", None)
    text = secret() if callable(secret) else value
    return "set" if text else "not set"


def cmd_check_config(_args: argparse.Namespace, settings: Settings | None) -> int:
    s = _settings_or_exit(settings)
    try:
        s.validate_for_startup()
    except ConfigError as exc:
        raise CliError(config_help(str(exc)), EXIT_CONFIG) from None
    rows = [
        ("Mode", s.mode),
        ("Listens on", f"{s.host}:{s.port}"),
        ("Public address", s.effective_public_origin),
        ("Data folder", str(s.data_dir)),
        ("Sign-in", _sign_in_state(s)),
        ("Owner password hash", _set(s.owner_password_hash)),
        ("Session secret", _set(s.session_secret)),
        ("OpenAI API key", "configured" if s.openai_configured else "not configured"),
        ("Models", f"transcribe {s.transcribe_model}, select {s.select_model}, review {s.review_model}"),
        ("Upload limit", f"{s.max_upload_mb} MB"),
        ("Trusted proxies", s.trusted_proxies or "none"),
    ]
    width = max(len(k) for k, _ in rows)
    for key, value in rows:
        _print(f"  {key:<{width}}  {value}")
    for check in check_config(s):
        if check.status == WARN:
            _print(f"\nWarning ({check.name}): {check.detail}")
    _print("\nConfiguration OK.")
    return EXIT_OK


def cmd_paths(_args: argparse.Namespace, settings: Settings | None) -> int:
    s = _settings_or_exit(settings)
    values = {
        "data_dir": Path(s.data_dir).resolve(),
        "log_dir": log_dir(s).resolve(),
        "run_dir": run_dir(s).resolve(),
        "pid_file": pid_file(s).resolve(),
        "children_file": children_file(s).resolve(),
        "url": s.effective_public_origin,
    }
    for key, value in values.items():
        _print(f"{key}={value}")
    return EXIT_OK


def worker_recently_alive(settings: Settings, *, this_host_only: bool = False,
                          within: float = WORKER_ALIVE_SECONDS) -> float | None:
    """Seconds since the most recent worker heartbeat, if within ``within`` seconds."""
    path = Path(settings.db_path)
    if not path.exists():
        return None
    conn = sqlite3.connect(str(path), timeout=10)
    try:
        sql, params = "SELECT MAX(heartbeat_at) FROM workers", ()
        if this_host_only:
            sql, params = sql + " WHERE hostname=?", (socket.gethostname(),)
        row = conn.execute(sql, params).fetchone()
    except sqlite3.DatabaseError:
        return None
    finally:
        conn.close()
    if not row or not row[0]:
        return None
    age = time.time() - float(row[0])
    return age if age <= within else None


def cmd_health(args: argparse.Namespace, settings: Settings | None) -> int:
    s = _settings_or_exit(settings)
    if args.worker:
        alive = worker_recently_alive(s, this_host_only=True) is not None
        if not args.quiet:
            _print("worker ok" if alive else "worker has not reported in the last 30 s")
        return EXIT_OK if alive else EXIT_FAIL
    url, host = health_target(s)
    deadline = time.monotonic() + max(0.0, args.wait)
    while True:
        healthy = check_health(url, host, timeout=3.0)
        # With --pid, an answer only counts while that process is alive: once it has exited,
        # whatever answers is another copy (e.g. one started with a different data folder).
        exited = bool(args.pid) and not pid_alive(args.pid)
        if healthy and not exited:
            if not args.quiet:
                _print(f"ok: {s.effective_public_origin}")
            return EXIT_OK
        if exited:
            if not args.quiet:
                _print("the server process exited before it became healthy")
            return EXIT_REFUSED
        if time.monotonic() >= deadline:
            if not args.quiet:
                _print("not answering")
            return EXIT_FAIL
        time.sleep(0.5)


def cmd_serve_api(_args: argparse.Namespace, settings: Settings | None) -> int:
    s = _settings_or_exit(settings)
    try:
        prepare(s)
    except ConfigError as exc:
        raise CliError(config_help(str(exc)), EXIT_CONFIG) from None
    except OSError as exc:
        raise CliError(f"Cannot prepare the data folder ({exc.strerror or type(exc).__name__}).") from None
    argv = api_command(s)
    sys.stdout.flush()
    sys.stderr.flush()
    os.execve(argv[0], argv, child_env(s))
    return EXIT_FAIL  # pragma: no cover - execve does not return


# ============================================================================ running-app detection
def app_running_reason(settings: Settings) -> str | None:
    """Why the app looks like it is running, or None. Conservative on purpose."""
    pid = read_pid(pid_file(settings))
    if pid is not None and pid != os.getpid() and pid_alive(pid):
        return (f"Local Clipper appears to be running (process {pid} from data/run/supervisor.pid). "
                "Stop it with scripts/stop.sh. If it is not running, delete data/run/supervisor.pid.")
    orphans = orphaned_children(settings)
    if orphans:
        return (f"Processes from an earlier run are still active ({', '.join(sorted(orphans))}). "
                "Run scripts/stop.sh first.")
    if app_healthy(settings, timeout=1.0):
        return f"Local Clipper is answering at {settings.effective_public_origin}. Stop it first."
    age = worker_recently_alive(settings)
    if age is not None:
        return (f"A worker updated the database {int(age)} s ago (is it running in another terminal or "
                "container?). Stop it first.")
    return None


# ============================================================================ backup
def _open_db(path: Path) -> sqlite3.Connection:
    """Read-only connection when possible (backups may sit on read-only media)."""
    conn: sqlite3.Connection | None = None
    try:
        conn = sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True, timeout=30)
        conn.execute("SELECT COUNT(*) FROM sqlite_master").fetchone()  # read-only WAL may fail lazily
        return conn
    except sqlite3.OperationalError:
        if conn is not None:
            conn.close()
        return sqlite3.connect(str(path), timeout=30)


def _job_counts(db_path: Path) -> tuple[int, int]:
    conn = _open_db(db_path)
    try:
        running = conn.execute("SELECT COUNT(*) FROM jobs WHERE state='running'").fetchone()[0]
        queued = conn.execute("SELECT COUNT(*) FROM jobs WHERE state='queued'").fetchone()[0]
    except sqlite3.DatabaseError as exc:
        raise CliError(f"The database cannot be read ({exc}). Run scripts/doctor.sh.") from None
    finally:
        conn.close()
    return int(running), int(queued)


def _db_facts(path: Path) -> tuple[int, str]:
    """(schema_version, integrity_check result) of a database file."""
    conn = _open_db(path)
    try:
        row = conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        integrity = conn.execute("PRAGMA integrity_check").fetchone()[0]
    except sqlite3.DatabaseError as exc:
        raise CliError(f"The database copy cannot be read ({exc}).") from None
    finally:
        conn.close()
    return (int(row[0]) if row else 0), str(integrity)


def snapshot_database(src: Path, dst: Path) -> None:
    """Consistent single-file copy via the SQLite online backup API (safe while in use)."""
    source = sqlite3.connect(str(src), timeout=30)
    target = sqlite3.connect(str(dst))
    try:
        source.execute("PRAGMA busy_timeout=30000")
        source.backup(target)  # pages=-1: copied in one step under one read transaction
        target.execute("PRAGMA journal_mode=DELETE")  # self-contained file, no -wal needed
    finally:
        target.close()
        source.close()
    # The copy briefly ran in WAL mode; switching to DELETE checkpointed it, leaving only a stale -shm.
    wal = Path(f"{dst}-wal")
    if wal.exists() and wal.stat().st_size > 0:
        raise CliError("The database copy could not be finalised (write-ahead log not merged).")
    for suffix in ("-wal", "-shm"):
        Path(f"{dst}{suffix}").unlink(missing_ok=True)


def _ignore_scratch(directory: str, names: list[str]) -> set[str]:
    """Skip symlinks and in-progress atomic-write temp files (``.name.*.tmp*``)."""
    skip = set()
    for name in names:
        if os.path.islink(os.path.join(directory, name)) or (name.startswith(".") and ".tmp" in name):
            skip.add(name)
    return skip


def _list_files(root: Path, top: Iterable[str]) -> dict[str, int]:
    files: dict[str, int] = {}
    for name in top:
        base = root / name
        if base.is_file():
            files[name] = base.stat().st_size
            continue
        for dirpath, _dirs, filenames in os.walk(base):
            for f in filenames:
                p = Path(dirpath) / f
                files[p.relative_to(root).as_posix()] = p.stat().st_size
    return dict(sorted(files.items()))


def create_backup(settings: Settings, out_dir: Path, *, include_uploads: bool = False,
                  include_cache: bool = False, archive: bool = False, force: bool = False,
                  log: Log = _print) -> Path:
    data_dir = Path(settings.data_dir).resolve()
    db_path = data_dir / DB_FILE
    if not db_path.is_file():
        raise CliError("There is no database in the data folder yet, so there is nothing to back up.")
    out_dir = Path(out_dir).expanduser().resolve()
    if out_dir == data_dir or out_dir.is_relative_to(data_dir):
        raise CliError("Choose a backup folder outside the data folder (ideally on another disk).", EXIT_REFUSED)

    running, queued = _job_counts(db_path)
    if running and not force:
        raise CliError(f"{running} job(s) are running. Wait until processing/rendering has finished (or cancel "
                       "it), then run the backup again. Use --force to back up anyway; in-progress work "
                       "may then be incomplete in the copy.", EXIT_REFUSED)
    if running:
        log(f"Warning: {running} job(s) are running; their in-progress files may be incomplete in this backup.")
    if queued:
        log(f"Note: {queued} job(s) are queued and may start during the backup.")
    if app_healthy(settings, timeout=1.0):
        log("Note: the app is running. That is fine while no job is running.")

    out_dir.mkdir(parents=True, exist_ok=True)
    name = _free_backup_name(out_dir, f"clipper-backup-{_timestamp()}")
    final = out_dir / (f"{name}.tar.gz" if archive else name)
    staging = out_dir / f"{name}.partial"
    partial_archive = out_dir / f"{name}.tar.gz.partial"
    staging.mkdir()
    try:
        snapshot_database(db_path, staging / DB_FILE)
        schema_version, integrity = _db_facts(staging / DB_FILE)
        if integrity != "ok":
            raise CliError("The database failed its integrity check, so no backup was written. "
                           "Run scripts/doctor.sh for details.")
        contents = [DB_FILE]
        trees = ["projects"] + [t for t, on in (("uploads", include_uploads), ("cache", include_cache)) if on]
        for tree in trees:
            src = data_dir / tree
            if src.is_dir():
                shutil.copytree(src, staging / tree, symlinks=True, ignore=_ignore_scratch)
                contents.append(tree)
        files = _list_files(staging, contents)
        manifest = {
            "format": BACKUP_FORMAT,
            "format_version": BACKUP_FORMAT_VERSION,
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "app_version": __version__,
            "schema_version": schema_version,
            "database": {"file": DB_FILE, "sha256": sha256_file(staging / DB_FILE),
                         "size": (staging / DB_FILE).stat().st_size, "integrity_check": integrity},
            "contents": contents,
            "running_jobs_at_backup": running,
            "forced": bool(force and running),
            "total_bytes": sum(files.values()),
            "files": files,
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if archive:
            with tarfile.open(partial_archive, "w:gz", compresslevel=1) as tar:
                tar.add(staging, arcname=name)
            os.replace(partial_archive, final)
            shutil.rmtree(staging)
        else:
            os.replace(staging, final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        partial_archive.unlink(missing_ok=True)
        raise
    log(f"Backup written: {final}")
    log(f"  {len(files)} file(s), {fmt_bytes(manifest['total_bytes'])}; database sha256 "
        f"{manifest['database']['sha256'][:16]}…")
    log("Keep a copy on another disk or machine.")
    return final


# ============================================================================ restore
def _safe_rel(rel: str) -> bool:
    p = PurePosixPath(rel)
    return bool(rel) and not p.is_absolute() and ".." not in p.parts


def _find_backup_root(base: Path) -> Path | None:
    if (base / "manifest.json").is_file():
        return base
    subdirs = [p for p in base.iterdir() if p.is_dir()] if base.is_dir() else []
    if len(subdirs) == 1 and (subdirs[0] / "manifest.json").is_file():
        return subdirs[0]
    return None


def verify_backup(root: Path, *, check_files: bool = True) -> dict:
    """Validate manifest, database checksum/integrity/schema and file sizes. Raises CliError."""
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise CliError("This is not a Local Clipper backup (manifest.json is missing or unreadable).") from None
    if not isinstance(manifest, dict) or manifest.get("format") != BACKUP_FORMAT:
        raise CliError("This is not a Local Clipper backup (unknown manifest format).")
    if int(manifest.get("format_version", 0)) > BACKUP_FORMAT_VERSION:
        raise CliError("This backup was made by a newer version of Local Clipper.")
    db_copy = root / DB_FILE
    expected = str((manifest.get("database") or {}).get("sha256", ""))
    if not db_copy.is_file() or sha256_file(db_copy) != expected:
        raise CliError("The backup's database does not match its checksum: the backup is damaged.")
    schema_version, integrity = _db_facts(db_copy)
    if integrity != "ok":
        raise CliError("The backup's database failed its integrity check: the backup is damaged.")
    if schema_version > db.SCHEMA_VERSION:
        raise CliError(f"The backup uses database schema v{schema_version}, newer than this app "
                       f"(v{db.SCHEMA_VERSION}). Install the matching version first.")
    if check_files:
        files = manifest.get("files") or {}
        bad = [rel for rel, size in files.items()
               if not _safe_rel(rel) or not (root / rel).is_file() or (root / rel).stat().st_size != size]
        if bad:
            raise CliError(f"The backup is incomplete: {len(bad)} file(s) are missing or have the wrong size.")
    return manifest


def _stage_backup(source: Path, staging: Path) -> tuple[Path, dict]:
    """Copy/extract the backup into ``staging`` (same disk as the data) and verify it."""
    if source.is_dir():
        root = _find_backup_root(source)
        if root is None:
            raise CliError("This folder is not a Local Clipper backup (manifest.json is missing).")
        manifest = verify_backup(root, check_files=False)  # cheap pre-check before copying media
        for name in manifest.get("contents", []):
            if name not in (DB_FILE, "projects", *OPTIONAL_TREES):
                continue
            src = root / name
            if src.is_dir():
                shutil.copytree(src, staging / name, symlinks=True, ignore=_ignore_scratch)
            elif src.is_file() and not src.is_symlink():
                shutil.copy2(src, staging / name)
        shutil.copy2(root / "manifest.json", staging / "manifest.json")
        staged = staging
    elif source.is_file() and tarfile.is_tarfile(source):
        with tarfile.open(source, "r:*") as tar:
            tar.extractall(staging, filter=_data_without_links)
        found = _find_backup_root(staging)
        if found is None:
            raise CliError("This archive is not a Local Clipper backup (manifest.json is missing).")
        staged = found
    else:
        raise CliError("The backup was not found or is not a backup folder / .tar.gz file.")
    return staged, verify_backup(staged)


def _aside_dir(data_dir: Path, ts: str) -> Path:
    """Sibling ``<data>.bak-<ts>`` when on the same disk and not a mount point, else inside the data dir."""
    parent = data_dir.parent
    try:
        same_disk = os.stat(parent).st_dev == os.stat(data_dir).st_dev
    except OSError:
        same_disk = False
    if same_disk and not os.path.ismount(data_dir) and os.access(parent, os.W_OK):
        return _unique(parent / f"{data_dir.name}.bak-{ts}")
    return _unique(data_dir / f"data.bak-{ts}")


def _move_aside(data_dir: Path, names: Iterable[str], ts: str) -> Path | None:
    present = [n for n in names if (data_dir / n).exists() or (data_dir / n).is_symlink()]
    if not present:
        return None
    aside = _aside_dir(data_dir, ts)
    aside.mkdir(parents=True)
    for name in present:
        os.replace(data_dir / name, aside / name)
    return aside


def _rollback(data_dir: Path, restored: Iterable[str], aside: Path | None, trash: Path) -> None:
    """Undo a failed restore: restored entries (and their DB sidecars) go to ``trash``,
    the previous data comes back from ``aside``."""
    trash.mkdir(parents=True, exist_ok=True)
    aside_names = {p.name for p in aside.iterdir()} if aside is not None else set()
    for name in {*restored, *DB_SIDECARS, *aside_names}:
        target = data_dir / name
        if target.exists() or target.is_symlink():
            os.replace(target, trash / name)
    if aside is not None:
        for name in aside_names:
            os.replace(aside / name, data_dir / name)
        aside.rmdir()


def restore_backup(settings: Settings, source: Path, *, log: Log = _print) -> Path | None:
    """Restore ``source`` into the data folder. Returns where the previous data was moved."""
    reason = app_running_reason(settings)
    if reason:
        raise CliError(f"Restore refused: {reason}", EXIT_REFUSED)
    data_dir = Path(settings.data_dir).resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    source = Path(source).expanduser().resolve()
    if source == data_dir or source.is_relative_to(data_dir):
        raise CliError("The backup must be outside the data folder.", EXIT_REFUSED)
    ts = _timestamp()
    staging = _unique(data_dir / f".restore-{ts}")
    staging.mkdir()
    try:
        root, manifest = _stage_backup(source, staging)
        restored = [n for n in (DB_FILE, "projects", *OPTIONAL_TREES) if (root / n).exists()]
        to_move = [DB_FILE, *DB_SIDECARS, "projects", "uploads", "tmp"]
        if "cache" in restored:  # keep the current provider cache unless the backup has one
            to_move.append("cache")
        aside = _move_aside(data_dir, to_move, ts)
        try:
            for name in restored:
                os.replace(root / name, data_dir / name)
            Storage(data_dir).ensure()
            db.init_db(data_dir / DB_FILE)
            _, integrity = _db_facts(data_dir / DB_FILE)
            if integrity != "ok":
                raise CliError("The restored database failed its integrity check.")
        except BaseException:
            _rollback(data_dir, restored, aside, staging / "failed-restore")
            raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    log(f"Restored the backup from {manifest.get('created_at', 'an unknown date')} "
        f"({len(manifest.get('files') or {})} file(s)).")
    if aside is not None:
        log(f"The previous data was moved to {aside} (nothing was deleted). Remove it once you are happy.")
    log("Start the app again with scripts/start.sh.")
    return aside


def _confirm(prompt: str) -> bool:
    if not sys.stdin.isatty():
        raise CliError("Add --yes to confirm the restore when not running in a terminal.", EXIT_REFUSED)
    return input(f"{prompt} [y/N] ").strip().lower() in ("y", "yes")


def cmd_backup(args: argparse.Namespace, settings: Settings | None) -> int:
    s = _settings_or_exit(settings)
    create_backup(s, Path(args.out), include_uploads=args.include_uploads, include_cache=args.include_cache,
                  archive=args.archive, force=args.force)
    return EXIT_OK


def cmd_restore(args: argparse.Namespace, settings: Settings | None) -> int:
    s = _settings_or_exit(settings)
    if not args.yes and not _confirm(f"Restore {args.from_path}? The current data will be moved aside, "
                                     "not deleted."):
        _print("Restore cancelled.")
        return EXIT_FAIL
    restore_backup(s, Path(args.from_path))
    return EXIT_OK


# ============================================================================ entry point
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m clipper.cli", description="Local Clipper maintenance tools.")
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")
    sub.add_parser("doctor", help="check the environment (PASS/WARN/FAIL)")
    hp = sub.add_parser("hash-password", help="print an argon2 hash for CLIPPER_OWNER_PASSWORD_HASH")
    hp.add_argument("--stdin", action="store_true", help="read the password from one line of standard input")
    sub.add_parser("check-config", help="validate the configuration")
    hc = sub.add_parser("health", help="check that the server (or worker) is up")
    hc.add_argument("--wait", type=float, default=0.0, metavar="SECONDS", help="keep trying for this long")
    hc.add_argument("--pid", type=int, default=0,
                    help="succeed only while this process is alive; give up early if it exits")
    hc.add_argument("--worker", action="store_true", help="check this machine's worker heartbeat instead")
    hc.add_argument("--quiet", action="store_true")
    sub.add_parser("paths", help="print data/log/run paths and the URL as key=value lines")
    sub.add_parser("serve-api", help="run the API server in this process (container entry point)")
    bk = sub.add_parser("backup", help="back up the database and projects")
    bk.add_argument("--out", required=True, metavar="DIR", help="folder to write the backup into")
    bk.add_argument("--include-uploads", action="store_true", help="also copy uploads that are not yet imported")
    bk.add_argument("--include-cache", action="store_true", help="also copy cached AI results (data/cache)")
    bk.add_argument("--archive", action="store_true", help="write a single .tar.gz instead of a folder")
    bk.add_argument("--force", action="store_true", help="back up even while a job is running")
    rs = sub.add_parser("restore", help="restore a backup (current data is moved aside)")
    rs.add_argument("--from", dest="from_path", required=True, metavar="PATH", help="backup folder or .tar.gz")
    rs.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    return parser


COMMANDS: dict[str, Callable[[argparse.Namespace, Settings | None], int]] = {
    "doctor": cmd_doctor,
    "hash-password": cmd_hash_password,
    "check-config": cmd_check_config,
    "health": cmd_health,
    "paths": cmd_paths,
    "serve-api": cmd_serve_api,
    "backup": cmd_backup,
    "restore": cmd_restore,
}


def main(argv: Iterable[str] | None = None, *, settings: Settings | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    try:
        return COMMANDS[args.command](args, settings)
    except CliError as exc:
        print(exc.message, file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
