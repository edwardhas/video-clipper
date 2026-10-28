"""YouTube import through yt-dlp (the venv's version-locked module), never with cookies.

* :func:`parse_youtube_url` accepts only single-video links on YouTube hosts and returns a
  canonical ``https://www.youtube.com/watch?v=ID`` URL. Only that canonical URL is ever
  handed to yt-dlp, after ``--``, so user input can never become a yt-dlp option.
* :func:`fetch_metadata` runs ``yt-dlp -J`` and rejects live/upcoming streams and videos
  without a length or over ``settings.max_source_minutes``.
* :func:`download` fetches ≤1080p merged MP4 into ``out_dir`` with machine-readable
  progress (``PROG {json}``) and final-path (``FILE <path>``) lines on stdout.
* :func:`classify_error` turns yt-dlp stderr into plain-language errors; raw stderr is
  never shown, only a short sanitised reason.

YouTube requires an external JavaScript runtime (deno >= 2.3 by default) for its player
challenges; it is resolved from ``settings.ytdlp_js_runtime`` (a binary name, a path, or
``runtime:path``) and passed explicitly with ``--js-runtimes``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import parse_qs, urlsplit

from ..config import Settings
from ..errors import (
    CancelledError,
    ClipperError,
    DiskSpaceError,
    InputError,
    InterruptedError_,
    MediaError,
    MissingDependencyError,
    SourceUnavailableError,
)
from ..procs import NEVER_CANCELLED, CancelToken, run

log = logging.getLogger(__name__)

ProgressCallback = Callable[[int, int | None], None]

MAX_URL_LENGTH = 500
METADATA_TIMEOUT_S = 180.0
DOWNLOAD_TIMEOUT_S = 3 * 3600.0
VERSION_TIMEOUT_S = 15.0
TEMP_DIR_NAME = ".part"

UPLOAD_INSTEAD = "You can download the video yourself (if you have the right to) and upload the file instead."

# --------------------------------------------------------------------------- URL parsing
_VIDEO_ID_RE = re.compile(r"[A-Za-z0-9_-]{11}")
_SHORT_HOST = "youtu.be"
_VIDEO_HOSTS = frozenset({
    "youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "www.youtube-nocookie.com",
})
_ID_PATH_KINDS = frozenset({"shorts", "live", "embed"})
# Links pasted without a scheme, e.g. "youtu.be/ID" or "www.youtube.com/watch?v=ID".
_SCHEMELESS_RE = re.compile(r"^(?:(?:www|m|music)\.)?(?:youtube\.com|youtu\.be)/", re.IGNORECASE)

MSG_EMPTY_URL = "Paste a YouTube video link."
MSG_URL_TOO_LONG = "That link is too long to be a YouTube video link."
MSG_NOT_A_LINK = (
    "That doesn't look like a YouTube video link. Paste a link such as "
    "https://www.youtube.com/watch?v=… or https://youtu.be/…"
)
MSG_BAD_SCHEME = "Only http:// or https:// YouTube links are supported."
MSG_NOT_YOUTUBE = (
    "Only YouTube video links (youtube.com or youtu.be) are supported. For other videos, download the "
    "file yourself (if you have the right to) and upload it instead."
)
MSG_PLAYLIST = "Playlists are not supported — paste a single video link."
MSG_NOT_A_VIDEO = (
    "That YouTube link doesn't point to a single video. Paste the link of the video itself, e.g. "
    "https://www.youtube.com/watch?v=… or https://youtu.be/…"
)
MSG_BAD_ID = "That YouTube link doesn't contain a valid video ID. Copy the link again from the video's Share button."


def canonical_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def parse_youtube_url(url: str) -> tuple[str, str]:
    """Return ``(video_id, canonical_url)`` for a single-video YouTube link, else raise InputError.

    Playlist/index/timestamp/share parameters are ignored; playlist-only links are rejected.
    """
    text = (url or "").strip() if isinstance(url, str) else ""
    if not text:
        raise InputError(MSG_EMPTY_URL)
    if len(text) > MAX_URL_LENGTH:
        raise InputError(MSG_URL_TOO_LONG)
    if any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in text):
        raise InputError(MSG_NOT_A_LINK)
    if _SCHEMELESS_RE.match(text):
        text = "https://" + text
    try:
        parts = urlsplit(text)
        port = parts.port
    except ValueError:
        raise InputError(MSG_NOT_A_LINK) from None
    scheme = parts.scheme.lower()
    if not scheme or not parts.netloc:
        raise InputError(MSG_NOT_A_LINK if scheme in ("", "http", "https") else MSG_BAD_SCHEME)
    if scheme not in ("http", "https"):
        raise InputError(MSG_BAD_SCHEME)
    if parts.username is not None or parts.password is not None or port not in (None, 80, 443):
        raise InputError(MSG_NOT_YOUTUBE)
    host = parts.hostname or ""
    if host == _SHORT_HOST:
        video_id = _id_from_short_path(parts.path)
    elif host in _VIDEO_HOSTS:
        video_id = _id_from_video_path(parts.path, parts.query)
    else:
        raise InputError(MSG_NOT_YOUTUBE)
    if not _VIDEO_ID_RE.fullmatch(video_id):
        raise InputError(MSG_BAD_ID)
    return video_id, canonical_url(video_id)


def _path_segments(path: str) -> list[str]:
    return [seg for seg in path.split("/") if seg]


def _id_from_short_path(path: str) -> str:
    segments = _path_segments(path)
    if len(segments) != 1:
        raise InputError(MSG_NOT_A_VIDEO)
    return segments[0]


def _id_from_video_path(path: str, query: str) -> str:
    segments = _path_segments(path)
    params = parse_qs(query, keep_blank_values=True)
    has_list = any(v.strip() for v in params.get("list", []))
    if segments == ["watch"]:
        video_ids = [v for v in params.get("v", []) if v]
        if video_ids:
            return video_ids[0]
        raise InputError(MSG_PLAYLIST if has_list else MSG_BAD_ID)
    if len(segments) == 2 and segments[0] in _ID_PATH_KINDS:
        if segments[0] == "embed" and segments[1] == "videoseries":
            raise InputError(MSG_PLAYLIST)
        return segments[1]
    if segments == ["playlist"] or has_list:
        raise InputError(MSG_PLAYLIST)
    if len(segments) == 1 and segments[0] in _ID_PATH_KINDS | {"watch"}:
        raise InputError(MSG_BAD_ID)
    raise InputError(MSG_NOT_A_VIDEO)


# --------------------------------------------------------------------------- tools & versions
_RUNTIME_ALIASES = {"deno": "deno", "node": "node", "nodejs": "node", "bun": "bun", "qjs": "quickjs",
                    "quickjs": "quickjs"}
_SUPPORTED_RUNTIMES = frozenset(_RUNTIME_ALIASES.values())

MSG_NO_JS_RUNTIME = (
    "YouTube imports need the Deno JavaScript runtime (version 2.3 or newer), but it was not found. "
    "Install it with \"brew install deno\" (or see https://deno.land), or point CLIPPER_YTDLP_JS_RUNTIME "
    "at it, then try again."
)


@dataclass(frozen=True)
class JsRuntime:
    name: str   # yt-dlp runtime key: deno | node | bun | quickjs
    path: str   # absolute path of the executable


def resolve_js_runtime(settings: Settings) -> JsRuntime:
    """Resolve ``settings.ytdlp_js_runtime`` ("deno", a path, or "node:/path") or raise MissingDependencyError."""
    raw = (settings.ytdlp_js_runtime or "").strip() or "deno"
    explicit: str | None = None
    head, sep, tail = raw.partition(":")
    if sep and head.lower() in _SUPPORTED_RUNTIMES and tail:
        explicit, raw = head.lower(), tail
    found = shutil.which(os.path.expanduser(raw))
    if not found:
        raise MissingDependencyError(MSG_NO_JS_RUNTIME, detail="JS runtime not found on PATH")
    path = os.path.abspath(found)
    name = explicit or _runtime_name_for(Path(path).name)
    if name is None:
        raise MissingDependencyError(
            "The configured JavaScript runtime is not one yt-dlp supports. Use Deno 2.3 or newer "
            "(\"brew install deno\" or https://deno.land) and set CLIPPER_YTDLP_JS_RUNTIME=deno.",
            detail="unsupported JS runtime binary name",
        )
    return JsRuntime(name=name, path=path)


def _runtime_name_for(binary_name: str) -> str | None:
    base = binary_name.lower().removesuffix(".exe")
    if base in _RUNTIME_ALIASES:
        return _RUNTIME_ALIASES[base]
    for alias, name in _RUNTIME_ALIASES.items():
        if base.startswith(alias + "-") or base.startswith(alias + "_"):
            return name
    return None


def js_runtime_args(settings: Settings) -> list[str]:
    """``--no-js-runtimes --js-runtimes NAME:PATH`` so exactly the configured runtime is used."""
    runtime = resolve_js_runtime(settings)
    return ["--no-js-runtimes", "--js-runtimes", f"{runtime.name}:{runtime.path}"]


def js_runtime_info(settings: Settings) -> str | None:
    """Short description such as ``"deno 2.8.2"``; None when the runtime is missing or broken."""
    try:
        runtime = resolve_js_runtime(settings)
        result = run([runtime.path, "--version"], timeout=VERSION_TIMEOUT_S, check=False,
                     error_cls=MissingDependencyError, error_message="JavaScript runtime version check failed")
    except (ClipperError, OSError):
        return None
    if result.returncode != 0:
        return None
    lines = result.stdout.decode("utf-8", "replace").strip().splitlines()
    if not lines:
        return runtime.name
    match = re.search(r"(\d+(?:[.-]\d+)+)", lines[0])
    return f"{runtime.name} {match.group(1)}" if match else runtime.name


@lru_cache(maxsize=1)
def ytdlp_version() -> str | None:
    """Version of the yt-dlp module installed in this environment (e.g. "2026.08.19")."""
    try:
        from yt_dlp.version import __version__
    except Exception:  # pragma: no cover - yt-dlp is a hard dependency
        return None
    return str(__version__)


def _ffmpeg_location(settings: Settings) -> str:
    found = shutil.which(os.path.expanduser(settings.ffmpeg))
    if not found:
        raise MissingDependencyError(
            "FFmpeg is required to download YouTube videos but was not found. "
            "Run scripts/doctor.sh for setup help."
        )
    return os.path.abspath(found)


# --------------------------------------------------------------------------- commands
_SECRET_ENV = frozenset({"OPENAI_API_KEY", "ANTHROPIC_API_KEY"})


def _child_env() -> dict[str, str]:
    """Environment for yt-dlp: no app secrets/config, unbuffered UTF-8 output."""
    env = {k: v for k, v in os.environ.items() if k not in _SECRET_ENV and not k.startswith("CLIPPER_")}
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _base_command(settings: Settings) -> list[str]:
    return [
        sys.executable, "-m", "yt_dlp",
        "--ignore-config",
        *js_runtime_args(settings),
        "--cache-dir", str(Path(settings.data_dir) / "cache" / "yt-dlp"),
        "--socket-timeout", "30",
    ]


def metadata_command(url: str, settings: Settings) -> list[str]:
    """yt-dlp argv that prints one video's metadata as JSON (no download, no playlist)."""
    return [*_base_command(settings), "-J", "--no-playlist", "--skip-download", "--", url]


def download_command(url: str, out_dir: Path, settings: Settings) -> list[str]:
    """yt-dlp argv that downloads ≤1080p merged MP4 into ``out_dir`` with parseable stdout."""
    home = Path(out_dir).resolve()
    return [
        *_base_command(settings),
        "--no-playlist",
        "--ffmpeg-location", _ffmpeg_location(settings),
        "-t", "mp4",
        "-S", "res:1080",
        "-P", f"home:{home}",
        "-P", f"temp:{home / TEMP_DIR_NAME}",
        "-o", "%(id)s.%(ext)s",
        "--restrict-filenames",
        "-R", "5",
        "--fragment-retries", "5",
        "--retry-sleep", "exp=1:30",
        "--retry-sleep", "fragment:exp=1:30",
        "--no-mtime",
        "--progress", "--newline", "--progress-delta", "1",
        "--progress-template",
        "download:PROG %(progress.{status,downloaded_bytes,total_bytes,total_bytes_estimate,speed,eta})j",
        "--print", "after_move:FILE %(filepath)s",
        "--", url,
    ]


# --------------------------------------------------------------------------- metadata
_LIVE_MESSAGES = {
    "is_live": ("This is a live stream that is still running. Live streams must finish and be processed by "
                "YouTube before they can be imported — try again after it has ended."),
    "is_upcoming": ("This live stream or premiere hasn't started yet. Live streams must finish and be processed "
                    "by YouTube before they can be imported — try again after it has ended."),
    "post_live": ("This live stream has just ended and YouTube is still processing the recording. "
                  "Try again later (this can take a few hours)."),
}
_RESTRICTED_AVAILABILITY = {
    "private": "This video is private. " + UPLOAD_INSTEAD,
    "premium_only": "This video is only available to YouTube Premium members. " + UPLOAD_INSTEAD,
    "subscriber_only": "This video is only available to channel members. " + UPLOAD_INSTEAD,
    "needs_auth": "YouTube requires a signed-in account to view this video. " + UPLOAD_INSTEAD,
}


def fetch_metadata(url: str, *, settings: Settings, cancel: CancelToken | None = None) -> dict[str, Any]:
    """Fetch and validate metadata for one video.

    Returns ``{id, title, duration, live_status, is_live, availability, filesize_approx, uploader,
    license, webpage_url}``. Raises InputError (live, no length, too long, bad link),
    SourceUnavailableError (private, removed, bot check …) or MissingDependencyError.
    """
    _, canonical = parse_youtube_url(url)
    info = _fetch_info_json(canonical, settings=settings, cancel=cancel)
    return validate_metadata(info, settings=settings)


def _fetch_info_json(url: str, *, settings: Settings, cancel: CancelToken | None) -> dict[str, Any]:
    result = run(
        metadata_command(url, settings),
        cancel=cancel, timeout=METADATA_TIMEOUT_S, check=False, env=_child_env(),
        error_cls=SourceUnavailableError,
        error_message="YouTube did not answer in time. Check the internet connection and try again.",
    )
    if result.returncode != 0:
        raise _failed("metadata", result.returncode, result.stderr_tail)
    try:
        info = json.loads(result.stdout.decode("utf-8", "replace"))
    except ValueError:
        info = None
    if not isinstance(info, dict):
        raise SourceUnavailableError(
            "Could not read the video information from YouTube. Try again later. " + UPLOAD_INSTEAD,
            detail="yt-dlp -J produced no JSON object",
        )
    return info


def validate_metadata(info: Mapping[str, Any], *, settings: Settings) -> dict[str, Any]:
    """Reduce yt-dlp's ``-J`` output to the fields we use and enforce import rules."""
    if info.get("_type") == "playlist":
        raise InputError(MSG_PLAYLIST)
    meta = {
        "id": _opt_str(info.get("id")),
        "title": _opt_str(info.get("title"), limit=500),
        "duration": _opt_float(info.get("duration")),
        "live_status": _opt_str(info.get("live_status")),
        "is_live": bool(info.get("is_live")),
        "availability": _opt_str(info.get("availability")),
        "filesize_approx": _opt_int(info.get("filesize_approx")),
        "uploader": _opt_str(info.get("uploader") or info.get("channel"), limit=200),
        "license": _opt_str(info.get("license"), limit=200),
    }
    video_id = meta["id"] or ""
    meta["webpage_url"] = canonical_url(video_id) if _VIDEO_ID_RE.fullmatch(video_id) else None
    live_status = "is_live" if meta["is_live"] and meta["live_status"] in (None, "not_live") else meta["live_status"]
    if live_status in _LIVE_MESSAGES:
        raise InputError(_LIVE_MESSAGES[live_status])
    if meta["availability"] in _RESTRICTED_AVAILABILITY:
        raise SourceUnavailableError(_RESTRICTED_AVAILABILITY[meta["availability"]])
    duration = meta["duration"]
    if not duration or duration <= 0:
        raise InputError("YouTube did not report a length for this video, so it can't be imported. " + UPLOAD_INSTEAD)
    limit_minutes = float(settings.max_source_minutes)
    if duration > limit_minutes * 60:
        raise InputError(
            f"This video is {format_duration(duration)} long, which is over the {limit_minutes:g}-minute limit. "
            "The limit is configurable via CLIPPER_MAX_SOURCE_MINUTES."
        )
    return meta


def format_duration(seconds: float) -> str:
    """``h:mm:ss`` (or ``m:ss`` under an hour)."""
    total = int(round(seconds))
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def _opt_str(value: Any, *, limit: int = 100) -> str | None:
    if value is None:
        return None
    text = "".join(ch for ch in str(value) if ch.isprintable()).strip()
    return text[:limit] or None


def _opt_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None and not isinstance(value, bool) else None
    except (TypeError, ValueError):
        return None


def _opt_int(value: Any) -> int | None:
    number = _opt_float(value)
    return int(number) if number is not None and number >= 0 else None


# --------------------------------------------------------------------------- download
def parse_output_line(line: str) -> tuple[str, Any] | None:
    """Parse one yt-dlp stdout line: ``("progress", dict)``, ``("file", path)`` or None."""
    text = line.rstrip("\r\n")
    if text.startswith("PROG "):
        try:
            data = json.loads(text[5:])
        except ValueError:
            return None
        return ("progress", data) if isinstance(data, dict) else None
    if text.startswith("FILE "):
        path = text[5:].strip()
        return ("file", path) if path and path != "NA" else None
    return None


@dataclass
class DownloadProgress:
    """Folds yt-dlp's per-file progress (video, then audio) into cumulative byte counts."""

    max_bytes: int | None = None
    completed_bytes: int = 0          # files that reported status "finished"
    downloaded_bytes: int = 0         # cumulative, including the current file
    total_bytes: int | None = None    # completed + current file's (estimated) total
    exceeded: bool = False

    def update(self, prog: Mapping[str, Any]) -> tuple[int, int | None]:
        current = _opt_int(prog.get("downloaded_bytes")) or 0
        exact_total = _opt_int(prog.get("total_bytes"))
        total = exact_total or _opt_int(prog.get("total_bytes_estimate"))
        if prog.get("status") == "finished":
            self.completed_bytes += total or current
            self.downloaded_bytes = self.total_bytes = self.completed_bytes
            known_final = self.completed_bytes
        else:
            self.downloaded_bytes = self.completed_bytes + current
            self.total_bytes = (self.completed_bytes + total) if total else None
            # Only an exact size may stop a download early; estimates can be far off.
            known_final = (self.completed_bytes + exact_total) if exact_total else self.downloaded_bytes
        if self.max_bytes is not None and max(self.downloaded_bytes, known_final) > self.max_bytes:
            self.exceeded = True
        return self.downloaded_bytes, self.total_bytes


@dataclass
class _DownloadOutput:
    progress: DownloadProgress
    progress_cb: ProgressCallback | None
    on_exceeded: Callable[[], None]
    final_path: str | None = None
    _reported_exceeded: bool = field(default=False, repr=False)

    def feed(self, line: str) -> None:
        parsed = parse_output_line(line)
        if parsed is None:
            return
        kind, value = parsed
        if kind == "file":
            self.final_path = value
            return
        downloaded, total = self.progress.update(value)
        if self.progress.exceeded and not self._reported_exceeded:
            self._reported_exceeded = True
            self.on_exceeded()
        if self.progress_cb is not None:
            self.progress_cb(downloaded, total)


def download(url: str, out_dir: Path, *, settings: Settings, cancel: CancelToken | None = None,
             progress_cb: ProgressCallback | None = None, max_bytes: int | None = None) -> Path:
    """Download one YouTube video (≤1080p, merged MP4) into ``out_dir`` and return its path.

    ``progress_cb(downloaded_bytes, total_bytes_or_None)`` receives cumulative bytes over all
    downloaded streams. Exceeding ``max_bytes`` stops the download with InputError. Partial
    files are kept on worker shutdown (so a resumed job continues) and removed otherwise.
    """
    _, canonical = parse_youtube_url(url)
    return _download_url(canonical, Path(out_dir), settings=settings, cancel=cancel,
                         progress_cb=progress_cb, max_bytes=max_bytes)


def _download_url(url: str, out_dir: Path, *, settings: Settings, cancel: CancelToken | None,
                  progress_cb: ProgressCallback | None, max_bytes: int | None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    home = out_dir.resolve()
    temp_dir = home / TEMP_DIR_NAME
    argv = download_command(url, home, settings)

    outer = cancel or NEVER_CANCELLED
    local = CancelToken(check=lambda: outer.cancelled)
    local.shutdown = outer.shutdown  # worker shutdown still means "interrupted", not "cancelled"
    output = _DownloadOutput(DownloadProgress(max_bytes=max_bytes), progress_cb, on_exceeded=local.cancel)

    try:
        result = run(
            argv, cancel=local, timeout=DOWNLOAD_TIMEOUT_S, check=False, env=_child_env(), cwd=home,
            stdout_line_cb=output.feed, error_cls=SourceUnavailableError,
            error_message="The YouTube download took too long and was stopped.",
        )
    except InterruptedError_:
        raise  # keep partial files: the resumed job continues the download
    except CancelledError:
        _remove_tree(temp_dir)
        if output.progress.exceeded:
            raise _too_large_error(max_bytes) from None
        raise
    except ClipperError:
        _remove_tree(temp_dir)
        raise

    if output.progress.exceeded:  # finished before the kill landed
        _discard_output(output.final_path, home)
        _remove_tree(temp_dir)
        raise _too_large_error(max_bytes)
    if result.returncode != 0:
        _remove_tree(temp_dir)
        raise _failed("download", result.returncode, result.stderr_tail)
    final = _verified_output(output.final_path, home)
    _remove_tree(temp_dir)
    return final


def _verified_output(printed: str | None, home: Path) -> Path:
    if not printed:
        raise MediaError("The YouTube download finished but no video file was produced.",
                         detail="yt-dlp printed no FILE line")
    candidate = Path(printed)
    if not candidate.is_absolute():
        candidate = home / candidate
    final = candidate.resolve()
    if not final.is_relative_to(home) or final == home or not final.is_file():
        raise MediaError("The YouTube download did not produce a usable video file.",
                         detail="yt-dlp output path missing or outside the download directory")
    return final


def _discard_output(printed: str | None, home: Path) -> None:
    if not printed:
        return
    candidate = Path(printed) if Path(printed).is_absolute() else home / printed
    try:
        final = candidate.resolve()
        if final.is_relative_to(home) and final.is_file():
            final.unlink()
    except OSError:  # pragma: no cover - best effort cleanup
        pass


def _remove_tree(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def _too_large_error(max_bytes: int | None) -> InputError:
    return InputError(
        f"This video is too large to import (over the {_format_size(max_bytes or 0)} limit), so the download "
        "was stopped. Try a shorter video."
    )


def _format_size(num_bytes: int) -> str:
    mib = num_bytes / (1024 * 1024)
    if mib >= 1024:
        return f"{mib / 1024:,.1f} GB".replace(".0 GB", " GB")
    if mib >= 1:
        return f"{mib:,.0f} MB"
    return f"{max(1, round(num_bytes / 1024)):,} KB"


# --------------------------------------------------------------------------- error classification
_JS_SETUP = (
    "Make sure Deno 2.3 or newer is installed (\"brew install deno\" or https://deno.land) and that "
    "yt-dlp is up to date."
)


@dataclass(frozen=True)
class _Rule:
    needles: tuple[str, ...]
    make: Callable[[str], ClipperError]


def _source(message: str, *, retryable: bool = False) -> Callable[[str], ClipperError]:
    return lambda reason: SourceUnavailableError(message, retryable=retryable, detail=reason or None)


def _missing(message: str) -> Callable[[str], ClipperError]:
    return lambda reason: MissingDependencyError(message, detail=reason or None)


def _input(message: str) -> Callable[[str], ClipperError]:
    return lambda reason: InputError(message, detail=reason or None)


_JS_RUNTIME_MISSING = _missing(
    "YouTube imports need a working JavaScript runtime (Deno 2.3 or newer), and none was usable. "
    "Install it with \"brew install deno\" (or see https://deno.land), then try again."
)
_JS_CHALLENGE_FAILED = _missing(
    "yt-dlp could not solve YouTube's player challenge, so no video formats were available. "
    + _JS_SETUP + " " + UPLOAD_INSTEAD
)
_JS_NEEDLES = ("no supported javascript runtime", "challenge solving failed", "signature solving failed")

# First match wins; specific source states before generic "unavailable".
_RULES: tuple[_Rule, ...] = (
    _Rule(("no space left on device", "disk quota exceeded"),
          lambda reason: DiskSpaceError("The disk is full, so the video could not be downloaded. Free up space and "
                                        "try again.", detail=reason or None)),
    _Rule(("http error 429", "too many requests", "isn't available, try again later", "rate-limited",
           "rate limited"),
          _source("YouTube is temporarily limiting requests from this computer (rate limited). Wait a while "
                  "(up to an hour) and try again. " + UPLOAD_INSTEAD, retryable=True)),
    _Rule(("drm protected", "drm-protected"),
          _source("This video is DRM-protected, so it can't be downloaded.")),
    _Rule(("unsupported url",),
          _input("That link isn't a supported YouTube video link. Paste the link of a single video.")),
    _Rule(("no supported javascript runtime",), _JS_RUNTIME_MISSING),
    _Rule(("private video", "this video is private"),
          _source("This video is private. " + UPLOAD_INSTEAD)),
    _Rule(("members-only", "join this channel", "available to this channel's members", "members only"),
          _source("This video is only available to channel members. " + UPLOAD_INSTEAD)),
    _Rule(("confirm your age", "age-restricted", "age restricted", "inappropriate for some users"),
          _source("This video is age-restricted and YouTube requires a signed-in account to watch it. "
                  "Local Clipper never uses your YouTube account. " + UPLOAD_INSTEAD)),
    _Rule(("live event will begin", "premieres in", "premiere will begin"),
          _source("This live stream or premiere hasn't started yet. Live streams must finish and be processed "
                  "by YouTube before they can be imported — try again after it has ended.")),
    _Rule(("not made this video available in your country", "geo restriction", "geo-restricted",
           "not available in your country", "not available from your location", "blocked it in your country"),
          _source("The uploader has not made this video available in this country. " + UPLOAD_INSTEAD)),
    _Rule(("not a bot", "sign in to confirm", "captcha", "sign in to view", "only available for registered users",
           "login required", "requires authentication"),
          _source("YouTube asked for a sign-in to confirm this isn't a bot, so the video can't be fetched from "
                  "this computer right now. Try again later. " + UPLOAD_INSTEAD)),
    _Rule(("challenge solving failed", "signature solving failed"), _JS_CHALLENGE_FAILED),
    _Rule(("video unavailable", "this video is unavailable", "has been removed", "no longer available",
           "account associated with this video has been terminated", "copyright claim",
           "this video does not exist"),
          _source("This video is unavailable on YouTube — it may have been removed or made private. Check that "
                  "the link plays in a browser. If you have your own copy, upload the file instead.")),
    _Rule(("requested format is not available", "no video formats found"),
          _source("YouTube didn't offer a downloadable video format. This usually means the JavaScript runtime "
                  "isn't working or yt-dlp needs an update. " + _JS_SETUP + " " + UPLOAD_INSTEAD)),
    _Rule(("http error 403",),
          _source("YouTube refused the download (HTTP 403). This is often temporary or fixed by updating "
                  "yt-dlp. Try again later. " + UPLOAD_INSTEAD, retryable=True)),
    _Rule(("ffmpeg is not installed", "ffmpeg not found", "ffprobe not found", "ffmpeg-location"),
          _missing("FFmpeg is required to download YouTube videos but yt-dlp could not use it. "
                   "Run scripts/doctor.sh for setup help.")),
    _Rule(("unable to download webpage", "unable to download api page", "timed out", "connection reset",
           "connection refused", "network is unreachable", "name or service not known",
           "nodename nor servname", "temporary failure in name resolution", "getaddrinfo failed",
           "failed to resolve", "remote end closed connection", "connection aborted", "ssl:", "certificate verify",
           "http error 5"),
          _source("Could not reach YouTube. Check the internet connection and try again.", retryable=True)),
)

_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "′": "'"})
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_URL_RE = re.compile(r"(?:https?|file|ftp)://\S+", re.IGNORECASE)
_PATH_RE = re.compile(r"(?:~|[A-Za-z]:)?(?:[\\/][^\s\\/:'\"]+){2,}[\\/]?")
_PREFIX_RE = re.compile(r"^\s*(?:error:\s*)?(?:\[[^\]]{1,40}\]\s*(?:[^\s:]{1,64}:\s+)?)?", re.IGNORECASE)
_NOISE_MARKERS = ("cookie", "--", "`", "see ", "please report this issue", "yt-dlp -u", "github.com",
                  "for more details", "refer to")


def _normalise(text: str) -> str:
    return _ANSI_RE.sub("", (text or "").translate(_APOSTROPHES))


def short_reason(stderr: str, *, limit: int = 160) -> str:
    """Last yt-dlp ERROR line without prefixes, cookie advice, URLs, paths or control characters."""
    lines = [ln.strip() for ln in _normalise(stderr).splitlines() if ln.strip()]
    errors = [ln for ln in lines if ln.upper().startswith("ERROR:")]
    if not errors:
        return ""
    reason = _PREFIX_RE.sub("", errors[-1], count=1)
    reason = re.sub(r"\s*\(caused by .*$", "", reason, flags=re.IGNORECASE)
    reason = re.split(r";\s*please report this issue", reason, flags=re.IGNORECASE)[0]
    sentences = re.split(r"(?<=[.!?])\s+", reason)
    kept = [s for s in sentences if not any(marker in s.lower() for marker in _NOISE_MARKERS)]
    reason = " ".join(kept)
    reason = _URL_RE.sub("", reason)
    reason = _PATH_RE.sub("…", reason)
    reason = "".join(ch for ch in reason if ch.isprintable())
    reason = re.sub(r"\s+", " ", reason).strip(" .:;,")
    if len(reason) > limit:
        reason = reason[: limit - 1].rstrip() + "…"
    return reason


def _failed(what: str, returncode: int, stderr: str) -> ClipperError:
    err = classify_error(stderr)
    log.warning("yt-dlp %s failed (exit %s, %s): %s", what, returncode, err.code, short_reason(stderr) or "no reason")
    return err


def classify_error(stderr: str) -> ClipperError:
    """Map yt-dlp stderr to a typed error with a plain-language message (never raw stderr)."""
    normalised = _normalise(stderr).lower()
    error_text = "\n".join(ln for ln in normalised.splitlines() if ln.strip().startswith("error:"))
    reason = short_reason(stderr)
    for scope in (error_text, normalised):
        if not scope:
            continue
        for rule in _RULES:
            if any(needle in scope for needle in rule.needles):
                if "requested format is not available" in rule.needles:
                    return _format_error(normalised, reason, rule)
                return rule.make(reason)
    if reason:
        return SourceUnavailableError(f"Could not get this video from YouTube ({reason}). " + UPLOAD_INSTEAD,
                                      detail=reason)
    return SourceUnavailableError("Could not get this video from YouTube. " + UPLOAD_INSTEAD)


def _format_error(normalised: str, reason: str, rule: _Rule) -> ClipperError:
    """Explain 'requested format is not available', which usually means the JS challenge failed."""
    if "no supported javascript runtime" in normalised:
        return _JS_RUNTIME_MISSING(reason)
    if any(needle in normalised for needle in _JS_NEEDLES):
        return _JS_CHALLENGE_FAILED(reason)
    return rule.make(reason)
