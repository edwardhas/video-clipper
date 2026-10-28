"""Tests for clipper.media.youtube.

* Pure tests: URL parsing, error classification, progress parsing, metadata validation.
* Command construction: ``run`` is replaced by a recorder so the argv can be inspected; the argv is also
  fed to the installed yt-dlp's own option parser to prove every flag is accepted by that version.
* Real integration (``ffmpeg`` marker): the real yt-dlp module downloads a small lavfi-generated MP4 from
  a local HTTP server (no internet), including too-large, cancel and shutdown paths.
* Network (``network`` marker, opt-in with CLIPPER_NETWORK_TESTS=1): live metadata for a CC-BY video.
"""

from __future__ import annotations

import http.server
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yt_dlp
import yt_dlp.version
from conftest import requires_ffmpeg

from clipper.config import Settings
from clipper.errors import (
    CancelledError,
    DiskSpaceError,
    InputError,
    InterruptedError_,
    MediaError,
    MissingDependencyError,
    SourceUnavailableError,
)
from clipper.media import youtube as yt
from clipper.procs import CancelToken, ProcResult


VIDEO_ID = "nKyWusznRgQ"
CANONICAL = f"https://www.youtube.com/watch?v={VIDEO_ID}"


# --------------------------------------------------------------------------- helpers & fixtures
def _write_script(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n" + body + "\n")
    path.chmod(0o755)
    return path


@pytest.fixture
def fake_bin(tmp_path: Path) -> Path:
    """A directory holding fake ``deno`` and ``ffmpeg`` executables."""
    bin_dir = tmp_path / "bin"
    _write_script(bin_dir / "deno", 'echo "deno 2.3.1 (stable, release, test-arch)"\necho "v8 13.0"')
    _write_script(bin_dir / "ffmpeg", "exit 0")
    return bin_dir


@pytest.fixture
def tool_settings(settings: Settings, fake_bin: Path) -> Settings:
    """Settings whose JS runtime and ffmpeg point at the fake executables (absolute paths)."""
    return settings.model_copy(update={"ytdlp_js_runtime": str(fake_bin / "deno"),
                                       "ffmpeg": str(fake_bin / "ffmpeg")})


class RecordingRun:
    """Stand-in for ``procs.run`` that records argv/kwargs and replays scripted output."""

    def __init__(self, *, returncode: int = 0, stdout: bytes = b"", stderr: str = "",
                 lines: list[str] | None = None, create: str | None = None) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.lines = lines or []
        self.create = create            # file name to create in the -P home: dir before replaying lines
        self.calls: list[tuple[list[Any], dict[str, Any]]] = []

    def __call__(self, args: list[Any], **kwargs: Any) -> ProcResult:
        self.calls.append((list(args), kwargs))
        if self.create is not None:
            home = Path(_option_value(args, "-P", prefix="home:"))
            (home / self.create).write_bytes(b"\x00" * 64)
        cb = kwargs.get("stdout_line_cb")
        for line in self.lines:
            assert cb is not None, "download must stream stdout lines"
            cb(line + "\n")
        return ProcResult(args=[str(a) for a in args], returncode=self.returncode, stdout=self.stdout,
                          stderr_tail=self.stderr, duration_s=0.01)

    @property
    def argv(self) -> list[Any]:
        assert len(self.calls) == 1
        return self.calls[0][0]

    @property
    def kwargs(self) -> dict[str, Any]:
        assert len(self.calls) == 1
        return self.calls[0][1]


def _option_value(argv: list[Any], flag: str, *, prefix: str = "") -> str:
    values = [str(argv[i + 1]) for i, a in enumerate(argv[:-1]) if a == flag and str(argv[i + 1]).startswith(prefix)]
    assert len(values) == 1, (flag, prefix, argv)
    return values[0][len(prefix):]


def _ok_info(**overrides: Any) -> dict[str, Any]:
    info: dict[str, Any] = {
        "id": VIDEO_ID, "title": "A lecture", "duration": 602, "live_status": "not_live", "is_live": False,
        "availability": "public", "filesize_approx": 43_505_154, "uploader": "Some Channel",
        "license": "Creative Commons Attribution license (reuse allowed)", "formats": [{"format_id": "18"}],
    }
    info.update(overrides)
    return info


# --------------------------------------------------------------------------- URL parsing
VALID_URLS = [
    ("https://www.youtube.com/watch?v=nKyWusznRgQ", VIDEO_ID),
    ("http://youtube.com/watch?v=nKyWusznRgQ", VIDEO_ID),
    ("https://m.youtube.com/watch?v=nKyWusznRgQ&feature=share", VIDEO_ID),
    ("https://music.youtube.com/watch?v=nKyWusznRgQ&list=RDAMVMnKyWusznRgQ", VIDEO_ID),
    ("https://www.youtube.com/watch?v=nKyWusznRgQ&list=PLabcdef&index=3&t=42s", VIDEO_ID),
    ("https://www.youtube.com/watch?feature=youtu.be&v=nKyWusznRgQ", VIDEO_ID),
    ("https://www.youtube.com/watch?v=nKyWusznRgQ#t=30", VIDEO_ID),
    ("https://youtu.be/nKyWusznRgQ", VIDEO_ID),
    ("https://youtu.be/nKyWusznRgQ?si=AbCdEf123_xyz&t=10", VIDEO_ID),
    ("https://www.youtube.com/shorts/a_b-C1d2E3f?si=xyz123", "a_b-C1d2E3f"),
    ("https://youtube.com/shorts/a_b-C1d2E3f/", "a_b-C1d2E3f"),
    ("https://www.youtube.com/live/nKyWusznRgQ?si=abc", VIDEO_ID),
    ("https://www.youtube.com/embed/nKyWusznRgQ?start=30", VIDEO_ID),
    ("https://www.youtube-nocookie.com/embed/nKyWusznRgQ", VIDEO_ID),
    ("  https://www.youtube.com/watch?v=nKyWusznRgQ \n", VIDEO_ID),
    ("HTTPS://WWW.YOUTUBE.COM/watch?v=nKyWusznRgQ", VIDEO_ID),
    ("https://www.youtube.com:443/watch?v=nKyWusznRgQ", VIDEO_ID),
    ("youtu.be/nKyWusznRgQ", VIDEO_ID),
    ("www.youtube.com/watch?v=nKyWusznRgQ", VIDEO_ID),
]


@pytest.mark.parametrize(("url", "video_id"), VALID_URLS)
def test_parse_valid_urls(url: str, video_id: str) -> None:
    assert yt.parse_youtube_url(url) == (video_id, f"https://www.youtube.com/watch?v={video_id}")


INVALID_URLS = [
    ("", "Paste a YouTube video link"),
    ("   ", "Paste a YouTube video link"),
    ("javascript:alert(1)", "http:// or https://"),
    ("JavaScript://www.youtube.com/watch?v=nKyWusznRgQ", "http:// or https://"),
    ("ftp://www.youtube.com/watch?v=nKyWusznRgQ", "http:// or https://"),
    ("file:///etc/passwd", "http:// or https://"),
    ("data:text/html,<script>alert(1)</script>", "http:// or https://"),
    ("https://vimeo.com/123456789", "Only YouTube video links"),
    ("https://youtube.com.evil.com/watch?v=nKyWusznRgQ", "Only YouTube video links"),
    ("https://evilyoutube.com/watch?v=nKyWusznRgQ", "Only YouTube video links"),
    ("https://www.youtube.com.example/watch?v=nKyWusznRgQ", "Only YouTube video links"),
    ("https://www.youtube.com@evil.com/watch?v=nKyWusznRgQ", "Only YouTube video links"),
    ("https://user:pw@www.youtube.com/watch?v=nKyWusznRgQ", "Only YouTube video links"),
    ("https://www.youtube.com:8080/watch?v=nKyWusznRgQ", "Only YouTube video links"),
    ("https://www.youtube.com./watch?v=nKyWusznRgQ", "Only YouTube video links"),
    ("https://www.youtube.com/watch", "valid video ID"),
    ("https://www.youtube.com/watch?v=", "valid video ID"),
    ("https://www.youtube.com/watch?v=nKyWusznRg", "valid video ID"),        # 10 chars
    ("https://www.youtube.com/watch?v=nKyWusznRgQQ", "valid video ID"),      # 12 chars
    ("https://www.youtube.com/watch?v=nKyWusznRgQ%0A", "valid video ID"),    # encoded newline
    ("https://youtu.be/nKyWusz$RgQ", "valid video ID"),
    ("https://youtu.be/nKyWusznRg", "valid video ID"),
    ("https://www.youtube.com/shorts/", "valid video ID"),
    ("https://youtu.be/", "doesn't point to a single video"),
    ("https://youtu.be/nKyWusznRgQ/extra", "doesn't point to a single video"),
    ("https://www.youtube.com/", "doesn't point to a single video"),
    ("https://www.youtube.com/@SomeChannel", "doesn't point to a single video"),
    ("https://www.youtube.com/channel/UC1234567890", "doesn't point to a single video"),
    ("https://www.youtube.com/playlist?list=PLabcdef", "Playlists are not supported"),
    ("https://www.youtube.com/watch?list=PLabcdef&index=2", "Playlists are not supported"),
    ("https://www.youtube.com/embed/videoseries?list=PLabcdef", "Playlists are not supported"),
    ("https://www.you tube.com/watch?v=nKyWusznRgQ", "doesn't look like a YouTube video link"),
    ("not a url", "doesn't look like a YouTube video link"),
    ("--exec=rm -rf /", "doesn't look like a YouTube video link"),
    ("-oevil", "doesn't look like a YouTube video link"),
    ("https://www.youtube.com/watch?v=" + "a" * 600, "too long"),
]


@pytest.mark.parametrize(("url", "message"), INVALID_URLS)
def test_parse_invalid_urls(url: str, message: str) -> None:
    with pytest.raises(InputError) as info:
        yt.parse_youtube_url(url)
    err = info.value
    assert message in err.user_message
    assert err.code == "invalid_input"
    # Never echo user input back (the messages only contain generic example links).
    assert VIDEO_ID not in err.user_message and "evil" not in err.user_message and "PLabc" not in err.user_message


def test_playlist_message_is_exact() -> None:
    with pytest.raises(InputError) as info:
        yt.parse_youtube_url("https://www.youtube.com/playlist?list=PLabcdef")
    assert info.value.user_message == "Playlists are not supported — paste a single video link."


def test_parse_rejects_non_string() -> None:
    with pytest.raises(InputError):
        yt.parse_youtube_url(None)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- error classification
BOT_STDERR = (
    "WARNING: [youtube] Some warning\n"
    "ERROR: [youtube] nKyWusznRgQ: Sign in to confirm you’re not a bot. Use --cookies-from-browser or "
    "--cookies for the authentication. See  https://github.com/yt-dlp/yt-dlp/wiki/FAQ#how-do-i-pass-cookies-to-yt-dlp"
    "  for how to manually pass cookies. Also see  https://github.com/yt-dlp/yt-dlp/wiki/Extractors#exporting-"
    "youtube-cookies  for tips on effectively exporting YouTube cookies"
)

CLASSIFY_CASES = [
    # (stderr, expected type, retryable, substring in message, ends with "upload the file instead.")
    (BOT_STDERR, SourceUnavailableError, False, "isn't a bot", True),
    ("ERROR: [youtube] abcdefghijk: Sign in to confirm your age. This video may be inappropriate for some users.",
     SourceUnavailableError, False, "age-restricted", True),
    ("ERROR: [youtube] abcdefghijk: This video is age-restricted and only available on YouTube.",
     SourceUnavailableError, False, "age-restricted", True),
    ("ERROR: [youtube] abcdefghijk: Private video. Sign in if you've been granted access to this video",
     SourceUnavailableError, False, "private", True),
    ("ERROR: [youtube] abcdefghijk: Video unavailable. This video has been removed by the uploader",
     SourceUnavailableError, False, "unavailable", False),
    ("ERROR: [youtube] abcdefghijk: Video unavailable",
     SourceUnavailableError, False, "unavailable", False),
    ("ERROR: [youtube] abcdefghijk: Join this channel to get access to members-only content like this video, "
     "and other exclusive perks.", SourceUnavailableError, False, "channel members", True),
    ("ERROR: [youtube] abcdefghijk: This live event will begin in 3 hours.",
     SourceUnavailableError, False, "hasn't started yet", False),
    ("ERROR: [youtube] abcdefghijk: Premieres in 2 days", SourceUnavailableError, False, "hasn't started yet", False),
    ("ERROR: [youtube] abcdefghijk: The uploader has not made this video available in your country",
     SourceUnavailableError, False, "country", True),
    ("ERROR: [youtube] abcdefghijk: Video unavailable. This video contains content from SME, who has blocked it "
     "in your country on copyright grounds", SourceUnavailableError, False, "country", True),
    ("ERROR: [youtube] abcdefghijk: Unable to download API page: HTTP Error 429: Too Many Requests",
     SourceUnavailableError, True, "rate limited", True),
    ("ERROR: [youtube] abcdefghijk: The current session has been rate-limited by YouTube for up to an hour. "
     "It is recommended to use `-t sleep` to add a delay between video requests",
     SourceUnavailableError, True, "rate limited", True),
    ("ERROR: [youtube] abcdefghijk: This content isn’t available, try again later.",
     SourceUnavailableError, True, "rate limited", True),
    ("ERROR: [youtube] abcdefghijk: This video is DRM protected", SourceUnavailableError, False, "DRM", False),
    ("ERROR: Unsupported URL: https://example.com/watch", InputError, False, "supported YouTube video link", False),
    ("ERROR: [youtube] abcdefghijk: Requested format is not available. Use --list-formats for a list of "
     "available formats", SourceUnavailableError, False, "JavaScript runtime", True),
    ("WARNING: [youtube] No supported JavaScript runtime could be found. Only deno is enabled by default; to use "
     "another runtime add --js-runtimes RUNTIME[:PATH] to your command/config.\n"
     "ERROR: [youtube] abcdefghijk: Requested format is not available. Use --list-formats for a list of "
     "available formats", MissingDependencyError, False, "Deno 2.3 or newer", False),
    ("WARNING: [youtube] abcdefghijk: n challenge solving failed: Some formats may be missing. Ensure you have a "
     "supported JavaScript runtime and challenge solver script distribution installed.\n"
     "ERROR: [youtube] abcdefghijk: Requested format is not available. Use --list-formats for a list of "
     "available formats", MissingDependencyError, False, "player challenge", True),
    ("ERROR: [youtube] abcdefghijk: Unable to download API page: <urlopen error [Errno 8] nodename nor servname "
     "provided, or not known> (caused by TransportError('...'))", SourceUnavailableError, True,
     "internet connection", False),
    ("ERROR: unable to write data: [Errno 28] No space left on device", DiskSpaceError, True, "disk is full", False),
    ("ERROR: Postprocessing: ffprobe and ffmpeg not found. Please install or provide the path using "
     "--ffmpeg-location", MissingDependencyError, False, "FFmpeg", False),
    ("\x1b[0;31mERROR:\x1b[0m [youtube] abcdefghijk: Private video. Sign in if you've been granted access",
     SourceUnavailableError, False, "private", True),
]


@pytest.mark.parametrize(("stderr", "etype", "retryable", "fragment", "suggests_upload"), CLASSIFY_CASES)
def test_classify_error_table(stderr: str, etype: type, retryable: bool, fragment: str,
                              suggests_upload: bool) -> None:
    err = yt.classify_error(stderr)
    assert type(err) is etype
    assert err.retryable is retryable
    assert fragment in err.user_message
    assert err.user_message.endswith(yt.UPLOAD_INSTEAD) is suggests_upload
    _assert_clean_message(err.user_message)
    if err.detail:
        assert "cookie" not in err.detail.lower()
        assert "http" not in err.detail.lower() or "http error" in err.detail.lower()


def _assert_clean_message(message: str) -> None:
    lowered = message.lower()
    assert "cookie" not in lowered
    assert "--" not in message
    assert "traceback" not in lowered
    assert "/users/" not in lowered and "/private/" not in lowered and "/home/" not in lowered
    assert "github.com" not in lowered
    assert "\x1b" not in message
    assert len(message) < 600


def test_classify_unknown_error_is_short_and_sanitised() -> None:
    stderr = (
        "ERROR: [youtube] abcdefghijk: Something odd at /Users/someone/secret/file.py; please report this issue on "
        " https://github.com/yt-dlp/yt-dlp/issues?q= , filling out the appropriate issue template. Confirm you are "
        "on the latest version using  yt-dlp -U"
    )
    err = yt.classify_error(stderr)
    assert isinstance(err, SourceUnavailableError)
    assert "Something odd at" in err.user_message
    assert "someone" not in err.user_message
    _assert_clean_message(err.user_message)
    assert err.user_message.endswith(yt.UPLOAD_INSTEAD)


@pytest.mark.parametrize("stderr", ["", "   \n", "WARNING: only a warning"])
def test_classify_empty_or_warning_only(stderr: str) -> None:
    err = yt.classify_error(stderr)
    assert isinstance(err, SourceUnavailableError)
    assert err.user_message == "Could not get this video from YouTube. " + yt.UPLOAD_INSTEAD


def test_classify_prefers_error_lines_over_warnings() -> None:
    stderr = ("WARNING: [youtube] abcdefghijk: HTTP Error 429 on a side request, retrying\n"
              "ERROR: [youtube] abcdefghijk: Private video. Sign in if you've been granted access")
    err = yt.classify_error(stderr)
    assert "private" in err.user_message and not err.retryable


def test_short_reason_strips_cookie_advice_urls_and_paths() -> None:
    assert yt.short_reason(BOT_STDERR) == "Sign in to confirm you're not a bot"
    reason = yt.short_reason("ERROR: unable to open for writing: [Errno 2] No such file: '/Users/x/y/z.part'")
    assert "/Users" not in reason and "No such file" in reason
    assert len(yt.short_reason("ERROR: " + "word " * 200)) <= 160


# --------------------------------------------------------------------------- JS runtime & versions
def test_resolve_js_runtime_by_name_on_path(settings: Settings, fake_bin: Path,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", str(fake_bin))
    runtime = yt.resolve_js_runtime(settings.model_copy(update={"ytdlp_js_runtime": "deno"}))
    assert runtime == yt.JsRuntime(name="deno", path=str(fake_bin / "deno"))
    assert yt.js_runtime_args(settings.model_copy(update={"ytdlp_js_runtime": "deno"})) == [
        "--no-js-runtimes", "--js-runtimes", f"deno:{fake_bin / 'deno'}"]


def test_resolve_js_runtime_by_path_and_explicit_kind(settings: Settings, tmp_path: Path) -> None:
    node = _write_script(tmp_path / "tools" / "node", "echo v22.11.0")
    assert yt.resolve_js_runtime(settings.model_copy(update={"ytdlp_js_runtime": str(node)})) == yt.JsRuntime(
        "node", str(node))
    custom = _write_script(tmp_path / "tools" / "my-runtime", "echo 1.0")
    assert yt.resolve_js_runtime(settings.model_copy(update={"ytdlp_js_runtime": f"bun:{custom}"})) == yt.JsRuntime(
        "bun", str(custom))


def test_missing_js_runtime_explains_deno(settings: Settings, tmp_path: Path,
                                          monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    for value in ("deno", str(tmp_path / "nope" / "deno")):
        with pytest.raises(MissingDependencyError) as info:
            yt.resolve_js_runtime(settings.model_copy(update={"ytdlp_js_runtime": value}))
        message = info.value.user_message
        assert "2.3" in message and "brew install deno" in message and "https://deno.land" in message
        assert str(tmp_path) not in message
    assert yt.js_runtime_info(settings.model_copy(update={"ytdlp_js_runtime": "deno"})) is None


def test_unsupported_js_runtime_binary(settings: Settings, tmp_path: Path) -> None:
    other = _write_script(tmp_path / "tools" / "python-ish", "echo 1")
    with pytest.raises(MissingDependencyError):
        yt.resolve_js_runtime(settings.model_copy(update={"ytdlp_js_runtime": str(other)}))


def test_js_runtime_info_reads_first_version_line(tool_settings: Settings, tmp_path: Path) -> None:
    assert yt.js_runtime_info(tool_settings) == "deno 2.3.1"
    broken = _write_script(tmp_path / "tools2" / "deno", "exit 3")
    assert yt.js_runtime_info(tool_settings.model_copy(update={"ytdlp_js_runtime": str(broken)})) is None


@pytest.mark.skipif(shutil.which("deno") is None, reason="deno not installed")
def test_js_runtime_info_real_deno(settings: Settings) -> None:
    info = yt.js_runtime_info(settings.model_copy(update={"ytdlp_js_runtime": "deno"}))
    assert info is not None and info.startswith("deno ")
    major, minor = (int(p) for p in info.split()[1].split(".")[:2])
    assert (major, minor) >= (2, 3), "YouTube needs deno >= 2.3"


def test_ytdlp_version_matches_installed_module() -> None:
    assert yt.ytdlp_version() == yt_dlp.version.__version__


# --------------------------------------------------------------------------- command construction
def _assert_safe_base(argv: list[Any], js_path: Path) -> None:
    assert isinstance(argv, list) and all(isinstance(a, str) for a in argv)
    assert argv[:3] == [sys.executable, "-m", "yt_dlp"]
    assert "--ignore-config" in argv and "--no-playlist" in argv
    assert not any("cookie" in a.lower() for a in argv)
    assert _option_value(argv, "--js-runtimes") == f"deno:{js_path}"
    assert argv.index("--no-js-runtimes") < argv.index("--js-runtimes")
    assert _option_value(argv, "--socket-timeout") == "30"
    assert argv[-2:] == ["--", CANONICAL]


def test_metadata_command_via_fetch(tool_settings: Settings, fake_bin: Path,
                                    monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-a-real-key")
    monkeypatch.setenv("CLIPPER_SESSION_SECRET", "s" * 40)
    rec = RecordingRun(stdout=json.dumps(_ok_info()).encode())
    monkeypatch.setattr(yt, "run", rec)
    meta = yt.fetch_metadata(f"https://youtu.be/{VIDEO_ID}?si=abc&list=PLx", settings=tool_settings,
                             cancel=CancelToken())
    argv, kwargs = rec.argv, rec.kwargs
    _assert_safe_base(argv, fake_bin / "deno")
    for flag in ("-J", "--skip-download"):
        assert flag in argv
    assert kwargs["check"] is False and kwargs["timeout"] > 0 and isinstance(kwargs["cancel"], CancelToken)
    env = kwargs["env"]
    assert "OPENAI_API_KEY" not in env and not any(k.startswith("CLIPPER_") for k in env)
    assert meta["id"] == VIDEO_ID and meta["duration"] == 602.0
    assert yt_dlp.parse_options(argv[3:]).ydl_opts["noplaylist"] is True


def test_download_command_via_download(tool_settings: Settings, fake_bin: Path, tmp_path: Path,
                                       monkeypatch: pytest.MonkeyPatch) -> None:
    out_dir = tmp_path / "dl"
    out_dir.mkdir()
    rec = RecordingRun(create=f"{VIDEO_ID}.mp4",
                       lines=["[download] noise", f"FILE {out_dir.resolve() / (VIDEO_ID + '.mp4')}"])
    monkeypatch.setattr(yt, "run", rec)
    path = yt.download(f"https://www.youtube.com/watch?v={VIDEO_ID}&list=PLx&index=4&t=9s", out_dir,
                       settings=tool_settings, cancel=CancelToken())
    assert path == out_dir.resolve() / f"{VIDEO_ID}.mp4"
    argv, kwargs = rec.argv, rec.kwargs
    _assert_safe_base(argv, fake_bin / "deno")
    home = Path(_option_value(argv, "-P", prefix="home:"))
    temp = Path(_option_value(argv, "-P", prefix="temp:"))
    assert home == out_dir.resolve() and temp == out_dir.resolve() / ".part"
    assert _option_value(argv, "-o") == "%(id)s.%(ext)s"
    assert _option_value(argv, "-t") == "mp4" and _option_value(argv, "-S") == "res:1080"
    assert _option_value(argv, "--ffmpeg-location") == str(fake_bin / "ffmpeg")
    assert _option_value(argv, "--print") == "after_move:FILE %(filepath)s"
    assert _option_value(argv, "--progress-template").startswith("download:PROG %(progress.")
    for flag in ("--restrict-filenames", "--no-mtime", "--progress", "--newline"):
        assert flag in argv
    assert kwargs["stdout_line_cb"] is not None and kwargs["timeout"] >= 3600
    assert kwargs["cwd"] == out_dir.resolve()
    assert not (out_dir / ".part").exists()

    # Every flag must be accepted by the installed yt-dlp version.
    opts = yt_dlp.parse_options(argv[3:]).ydl_opts
    assert opts["noplaylist"] is True and opts["restrictfilenames"] is True
    assert opts["paths"] == {"home": str(home), "temp": str(temp)}
    assert opts["forceprint"] == {"after_move": ["FILE %(filepath)s"]}
    assert opts["js_runtimes"] == {"deno": {"path": str(fake_bin / "deno")}}
    assert opts["format_sort"][0] == "res:1080" and opts["merge_output_format"] == "mp4"
    assert opts["noprogress"] is False and opts["progress_delta"] == 1.0
    assert opts["retries"] == 5 and opts["fragment_retries"] == 5 and opts["socket_timeout"] == 30.0


def test_download_rejects_bad_url_before_running(tool_settings: Settings, tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    rec = RecordingRun()
    monkeypatch.setattr(yt, "run", rec)
    with pytest.raises(InputError):
        yt.download("https://example.com/video.mp4", tmp_path / "dl", settings=tool_settings)
    with pytest.raises(InputError):
        yt.fetch_metadata("https://www.youtube.com/playlist?list=PLx", settings=tool_settings)
    assert rec.calls == []


def test_commands_require_js_runtime(settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rec = RecordingRun()
    monkeypatch.setattr(yt, "run", rec)
    missing = settings.model_copy(update={"ytdlp_js_runtime": str(tmp_path / "missing" / "deno")})
    with pytest.raises(MissingDependencyError):
        yt.fetch_metadata(CANONICAL, settings=missing)
    assert rec.calls == []


# --------------------------------------------------------------------------- progress parsing
def test_parse_output_line_samples() -> None:
    assert yt.parse_output_line('PROG {"status": "downloading", "downloaded_bytes": 1024, "total_bytes": 294314}\n') \
        == ("progress", {"status": "downloading", "downloaded_bytes": 1024, "total_bytes": 294314})
    assert yt.parse_output_line("FILE /tmp/x/abc.mp4\r\n") == ("file", "/tmp/x/abc.mp4")
    assert yt.parse_output_line("FILE NA") is None
    assert yt.parse_output_line("PROG not-json") is None
    assert yt.parse_output_line("PROG [1, 2]") is None
    assert yt.parse_output_line("[download] Destination: x.mp4") is None
    assert yt.parse_output_line("") is None


def test_progress_is_cumulative_over_video_and_audio() -> None:
    prog = yt.DownloadProgress()
    assert prog.update({"status": "downloading", "downloaded_bytes": 100, "total_bytes": 1000}) == (100, 1000)
    assert prog.update({"status": "downloading", "downloaded_bytes": 600, "total_bytes_estimate": 1100}) == (600, 1100)
    assert prog.update({"status": "finished", "downloaded_bytes": 1000, "total_bytes": 1000}) == (1000, 1000)
    assert prog.update({"status": "downloading", "downloaded_bytes": 50}) == (1050, None)
    assert prog.update({"status": "downloading", "downloaded_bytes": 50, "total_bytes": 200}) == (1050, 1200)
    assert prog.update({"status": "finished", "downloaded_bytes": 200, "total_bytes": 200}) == (1200, 1200)
    assert not prog.exceeded


def test_progress_limit_uses_exact_size_not_estimates() -> None:
    prog = yt.DownloadProgress(max_bytes=1000)
    prog.update({"status": "downloading", "downloaded_bytes": 10, "total_bytes_estimate": 50_000})
    assert not prog.exceeded           # estimates alone never stop a download
    prog.update({"status": "downloading", "downloaded_bytes": 1001, "total_bytes_estimate": 50_000})
    assert prog.exceeded
    exact = yt.DownloadProgress(max_bytes=1000)
    exact.update({"status": "downloading", "downloaded_bytes": 1, "total_bytes": 5000})
    assert exact.exceeded              # an exact size over the limit stops at once


def test_download_reports_progress_and_stops_when_too_large(tool_settings: Settings, tmp_path: Path,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    out_dir = tmp_path / "dl"
    out_dir.mkdir()
    final = out_dir.resolve() / f"{VIDEO_ID}.mp4"
    lines = [
        'PROG {"status": "downloading", "downloaded_bytes": 1024, "total_bytes": 4096}',
        "some unrelated line",
        'PROG {"status": "finished", "downloaded_bytes": 4096, "total_bytes": 4096}',
        'PROG {"status": "downloading", "downloaded_bytes": 100, "total_bytes_estimate": 1000}',
        'PROG {"status": "finished", "downloaded_bytes": 1000, "total_bytes": 1000}',
        f"FILE {final}",
    ]
    events: list[tuple[int, int | None]] = []
    monkeypatch.setattr(yt, "run", RecordingRun(lines=lines, create=final.name))
    path = yt.download(CANONICAL, out_dir, settings=tool_settings, progress_cb=lambda d, t: events.append((d, t)))
    assert path == final
    assert events == [(1024, 4096), (4096, 4096), (4196, 5096), (5096, 5096)]

    # Same output with a 2000-byte limit: the fake process "finishes" before the kill lands.
    monkeypatch.setattr(yt, "run", RecordingRun(lines=lines, create=final.name))
    with pytest.raises(InputError) as info:
        yt.download(CANONICAL, out_dir, settings=tool_settings, max_bytes=2000)
    assert "too large" in info.value.user_message
    assert not final.exists()


def test_download_failure_is_classified_and_cleans_temp(tool_settings: Settings, tmp_path: Path,
                                                        monkeypatch: pytest.MonkeyPatch) -> None:
    out_dir = tmp_path / "dl"
    (out_dir / ".part").mkdir(parents=True)
    (out_dir / ".part" / "x.mp4.part").write_bytes(b"partial")
    rec = RecordingRun(returncode=1, stderr="ERROR: [youtube] abcdefghijk: Private video. Sign in if you've been "
                                            "granted access to this video")
    monkeypatch.setattr(yt, "run", rec)
    with pytest.raises(SourceUnavailableError) as info:
        yt.download(CANONICAL, out_dir, settings=tool_settings)
    assert "private" in info.value.user_message
    assert not (out_dir / ".part").exists()


def test_download_without_file_line_is_media_error(tool_settings: Settings, tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(yt, "run", RecordingRun(lines=['PROG {"status": "finished", "downloaded_bytes": 5}']))
    with pytest.raises(MediaError):
        yt.download(CANONICAL, tmp_path / "dl", settings=tool_settings)


def test_download_rejects_output_outside_out_dir(tool_settings: Settings, tmp_path: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
    outside = tmp_path / "elsewhere.mp4"
    outside.write_bytes(b"x")
    out_dir = tmp_path / "dl"
    out_dir.mkdir()
    for printed in (str(outside), f"{out_dir.resolve()}/../elsewhere.mp4", "../elsewhere.mp4"):
        monkeypatch.setattr(yt, "run", RecordingRun(lines=[f"FILE {printed}"]))
        with pytest.raises(MediaError) as info:
            yt.download(CANONICAL, out_dir, settings=tool_settings)
        assert str(tmp_path) not in info.value.user_message
    # A symlink inside out_dir that points outside is rejected as well.
    link = out_dir / "link.mp4"
    link.symlink_to(outside)
    monkeypatch.setattr(yt, "run", RecordingRun(lines=[f"FILE {link}"]))
    with pytest.raises(MediaError):
        yt.download(CANONICAL, out_dir, settings=tool_settings)
    assert outside.exists()


# --------------------------------------------------------------------------- metadata validation
def test_validate_metadata_ok(settings: Settings) -> None:
    meta = yt.validate_metadata(_ok_info(title="  Title\u0007 with control  "), settings=settings)
    assert meta == {
        "id": VIDEO_ID, "title": "Title with control", "duration": 602.0, "live_status": "not_live",
        "is_live": False, "availability": "public", "filesize_approx": 43_505_154, "uploader": "Some Channel",
        "license": "Creative Commons Attribution license (reuse allowed)", "webpage_url": CANONICAL,
    }
    minimal = yt.validate_metadata({"id": VIDEO_ID, "duration": 30.5}, settings=settings)
    assert minimal["license"] is None and minimal["filesize_approx"] is None and minimal["is_live"] is False


@pytest.mark.parametrize(("overrides", "fragment"), [
    ({"live_status": "is_live", "is_live": True}, "still running"),
    ({"live_status": None, "is_live": True}, "still running"),
    ({"live_status": "is_upcoming", "duration": None}, "hasn't started yet"),
    ({"live_status": "post_live"}, "still processing"),
    ({"duration": None}, "length"),
    ({"duration": 0}, "length"),
    ({"duration": "NA"}, "length"),
    ({"_type": "playlist"}, "Playlists are not supported"),
])
def test_validate_metadata_rejects(settings: Settings, overrides: dict[str, Any], fragment: str) -> None:
    with pytest.raises(InputError) as info:
        yt.validate_metadata(_ok_info(**overrides), settings=settings)
    assert fragment in info.value.user_message
    _assert_clean_message(info.value.user_message)


def test_validate_metadata_too_long_states_limit(settings: Settings) -> None:
    limited = settings.model_copy(update={"max_source_minutes": 120.0})
    yt.validate_metadata(_ok_info(duration=7200), settings=limited)  # exactly at the limit is fine
    with pytest.raises(InputError) as info:
        yt.validate_metadata(_ok_info(duration=7201), settings=limited)
    message = info.value.user_message
    assert "120-minute limit" in message and "CLIPPER_MAX_SOURCE_MINUTES" in message and "2:00:01" in message
    with pytest.raises(InputError) as info:
        yt.validate_metadata(_ok_info(duration=700), settings=settings.model_copy(update={"max_source_minutes": 7.5}))
    assert "7.5-minute limit" in info.value.user_message and "11:40" in info.value.user_message


@pytest.mark.parametrize("availability", ["private", "premium_only", "subscriber_only", "needs_auth"])
def test_validate_metadata_restricted_availability(settings: Settings, availability: str) -> None:
    with pytest.raises(SourceUnavailableError) as info:
        yt.validate_metadata(_ok_info(availability=availability), settings=settings)
    assert info.value.user_message.endswith(yt.UPLOAD_INSTEAD)


def test_fetch_metadata_live_and_too_long_via_fake_run(tool_settings: Settings,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(yt, "run", RecordingRun(stdout=json.dumps(_ok_info(live_status="is_live")).encode()))
    with pytest.raises(InputError, match="live stream"):
        yt.fetch_metadata(CANONICAL, settings=tool_settings)
    monkeypatch.setattr(yt, "run", RecordingRun(stdout=json.dumps(_ok_info(duration=10 * 3600)).encode()))
    with pytest.raises(InputError, match="CLIPPER_MAX_SOURCE_MINUTES"):
        yt.fetch_metadata(CANONICAL, settings=tool_settings)


@pytest.mark.parametrize("stdout", [b"", b"not json", b"[1, 2, 3]", b"null"])
def test_fetch_metadata_bad_json(tool_settings: Settings, monkeypatch: pytest.MonkeyPatch, stdout: bytes) -> None:
    monkeypatch.setattr(yt, "run", RecordingRun(stdout=stdout))
    with pytest.raises(SourceUnavailableError):
        yt.fetch_metadata(CANONICAL, settings=tool_settings)


def test_fetch_metadata_failure_is_classified(tool_settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(yt, "run", RecordingRun(returncode=1, stderr=BOT_STDERR))
    with pytest.raises(SourceUnavailableError) as info:
        yt.fetch_metadata(CANONICAL, settings=tool_settings)
    assert "bot" in info.value.user_message
    _assert_clean_message(info.value.user_message)


def test_format_duration() -> None:
    assert yt.format_duration(59.6) == "1:00"
    assert yt.format_duration(602) == "10:02"
    assert yt.format_duration(7201) == "2:00:01"


# --------------------------------------------------------------------------- real yt-dlp + local HTTP server
class _MediaHandler(http.server.BaseHTTPRequestHandler):
    """Serves one in-memory MP4, optionally throttled (chunk bytes every ``delay`` seconds)."""

    payload: bytes = b""
    chunk: int = 1 << 20
    delay: float = 0.0

    def _headers(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Length", str(len(self.payload)))
        self.end_headers()

    def do_HEAD(self) -> None:  # noqa: N802 - http.server API
        self._headers()

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        self._headers()
        try:
            for start in range(0, len(self.payload), self.chunk):
                self.wfile.write(self.payload[start:start + self.chunk])
                self.wfile.flush()
                if self.delay:
                    time.sleep(self.delay)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, *args: Any) -> None:
        pass


@pytest.fixture(scope="module")
def sample_mp4(tmp_path_factory: pytest.TempPathFactory) -> bytes:
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    path = tmp_path_factory.mktemp("media") / "clip.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=25",
         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-t", "3", "-shortest",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac",
         "-movflags", "+faststart", str(path)],
        check=True, timeout=60,
    )
    data = path.read_bytes()
    path.unlink()
    return data


@pytest.fixture
def media_server(sample_mp4: bytes) -> Iterator[tuple[str, type[_MediaHandler]]]:
    handler = type("Handler", (_MediaHandler,), {"payload": sample_mp4})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/clip.mp4", handler
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture
def real_settings(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Real yt-dlp settings; the JS runtime is irrelevant for direct HTTP media, so skip it if deno is absent."""
    if shutil.which("deno") is None:
        monkeypatch.setattr(yt, "js_runtime_args", lambda _settings: [])
    return settings


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_real_download_from_local_server(real_settings: Settings, media_server: tuple[str, Any],
                                         sample_mp4: bytes, tmp_path: Path) -> None:
    url, _ = media_server
    out_dir = tmp_path / "dl"
    events: list[tuple[int, int | None]] = []
    path = yt._download_url(url, out_dir, settings=real_settings, cancel=CancelToken(),
                            progress_cb=lambda d, t: events.append((d, t)), max_bytes=None)
    assert path == out_dir.resolve() / "clip.mp4"
    assert path.read_bytes() == sample_mp4
    assert events and events[-1] == (len(sample_mp4), len(sample_mp4))
    assert [d for d, _ in events] == sorted(d for d, _ in events)
    assert sorted(p.name for p in out_dir.iterdir()) == ["clip.mp4"]  # .part temp dir removed


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_real_download_too_large_is_stopped(real_settings: Settings, media_server: tuple[str, Any],
                                            tmp_path: Path) -> None:
    url, handler = media_server
    handler.chunk, handler.delay = 16 * 1024, 0.05
    out_dir = tmp_path / "dl"
    with pytest.raises(InputError) as info:
        yt._download_url(url, out_dir, settings=real_settings, cancel=CancelToken(), progress_cb=None,
                         max_bytes=50_000)
    assert "too large" in info.value.user_message and "49 KB" in info.value.user_message
    assert list(out_dir.iterdir()) == []


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_real_download_cancel_kills_and_cleans_up(real_settings: Settings, media_server: tuple[str, Any],
                                                  tmp_path: Path) -> None:
    url, handler = media_server
    handler.chunk, handler.delay = 8 * 1024, 0.2   # ~7 s for the whole file unless cancelled
    out_dir = tmp_path / "dl"
    token = CancelToken()
    started = time.monotonic()
    with pytest.raises(CancelledError):
        yt._download_url(url, out_dir, settings=real_settings, cancel=token,
                         progress_cb=lambda d, t: token.cancel(), max_bytes=None)
    assert time.monotonic() - started < 6
    assert list(out_dir.iterdir()) == []


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_real_download_shutdown_keeps_partial_files(real_settings: Settings, media_server: tuple[str, Any],
                                                    tmp_path: Path) -> None:
    url, handler = media_server
    handler.chunk, handler.delay = 8 * 1024, 0.2
    out_dir = tmp_path / "dl"
    token = CancelToken()
    with pytest.raises(InterruptedError_):
        yt._download_url(url, out_dir, settings=real_settings, cancel=token,
                         progress_cb=lambda d, t: token.shutdown.set(), max_bytes=None)
    assert (out_dir / ".part").is_dir()          # resumable: partial download kept
    assert not (out_dir / "clip.mp4").exists()


# --------------------------------------------------------------------------- live network (opt-in)
@pytest.mark.network
@pytest.mark.skipif(os.environ.get("CLIPPER_NETWORK_TESTS") != "1", reason="set CLIPPER_NETWORK_TESTS=1")
def test_live_fetch_metadata(settings: Settings) -> None:
    if shutil.which(settings.ytdlp_js_runtime) is None:
        pytest.skip("deno not installed")
    meta = yt.fetch_metadata(CANONICAL, settings=settings, cancel=CancelToken())
    assert meta["id"] == VIDEO_ID
    assert abs(meta["duration"] - 602) <= 2
    assert meta["live_status"] not in ("is_live", "is_upcoming", "post_live")
