"""Runtime configuration for Local Clipper.

All settings come from environment variables (prefix ``CLIPPER_``) or an ignored
``.env`` file at the repository root. ``OPENAI_API_KEY`` is read here and only
here; it is never returned to clients or written to logs.

Two execution modes share one code base:

* ``local``  – native run on the owner's computer, bound to loopback. Auth may be
  disabled, but only while the bind address, the public origin and every allowed
  host/origin are loopback (see :attr:`Settings.exposure_reasons`).
* ``hosted`` – container/server deployment behind an HTTPS reverse proxy. Auth is
  mandatory and startup fails if secrets are missing.
"""

from __future__ import annotations

import ipaddress
import os
from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = Path(__file__).resolve().parent / "assets"


class ConfigError(RuntimeError):
    """Raised when the configuration is unsafe or incomplete for the chosen mode."""


def _is_loopback(host: str) -> bool:
    host = (host or "").strip().lower()
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    if host in ("localhost",):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _host_entry_name(entry: str) -> str:
    """Hostname of an allowed-host entry (``name``, ``name:port``, ``[v6]:port`` or bare v6)."""
    entry = entry.strip().lower()
    if entry.startswith("["):
        end = entry.find("]")
        return entry[1:end] if end != -1 else entry
    name, sep, port = entry.rpartition(":")
    if sep and port.isdigit() and ":" not in name:
        return name
    return entry


def _origin_name(origin: str) -> str:
    """Hostname of an origin URL; the raw value if it cannot be parsed (treated as non-loopback)."""
    try:
        return urlsplit(origin.strip()).hostname or origin.strip()
    except ValueError:
        return origin.strip()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CLIPPER_",
        env_file=(str(REPO_ROOT / ".env"),),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Mode & network -------------------------------------------------------
    mode: Literal["local", "hosted"] = "local"
    host: str = "127.0.0.1"
    port: int = 8765
    # Origin the browser uses to reach the app, e.g. http://127.0.0.1:8765 or
    # https://clipper.example.com (example only; the real hostname is configurable).
    public_origin: str = ""
    # Extra allowed Host header values / origins (comma separated). Useful for the
    # Vite dev server (http://localhost:5173) during development.
    allowed_hosts: str = ""
    allowed_origins: str = ""
    # Comma separated proxy IPs/CIDRs whose X-Forwarded-For header is trusted (both modes; in
    # local mode 127.0.0.1 for a same-host reverse proxy or tunnel). Empty = trust nobody.
    trusted_proxies: str = ""

    # --- Storage --------------------------------------------------------------
    data_dir: Path = REPO_ROOT / "data"
    frontend_dist: Path = REPO_ROOT / "frontend" / "dist"
    min_free_disk_mb: int = 1024

    # --- Auth -----------------------------------------------------------------
    auth: Literal["auto", "required", "disabled"] = "auto"
    owner_username: str = "owner"
    owner_password_hash: SecretStr | None = None
    session_secret: SecretStr | None = None
    session_ttl_hours: float = 12.0
    session_idle_minutes: float = 120.0
    cookie_secure: Literal["auto", "true", "false"] = "auto"
    login_max_failures: int = 5
    login_window_seconds: int = 900

    # --- Limits & defaults ----------------------------------------------------
    max_source_minutes: float = 120.0
    max_upload_mb: int = 4096
    max_clips: int = 10
    default_clip_count: int = 5
    default_min_seconds: float = 20.0
    default_max_seconds: float = 60.0

    # --- AI providers ---------------------------------------------------------
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    transcribe_model: str = "whisper-1"
    # Defaults verified against developers.openai.com model pages on 2026-10-05:
    # gpt-6.1-sol supports image input + strict Structured Outputs via the Responses API;
    # reasoning effort values: low, medium (default), high, xhigh, max (no none/minimal).
    # Cheaper alternative: gpt-6-luna. See docs/MODELS.md.
    select_model: str = "gpt-6.1-sol"
    review_model: str = "gpt-6.1-sol"
    select_reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh", "max", ""] = "low"
    review_reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh", "max", ""] = "low"
    select_max_output_tokens: int = 32000
    review_max_output_tokens: int = 32000
    openai_timeout_seconds: float = 300.0
    openai_max_retries: int = 3
    transcribe_chunk_seconds: float = 600.0
    transcribe_overlap_seconds: float = 3.0
    # Chunk container sent to /v1/audio/transcriptions (25 MB limit per request).
    # flac: lossless + compressed, no encoder priming offset (listed in the API reference);
    # wav: listed everywhere but larger; mp3: smallest, ~69 ms priming offset at 16 kHz.
    transcribe_audio_format: Literal["flac", "wav", "mp3"] = "flac"

    # --- Media tools ----------------------------------------------------------
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"
    ytdlp_js_runtime: str = "deno"
    hw_encode: Literal["auto", "off"] = "auto"
    render_preset: str = "veryfast"
    render_crf: int = 20

    log_level: str = "INFO"

    # ------------------------------------------------------------------------
    @field_validator("data_dir", "frontend_dist", mode="before")
    @classmethod
    def _expand(cls, v: object) -> object:
        if isinstance(v, str):
            return Path(os.path.expanduser(v))
        return v

    # Derived values ----------------------------------------------------------
    @property
    def effective_public_origin(self) -> str:
        if self.public_origin:
            return self.public_origin.rstrip("/")
        host = f"[{self.host}]" if ":" in self.host and not self.host.startswith("[") else self.host
        return f"http://{host}:{self.port}"

    @property
    def auth_enabled(self) -> bool:
        if self.mode == "hosted":
            return True
        if self.auth == "required":
            return True
        if self.auth == "disabled":
            return False
        # auto in local mode: off only while every address the app accepts is loopback
        return bool(self.exposure_reasons)

    @property
    def exposure_reasons(self) -> list[str]:
        """Settings that make the app reachable from other machines (non-loopback names).

        The bind address alone is not enough: a same-machine reverse proxy or tunnel can
        forward an external name to 127.0.0.1, and naming it in the public origin or the
        allowed hosts/origins is what lets those requests through the Host/Origin checks.
        """
        reasons: list[str] = []
        if not _is_loopback(self.host):
            reasons.append(f"CLIPPER_HOST={self.host}")
        if self.public_origin and not _is_loopback(_origin_name(self.public_origin)):
            reasons.append(f"CLIPPER_PUBLIC_ORIGIN={self.public_origin}")
        for entry in (h.strip() for h in self.allowed_hosts.split(",") if h.strip()):
            if not _is_loopback(_host_entry_name(entry)):
                reasons.append(f"CLIPPER_ALLOWED_HOSTS entry {entry}")
        for entry in (o.strip() for o in self.allowed_origins.split(",") if o.strip()):
            if not _is_loopback(_origin_name(entry)):
                reasons.append(f"CLIPPER_ALLOWED_ORIGINS entry {entry}")
        return reasons

    @property
    def secure_cookies(self) -> bool:
        if self.cookie_secure == "true":
            return True
        if self.cookie_secure == "false":
            return False
        return self.mode == "hosted" or self.effective_public_origin.startswith("https://")

    @property
    def allowed_host_list(self) -> list[str]:
        hosts: set[str] = set()
        origin_host = urlsplit(self.effective_public_origin).hostname
        if origin_host:
            hosts.add(origin_host)
        if self.mode == "local":
            hosts.update({"127.0.0.1", "localhost", "[::1]", "::1"})
        hosts.update(h.strip() for h in self.allowed_hosts.split(",") if h.strip())
        return sorted(hosts)

    @property
    def allowed_origin_list(self) -> list[str]:
        origins = {self.effective_public_origin}
        if self.mode == "local":
            origins.update({f"http://127.0.0.1:{self.port}", f"http://localhost:{self.port}"})
        origins.update(o.strip().rstrip("/") for o in self.allowed_origins.split(",") if o.strip())
        return sorted(origins)

    @property
    def db_path(self) -> Path:
        return self.data_dir / "clipper.db"

    @property
    def openai_configured(self) -> bool:
        return bool(self.openai_api_key and self.openai_api_key.get_secret_value().strip())

    def validate_for_startup(self) -> None:
        """Reject unsafe combinations. Called by both the API and the worker."""
        problems: list[str] = []
        exposure = self.exposure_reasons
        if self.mode == "local" and self.auth == "disabled" and exposure:
            problems.append(
                "CLIPPER_AUTH=disabled is only allowed in local mode when every configured address is "
                f"loopback (127.0.0.1, ::1, localhost); not loopback: {', '.join(exposure)}."
            )
        if self.mode == "hosted" and self.auth == "disabled":
            problems.append("Authentication cannot be disabled in hosted mode.")
        if self.auth_enabled:
            why = (f" Sign-in is on because the app is reachable from other machines ({', '.join(exposure)})."
                   if self.mode == "local" and self.auth == "auto" else "")
            if not self.owner_password_hash or not self.owner_password_hash.get_secret_value().startswith("$argon2"):
                problems.append(
                    "CLIPPER_OWNER_PASSWORD_HASH must be set to an argon2 hash "
                    f"(generate one with scripts/hash-password.sh).{why}"
                )
            secret = self.session_secret.get_secret_value() if self.session_secret else ""
            if len(secret) < 32:
                problems.append(f"CLIPPER_SESSION_SECRET must be set to at least 32 random characters.{why}")
        if self.mode == "hosted":
            if not self.public_origin.startswith("https://"):
                problems.append("CLIPPER_PUBLIC_ORIGIN must be an https:// origin in hosted mode.")
        if self.default_min_seconds >= self.default_max_seconds:
            problems.append("CLIPPER_DEFAULT_MIN_SECONDS must be below CLIPPER_DEFAULT_MAX_SECONDS.")
        if problems:
            raise ConfigError("Unsafe or incomplete configuration:\n  - " + "\n  - ".join(problems))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    get_settings.cache_clear()
