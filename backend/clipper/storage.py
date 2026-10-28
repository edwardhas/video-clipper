"""App-owned storage layout, safe path handling, atomic writes and disk checks.

Layout under ``settings.data_dir``::

    clipper.db                       SQLite database (WAL)
    uploads/<upload_id>/<safe-name>  streamed uploads awaiting a project
    projects/<project_id>/
        source/source.<ext>          imported media (a copy; never the user's original)
        work/                        intermediate artifacts (audio, chunks, frames, transcript, llm)
        clips/<clip_id>/             rendered outputs  <render_hash>.mp4 / .srt / thumb.jpg
    cache/<namespace>/<key>.json     content-addressed cache shared across projects
    tmp/                             scratch space for in-progress writes

Database rows store paths *relative* to ``data_dir``; :func:`resolve` turns them
back into absolute paths and refuses anything that escapes the data directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .errors import DiskSpaceError

_ID_RE = re.compile(r"^[a-z]{1,3}_[0-9a-f]{12,32}$")
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def new_id(prefix: str) -> str:
    """Random, URL-safe identifier like ``p_3f9a0c1b2d4e``."""
    return f"{prefix}_{secrets.token_hex(8)}"


def is_valid_id(value: str) -> bool:
    return bool(_ID_RE.match(value or ""))


def safe_filename(name: str, default: str = "file", max_len: int = 80) -> str:
    """Reduce an untrusted filename to ``[A-Za-z0-9._-]`` without path parts."""
    base = os.path.basename((name or "").replace("\\", "/"))
    base = _SAFE_NAME_RE.sub("_", base).strip("._")
    if not base:
        base = default
    if len(base) > max_len:
        stem, dot, ext = base.rpartition(".")
        if dot and len(ext) <= 8:
            base = stem[: max_len - len(ext) - 1] + "." + ext
        else:
            base = base[:max_len]
    return base


class Storage:
    def __init__(self, data_dir: Path) -> None:
        self.root = Path(data_dir).resolve()
        # The data dir as configured; it may reach ``root`` through a symlink (/var -> /private/var).
        self._configured_root = Path(os.path.abspath(data_dir))

    # -- directories -----------------------------------------------------------
    def ensure(self) -> None:
        for sub in ("uploads", "projects", "cache", "tmp"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)

    def project_dir(self, project_id: str) -> Path:
        self._check_id(project_id)
        return self.root / "projects" / project_id

    def source_dir(self, project_id: str) -> Path:
        return self.project_dir(project_id) / "source"

    def work_dir(self, project_id: str) -> Path:
        return self.project_dir(project_id) / "work"

    def clip_dir(self, project_id: str, clip_id: str) -> Path:
        self._check_id(clip_id)
        return self.project_dir(project_id) / "clips" / clip_id

    def upload_dir(self, upload_id: str) -> Path:
        self._check_id(upload_id)
        return self.root / "uploads" / upload_id

    def cache_path(self, namespace: str, key: str, suffix: str = ".json") -> Path:
        if not re.fullmatch(r"[a-z0-9_-]{1,40}", namespace) or not re.fullmatch(r"[0-9a-f]{16,128}", key):
            raise ValueError("invalid cache namespace/key")
        return self.root / "cache" / namespace / f"{key}{suffix}"

    def tmp_dir(self) -> Path:
        d = self.root / "tmp"
        d.mkdir(parents=True, exist_ok=True)
        return d

    @staticmethod
    def _check_id(value: str) -> None:
        if not is_valid_id(value):
            raise ValueError("invalid identifier")

    # -- relative/absolute -----------------------------------------------------
    def rel(self, path: Path) -> str:
        """Path relative to the data dir (for storing in the DB)."""
        p = Path(path).resolve()
        if not p.is_relative_to(self.root):
            raise ValueError("path outside data directory")
        return p.relative_to(self.root).as_posix()

    def resolve(self, rel_path: str | None) -> Path:
        """Resolve a DB-stored relative path, refusing traversal outside data_dir."""
        if not rel_path:
            raise ValueError("empty path")
        if os.path.isabs(rel_path) or "\x00" in rel_path:
            raise ValueError("absolute or invalid path")
        p = (self.root / rel_path).resolve()
        if not p.is_relative_to(self.root) or p == self.root:
            raise ValueError("path outside data directory")
        return p

    # -- deletion (app-managed artifacts only) -----------------------------------
    def _deletion_target(self, path: Path) -> Path:
        """``path`` as a location under ``root`` whose last component is NOT resolved.

        Deleting acts on what sits at that location: a symbolic link is removed itself, never
        its target (``resolve()`` would turn the link into its target). The path must lie under
        the data dir, and no directory between the data dir and the last component may be a
        symlink, so a linked ``clips/<id>`` (e.g. from a restored backup) cannot redirect a
        deletion into another project or onto ``clipper.db``.
        """
        raw = Path(os.path.abspath(path))
        for base in (self.root, self._configured_root):
            if raw.is_relative_to(base) and raw != base:
                break
        else:
            raise ValueError("refusing to delete outside data directory")
        parts = raw.relative_to(base).parts
        current = self.root
        for part in parts[:-1]:
            current = current / part
            if current.is_symlink():
                raise ValueError("refusing to delete through a symbolic link")
        return self.root.joinpath(*parts)

    def delete_tree(self, path: Path) -> None:
        """Delete a directory that must live under projects/, uploads/, cache/ or tmp/."""
        p = self._deletion_target(path)
        allowed = [self.root / s for s in ("projects", "uploads", "cache", "tmp")]
        if not any(p.is_relative_to(a) and p != a for a in allowed):
            raise ValueError("refusing to delete outside app-managed directories")
        if p.is_symlink():
            p.unlink()                  # the link only; rmtree would refuse it anyway
        elif p.is_dir():
            shutil.rmtree(p)            # does not follow symlinks inside the tree
        elif p.exists():
            p.unlink()

    def delete_file(self, path: Path) -> None:
        p = self._deletion_target(path)
        if not p.is_relative_to(self.root) or p == self.root:
            raise ValueError("refusing to delete outside data directory")
        if p.is_symlink() or p.is_file():
            p.unlink()

    # -- disk ------------------------------------------------------------------
    def free_bytes(self) -> int:
        self.root.mkdir(parents=True, exist_ok=True)
        return shutil.disk_usage(self.root).free

    def require_free(self, needed_bytes: int, reserve_mb: int, what: str) -> None:
        free = self.free_bytes()
        reserve = reserve_mb * 1024 * 1024
        if free - needed_bytes < reserve:
            raise DiskSpaceError(
                f"Not enough disk space for {what}: about {_fmt_bytes(needed_bytes)} needed, "
                f"{_fmt_bytes(free)} free, and {_fmt_bytes(reserve)} is kept in reserve. "
                "Delete old projects in Settings or free disk space, then retry."
            )

    def usage_bytes(self, path: Path | None = None) -> int:
        target = Path(path) if path else self.root
        total = 0
        for dirpath, _dirs, files in os.walk(target):
            for f in files:
                try:
                    total += os.lstat(os.path.join(dirpath, f)).st_size
                except OSError:
                    pass
        return total


def _fmt_bytes(n: int | float) -> str:
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


fmt_bytes = _fmt_bytes


# -- atomic writes ---------------------------------------------------------------
@contextmanager
def atomic_path(final: Path) -> Iterator[Path]:
    """Yield a temp path in the same directory; rename over ``final`` on success.

    The temp file keeps the final suffix so tools that sniff extensions (ffmpeg)
    behave. On exception the temp file is removed and ``final`` is untouched.
    """
    final = Path(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{final.stem}.", suffix=f".tmp{final.suffix}", dir=final.parent)
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        yield tmp_path
        os.replace(tmp_path, final)
    except BaseException:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass
        raise


def write_json_atomic(path: Path, data: Any) -> None:
    with atomic_path(path) as tmp:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, separators=(",", ":"))
            fh.flush()
            os.fsync(fh.fileno())


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def sha256_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def stable_hash(*parts: Any) -> str:
    """Deterministic sha256 over JSON-serialisable parts (used for cache keys)."""
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
