"""Shared pytest fixtures.

* ``settings`` – isolated Settings pointing at a temp data dir (local mode, auth off).
* ``storage``  – Storage for that data dir with directories created.
* ``db_path``  – initialised SQLite database path.
* ``ffmpeg_available`` – skip marker helper for real FFmpeg integration tests.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from clipper import db
from clipper.config import Settings, reset_settings_cache
from clipper.storage import Storage


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    for var in ("OPENAI_API_KEY",):
        monkeypatch.delenv(var, raising=False)
    reset_settings_cache()
    s = Settings(_env_file=None, data_dir=tmp_path / "data", mode="local", host="127.0.0.1", port=8765,
                 auth="disabled", hw_encode="off")
    return s


@pytest.fixture
def storage(settings: Settings) -> Storage:
    st = Storage(settings.data_dir)
    st.ensure()
    return st


@pytest.fixture
def db_path(settings: Settings, storage: Storage) -> Path:
    db.init_db(settings.db_path)
    return settings.db_path


requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None, reason="ffmpeg/ffprobe not installed"
)
