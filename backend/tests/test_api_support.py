"""Shared fixtures and seeding helpers for the API tests (this module has no tests itself).

Two configurations are used throughout:

* local  – loopback, auth disabled (CSRF + Origin still enforced on mutations)
* hosted – https public origin, argon2 owner password, 40-char session secret
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterator

import pytest
from argon2 import PasswordHasher
from fastapi.testclient import TestClient

from clipper import db
from clipper.api.app import create_app
from clipper.config import Settings
from clipper.models import (
    ClipEdit,
    ClipSelection,
    MediaInfo,
    ProjectOptions,
    Segment,
    Transcript,
    Word,
)
from clipper.storage import Storage, new_id

LOCAL_ORIGIN = "http://127.0.0.1:8765"
HOSTED_ORIGIN = "https://clipper.example.test"
OWNER = "owner"
PASSWORD = "correct horse battery staple"
SESSION_SECRET = "x" * 40
# Cheap argon2 parameters keep the tests fast; verification reads them from the hash.
_HASHER = PasswordHasher(time_cost=1, memory_cost=8192, parallelism=1)


def make_local_settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = dict(
        data_dir=tmp_path / "data", frontend_dist=tmp_path / "no-frontend", mode="local",
        host="127.0.0.1", port=8765, auth="disabled", hw_encode="off",
    )
    values.update(overrides)
    return Settings(_env_file=None, **values)


def make_hosted_settings(tmp_path: Path, **overrides: Any) -> Settings:
    values: dict[str, Any] = dict(
        data_dir=tmp_path / "data", frontend_dist=tmp_path / "no-frontend", mode="hosted",
        host="0.0.0.0", port=8765, public_origin=HOSTED_ORIGIN,
        owner_username=OWNER, owner_password_hash=_HASHER.hash(PASSWORD),
        session_secret=SESSION_SECRET, login_max_failures=3, login_window_seconds=900,
        hw_encode="off",
    )
    values.update(overrides)
    return Settings(_env_file=None, **values)


@pytest.fixture
def local_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return make_local_settings(tmp_path)


@pytest.fixture
def hosted_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    return make_hosted_settings(tmp_path)


@pytest.fixture
def local_client(local_settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(local_settings), base_url=LOCAL_ORIGIN) as client:
        yield client


@pytest.fixture
def hosted_client(hosted_settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(hosted_settings), base_url=HOSTED_ORIGIN) as client:
        yield client


@pytest.fixture
def conn(local_settings: Settings, local_client: TestClient) -> Iterator[sqlite3.Connection]:
    """A DB connection to the local app's database (the app creates the schema)."""
    c = db.connect(local_settings.db_path)
    try:
        yield c
    finally:
        c.close()


@pytest.fixture
def app_storage(local_settings: Settings, local_client: TestClient) -> Storage:
    return Storage(local_settings.data_dir)


# ----------------------------------------------------------------------------- request helpers
def csrf_token(client: TestClient) -> str:
    token = client.get("/api/auth/session").json()["csrf_token"]
    assert token
    return token


def mutation_headers(client: TestClient, origin: str = LOCAL_ORIGIN) -> dict[str, str]:
    return {"Origin": origin, "X-CSRF-Token": csrf_token(client)}


def login(client: TestClient, password: str = PASSWORD, origin: str = HOSTED_ORIGIN):
    return client.post("/api/auth/login", json={"username": OWNER, "password": password},
                       headers={"Origin": origin})


# ----------------------------------------------------------------------------- seeding
def media_info(duration: float = 300.0, width: int = 1280, height: int = 720) -> MediaInfo:
    return MediaInfo(duration=duration, size_bytes=1000, container="mov,mp4,m4a,3gp,3g2,mj2", has_video=True,
                     has_audio=True, width=width, height=height, coded_width=width, coded_height=height,
                     fps=30.0, video_codec="h264", audio_codec="aac")


def seed_project(conn: sqlite3.Connection, storage: Storage, *, status: str = "ready",
                 duration: float = 300.0, source_file: Path | None = None, title: str = "Seeded talk",
                 options: ProjectOptions | None = None) -> str:
    """Insert a project row (YouTube kind); optionally copy ``source_file`` in as its source."""
    pid = new_id("p")
    source_rel = None
    if source_file is not None:
        src_dir = storage.source_dir(pid)
        src_dir.mkdir(parents=True, exist_ok=True)
        dest = src_dir / f"source{source_file.suffix}"
        dest.write_bytes(source_file.read_bytes())
        source_rel = storage.rel(dest)
    ts = db.now()
    conn.execute(
        """INSERT INTO projects(id, title, source_kind, source_url, youtube_id, source_path, source_sha256,
               media_info, options, status, created_at, updated_at)
           VALUES (?, ?, 'youtube', 'https://www.youtube.com/watch?v=nKyWusznRgQ', 'nKyWusznRgQ', ?, ?, ?, ?, ?, ?, ?)""",
        (pid, title, source_rel or "projects/x/source/missing.mp4", "0" * 64,
         media_info(duration).model_dump_json(), (options or ProjectOptions()).model_dump_json(),
         status, ts, ts),
    )
    return pid


def make_transcript(duration: float = 300.0, step: float = 0.5) -> Transcript:
    """One word every ``step`` seconds (each 0.4 s long) over the whole duration."""
    words: list[Word] = []
    i = 0
    while i * step + 0.4 <= duration:
        words.append(Word(id=f"w{i + 1:06d}", text=f"word{i + 1}", raw=f"word{i + 1}", start=round(i * step, 3),
                          end=round(i * step + 0.4, 3), segment_id="s00001", chunk_index=0))
        i += 1
    segment = Segment(id="s00001", start=0.0, end=words[-1].end, text=" ".join(w.text for w in words),
                      word_start=0, word_end=len(words) - 1, chunk_index=0)
    return Transcript(provider="openai", model="whisper-1", language="en", duration=duration, words=words,
                      segments=[segment], units=[], chunks=[], created_at=db.now())


def write_transcript(storage: Storage, project_id: str, transcript: Transcript) -> Path:
    path = storage.work_dir(project_id) / "transcript.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(transcript.model_dump_json(), encoding="utf-8")
    return path


def seed_clip(conn: sqlite3.Connection, project_id: str, *, rank: int = 1, src_start: float = 100.0,
              src_end: float = 130.0, title: str = "A great moment", render_state: str = "none",
              render_path: str | None = None, srt_path: str | None = None, thumb_path: str | None = None,
              render_hash: str | None = None) -> str:
    cid = new_id("c")
    selection = ClipSelection(candidate_id="c001", start_unit_id="u0001", end_unit_id="u0002",
                              src_start=src_start, src_end=src_end, title=title, hook="Hook line",
                              reason="Clear, self-contained point", topic="Topic", score=80.0,
                              sub_scores={"hook": 4.0}, select_model="gpt-6.1-sol", review_model="gpt-6.1-sol",
                              prompt_version="test")
    edit = ClipEdit(src_start=src_start, src_end=src_end)
    ts = db.now()
    conn.execute(
        """INSERT INTO clips(id, project_id, rank, title, reason, score, score_detail, selection, edit, framing,
               render_state, render_hash, render_path, srt_path, thumb_path, render_info, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (cid, project_id, rank, title, selection.reason, selection.score, db.dumps({"hook": 4.0}),
         selection.model_dump_json(), edit.model_dump_json(), render_state, render_hash, render_path,
         srt_path, thumb_path,
         db.dumps({"width": 1080, "height": 1920, "duration": src_end - src_start, "size_bytes": 10,
                   "video_codec": "h264", "audio_codec": "aac"}) if render_path else None, ts, ts),
    )
    return cid


def write_clip_file(storage: Storage, project_id: str, clip_id: str, name: str, data: bytes) -> str:
    """Create a file in the clip's dir and return its DB-style relative path."""
    cdir = storage.clip_dir(project_id, clip_id)
    cdir.mkdir(parents=True, exist_ok=True)
    path = cdir / name
    path.write_bytes(data)
    return storage.rel(path)


def jobs_for(conn: sqlite3.Connection, project_id: str) -> list[sqlite3.Row]:
    return conn.execute("SELECT * FROM jobs WHERE project_id=? ORDER BY created_at", (project_id,)).fetchall()
