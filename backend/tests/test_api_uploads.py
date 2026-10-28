"""POST /api/uploads: streamed writes, limits, extension allow-list, filename sanitising, disk checks."""

from __future__ import annotations

from pathlib import Path

import anyio
import pytest
from fastapi.testclient import TestClient

from clipper import db
from clipper.api import routes_uploads
from clipper.api.app import create_app
from clipper.api.deps import ApiError
from clipper.storage import Storage
from test_api_support import (
    LOCAL_ORIGIN,
    app_storage,
    conn,
    local_client,
    local_settings,
    make_local_settings,
    mutation_headers,
)

__all__ = ["app_storage", "conn", "local_client", "local_settings"]

MB = 1024 * 1024


def _upload(client: TestClient, data: bytes, name: str, **extra: str):
    headers = dict(mutation_headers(client), **{"X-Filename": name, "Content-Type": "application/octet-stream"})
    headers.update(extra)
    return client.post("/api/uploads", content=data, headers=headers)


def _files_under(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file())


def test_streamed_upload_success(local_client: TestClient, conn, app_storage: Storage) -> None:
    payload = bytes(range(256)) * 9000            # ~2.3 MB: several receive chunks
    r = _upload(local_client, payload, "My%20Talk%20(final).MP4")
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["filename"] == "My_Talk_final_.MP4" and body["size_bytes"] == len(payload)
    row = conn.execute("SELECT * FROM uploads WHERE id=?", (body["id"],)).fetchone()
    assert row["state"] == "complete" and row["size_bytes"] == len(payload) and row["project_id"] is None
    stored = app_storage.resolve(row["path"])
    assert stored.parent == app_storage.upload_dir(body["id"])
    assert stored.read_bytes() == payload
    assert not list(stored.parent.glob("*.part"))


def test_too_large_is_rejected_before_reading(tmp_path: Path) -> None:
    settings = make_local_settings(tmp_path, max_upload_mb=1)
    with TestClient(create_app(settings), base_url=LOCAL_ORIGIN) as client:
        r = _upload(client, b"\0" * (MB + 1), "big.mp4")
        assert r.status_code == 413 and r.json()["code"] == "too_large"
        assert _files_under(Storage(settings.data_dir).root / "uploads") == []


def test_missing_content_length_is_refused(local_client: TestClient) -> None:
    def chunks():
        yield b"abc"
        yield b"def"

    headers = dict(mutation_headers(local_client), **{"X-Filename": "a.mp4"})
    r = local_client.post("/api/uploads", content=chunks(), headers=headers)
    assert r.status_code == 411


@pytest.mark.parametrize("name", ["evil.exe", "notes.txt", "video", "archive.mp4.zip", ""])
def test_bad_extension_or_missing_name(local_client: TestClient, app_storage: Storage, name: str) -> None:
    r = _upload(local_client, b"data", name) if name else local_client.post(
        "/api/uploads", content=b"data", headers=mutation_headers(local_client))
    assert r.status_code == 422 and r.json()["code"] == "invalid_input"
    assert _files_under(app_storage.root / "uploads") == []


def test_empty_upload_is_rejected(local_client: TestClient) -> None:
    assert _upload(local_client, b"", "empty.mp4").status_code == 422


@pytest.mark.parametrize("name,expected", [
    ("..%2F..%2F..%2Fevil.mp4", "evil.mp4"),
    ("../../etc/passwd.mov", "passwd.mov"),
    ("..\\..\\windows\\x.mkv", "x.mkv"),
    ("%2e%2e%2f.webm", "webm"),
])
def test_filename_traversal_is_neutralised(local_client: TestClient, app_storage: Storage, tmp_path: Path,
                                           name: str, expected: str) -> None:
    r = _upload(local_client, b"0123456789", name)
    if not expected.count("."):                    # nothing but an extension left -> rejected
        assert r.status_code == 422
        return
    assert r.status_code == 201, r.text
    assert r.json()["filename"] == expected
    files = _files_under(tmp_path)
    uploaded = [f for f in files if f.name == expected]
    assert len(uploaded) == 1 and uploaded[0].is_relative_to(app_storage.root / "uploads")


def test_disk_space_check(local_client: TestClient, app_storage: Storage, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(Storage, "free_bytes", lambda self: 10 * MB)
    r = _upload(local_client, b"x" * 1000, "clip.mp4")
    assert r.status_code == 507 and r.json()["code"] == "disk_full"
    assert str(app_storage.root) not in r.text
    assert _files_under(app_storage.root / "uploads") == []


def test_upload_can_be_used_by_exactly_one_project(local_client: TestClient, conn) -> None:
    upload_id = _upload(local_client, b"0123456789", "Keynote_2026.mp4").json()["id"]
    body = {"source": {"kind": "upload", "upload_id": upload_id}}
    r = local_client.post("/api/projects", json=body, headers=mutation_headers(local_client))
    assert r.status_code == 201, r.text
    assert r.json()["title"] == "Keynote 2026" and r.json()["source_kind"] == "upload"
    row = conn.execute("SELECT state, project_id FROM uploads WHERE id=?", (upload_id,)).fetchone()
    assert row["state"] == "attached" and row["project_id"] == r.json()["id"]
    again = local_client.post("/api/projects", json=body, headers=mutation_headers(local_client))
    assert again.status_code == 409


def test_unknown_or_incomplete_upload(local_client: TestClient, conn) -> None:
    headers = mutation_headers(local_client)
    r = local_client.post("/api/projects", json={"source": {"kind": "upload", "upload_id": "u_0123456789abcdef"}},
                          headers=headers)
    assert r.status_code == 404
    r = local_client.post("/api/projects", json={"source": {"kind": "upload", "upload_id": "../../etc"}},
                          headers=headers)
    assert r.status_code == 404
    conn.execute("INSERT INTO uploads(id, filename, path, size_bytes, state, created_at) "
                 "VALUES ('u_aaaaaaaaaaaaaaaa', 'a.mp4', 'uploads/u_aaaaaaaaaaaaaaaa/a.mp4', 0, 'receiving', ?)",
                 (db.now(),))
    r = local_client.post("/api/projects", json={"source": {"kind": "upload", "upload_id": "u_aaaaaaaaaaaaaaaa"}},
                          headers=headers)
    assert r.status_code == 409
    assert conn.execute("SELECT COUNT(*) FROM projects").fetchone()[0] == 0


def test_stale_uploads_are_purged(local_client: TestClient, conn, app_storage: Storage) -> None:
    old_dir = app_storage.upload_dir("u_bbbbbbbbbbbbbbbb")
    old_dir.mkdir(parents=True)
    (old_dir / "old.mp4").write_bytes(b"old")
    conn.execute("INSERT INTO uploads(id, filename, path, size_bytes, state, created_at) "
                 "VALUES ('u_bbbbbbbbbbbbbbbb', 'old.mp4', 'uploads/u_bbbbbbbbbbbbbbbb/old.mp4', 3, 'complete', ?)",
                 (db.now() - 4 * 24 * 3600,))
    assert _upload(local_client, b"new", "new.mp4").status_code == 201
    assert not old_dir.exists()
    assert conn.execute("SELECT COUNT(*) FROM uploads WHERE id='u_bbbbbbbbbbbbbbbb'").fetchone()[0] == 0


# ----------------------------------------------------------------------------- unit: streaming writer
def test_write_stream_enforces_limit(tmp_path: Path) -> None:
    async def chunks():
        for _ in range(5):
            yield b"x" * 1000

    dest = tmp_path / "out.part"
    too_large = ApiError(413, "too large", "too_large")

    async def main() -> None:
        with pytest.raises(ApiError):
            await routes_uploads.write_stream(chunks(), dest, limit=2500, too_large=too_large)
        assert await routes_uploads.write_stream(chunks(), dest, limit=5000, too_large=too_large) == 5000

    anyio.run(main)
    assert dest.stat().st_size == 5000


def test_clean_filename_unit() -> None:
    assert routes_uploads.clean_filename("clip%20one.mov") == "clip_one.mov"
    with pytest.raises(ApiError):
        routes_uploads.clean_filename("payload.sh")
    with pytest.raises(ApiError):
        routes_uploads.clean_filename(None)
