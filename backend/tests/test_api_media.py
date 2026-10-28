"""Media routes: Range playback, downloads, thumbnails, source frames (real FFmpeg) and ZIP export."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from clipper.api import routes_media
from clipper.procs import run
from clipper.storage import Storage
from conftest import requires_ffmpeg
from test_api_support import (
    app_storage,
    conn,
    local_client,
    local_settings,
    seed_clip,
    seed_project,
    write_clip_file,
)

__all__ = ["app_storage", "conn", "local_client", "local_settings"]

FAKE_MP4 = bytes(range(256)) * 40          # 10 240 bytes of deterministic content


def _ready_clip(conn, storage: Storage, pid: str, *, title: str, rank: int, data: bytes = FAKE_MP4,
                srt: bool = True, state: str = "ready", render_hash: str = "abc123") -> str:
    cid = seed_clip(conn, pid, title=title, rank=rank, render_state=state, render_hash=render_hash)
    mp4 = write_clip_file(storage, pid, cid, f"{render_hash}.mp4", data)
    srt_rel = write_clip_file(storage, pid, cid, f"{render_hash}.srt", b"1\n00:00:00,000 --> 00:00:01,000\nHi\n") \
        if srt else None
    thumb = write_clip_file(storage, pid, cid, "thumb.jpg", b"\xff\xd8\xff\xe0fakejpeg")
    conn.execute("UPDATE clips SET render_path=?, srt_path=?, thumb_path=? WHERE id=?", (mp4, srt_rel, thumb, cid))
    return cid


def _make_mp4(path: Path, seconds: float = 1.0) -> None:
    run(["ffmpeg", "-hide_banner", "-nostdin", "-v", "error", "-y",
         "-f", "lavfi", "-i", f"testsrc2=size=160x284:rate=15:duration={seconds}",
         "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={seconds}",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
         "-movflags", "+faststart", str(path)], timeout=60)


# ----------------------------------------------------------------------------- clip files
def test_video_range_requests(local_client: TestClient, conn, app_storage: Storage) -> None:
    pid = seed_project(conn, app_storage)
    cid = _ready_clip(conn, app_storage, pid, title="Range test", rank=1)
    full = local_client.get(f"/media/clips/{cid}/video.mp4")
    assert full.status_code == 200 and full.content == FAKE_MP4
    assert full.headers["content-type"] == "video/mp4" and full.headers["accept-ranges"] == "bytes"
    assert "attachment" not in full.headers.get("content-disposition", "")
    assert full.headers["cache-control"] == "private, no-cache"
    part = local_client.get(f"/media/clips/{cid}/video.mp4?v=abc123", headers={"Range": "bytes=100-199"})
    assert part.status_code == 206
    assert part.headers["content-range"] == f"bytes 100-199/{len(FAKE_MP4)}"
    assert part.content == FAKE_MP4[100:200]
    assert part.headers["cache-control"] == "private, max-age=3600"
    tail = local_client.get(f"/media/clips/{cid}/video.mp4", headers={"Range": "bytes=-16"})
    assert tail.status_code == 206 and tail.content == FAKE_MP4[-16:]
    stale_v = local_client.get(f"/media/clips/{cid}/video.mp4?v=old")
    assert stale_v.headers["cache-control"] == "private, no-cache"
    head = local_client.head(f"/media/clips/{cid}/video.mp4")
    assert head.status_code == 200 and int(head.headers["content-length"]) == len(FAKE_MP4)


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_real_mp4_range_playback(local_client: TestClient, conn, app_storage: Storage, tmp_path: Path) -> None:
    src = tmp_path / "clip.mp4"
    _make_mp4(src)
    data = src.read_bytes()
    pid = seed_project(conn, app_storage)
    cid = _ready_clip(conn, app_storage, pid, title="Real clip", rank=1, data=data)
    r = local_client.get(f"/media/clips/{cid}/video.mp4", headers={"Range": "bytes=0-1023"})
    assert r.status_code == 206 and r.headers["content-range"] == f"bytes 0-1023/{len(data)}"
    assert r.content[4:8] == b"ftyp"                       # an MP4 box header, as a browser would read it
    out = tmp_path / "roundtrip.mp4"
    out.write_bytes(local_client.get(f"/media/clips/{cid}/download.mp4").content)
    probe = run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", str(out)])
    assert set(probe.stdout.decode().split()) == {"video", "audio"}


def test_download_srt_and_thumbs(local_client: TestClient, conn, app_storage: Storage) -> None:
    pid = seed_project(conn, app_storage, title="Talk")
    cid = _ready_clip(conn, app_storage, pid, title="Why Sleep Matters: A Story!", rank=3)
    dl = local_client.get(f"/media/clips/{cid}/download.mp4")
    assert dl.status_code == 200
    assert dl.headers["content-disposition"] == 'attachment; filename="why-sleep-matters-a-story-3.mp4"'
    srt = local_client.get(f"/media/clips/{cid}/captions.srt")
    assert srt.status_code == 200 and "attachment" in srt.headers["content-disposition"]
    assert srt.text.startswith("1\n")
    thumb = local_client.get(f"/media/clips/{cid}/thumb.jpg")
    assert thumb.status_code == 200 and thumb.headers["content-type"] == "image/jpeg"
    project_thumb = app_storage.source_dir(pid) / "thumb.jpg"
    project_thumb.parent.mkdir(parents=True, exist_ok=True)
    project_thumb.write_bytes(b"\xff\xd8\xffproject")
    conn.execute("UPDATE projects SET thumb_path=? WHERE id=?", (app_storage.rel(project_thumb), pid))
    assert local_client.get(f"/media/projects/{pid}/thumb.jpg").content == b"\xff\xd8\xffproject"


def test_missing_files_and_traversal(local_client: TestClient, conn, app_storage: Storage, tmp_path: Path) -> None:
    (tmp_path / "secret.txt").write_text("secret")
    pid = seed_project(conn, app_storage)
    cid = seed_clip(conn, pid)                         # never rendered
    for path in ("video.mp4", "download.mp4", "captions.srt", "thumb.jpg"):
        assert local_client.get(f"/media/clips/{cid}/{path}").status_code == 404
    assert local_client.get(f"/media/projects/{pid}/thumb.jpg").status_code == 404
    for bad in ("../../secret.txt", "/etc/hosts", f"projects/{pid}/../../../secret.txt", "clipper.db"):
        conn.execute("UPDATE clips SET render_path=? WHERE id=?", (bad, cid))
        r = local_client.get(f"/media/clips/{cid}/video.mp4")
        assert r.status_code == 404 and "secret" not in r.text
    conn.execute("UPDATE clips SET render_path=? WHERE id=?", (f"projects/{pid}/clips/{cid}/gone.mp4", cid))
    assert local_client.get(f"/media/clips/{cid}/video.mp4").status_code == 404
    for bad_id in ("..%2F..%2Fsecret.txt", "c_..", "C_0123456789ABCDEF", "x" * 300):
        r = local_client.get(f"/media/clips/{bad_id}/video.mp4")
        assert r.status_code in (404, 422) and "secret" not in r.text


# ----------------------------------------------------------------------------- frames
@pytest.mark.ffmpeg
@requires_ffmpeg
def test_source_frame_extraction(local_client: TestClient, conn, app_storage: Storage, tmp_path: Path) -> None:
    src = tmp_path / "source.mp4"
    _make_mp4(src, seconds=2.0)
    pid = seed_project(conn, app_storage, duration=2.0, source_file=src)
    r = local_client.get(f"/media/projects/{pid}/frame.jpg?t=0.5")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert r.content[:3] == b"\xff\xd8\xff"
    cached = app_storage.work_dir(pid) / "frames" / "preview_500.jpg"
    assert cached.is_file()
    assert local_client.get(f"/media/projects/{pid}/frame.jpg?t=2.0").status_code == 200   # end clamps
    assert local_client.get(f"/media/projects/{pid}/frame.jpg?t=2.5").status_code == 422
    assert local_client.get(f"/media/projects/{pid}/frame.jpg?t=-1").status_code == 422
    assert local_client.get(f"/media/projects/{pid}/frame.jpg?t=abc").status_code == 422
    assert local_client.get(f"/media/projects/{pid}/frame.jpg").status_code == 422


def test_frame_cache_is_bounded(local_client: TestClient, conn, app_storage: Storage,
                                monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import os
    import clipper.media.frames as frames

    src = tmp_path / "source.mp4"
    src.write_bytes(b"not decoded in this test")
    pid = seed_project(conn, app_storage, duration=100.0, source_file=src)
    fdir = app_storage.work_dir(pid) / "frames"
    fdir.mkdir(parents=True)
    for i in range(60):
        f = fdir / f"preview_{i}.jpg"
        f.write_bytes(b"old")
        os.utime(f, (1_000_000 + i, 1_000_000 + i))
    (fdir / "c001_0.jpg").write_bytes(b"review frame")      # review frames are not touched

    def fake_extract(src_path, t, out_jpg, *, max_width=640, settings, cancel=None):
        Path(out_jpg).write_bytes(b"\xff\xd8\xffnew")
        return Path(out_jpg)

    monkeypatch.setattr(frames, "extract_frame", fake_extract)
    assert local_client.get(f"/media/projects/{pid}/frame.jpg?t=50").content == b"\xff\xd8\xffnew"
    previews = list(fdir.glob("preview_*.jpg"))
    assert len(previews) == routes_media.MAX_PREVIEW_FRAMES
    assert (fdir / "preview_50000.jpg").exists() and not (fdir / "preview_0.jpg").exists()
    assert (fdir / "c001_0.jpg").exists()


def test_frame_extraction_failure_is_plain(local_client: TestClient, conn, app_storage: Storage,
                                           monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    import clipper.media.frames as frames
    from clipper.errors import MediaError

    src = tmp_path / "source.mp4"
    src.write_bytes(b"x")
    pid = seed_project(conn, app_storage, duration=10.0, source_file=src)

    def failing(*_a, **_k):
        raise MediaError("Could not read a video frame from the source.", detail=f"ffmpeg failed on {src}")

    monkeypatch.setattr(frames, "extract_frame", failing)
    r = local_client.get(f"/media/projects/{pid}/frame.jpg?t=1")
    assert r.status_code == 422 and r.json()["detail"] == "Could not read a video frame from the source."
    assert str(tmp_path) not in r.text


# ----------------------------------------------------------------------------- export
def _zip_names(content: bytes) -> dict[str, zipfile.ZipInfo]:
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        assert zf.testzip() is None
        return {i.filename: i for i in zf.infolist()}


def test_export_zip(local_client: TestClient, conn, app_storage: Storage) -> None:
    pid = seed_project(conn, app_storage, title="My Talk")
    a = _ready_clip(conn, app_storage, pid, title="Same Title", rank=1)
    b = _ready_clip(conn, app_storage, pid, title="Same Title", rank=1, render_hash="def456", srt=False)
    stale = _ready_clip(conn, app_storage, pid, title="Old edit", rank=3, state="stale")
    r = local_client.get(f"/media/projects/{pid}/export.zip?clips={a},{b},{stale}&srt=1")
    assert r.status_code == 200
    assert r.headers["content-disposition"] == 'attachment; filename="my-talk-clips.zip"'
    infos = _zip_names(r.content)
    assert set(infos) == {"same-title-1.mp4", "same-title-1.srt", "same-title-1-2.mp4"}
    assert infos["same-title-1.mp4"].compress_type == zipfile.ZIP_STORED
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        assert zf.read("same-title-1.mp4") == FAKE_MP4
    assert list(app_storage.tmp_dir().glob("export-*")) == []     # temp ZIP removed after sending

    no_srt = _zip_names(local_client.get(f"/media/projects/{pid}/export.zip?clips={a}&srt=0").content)
    assert set(no_srt) == {"same-title-1.mp4"}
    everything = _zip_names(local_client.get(f"/media/projects/{pid}/export.zip").content)
    assert len([n for n in everything if n.endswith(".mp4")]) == 2


def test_export_rejects_bad_requests(local_client: TestClient, conn, app_storage: Storage) -> None:
    pid = seed_project(conn, app_storage)
    other = seed_project(conn, app_storage)
    foreign = _ready_clip(conn, app_storage, other, title="Not yours", rank=1)
    stale = _ready_clip(conn, app_storage, pid, title="Stale", rank=1, state="stale")
    assert local_client.get(f"/media/projects/{pid}/export.zip?clips={foreign}").status_code == 404
    assert local_client.get(f"/media/projects/{pid}/export.zip?clips={stale}").status_code == 404
    assert local_client.get(f"/media/projects/{pid}/export.zip?clips=../../etc").status_code == 404
    assert local_client.get(f"/media/projects/{pid}/export.zip?srt=7").status_code == 422
    assert local_client.get("/media/projects/p_0123456789abcdef/export.zip").status_code == 404


def test_unique_name_helper() -> None:
    taken: set[str] = set()
    assert [routes_media.unique_name("a.mp4", taken) for _ in range(3)] == ["a.mp4", "a-2.mp4", "a-3.mp4"]
    assert routes_media.clip_filename("Ünïcode — title?", 2, "mp4") == "unicode-title-2.mp4"
    assert routes_media.clip_filename("!!!", 1, "srt") == "clip-1.srt"
