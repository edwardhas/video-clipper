"""Second review round: resuming an interrupted YouTube import, and a re-selection that finds nothing.

Real FFmpeg media, FAKE providers and a fake yt-dlp (never the network).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from clipper.errors import CancelledError, InterruptedError_, SourceUnavailableError
from clipper.models import ProjectOptions
from clipper.storage import Storage
from synthetic import FakeSelector, FakeTranscriber, make_source
from test_lifecycle_fixes import OPTS, _analyze, _clips, _project, _queue_reselect, _ready_project, _youtube_project
from test_pipeline_e2e import _install_fakes, _run_next, env  # noqa: F401 (fixture)

pytestmark = [
    pytest.mark.ffmpeg,
    pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed"),
]

MB = 1024 * 1024


# ----------------------------------------------------------------------------- YouTube resume (finding 3)
def _partial(out_dir: Path) -> Path:
    return Path(out_dir) / ".part" / "abcdefghijk.mp4.part"


def _tight_disk(monkeypatch, storage, *, free_besides_download: int) -> None:
    """Free space = a fixed amount minus whatever the project's download directory holds."""
    def free_bytes(self) -> int:
        held = sum(f.stat().st_size for f in storage.root.glob("projects/*/source/download/**/*") if f.is_file())
        return free_besides_download - held

    monkeypatch.setattr(Storage, "free_bytes", free_bytes)


def _interrupted_youtube_import(env, monkeypatch, tmp_path, *, estimate: int, partial_bytes: int):
    """A YouTube project whose first download stopped (worker shutdown) after ``partial_bytes``."""
    settings, storage, db_path, worker = env
    pid = _youtube_project(db_path)
    _install_fakes(monkeypatch, FakeTranscriber(db_path, pid), FakeSelector())
    video = tmp_path / "video.mp4"
    make_source(video)
    import clipper.media.youtube as youtube
    meta = {"title": "Talk", "duration": 48.0, "filesize_approx": estimate}
    monkeypatch.setattr(youtube, "fetch_metadata", lambda url, **k: meta)
    resumed_with: list[int] = []

    def download(url, out_dir, **k):
        part = _partial(out_dir)
        if not resumed_with and not part.exists():
            part.parent.mkdir(parents=True, exist_ok=True)
            with open(part, "wb") as fh:
                fh.truncate(partial_bytes)      # sparse: counts as its size, costs no real disk
            raise InterruptedError_("Processing was interrupted because the worker stopped.")
        resumed_with.append(part.stat().st_size if part.exists() else 0)
        final = Path(out_dir) / "abcdefghijk.mp4"
        shutil.copy(video, final)
        return final

    monkeypatch.setattr(youtube, "download", download)
    assert _analyze(worker, db_path, pid)["state"] == "interrupted"
    assert _partial(storage.source_dir(pid) / "download").stat().st_size == partial_bytes
    return pid, youtube, resumed_with


def test_resumed_youtube_download_is_not_charged_twice_for_its_kept_partial(env, monkeypatch, tmp_path):
    settings, storage, db_path, worker = env
    estimate, partial = 100 * MB, 30 * MB
    reserve = settings.min_free_disk_mb * MB
    # enough for the whole download (+20 %) when nothing was downloaded yet, with 5 MB to spare
    _tight_disk(monkeypatch, storage, free_besides_download=reserve + int(estimate * 1.2) + 5 * MB)
    pid, _, resumed_with = _interrupted_youtube_import(env, monkeypatch, tmp_path, estimate=estimate,
                                                       partial_bytes=partial)
    job = _analyze(worker, db_path, pid, resume=True)
    assert job["state"] == "succeeded", (job["error_code"], job["error_message"])
    assert resumed_with == [partial]            # the download continued from the kept partial
    assert not (storage.source_dir(pid) / "download").exists()


def test_resume_that_still_lacks_disk_space_removes_the_partial(env, monkeypatch, tmp_path):
    settings, storage, db_path, worker = env
    estimate, partial = 100 * MB, 30 * MB
    reserve = settings.min_free_disk_mb * MB
    _tight_disk(monkeypatch, storage, free_besides_download=reserve + int(estimate * 1.2) + 5 * MB)
    pid, _, _ = _interrupted_youtube_import(env, monkeypatch, tmp_path, estimate=estimate, partial_bytes=partial)
    _tight_disk(monkeypatch, storage, free_besides_download=reserve + 50 * MB)   # other files filled the disk
    job = _analyze(worker, db_path, pid, resume=True)
    assert (job["state"], job["error_code"]) == ("failed", "disk_full")
    assert not (storage.source_dir(pid) / "download").exists()      # its space is given back


@pytest.mark.parametrize(("error", "state"), [
    (SourceUnavailableError("This video is no longer available."), "failed"),
    (CancelledError("Cancelled."), "cancelled"),
])
def test_resumed_youtube_import_stopping_before_its_download_removes_the_partial(env, monkeypatch, tmp_path,
                                                                                  error, state):
    settings, storage, db_path, worker = env
    pid, youtube, _ = _interrupted_youtube_import(env, monkeypatch, tmp_path, estimate=10 * MB,
                                                  partial_bytes=3 * MB)

    def fetch_metadata(url, **k):               # the video was removed / Cancel pressed while checking it
        raise error

    monkeypatch.setattr(youtube, "fetch_metadata", fetch_metadata)
    assert _analyze(worker, db_path, pid, resume=True)["state"] == state
    assert not (storage.source_dir(pid) / "download").exists()


def test_youtube_import_interrupted_while_checking_the_video_keeps_the_partial(env, monkeypatch, tmp_path):
    settings, storage, db_path, worker = env
    pid, youtube, resumed_with = _interrupted_youtube_import(env, monkeypatch, tmp_path, estimate=10 * MB,
                                                             partial_bytes=3 * MB)
    real = youtube.fetch_metadata

    def stopped(url, **k):
        raise InterruptedError_("Processing was interrupted because the worker stopped.")

    monkeypatch.setattr(youtube, "fetch_metadata", stopped)
    assert _analyze(worker, db_path, pid, resume=True)["state"] == "interrupted"
    assert _partial(storage.source_dir(pid) / "download").exists()
    monkeypatch.setattr(youtube, "fetch_metadata", real)
    job = _analyze(worker, db_path, pid, resume=True)
    assert job["state"] == "succeeded", job["error_message"]
    assert resumed_with == [3 * MB]


# ----------------------------------------------------------------------------- re-selection (finding 5)
class _NothingOnSecondCall(FakeSelector):
    def find_candidates(self, *a, **k):
        out = super().find_candidates(*a, **k)
        return out if self.find_calls == 1 else []


def test_reselect_finding_no_new_clips_restores_the_previous_options(env, monkeypatch):
    settings, storage, db_path, worker = env
    pid, _, _ = _ready_project(env, monkeypatch, selector=_NothingOnSecondCall())
    clips = _clips(db_path, pid)
    _queue_reselect(db_path, pid, ProjectOptions(clip_count=2, min_seconds=8, max_seconds=20,
                                                 captions_enabled=False))
    assert _run_next(worker, db_path)["state"] == "succeeded"
    p = _project(db_path, pid)
    assert p["status"] == "ready" and "previous clips are kept" in p["status_detail"]
    assert [c["id"] for c in _clips(db_path, pid)] == [c["id"] for c in clips]
    assert ProjectOptions.model_validate_json(p["options"]) == OPTS     # what the kept clips were made with


def test_first_analysis_finding_no_clips_keeps_its_options(env, monkeypatch):
    # control: no previous clips → nothing to restore (the options are the ones that were used)
    settings, storage, db_path, worker = env
    from test_lifecycle_fixes import _NoMoments
    from test_pipeline_e2e import _create_upload_project
    pid = _create_upload_project(settings, storage, db_path, options=OPTS)
    _install_fakes(monkeypatch, FakeTranscriber(db_path, pid), _NoMoments())
    assert _analyze(worker, db_path, pid)["state"] == "succeeded"
    other = ProjectOptions(clip_count=2, min_seconds=8, max_seconds=20)
    _queue_reselect(db_path, pid, other)
    assert _run_next(worker, db_path)["state"] == "succeeded"
    p = _project(db_path, pid)
    assert p["status"] == "no_clips" and ProjectOptions.model_validate_json(p["options"]) == other
