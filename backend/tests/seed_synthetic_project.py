"""Seed a SYNTHETIC test project into a separate data directory for UI testing.

    cd backend && .venv/bin/python tests/seed_synthetic_project.py /path/to/scratch-data

Runs the real pipeline (real FFmpeg import/probe/audio/framing/render) with the
deterministic FAKE providers from ``synthetic.py`` instead of OpenAI. The project
title is prefixed with "[SYNTHETIC FIXTURE]" so it can never be mistaken for a
real analysis. It refuses to write into the app's default data directory.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/ on path when run as a script

from clipper import db  # noqa: E402
from clipper.config import REPO_ROOT, Settings  # noqa: E402
from clipper.jobs import queue  # noqa: E402
from clipper.models import ProjectOptions  # noqa: E402
from clipper.storage import Storage, new_id  # noqa: E402
from synthetic import FakeSelector, FakeTranscriber, make_source  # noqa: E402


def main(data_dir: Path) -> int:
    data_dir = data_dir.resolve()
    if data_dir == (REPO_ROOT / "data").resolve():
        print("Refusing to seed synthetic data into the real data directory.", file=sys.stderr)
        return 2
    settings = Settings(_env_file=None, data_dir=data_dir, auth="disabled", transcribe_chunk_seconds=20.0,
                        min_free_disk_mb=200)
    storage = Storage(data_dir)
    storage.ensure()
    db.init_db(settings.db_path)

    upload_id = new_id("u")
    udir = storage.upload_dir(upload_id)
    udir.mkdir(parents=True)
    src = udir / "synthetic_talk.mp4"
    make_source(src)
    pid = new_id("p")
    ts = db.now()
    opts = ProjectOptions(clip_count=3, min_seconds=8, max_seconds=16)
    with db.session(settings.db_path) as c:
        c.execute("INSERT INTO uploads(id, filename, path, size_bytes, state, created_at) VALUES (?,?,?,?,?,?)",
                  (upload_id, "synthetic_talk.mp4", storage.rel(src), src.stat().st_size, "complete", ts))
        c.execute("""INSERT INTO projects(id, title, source_kind, upload_id, options, status, created_at, updated_at)
                     VALUES (?, ?, 'upload', ?, ?, 'queued', ?, ?)""",
                  (pid, "[SYNTHETIC FIXTURE] Tone-burst talk (fake providers, not a real analysis)", upload_id,
                   opts.model_dump_json(), ts, ts))
        queue.create_job(c, project_id=pid, kind="analyze")

    import clipper.ai.factory as factory
    fake_t, fake_s = FakeTranscriber(settings.db_path, pid), FakeSelector()
    factory.get_transcriber = lambda s: fake_t  # type: ignore[assignment]
    factory.get_selector = lambda s: fake_s  # type: ignore[assignment]

    from clipper.worker import Worker
    worker = Worker(settings)
    worker.setup()
    conn = db.connect(settings.db_path)
    try:
        job = queue.claim_next(conn, worker.worker_id)
        assert job is not None
        worker.run_job(conn, dict(job))
        row = conn.execute("SELECT state, error_message FROM jobs WHERE id=?", (job["id"],)).fetchone()
        clips = conn.execute("SELECT COUNT(*) FROM clips WHERE project_id=?", (pid,)).fetchone()[0]
    finally:
        conn.close()
    print(f"project {pid}: job {row['state']} {row['error_message'] or ''}; {clips} clips in {data_dir}")
    return 0 if row["state"] == "succeeded" else 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(Path(sys.argv[1])))
