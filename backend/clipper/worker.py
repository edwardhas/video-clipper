"""Background worker: runs one media job at a time, independent of HTTP requests.

Start with ``python -m clipper.worker`` (the supervisor does this). Closing the
browser has no effect on running jobs. On SIGTERM/SIGINT the current subprocess
group is stopped and the job is marked ``interrupted`` so it can be resumed.
"""

from __future__ import annotations

import logging
import os
import signal
import socket
import sys
import threading
import time
from pathlib import Path

from . import __version__, db
from .config import ConfigError, Settings, get_settings
from .errors import CancelledError, ClipperError, InterruptedError_
from .jobs import queue
from .jobs.context import JobContext
from .media.encoders import EncoderChoice, choose_encoder
from .procs import CancelToken
from .storage import Storage, new_id

log = logging.getLogger("clipper.worker")

HEARTBEAT_SECONDS = 5.0
RECOVER_SECONDS = 30.0          # periodic orphan recovery (a killed worker's job is freed without a restart)
IDLE_POLL_SECONDS = 1.0
TMP_MAX_AGE_SECONDS = 24 * 3600


class Worker:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.storage = Storage(settings.data_dir)
        self.worker_id = new_id("w")
        self.stop_event = threading.Event()
        self.current_job_id: str | None = None
        self.current_token: CancelToken | None = None
        self.encoder: EncoderChoice | None = None
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ lifecycle
    def setup(self) -> None:
        self.settings.validate_for_startup()
        self.storage.ensure()
        db.init_db(self.settings.db_path)
        conn = db.connect(self.settings.db_path)
        try:
            ts = db.now()
            conn.execute(
                "INSERT INTO workers(id, pid, hostname, started_at, heartbeat_at, info) VALUES (?,?,?,?,?,?)",
                (self.worker_id, os.getpid(), socket.gethostname(), ts, ts, db.dumps({"version": __version__})),
            )
            self._recover_orphans(conn)
            self.encoder = choose_encoder(self.settings)
            conn.execute("UPDATE workers SET info=? WHERE id=?",
                         (db.dumps({"version": __version__, "encoder": self.encoder.name,
                                    "encoder_note": self.encoder.note}), self.worker_id))
            log.info("worker %s ready (encoder: %s)", self.worker_id, self.encoder.note)
        finally:
            conn.close()

    def _recover_orphans(self, conn) -> list[str]:
        """Mark jobs of dead or silent workers interrupted (at startup and every RECOVER_SECONDS).

        Repeating this matters in containers: a worker killed during a long provider call
        can leave a ``running`` job whose heartbeat is still fresh when the next worker
        starts (and, after a recreate, on a different hostname), so a one-time check at
        startup may legitimately skip it; a later pass reclaims it once the heartbeat is stale.
        """
        recovered = queue.recover_interrupted(conn, self.worker_id, on_interrupt=self._on_recovered)
        if recovered:
            log.info("marked %d orphaned job(s) as interrupted", len(recovered))
        return recovered

    @staticmethod
    def _on_recovered(conn, job) -> None:
        """Project/clip bookkeeping for jobs orphaned by a dead worker."""
        from .jobs import stages
        from .jobs.pipeline import restore_reselect_options
        params = db.loads(job["params"], {}) or {}
        if job["kind"] == "analyze":
            stages.reset_running(conn, job["project_id"])
            has_clips = conn.execute("SELECT COUNT(*) FROM clips WHERE project_id=?",
                                     (job["project_id"],)).fetchone()[0]
            if params.get("reselect") and has_clips:
                conn.execute("UPDATE projects SET status='ready', status_detail=? WHERE id=?",
                             ("Re-selection was interrupted; your previous clips are kept.", job["project_id"]))
                restore_reselect_options(conn, job["project_id"], params)
            else:
                state = "cancelled" if job["cancel_requested"] else "interrupted"
                conn.execute(
                    "UPDATE projects SET status=?, status_detail=?, updated_at=? WHERE id=?",
                    (state, "Cancelled." if state == "cancelled" else
                     "Processing was interrupted (the app or worker stopped). Resume to continue where it left off.",
                     db.now(), job["project_id"]))
        elif job["clip_id"]:
            row = conn.execute("SELECT render_path FROM clips WHERE id=?", (job["clip_id"],)).fetchone()
            if row is not None:
                conn.execute("UPDATE clips SET render_state=? WHERE id=?",
                             ("stale" if row["render_path"] else "none", job["clip_id"]))

    def request_stop(self, *_args) -> None:
        if not self.stop_event.is_set():
            log.info("stop requested; interrupting current job if any")
        self.stop_event.set()
        with self._lock:
            if self.current_token is not None:
                self.current_token.shutdown.set()

    # ------------------------------------------------------------------ heartbeat
    def _heartbeat_loop(self) -> None:
        # Runs while a job executes as well as when idle (the main loop blocks in run_job), so
        # orphan recovery is repeated here rather than in the main loop.
        conn = db.connect(self.settings.db_path)
        last_recover = time.monotonic()
        try:
            while not self.stop_event.wait(HEARTBEAT_SECONDS):
                try:
                    conn.execute("UPDATE workers SET heartbeat_at=? WHERE id=?", (db.now(), self.worker_id))
                    with self._lock:
                        job_id = self.current_job_id
                    if job_id:
                        queue.heartbeat(conn, job_id, self.worker_id)
                except Exception:  # pragma: no cover - transient db lock etc.
                    log.exception("heartbeat failed")
                if time.monotonic() - last_recover >= RECOVER_SECONDS:
                    last_recover = time.monotonic()
                    try:
                        self._recover_orphans(conn)
                    except Exception:  # pragma: no cover - transient db lock etc.
                        log.exception("orphan recovery failed")
        finally:
            conn.close()

    # ------------------------------------------------------------------ main loop
    def run_forever(self) -> None:
        hb = threading.Thread(target=self._heartbeat_loop, name="heartbeat", daemon=True)
        hb.start()
        conn = db.connect(self.settings.db_path)
        last_housekeeping = 0.0
        try:
            while not self.stop_event.is_set():
                if time.monotonic() - last_housekeeping > 600:
                    self._housekeeping()
                    last_housekeeping = time.monotonic()
                job = queue.claim_next(conn, self.worker_id)
                if job is None:
                    self.stop_event.wait(IDLE_POLL_SECONDS)
                    continue
                self.run_job(conn, dict(job))
        finally:
            conn.close()
            self.stop_event.set()
            try:
                with db.session(self.settings.db_path) as c:
                    c.execute("UPDATE workers SET heartbeat_at=? WHERE id=?", (0, self.worker_id))
            except Exception:  # pragma: no cover
                pass

    def run_job(self, conn, job: dict) -> None:
        from .jobs.pipeline import on_analyze_finished, run_analyze
        from .jobs.render_job import on_render_finished, run_render

        check_conn = db.connect(self.settings.db_path)
        token = CancelToken(check=lambda: queue.is_cancel_requested(check_conn, job["id"]))
        if self.stop_event.is_set():
            token.shutdown.set()
        with self._lock:
            self.current_job_id = job["id"]
            self.current_token = token
        assert self.encoder is not None
        ctx = JobContext(settings=self.settings, storage=self.storage, conn=conn, job=job,
                         worker_id=self.worker_id, cancel=token, encoder=self.encoder)
        kind = job["kind"]
        started = time.monotonic()
        log.info("job %s (%s) started for project %s", job["id"], kind, job["project_id"])
        on_finish = on_analyze_finished if kind == "analyze" else on_render_finished

        def finish(state: str, code: str | None = None, message: str | None = None,
                   retryable: bool | None = None) -> None:
            final = queue.finish(conn, job["id"], self.worker_id, state=state, error_code=code,
                                 error_message=message, retryable=retryable,
                                 on_finish=on_finish(ctx, message))
            log.info("job %s ended: %s (%.1f s)", job["id"], final, time.monotonic() - started)

        try:
            if kind == "analyze":
                run_analyze(ctx)
            elif kind == "render":
                run_render(ctx)
            else:  # pragma: no cover - schema CHECK prevents this
                raise ClipperError(f"Unknown job kind {kind!r}")
            log.info("job %s succeeded (%.1f s)", job["id"], time.monotonic() - started)
        except queue.JobLost:
            # Cancelled (or reclaimed) right before publishing: nothing was published.
            finish("cancelled")
        except CancelledError:
            finish("cancelled")
        except InterruptedError_:
            finish("interrupted", "interrupted", "Processing was interrupted because the worker stopped.", True)
        except ClipperError as exc:
            if exc.detail:
                log.warning("job %s failed: %s | %s", job["id"], exc.code, exc.detail)
            else:
                log.warning("job %s failed: %s", job["id"], exc.code)
            if token.shutdown.is_set():
                finish("interrupted", "interrupted", "Processing was interrupted because the worker stopped.", True)
            else:
                finish("failed", exc.code, exc.user_message, exc.retryable)
        except Exception:
            log.exception("job %s crashed", job["id"])
            stage = conn.execute("SELECT stage FROM jobs WHERE id=?", (job["id"],)).fetchone()
            where = f" during “{stage['stage']}”" if stage and stage["stage"] else ""
            finish("failed", "internal_error",
                   f"Unexpected error{where}. Details are in the worker log (data/logs/worker.log). "
                   "You can retry; completed steps will not be repeated.", True)
        finally:
            with self._lock:
                self.current_job_id = None
                self.current_token = None
            check_conn.close()

    def _housekeeping(self) -> None:
        """Remove stale scratch files (never user media)."""
        tmp = self.storage.tmp_dir()
        cutoff = time.time() - TMP_MAX_AGE_SECONDS
        for p in tmp.iterdir():
            try:
                if p.stat().st_mtime < cutoff:
                    self.storage.delete_tree(p)
            except (OSError, ValueError):
                pass


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    # The OpenAI SDK / HTTP clients must never log request headers (API key).
    for noisy in ("openai", "httpx", "httpx2", "httpcore", "httpcore2"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main() -> int:
    settings = get_settings()
    configure_logging(settings.log_level)
    worker = Worker(settings)
    try:
        worker.setup()
    except ConfigError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    signal.signal(signal.SIGTERM, worker.request_stop)
    signal.signal(signal.SIGINT, worker.request_stop)
    worker.run_forever()
    log.info("worker stopped")
    return 0


if __name__ == "__main__":
    sys.exit(main())
