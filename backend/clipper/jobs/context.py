"""Shared state handed to job runners by the worker."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import db
from ..config import Settings
from ..media.encoders import EncoderChoice
from ..procs import CancelToken
from ..storage import Storage
from . import queue
from .usage import DbUsageSink


@dataclass
class JobContext:
    settings: Settings
    storage: Storage
    conn: sqlite3.Connection           # connection owned by the job thread
    job: dict[str, Any]
    worker_id: str
    cancel: CancelToken
    encoder: EncoderChoice
    _last_progress: float = field(default=0.0, repr=False)

    @property
    def job_id(self) -> str:
        return self.job["id"]

    @property
    def project_id(self) -> str:
        return self.job["project_id"]

    @property
    def db_path(self) -> Path:
        return self.settings.db_path

    @property
    def params(self) -> dict[str, Any]:
        return db.loads(self.job.get("params"), {}) or {}

    def usage(self) -> DbUsageSink:
        return DbUsageSink(self.db_path, self.project_id, self.job_id)

    def progress(self, stage: str | None = None, detail: str | None = None, current: int | None = None,
                 total: int | None = None, *, throttle: float = 0.0) -> None:
        """Record real progress. ``throttle`` limits write frequency for byte counters."""
        now = time.monotonic()
        if throttle and now - self._last_progress < throttle:
            return
        self._last_progress = now
        queue.set_progress(self.conn, self.job_id, stage=stage, detail=detail, current=current, total=total)
