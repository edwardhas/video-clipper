"""Persist provider usage (tokens, audio seconds, request ids) per project/job."""

from __future__ import annotations

import logging
from pathlib import Path

from .. import db

log = logging.getLogger(__name__)


class DbUsageSink:
    def __init__(self, db_path: Path, project_id: str | None, job_id: str | None) -> None:
        self.db_path = db_path
        self.project_id = project_id
        self.job_id = job_id

    def record(self, *, stage: str, provider: str, model: str, request_id: str | None, status: str,
               input_tokens: int | None = None, output_tokens: int | None = None,
               reasoning_tokens: int | None = None, cached_tokens: int | None = None,
               audio_seconds: float | None = None, duration_s: float | None = None,
               error_code: str | None = None) -> None:
        try:
            with db.session(self.db_path) as conn:
                conn.execute(
                    """INSERT INTO api_usage(project_id, job_id, stage, provider, model, request_id, status,
                           input_tokens, output_tokens, reasoning_tokens, cached_tokens, audio_seconds,
                           duration_s, error_code, created_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (self.project_id, self.job_id, stage, provider, model, request_id, status,
                     input_tokens, output_tokens, reasoning_tokens, cached_tokens, audio_seconds,
                     duration_s, error_code, db.now()),
                )
        except Exception:  # usage accounting must never break a job
            log.exception("failed to record API usage")
