"""Login throttling holds under concurrency: simultaneous wrong-password requests cannot
slip past the per-IP limit or the global cap (check-and-record must be atomic)."""

from __future__ import annotations

import collections
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from clipper import db
from clipper.api import auth
from clipper.api.app import create_app
from test_api_support import HOSTED_ORIGIN, login, make_hosted_settings

BURST = 20
SLOW_VERIFY_SECONDS = 0.25      # stands in for production-cost argon2 so the requests overlap


@pytest.fixture
def slow_verify(monkeypatch: pytest.MonkeyPatch) -> None:
    real = auth.verify_credentials

    def slow(*args, **kwargs):
        time.sleep(SLOW_VERIFY_SECONDS)
        return real(*args, **kwargs)

    monkeypatch.setattr(auth, "verify_credentials", slow)


def _burst(client: TestClient, n: int, password: str = "wrong") -> collections.Counter:
    codes: collections.Counter = collections.Counter()
    lock = threading.Lock()
    barrier = threading.Barrier(n)

    def go() -> None:
        barrier.wait()
        r = login(client, password=password)
        with lock:
            codes[r.status_code] += 1

    threads = [threading.Thread(target=go) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    return codes


def _failure_rows(settings) -> int:
    conn = db.connect(settings.db_path)
    try:
        return conn.execute("SELECT COUNT(*) FROM login_attempts WHERE success=0").fetchone()[0]
    finally:
        conn.close()


def test_concurrent_wrong_passwords_respect_per_ip_limit(tmp_path: Path, slow_verify) -> None:
    settings = make_hosted_settings(tmp_path, login_max_failures=3)
    with TestClient(create_app(settings), base_url=HOSTED_ORIGIN) as client:
        codes = _burst(client, BURST)
        assert sum(codes.values()) == BURST
        assert codes[401] <= 3, dict(codes)                 # at most the configured number of guesses
        assert codes[429] == BURST - codes[401], dict(codes)
        assert _failure_rows(settings) == codes[401]
        assert login(client).status_code == 429             # still locked for the right password


def test_concurrent_wrong_passwords_respect_global_cap(tmp_path: Path, slow_verify) -> None:
    settings = make_hosted_settings(tmp_path, login_max_failures=1000)   # only the global cap applies
    with TestClient(create_app(settings), base_url=HOSTED_ORIGIN) as client:
        conn = db.connect(settings.db_path)
        try:
            for i in range(auth.GLOBAL_FAILURE_CAP - 2):
                auth.record_failure(conn, f"10.0.{i // 250}.{i % 250}")
        finally:
            conn.close()
        codes = _burst(client, BURST)
        assert codes[401] <= 2, dict(codes)
        assert _failure_rows(settings) <= auth.GLOBAL_FAILURE_CAP


def test_right_password_after_reservation_clears_the_attempt(tmp_path: Path, slow_verify) -> None:
    """The attempt reserved before verification is removed again by a successful sign-in."""
    settings = make_hosted_settings(tmp_path, login_max_failures=3)
    with TestClient(create_app(settings), base_url=HOSTED_ORIGIN) as client:
        assert login(client, password="wrong").status_code == 401
        assert login(client, password="wrong").status_code == 401
        assert login(client).status_code == 200            # third attempt (limit 3) still allowed
        assert _failure_rows(settings) == 0
