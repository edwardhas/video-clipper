"""Dated list-price estimates (clipper.ai.pricing)."""

from __future__ import annotations

import pytest

from clipper.ai import pricing


def test_known_prices() -> None:
    assert pricing.estimate_cost("whisper-1", audio_seconds=60.0) == pytest.approx(0.006)
    assert pricing.estimate_cost("gpt-6.1-sol", input_tokens=1_000_000, output_tokens=1_000_000) == pytest.approx(12.0)
    assert pricing.estimate_cost("gpt-6-luna", input_tokens=2_000_000, output_tokens=1_000_000) == pytest.approx(0.7)
    assert pricing.estimate_cost(" GPT-6.1-SOL ", input_tokens=500_000) == pytest.approx(1.0)


def test_unknown_or_snapshot_models_get_no_estimate() -> None:
    assert pricing.estimate_cost("gpt-6.1-sol-2026-09-01", input_tokens=10) is None
    assert pricing.estimate_cost("", input_tokens=10) is None


def test_summarize_usage_skips_cached_rows_and_counts_errors() -> None:
    rows = [
        {"model": "whisper-1", "status": "ok", "audio_seconds": 120.0},
        {"model": "whisper-1", "status": "cached", "audio_seconds": 600.0},
        {"model": "gpt-6.1-sol", "status": "error", "input_tokens": None, "output_tokens": None},
        {"model": "gpt-6.1-sol", "status": "ok", "input_tokens": 100_000, "output_tokens": 20_000},
    ]
    totals = pricing.summarize_usage(rows)
    assert totals.requests == 3 and totals.audio_seconds == 120.0
    assert totals.input_tokens == 100_000 and totals.output_tokens == 20_000
    assert totals.estimated_cost_usd == pytest.approx(0.012 + 0.2 + 0.2)
    assert totals.cost_note == pricing.COST_NOTE
    assert "2026-10-05" in pricing.COST_NOTE and "cached-token discounts not applied" in pricing.COST_NOTE


def test_summarize_usage_without_dated_price() -> None:
    totals = pricing.summarize_usage([{"model": "mystery-model", "status": "ok", "input_tokens": 5}])
    assert totals.estimated_cost_usd is None and "mystery-model" in totals.cost_note
    empty = pricing.summarize_usage([])
    assert empty.requests == 0 and empty.estimated_cost_usd == 0.0
