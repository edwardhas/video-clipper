"""Dated list prices, used only for clearly labelled cost *estimates*.

Prices were checked on developers.openai.com on :data:`PRICES_CHECKED_ON` (see
docs/RESEARCH_NOTES.md). They are list prices: cached-input discounts, batch
discounts and taxes are not applied. A model without a dated entry here gets no
estimate at all; we never guess a price.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

PRICES_CHECKED_ON = "2026-10-05"
COST_NOTE = f"Estimate from list prices checked {PRICES_CHECKED_ON}; cached-token discounts not applied."


@dataclass(frozen=True)
class ModelPrice:
    input_per_mtok: float = 0.0      # USD per 1M input tokens
    output_per_mtok: float = 0.0     # USD per 1M output tokens (reasoning tokens are part of output)
    audio_per_minute: float = 0.0    # USD per minute of audio (transcription models)


PRICES: dict[str, ModelPrice] = {
    "whisper-1": ModelPrice(audio_per_minute=0.006),
    "gpt-transcribe": ModelPrice(audio_per_minute=0.0045),
    "gpt-6.1-sol": ModelPrice(input_per_mtok=2.00, output_per_mtok=10.00),
    "gpt-6-luna": ModelPrice(input_per_mtok=0.10, output_per_mtok=0.50),
}


def price_for(model: str) -> ModelPrice | None:
    """Exact-match lookup; snapshots/aliases without a dated entry return None."""
    return PRICES.get((model or "").strip().lower())


def estimate_cost(model: str, *, input_tokens: int = 0, output_tokens: int = 0,
                  audio_seconds: float = 0.0) -> float | None:
    """USD estimate for one model's usage, or None when the model has no dated price."""
    price = price_for(model)
    if price is None:
        return None
    return (
        max(0, input_tokens) / 1_000_000 * price.input_per_mtok
        + max(0, output_tokens) / 1_000_000 * price.output_per_mtok
        + max(0.0, audio_seconds) / 60.0 * price.audio_per_minute
    )


@dataclass
class UsageTotals:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    audio_seconds: float = 0.0
    estimated_cost_usd: float | None = 0.0
    cost_note: str = COST_NOTE


def summarize_usage(rows: Iterable[Mapping[str, Any]]) -> UsageTotals:
    """Aggregate ``api_usage`` rows (status ok | error | cached).

    * ``cached`` rows were served from the local cache: no request, no cost.
    * ``error`` rows count as requests; their token counts (if any) are included.
    * The estimate is None if any billed row uses a model without a dated price.
    """
    totals = UsageTotals()
    cost = 0.0
    unpriced: set[str] = set()
    for row in rows:
        if row.get("status") == "cached":
            continue
        model = str(row.get("model") or "")
        inp = int(row.get("input_tokens") or 0)
        out = int(row.get("output_tokens") or 0)
        audio = float(row.get("audio_seconds") or 0.0)
        totals.requests += 1
        totals.input_tokens += inp
        totals.output_tokens += out
        totals.audio_seconds += audio
        est = estimate_cost(model, input_tokens=inp, output_tokens=out, audio_seconds=audio)
        if est is None:
            if inp or out or audio:
                unpriced.add(model or "unknown model")
        else:
            cost += est
    if unpriced:
        totals.estimated_cost_usd = None
        totals.cost_note = ("No cost estimate: no dated list price is recorded for "
                            + ", ".join(sorted(unpriced)) + ".")
    else:
        totals.estimated_cost_usd = round(cost, 4)
    return totals
