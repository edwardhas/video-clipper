"""Round-2 regression tests for clipper.ai.selection (fake OpenAI client; no live API calls).

* A truncated review answer is asked again for *every* decision, and candidates a partial answer
  left out are reviewed once more on their own, so no shortlisted candidate is silently dropped.
* Every paid attempt inside a window/batch is checkpointed before a cancel is honoured, so a
  resume never pays for it again; a window/batch that finally fails is cleared so "Try again" asks afresh.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from clipper.ai import prompts
from clipper.config import Settings
from clipper.errors import CancelledError, ProviderError
from clipper.jobs.cache import FileJsonCache
from clipper.models import ProjectOptions
from clipper.procs import CancelToken
from clipper.storage import Storage
from test_selection import (DictCache, RecordingUsage, Scripted, _selector, answer, cand_json, decision_json,
                            make_candidate, texts_of, uniform_transcript)


@pytest.fixture
def options() -> ProjectOptions:
    return ProjectOptions(clip_count=3, min_seconds=20.0, max_seconds=60.0)


def _ids_in(kwargs: dict[str, Any]) -> list[str]:
    text = texts_of(kwargs["input"][1])
    return [part.split(" ===", 1)[0] for part in text.split("=== Candidate ")[1:]]


# ============================================================================ truncated review
def test_truncated_review_is_asked_for_every_decision_and_none_is_dropped(settings: Settings,
                                                                          options: ProjectOptions) -> None:
    t = uniform_transcript(120)
    cands = [make_candidate(t, f"c{i:03d}", i * 10, i * 10 + 2) for i in range(1, 9)]       # one batch of 8
    by_id = {c.id: c for c in cands}
    full = [decision_json(c.id, c.start_unit_id, c.end_unit_id) for c in cands]
    cut = Scripted(text=json.dumps({"decisions": full})[:700], status="incomplete",
                   incomplete_reason="max_output_tokens")                 # JSON cut off at max_output_tokens

    def four_of_them(kwargs: dict[str, Any]) -> Scripted:                 # still answers only half
        return answer({"decisions": full[:4]})

    def the_rest(kwargs: dict[str, Any]) -> Scripted:
        return answer({"decisions": [decision_json(i, by_id[i].start_unit_id, by_id[i].end_unit_id)
                                     for i in _ids_in(kwargs)]})

    selector, client = _selector(settings, [cut, four_of_them, the_rest])
    usage = RecordingUsage()
    decisions = selector.review(t, cands, {}, options, cancel=CancelToken(), usage=usage)
    note = texts_of(client.calls[1]["input"][-1])
    assert "cut off" in note and "at most" not in note and "each of the 8 candidate ids" in note
    assert note == prompts.review_truncated_correction(8)
    assert _ids_in(client.calls[1]) == [c.id for c in cands]               # the retry still covers all 8
    assert _ids_in(client.calls[2]) == ["c005", "c006", "c007", "c008"]    # the missing ones, on their own
    assert [d.candidate_id for d in decisions] == [c.id for c in cands]
    assert [(r["status"], r.get("error_code")) for r in usage.rows] == [
        ("error", "incomplete_max_output_tokens"), ("ok", None), ("ok", None)]


def test_follow_up_that_gets_no_answer_keeps_the_decisions_already_given(settings: Settings,
                                                                         options: ProjectOptions) -> None:
    t = uniform_transcript(120)
    cands = [make_candidate(t, f"c{i:03d}", i * 10, i * 10 + 2) for i in range(1, 9)]
    six = answer({"decisions": [decision_json(c.id, c.start_unit_id, c.end_unit_id) for c in cands[:6]]})
    refused = Scripted(refusal="I can't help with that.")
    selector, client = _selector(settings, [six, refused, refused, refused])
    decisions = selector.review(t, cands, {}, options, cancel=CancelToken(), usage=RecordingUsage())
    assert [d.candidate_id for d in decisions] == [c.id for c in cands[:6]]
    assert len(client.calls) == 4                                          # bounded: one follow-up batch
    assert all(_ids_in(c) == ["c007", "c008"] for c in client.calls[1:])


def test_follow_up_network_failure_is_not_hidden(settings: Settings, options: ProjectOptions) -> None:
    t = uniform_transcript(120)
    cands = [make_candidate(t, f"c{i:03d}", i * 10, i * 10 + 2) for i in range(1, 9)]
    six = answer({"decisions": [decision_json(c.id, c.start_unit_id, c.end_unit_id) for c in cands[:6]]})
    selector, _ = _selector(settings, [six, ProviderError("OpenAI is unavailable.", retryable=True)])
    with pytest.raises(ProviderError, match="unavailable"):
        selector.review(t, cands, {}, options, cancel=CancelToken(), usage=RecordingUsage())


# ============================================================================ per-attempt checkpoints
def test_cancel_during_a_window_attempt_that_needs_a_retry_does_not_rebill_it(settings: Settings,
                                                                              options: ProjectOptions) -> None:
    t = uniform_transcript(30)                                             # one window
    token, cache = CancelToken(), DictCache()
    first = {"candidates": [cand_json("u0002", "u0004"), cand_json("u9998", "u9999"), cand_json("x", "y")]}
    second = {"candidates": [cand_json("u0015", "u0017")]}

    def paid_then_cancel(kwargs: dict[str, Any]) -> Scripted:
        token.cancel()                      # the user presses Cancel while this billed request is in flight
        return answer(first)

    selector, client = _selector(settings, [paid_then_cancel])
    with pytest.raises(CancelledError):
        selector.find_candidates(t, options, cancel=token, usage=RecordingUsage(), cache=cache)
    assert len(client.calls) == 1
    resumed, client2 = _selector(settings, [answer(second)])
    usage = RecordingUsage()
    found = resumed.find_candidates(t, options, cancel=CancelToken(), usage=usage, cache=cache)
    assert len(client2.calls) == 1 and len(usage.rows) == 1               # only the corrective attempt is paid
    assert "u9998" in texts_of(client2.calls[0]["input"][-1])             # rebuilt from the stored answer
    assert [(c.start_unit_id, c.end_unit_id) for c in found] == [("u0002", "u0004"), ("u0015", "u0017")]


def test_cancel_during_a_review_attempt_that_needs_a_retry_does_not_rebill_it(settings: Settings,
                                                                              options: ProjectOptions) -> None:
    t = uniform_transcript(60)
    cands = [make_candidate(t, f"c{i:03d}", i * 10, i * 10 + 2) for i in range(1, 5)]       # one batch of 4
    partial = {"decisions": [decision_json(cands[0].id, cands[0].start_unit_id, cands[0].end_unit_id)]}
    full = {"decisions": [decision_json(c.id, c.start_unit_id, c.end_unit_id) for c in cands]}
    token, cache = CancelToken(), DictCache()

    def paid_then_cancel(kwargs: dict[str, Any]) -> Scripted:
        token.cancel()
        return answer(partial)

    selector, client = _selector(settings, [paid_then_cancel])
    with pytest.raises(CancelledError):
        selector.review(t, cands, {}, options, cancel=token, usage=RecordingUsage(), cache=cache)
    assert len(client.calls) == 1
    resumed, client2 = _selector(settings, [answer(full)])
    decisions = resumed.review(t, cands, {}, options, cancel=CancelToken(), usage=RecordingUsage(), cache=cache)
    assert len(client2.calls) == 1                                         # the first answer was not paid again
    assert "Missing: c002, c003, c004" in texts_of(client2.calls[0]["input"][-1])
    assert [d.candidate_id for d in decisions] == [c.id for c in cands]


def test_provider_error_on_the_corrective_attempt_keeps_the_paid_first_one(settings: Settings,
                                                                           options: ProjectOptions) -> None:
    t = uniform_transcript(30)
    cache = DictCache()
    first = answer({"candidates": [cand_json("u0002", "u0004"), cand_json("u9998", "u9999")]})
    selector, client = _selector(settings, [first, ProviderError("OpenAI is unavailable.", retryable=True)])
    with pytest.raises(ProviderError):
        selector.find_candidates(t, options, cancel=CancelToken(), usage=RecordingUsage(), cache=cache)
    retry, client2 = _selector(settings, [answer({"candidates": []})])
    found = retry.find_candidates(t, options, cancel=CancelToken(), usage=RecordingUsage(), cache=cache)
    assert len(client.calls) == 2 and len(client2.calls) == 1
    assert [(c.start_unit_id, c.end_unit_id) for c in found] == [("u0002", "u0004")]


def test_window_that_failed_every_attempt_is_asked_afresh_on_try_again(settings: Settings, options: ProjectOptions,
                                                                       tmp_path: Path) -> None:
    # The pipeline's file cache: stored attempts of a window that finally failed are cleared, so a
    # "Try again" really asks the model again instead of replaying three failures.
    t = uniform_transcript(30)
    storage = Storage(tmp_path / "data")
    storage.ensure()
    cache = FileJsonCache(storage, "candidates-window", "stage-key")
    bad = {"candidates": [cand_json("x1", "x2")]}
    selector, client = _selector(settings, [answer(bad) for _ in range(3)])
    with pytest.raises(ProviderError, match="transcript positions"):
        selector.find_candidates(t, options, cancel=CancelToken(), usage=RecordingUsage(), cache=cache)
    assert len(client.calls) == 3
    retry, client2 = _selector(settings, [answer({"candidates": [cand_json("u0002", "u0004")]})])
    found = retry.find_candidates(t, options, cancel=CancelToken(), usage=RecordingUsage(), cache=cache)
    assert len(client2.calls) == 1 and [c.start_unit_id for c in found] == ["u0002"]
    note = texts_of(client2.calls[0]["input"][-1])
    assert "x1" not in note                                                # a fresh first attempt


def test_review_that_failed_every_attempt_is_asked_afresh_on_try_again(settings: Settings,
                                                                       options: ProjectOptions) -> None:
    t = uniform_transcript(60)
    cands = [make_candidate(t, f"c{i:03d}", i * 10, i * 10 + 2) for i in range(1, 3)]
    cache = DictCache()
    cut = Scripted(status="incomplete", incomplete_reason="max_output_tokens")
    selector, client = _selector(settings, [cut, cut, cut])
    with pytest.raises(ProviderError, match="CLIPPER_REVIEW_MAX_OUTPUT_TOKENS"):
        selector.review(t, cands, {}, options, cancel=CancelToken(), usage=RecordingUsage(), cache=cache)
    full = answer({"decisions": [decision_json(c.id, c.start_unit_id, c.end_unit_id) for c in cands]})
    retry, client2 = _selector(settings, [full])
    decisions = retry.review(t, cands, {}, options, cancel=CancelToken(), usage=RecordingUsage(), cache=cache)
    assert len(client2.calls) == 1 and len(decisions) == 2


def test_unreadable_stored_attempt_is_requested_again(settings: Settings, options: ProjectOptions) -> None:
    t = uniform_transcript(30)
    token, cache = CancelToken(), DictCache()
    first = {"candidates": [cand_json("u0002", "u0004"), cand_json("u9998", "u9999")]}

    def paid_then_cancel(kwargs: dict[str, Any]) -> Scripted:
        token.cancel()
        return answer(first)

    selector, _ = _selector(settings, [paid_then_cancel])
    with pytest.raises(CancelledError):
        selector.find_candidates(t, options, cancel=token, usage=RecordingUsage(), cache=cache)
    for key in cache.data:
        cache.data[key] = {"parsed": {"not": "candidates"}, "problem": None}
    resumed, client2 = _selector(settings, [answer({"candidates": [cand_json("u0002", "u0004")]})])
    found = resumed.find_candidates(t, options, cancel=CancelToken(), usage=RecordingUsage(), cache=cache)
    assert len(client2.calls) == 1 and [c.start_unit_id for c in found] == ["u0002"]
