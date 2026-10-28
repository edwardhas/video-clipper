"""Caption cue building, owner text edits and SRT export (pure logic, no FFmpeg)."""

from __future__ import annotations

import pytest

from clipper.errors import InputError
from clipper.models import CaptionStyle, ClipEdit, CueOverride, Word
from clipper.render.caption_image import lines_needed
from clipper.render.captions import HOLD_AFTER, apply_caption_edit, build_cues, to_srt


def mk_words(spec: list[tuple[str, float, float]]) -> list[Word]:
    return [Word(id=f"w{i + 1:06d}", text=text, raw=text.strip(".,!?").lower(), start=start, end=end,
                 segment_id="s00001", chunk_index=0)
            for i, (text, start, end) in enumerate(spec)]


def edit_for(src_start: float, src_end: float, **kw) -> ClipEdit:
    return ClipEdit(src_start=src_start, src_end=src_end, **kw)


# ----------------------------------------------------------------------------- trimming / times
def test_words_kept_by_midpoint_in_half_open_range() -> None:
    words = mk_words([
        ("before", 9.5, 10.1),     # midpoint 9.8  -> out
        ("straddle", 9.8, 10.4),   # midpoint 10.1 -> in
        ("middle", 11.0, 11.4),
        ("late", 13.7, 14.1),      # midpoint 13.9 -> in
        ("after", 13.8, 14.4),     # midpoint 14.1 -> out
        ("edge", 14.0, 14.0),      # midpoint == src_end -> out (half-open)
    ])
    cues = build_cues(words, edit_for(10.0, 14.0), pause_break=10.0, max_cue_seconds=10.0, max_chars=80)
    ids = [w.word_id for c in cues for w in c.words]
    assert ids == ["w000002", "w000003", "w000004"]


def test_word_exactly_at_start_is_kept() -> None:
    words = mk_words([("start", 9.9, 10.1)])           # midpoint == src_start -> in
    cues = build_cues(words, edit_for(10.0, 12.0))
    assert [w.word_id for w in cues[0].words] == ["w000001"]


def test_rel_times_are_exact_source_minus_clip_start() -> None:
    words = mk_words([("hello", 12.30, 12.70)])
    cue = build_cues(words, edit_for(10.0, 20.0))[0]
    assert cue.words[0].rel_start == 2.3 and cue.words[0].rel_end == 2.7
    assert cue.rel_start == 2.3


def test_rel_times_clamped_to_clip_edges() -> None:
    words = mk_words([("first", 9.8, 10.4), ("last", 13.7, 14.1)])
    edit = edit_for(10.0, 14.0)
    cues = build_cues(words, edit, pause_break=10.0, max_cue_seconds=10.0)
    all_words = [w for c in cues for w in c.words]
    assert all_words[0].rel_start == 0.0 and all_words[0].rel_end == pytest.approx(0.4)
    assert all_words[-1].rel_end == 4.0
    for c in cues:
        assert 0.0 <= c.rel_start <= c.rel_end <= 4.0


def test_uppercase_is_not_applied_when_building() -> None:
    words = mk_words([("Hello", 1.0, 1.4), ("there.", 1.5, 1.9)])
    cues = build_cues(words, edit_for(0.0, 5.0, style=CaptionStyle(uppercase=True)))
    assert cues[0].text == "Hello there."


# ----------------------------------------------------------------------------- grouping
def test_breaks_after_sentence_punctuation() -> None:
    words = mk_words([("One", 1.0, 1.2), ("two.", 1.3, 1.5), ("Three", 1.6, 1.8), ("four?", 1.9, 2.1),
                      ("Five", 2.2, 2.4)])
    cues = build_cues(words, edit_for(0.0, 10.0))
    assert [c.text for c in cues] == ["One two.", "Three four?", "Five"]


def test_abbreviation_does_not_end_sentence() -> None:
    words = mk_words([("Ask", 1.0, 1.2), ("Dr.", 1.3, 1.5), ("Smith", 1.6, 1.8)])
    assert [c.text for c in build_cues(words, edit_for(0.0, 10.0))] == ["Ask Dr. Smith"]


def test_breaks_at_pause() -> None:
    words = mk_words([("a", 1.0, 1.2), ("b", 1.3, 1.5), ("c", 2.1, 2.3), ("d", 2.4, 2.6)])
    cues = build_cues(words, edit_for(0.0, 10.0), pause_break=0.6)
    assert [c.text for c in cues] == ["a b", "c d"]        # gap 1.5 -> 2.1 is exactly 0.6


def test_breaks_when_text_exceeds_line_budget() -> None:
    words = mk_words([(t, 1.0 + i * 0.2, 1.15 + i * 0.2) for i, t in enumerate(
        ["alpha", "beta", "gamma", "delta", "omega"])])
    cues = build_cues(words, edit_for(0.0, 10.0), max_chars=10, max_lines=2)
    # "alpha beta" fills line 1, "gamma" line 2; "delta" would need a third line.
    assert [c.text for c in cues] == ["alpha beta gamma", "delta omega"]


def test_breaks_when_cue_would_exceed_max_seconds() -> None:
    words = mk_words([(f"w{i}", 1.0 + i * 0.5, 1.4 + i * 0.5) for i in range(8)])
    cues = build_cues(words, edit_for(0.0, 10.0), max_cue_seconds=1.5, max_chars=80)
    for c in cues:
        assert c.words[-1].rel_end - c.words[0].rel_start <= 1.5 + 1e-9
    assert len(cues) == 3


def test_cues_fit_two_lines_at_the_rendered_size() -> None:
    texts = "WONDERFUL extraordinary MEASUREMENTS everywhere considered thoroughly beautiful".split() * 3
    words = mk_words([(t, i * 0.3, i * 0.3 + 0.25) for i, t in enumerate(texts)])
    style = CaptionStyle(size="L", uppercase=True)
    cues = build_cues(words, edit_for(0.0, 30.0, style=style), max_cue_seconds=30.0, pause_break=10.0)
    assert len(cues) > 1
    for c in cues:
        assert lines_needed([w.text for w in c.words], style) <= 2


# ----------------------------------------------------------------------------- timing adjustment
def test_small_gap_is_closed_and_silence_gets_hold() -> None:
    words = mk_words([("One.", 1.0, 1.5), ("Two.", 1.6, 2.0), ("Three.", 4.0, 4.5)])
    cues = build_cues(words, edit_for(0.0, 4.8))
    assert cues[0].rel_end == cues[1].rel_start == 1.6               # 0.1 s gap closed
    assert cues[1].rel_end == pytest.approx(2.0 + HOLD_AFTER)        # held into the silence
    assert cues[2].rel_end == 4.8                                    # hold clipped at the clip end


def test_hold_never_reaches_past_next_cue() -> None:
    words = mk_words([("One.", 1.0, 1.5), ("Two.", 1.8, 2.2)])
    cues = build_cues(words, edit_for(0.0, 10.0))
    assert cues[0].rel_end == 1.8                                    # min(1.5 + 0.4, next start)


# ----------------------------------------------------------------------------- overrides
def test_word_override_keeps_timing_and_highlight() -> None:
    words = mk_words([("their", 1.0, 1.3), ("going", 1.4, 1.8)])
    edit = edit_for(0.0, 5.0, word_overrides={"w000001": "they're"})
    cue = build_cues(words, edit)[0]
    assert cue.text == "they're going"
    assert cue.highlight is True and cue.edited is True
    assert (cue.words[0].rel_start, cue.words[0].rel_end) == (1.0, 1.3)


def test_empty_word_override_hides_word() -> None:
    words = mk_words([("um", 1.0, 1.2), ("hello", 1.3, 1.6)])
    cues = build_cues(words, edit_for(0.0, 5.0, word_overrides={"w000001": ""}))
    assert [c.text for c in cues] == ["hello"]
    assert cues[0].rel_start == 1.3


def test_cue_override_is_single_cue_without_highlight() -> None:
    words = mk_words([("gonna", 1.0, 1.3), ("do", 1.4, 1.6), ("it.", 1.7, 2.0), ("Next", 3.0, 3.3)])
    ov = CueOverride(first_word_id="w000001", last_word_id="w000003", text="going to  do it")
    cues = build_cues(words, edit_for(0.0, 5.0, cue_overrides=[ov]))
    first = cues[0]
    assert first.text == "going to do it"
    assert first.highlight is False and first.edited is True
    assert first.words == []                                   # no fabricated per-word timings
    assert (first.first_word_id, first.last_word_id) == ("w000001", "w000003")
    assert first.rel_start == 1.0
    assert [c.text for c in cues[1:]] == ["Next"]


def test_cue_override_ignored_when_range_leaves_clip() -> None:
    words = mk_words([("out", 0.5, 0.8), ("in", 1.5, 1.8)])
    ov = CueOverride(first_word_id="w000001", last_word_id="w000002", text="replaced text here")
    cues = build_cues(words, edit_for(1.0, 5.0, cue_overrides=[ov]))
    assert [c.text for c in cues] == ["in"] and cues[0].highlight is True


def test_no_words_gives_no_cues() -> None:
    assert build_cues([], edit_for(0.0, 5.0)) == []
    assert to_srt([]) == ""


# ----------------------------------------------------------------------------- apply_caption_edit
@pytest.fixture
def five_words() -> list[Word]:
    return mk_words([("we", 1.0, 1.2), ("gonna", 1.3, 1.6), ("win", 1.7, 2.0), ("this", 2.1, 2.4),
                     ("year.", 2.5, 2.9)])


def test_apply_same_token_count_sets_word_overrides(five_words: list[Word]) -> None:
    edit = edit_for(0.0, 5.0)
    new = apply_caption_edit(edit, five_words, "w000001", "w000003", "We gotta win")
    assert new.word_overrides == {"w000001": "We", "w000002": "gotta"}     # unchanged word not stored
    assert new.cue_overrides == []
    cue = build_cues(five_words, new)[0]
    assert cue.highlight is True
    assert [w.rel_start for w in cue.words] == [1.0, 1.3, 1.7, 2.1, 2.5]   # timings kept


def test_apply_restoring_original_text_drops_overrides(five_words: list[Word]) -> None:
    edit = edit_for(0.0, 5.0, word_overrides={"w000002": "gotta"})
    new = apply_caption_edit(edit, five_words, "w000002", "w000002", "gonna")
    assert new.word_overrides == {}


def test_apply_different_token_count_creates_cue_override(five_words: list[Word]) -> None:
    edit = edit_for(0.0, 5.0, word_overrides={"w000002": "gotta"})
    new = apply_caption_edit(edit, five_words, "w000001", "w000003", "we are going to win")
    assert new.word_overrides == {}
    assert new.cue_overrides == [CueOverride(first_word_id="w000001", last_word_id="w000003",
                                             text="we are going to win")]
    cues = build_cues(five_words, new)
    assert cues[0].highlight is False and cues[0].words == []
    assert cues[0].text == "we are going to win"


def test_apply_replaces_overlapping_cue_overrides(five_words: list[Word]) -> None:
    old = [CueOverride(first_word_id="w000002", last_word_id="w000004", text="x y"),
           CueOverride(first_word_id="w000005", last_word_id="w000005", text="year!")]
    edit = edit_for(0.0, 5.0, cue_overrides=old)
    new = apply_caption_edit(edit, five_words, "w000001", "w000003", "we will win big")
    assert [o.first_word_id for o in new.cue_overrides] == ["w000005", "w000001"]
    assert new.cue_overrides[-1].text == "we will win big"


def test_apply_empty_text_hides_words(five_words: list[Word]) -> None:
    new = apply_caption_edit(edit_for(0.0, 5.0), five_words, "w000004", "w000005", "   ")
    assert new.word_overrides == {"w000004": "", "w000005": ""}
    assert [c.text for c in build_cues(five_words, new)] == ["we gonna win"]


def test_hidden_caption_is_listed_for_the_editor_and_can_be_restored() -> None:
    # Review finding 17: hiding a cue made it vanish from ClipDetailOut.cues, so it could only
    # come back via "Reset caption edits" (which also drops every other correction).
    words = mk_words([("Hello.", 1.0, 1.4), ("Their", 2.0, 2.3), ("going.", 2.4, 2.8), ("Big", 3.5, 3.8),
                      ("win.", 3.9, 4.2)])
    edit = edit_for(0.0, 6.0)
    assert [c.text for c in build_cues(words, edit)] == ["Hello.", "Their going.", "Big win."]
    edit = apply_caption_edit(edit, words, "w000002", "w000003", "They're going.")
    edit = apply_caption_edit(edit, words, "w000001", "w000001", "")          # hide cue 1 by mistake
    rendered = build_cues(words, edit)
    assert [c.text for c in rendered] == ["They're going.", "Big win."]        # render/SRT: still hidden
    listed = build_cues(words, edit, include_hidden=True)
    assert [c.text for c in listed] == ["", "They're going.", "Big win."]
    hidden = listed[0]
    assert (hidden.first_word_id, hidden.last_word_id) == ("w000001", "w000001")
    assert hidden.edited is True and hidden.highlight is False and hidden.words == []
    assert (hidden.rel_start, hidden.rel_end) == (1.0, 1.4)
    assert listed[1:] == rendered                                             # visible cues are identical
    assert "Hello" not in to_srt(listed) and to_srt(listed) == to_srt(rendered)
    # Restoring just that caption keeps the other correction.
    restored = apply_caption_edit(edit, words, hidden.first_word_id, hidden.last_word_id, "Hello.")
    assert restored.word_overrides == {"w000002": "They're"}
    assert [c.text for c in build_cues(words, restored, include_hidden=True)] == \
        ["Hello.", "They're going.", "Big win."]


def test_hidden_runs_are_grouped_like_cues_and_cue_overrides_can_be_hidden() -> None:
    words = mk_words([("One", 1.0, 1.2), ("two.", 1.3, 1.5), ("Three", 1.6, 1.8), ("four.", 1.9, 2.1),
                      ("Five", 4.0, 4.2)])
    edit = edit_for(0.0, 6.0, word_overrides={"w000001": "", "w000002": "", "w000003": "", "w000004": ""})
    listed = build_cues(words, edit, include_hidden=True)
    assert [(c.text, c.first_word_id, c.last_word_id) for c in listed] == [
        ("", "w000001", "w000002"), ("", "w000003", "w000004"), ("Five", "w000005", "w000005")]
    assert [c.text for c in build_cues(words, edit)] == ["Five"]
    blank_ov = edit_for(0.0, 6.0, cue_overrides=[CueOverride(first_word_id="w000003", last_word_id="w000005",
                                                             text="  ")])
    assert [c.text for c in build_cues(words, blank_ov)] == ["One two."]
    assert [(c.text, c.first_word_id, c.last_word_id) for c in build_cues(words, blank_ov, include_hidden=True)] \
        == [("One two.", "w000001", "w000002"), ("", "w000003", "w000005")]


def test_apply_unknown_word_id_is_input_error(five_words: list[Word]) -> None:
    with pytest.raises(InputError):
        apply_caption_edit(edit_for(0.0, 5.0), five_words, "w000004", "w999999", "x")
    with pytest.raises(InputError):
        apply_caption_edit(edit_for(0.0, 5.0), five_words, "w000004", "w000002", "x")


def test_apply_rejects_overlong_replacement(five_words: list[Word]) -> None:
    with pytest.raises(InputError, match="300 characters"):
        apply_caption_edit(edit_for(0.0, 5.0), five_words, "w000001", "w000002", "word " * 80)


def test_apply_keeps_other_edit_fields(five_words: list[Word]) -> None:
    edit = edit_for(0.5, 4.0, style=CaptionStyle(color="yellow"))
    new = apply_caption_edit(edit, five_words, "w000001", "w000001", "We")
    assert (new.src_start, new.src_end, new.style.color) == (0.5, 4.0, "yellow")
    assert edit.word_overrides == {}                                        # input not mutated


# ----------------------------------------------------------------------------- SRT
def test_srt_golden() -> None:
    words = mk_words([("Hello", 10.5, 10.9), ("world.", 11.0, 11.6), ("um", 13.0, 13.2), ("Next", 14.25, 14.6),
                      ("one", 14.7, 15.123)])
    edit = edit_for(10.0, 3700.0, word_overrides={"w000003": ""})
    cues = build_cues(words, edit)
    expected = (
        "1\n00:00:00,500 --> 00:00:02,000\nHello world.\n"
        "\n"
        "2\n00:00:04,250 --> 00:00:05,523\nNext one\n"
    )
    assert to_srt(cues) == expected


def test_srt_skips_empty_cues_and_uses_hour_field() -> None:
    words = mk_words([("late", 3725.0, 3725.5)])
    cues = build_cues(words, edit_for(0.0, 3726.0))
    srt = to_srt(cues)
    assert srt == "1\n01:02:05,000 --> 01:02:05,900\nlate\n"
    assert "\r" not in srt
    blank = cues[0].model_copy(update={"text": "  "})
    assert to_srt([blank, *cues]) == srt
