"""Tests for clipper.media.audio.

* Pure: silencedetect log parsing, chunk planning (table-driven cases + a seeded random
  property loop checking the keep-range partition and chunk-length bounds).
* Real FFmpeg (marked ``ffmpeg``): analysis-audio extraction, pause detection, volume
  measurement and chunk encoding on small synthetic media built with lavfi in tmp_path.
"""

from __future__ import annotations

import dataclasses
import math
import random
import wave
from pathlib import Path

import pytest
from conftest import requires_ffmpeg

from clipper.config import Settings
from clipper.errors import CancelledError, MediaError, NoSpeechError
from clipper.media import audio
from clipper.media.audio import (
    AUDIO_VERSION,
    MIN_TAIL_SECONDS,
    NO_AUDIO_MESSAGE,
    ChunkPlan,
    chunk_extension,
    detect_silences,
    encode_chunk,
    extract_analysis_audio,
    mean_volume_db,
    parse_silencedetect,
    pcm_sha256,
    plan_chunks,
    wav_duration,
)
from clipper.media.probe import probe
from clipper.procs import CancelToken, run

T, OV, SR = 600.0, 3.0, 60.0  # pipeline defaults: target, overlap, search


# ----------------------------------------------------------------------------- helpers
def _cuts(plans: list[ChunkPlan]) -> list[float]:
    return [p.keep_start for p in plans[1:]]


def _with_format(settings: Settings, fmt: str) -> Settings:
    return settings.model_copy(update={"transcribe_audio_format": fmt})


def _ffmpeg(settings: Settings, *args: str) -> None:
    run([settings.ffmpeg, "-hide_banner", "-nostdin", "-v", "error", "-y", *args], timeout=120)


# Tones separated by digital silence: gaps at [4, 5], [10, 12], [18, 18.8]; total 25 s.
TONE_LAYOUT: list[tuple[str, float]] = [
    ("tone", 4.0), ("gap", 1.0), ("tone", 5.0), ("gap", 2.0), ("tone", 6.0), ("gap", 0.8), ("tone", 6.2),
]
TONE_GAPS = [(4.0, 5.0), (10.0, 12.0), (18.0, 18.8)]
TONE_DURATION = 25.0


def _segment_inputs(layout: list[tuple[str, float]], sample_rate: int, first_input: int = 0) -> tuple[list[str], str]:
    """lavfi inputs (sine for tones, anullsrc for gaps) + a concat graph producing ``[a]``."""
    args: list[str] = []
    for i, (kind, dur) in enumerate(layout):
        if kind == "tone":
            args += ["-f", "lavfi", "-i", f"sine=frequency={300 + 70 * i}:sample_rate={sample_rate}:duration={dur}"]
        else:
            args += ["-f", "lavfi", "-i", f"anullsrc=r={sample_rate}:cl=mono:d={dur}"]
    labels = "".join(f"[{first_input + i}:a]" for i in range(len(layout)))
    return args, f"{labels}concat=n={len(layout)}:v=0:a=1[a]"


def _make_tone_wav(settings: Settings, out: Path, layout: list[tuple[str, float]] = TONE_LAYOUT,
                   sample_rate: int = 16000) -> Path:
    inputs, graph = _segment_inputs(layout, sample_rate)
    _ffmpeg(settings, *inputs, "-filter_complex", graph, "-map", "[a]", "-ac", "1", "-ar", str(sample_rate),
            "-c:a", "pcm_s16le", str(out))
    return out


def _make_video_with_layout(settings: Settings, out: Path, layout: list[tuple[str, float]]) -> Path:
    """Small H.264 + stereo 44.1 kHz AAC mp4 whose audio follows ``layout``."""
    total = sum(d for _, d in layout)
    inputs, graph = _segment_inputs(layout, 44100, first_input=1)
    _ffmpeg(settings, "-f", "lavfi", "-i", f"testsrc2=size=160x120:rate=10:duration={total}", *inputs,
            "-filter_complex", graph, "-map", "0:v", "-map", "[a]",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "96k", "-ac", "2", "-ar", "44100", str(out))
    return out


def _assert_ranges_close(got: list[tuple[float, float]], want: list[tuple[float, float]], tol: float) -> None:
    assert len(got) == len(want), (got, want)
    for (gs, ge), (ws, we) in zip(got, want):
        assert gs == pytest.approx(ws, abs=tol), (got, want)
        assert ge == pytest.approx(we, abs=tol), (got, want)


def _wav_params(path: Path) -> tuple[int, int, int]:
    with wave.open(str(path), "rb") as fh:
        return fh.getnchannels(), fh.getsampwidth(), fh.getframerate()


def _leftovers(directory: Path, keep: set[str]) -> list[str]:
    return sorted(p.name for p in directory.iterdir() if p.name not in keep)


# ============================================================================= pure unit tests
def test_audio_version_and_frozen_plan() -> None:
    # 2: the WAV origin is the container start for MPEG-TS/PS with late audio (review finding 14).
    # (Cached chunk transcripts are keyed by chunk content, pcm_sha256, not by this version.)
    assert AUDIO_VERSION == 2
    plan = ChunkPlan(index=0, start=0.0, end=1.0, keep_start=0.0, keep_end=2.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.start = 5.0  # type: ignore[misc]
    assert dataclasses.asdict(plan) == {"index": 0, "start": 0.0, "end": 1.0, "keep_start": 0.0, "keep_end": 2.0}


@pytest.mark.parametrize(("fmt", "ext"), [("flac", ".flac"), ("wav", ".wav"), ("mp3", ".mp3")])
def test_chunk_extension(settings: Settings, fmt: str, ext: str) -> None:
    assert chunk_extension(_with_format(settings, fmt)) == ext


def test_wav_duration_rejects_non_wav(tmp_path: Path) -> None:
    junk = tmp_path / "junk.wav"
    junk.write_bytes(b"not a wav file at all" * 10)
    with pytest.raises(MediaError) as exc:
        wav_duration(junk)
    assert str(tmp_path) not in exc.value.user_message


# ----------------------------------------------------------------------------- silencedetect parsing
def test_parse_silencedetect_pairs_lines() -> None:
    lines = [
        "[Parsed_silencedetect_0 @ 0xa9] silence_start: 3.999938\n",
        "[Parsed_silencedetect_0 @ 0xa9] silence_end: 5.000063 | silence_duration: 1.000125\n",
        "size=N/A time=00:00:10.00 bitrate=N/A speed= 500x\n",
        "[Parsed_silencedetect_0 @ 0xa9] silence_start: 10\n",
        "[Parsed_silencedetect_0 @ 0xa9] silence_end: 12.000063 | silence_duration: 2.000063\n",
    ]
    assert parse_silencedetect(lines, 25.0) == [(3.999938, 5.000063), (10.0, 12.000063)]


def test_parse_silencedetect_closes_trailing_silence_at_duration() -> None:
    lines = ["silence_start: 1", "silence_end: 2 | silence_duration: 1", "silence_start: 20.5"]
    assert parse_silencedetect(lines, 25.0) == [(1.0, 2.0), (20.5, 25.0)]


def test_parse_silencedetect_clamps_and_ignores_garbage() -> None:
    lines = [
        "silence_start: -0.00133", "silence_end: 0.8 | silence_duration: 0.80133",
        "silence_start: nan", "silence_start: abc", "silence_end: 3.0",   # no open start -> ignored
        "silence_start: 6", "silence_start: 7", "silence_end: 8",           # repeated start keeps the first
        "silence_start: 24.0", "silence_end: 26.0 | silence_duration: 2",  # clamped to the duration
        "silence_start: 30", "silence_end: 31",                              # entirely past the end -> dropped
    ]
    assert parse_silencedetect(lines, 25.0) == [(0.0, 0.8), (6.0, 8.0), (24.0, 25.0)]


def test_parse_silencedetect_empty() -> None:
    assert parse_silencedetect([], 10.0) == []


# ----------------------------------------------------------------------------- plan_chunks: table
def test_short_audio_is_one_chunk() -> None:
    assert plan_chunks(300.0, [(100.0, 101.0)], target=T, overlap=OV) == [
        ChunkPlan(index=0, start=0.0, end=300.0, keep_start=0.0, keep_end=301.0)
    ]


def test_up_to_ten_percent_over_target_is_one_chunk() -> None:
    plans = plan_chunks(660.0, [], target=T, overlap=OV)
    assert len(plans) == 1 and plans[0].end == 660.0 and plans[0].keep_end == 661.0


def test_just_over_threshold_splits_at_ideal_without_silences() -> None:
    assert plan_chunks(661.0, [], target=T, overlap=OV) == [
        ChunkPlan(index=0, start=0.0, end=603.0, keep_start=0.0, keep_end=600.0),
        ChunkPlan(index=1, start=597.0, end=661.0, keep_start=600.0, keep_end=662.0),
    ]


@pytest.mark.parametrize(
    ("duration", "silences", "expected_cuts"),
    [
        # no silences -> ideal multiples of the target
        (1800.0, [], [600.0, 1200.0]),
        # silence inside the ±60 s window is used (its midpoint); one outside (mid 710) is not
        (1500.0, [(570.0, 571.0), (610.0, 614.0), (700.0, 720.0)], [612.0, 1200.0]),
        # the longest silence wins over the closest one
        (1500.0, [(599.5, 600.5), (640.0, 650.0)], [645.0, 1200.0]),
        # equal lengths -> closest to the ideal cut
        (1500.0, [(560.0, 562.0), (601.0, 603.0)], [602.0, 1200.0]),
        # midpoint exactly on the window edge counts; just outside does not
        (1500.0, [(655.0, 665.0)], [660.0, 1200.0]),
        (1500.0, [(656.0, 666.0)], [600.0, 1200.0]),
        (1500.0, [(535.0, 545.0)], [540.0, 1200.0]),
        (1500.0, [(534.0, 544.0)], [600.0, 1200.0]),
        # each ideal cut picks its own silence
        (1500.0, [(590.0, 592.0), (1230.0, 1236.0)], [591.0, 1233.0]),
        # a tiny tail (10 s) is merged into the previous chunk ...
        (1210.0, [], [600.0]),
        # ... a 30 s tail is kept ...
        (1230.0, [], [600.0, 1200.0]),
        # ... and a silence that would leave < 30 s for the last chunk is not used
        (1250.0, [(1225.0, 1235.0)], [600.0, 1200.0]),
        # silences outside [0, duration] or non-finite are ignored
        (1500.0, [(-5.0, -1.0), (2000.0, 2100.0), (math.nan, 601.0), (599.0, math.inf)], [600.0, 1200.0]),
    ],
)
def test_plan_chunks_cut_selection(duration: float, silences: list[tuple[float, float]],
                                   expected_cuts: list[float]) -> None:
    plans = plan_chunks(duration, silences, target=T, overlap=OV, search=SR)
    assert _cuts(plans) == pytest.approx(expected_cuts, abs=1e-6)
    _check_partition(plans, duration, target=T, overlap=OV, search=SR, silences=silences,
                     rng=random.Random(duration))


def test_tiny_tail_merge_extends_last_chunk() -> None:
    plans = plan_chunks(1210.0, [], target=T, overlap=OV)
    assert plans[-1] == ChunkPlan(index=1, start=597.0, end=1210.0, keep_start=600.0, keep_end=1211.0)


def test_search_window_is_capped_at_quarter_target() -> None:
    # target 100 -> effective search 25 s even though 60 s was requested
    assert _cuts(plan_chunks(350.0, [(130.0, 131.0)], target=100.0, overlap=1.0, search=60.0))[0] == 100.0
    assert _cuts(plan_chunks(350.0, [(120.0, 121.0)], target=100.0, overlap=1.0, search=60.0))[0] == 120.5


def test_zero_overlap_chunks_equal_keep_ranges() -> None:
    plans = plan_chunks(2000.0, [(1190.0, 1194.0)], target=T, overlap=0.0)
    assert [(p.start, p.end) for p in plans] == [(0.0, 600.0), (600.0, 1192.0), (1192.0, 1800.0),
                                                 (1800.0, 2000.0)]


def test_two_hour_source_without_silences() -> None:
    plans = plan_chunks(7200.0, [], target=T, overlap=OV)
    assert len(plans) == 12
    assert _cuts(plans) == [600.0 * k for k in range(1, 12)]
    assert all(p.end - p.start <= T + 2 * OV for p in plans)
    _check_partition(plans, 7200.0, target=T, overlap=OV, search=SR)


def test_two_hour_source_snaps_to_long_pauses() -> None:
    # Short 0.4 s pauses every 25 s plus one 2 s pause near each ideal cut (alternating ±).
    silences = [(12.5 + 25 * j, 12.9 + 25 * j) for j in range(288)]
    expected = []
    for k in range(1, 12):
        mid = 600.0 * k + (40.0 if k % 2 == 0 else -45.0)
        silences.append((mid - 1.0, mid + 1.0))
        expected.append(mid)
    random.Random(7).shuffle(silences)  # input order must not matter
    plans = plan_chunks(7200.0, silences, target=T, overlap=OV, search=SR)
    assert _cuts(plans) == pytest.approx(expected, abs=1e-6)
    _check_partition(plans, 7200.0, target=T, overlap=OV, search=SR, silences=silences)


def test_plan_chunks_accepts_json_style_lists_and_does_not_mutate() -> None:
    silences = [[610.0, 614.0], [1190.0, 1191.0]]
    snapshot = [list(s) for s in silences]
    plans = plan_chunks(1500.0, silences, target=T, overlap=OV)  # type: ignore[arg-type]
    assert _cuts(plans) == [612.0, 1190.5]
    assert silences == snapshot
    assert plan_chunks(1500.0, silences, target=T, overlap=OV) == plans  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"duration": 0.0}, {"duration": -1.0}, {"duration": math.nan}, {"duration": math.inf},
        {"target": 0.0}, {"target": -600.0}, {"target": 0.5}, {"overlap": -1.0}, {"search": -1.0},
        {"overlap": math.nan},
    ],
)
def test_plan_chunks_rejects_invalid_arguments(kwargs: dict[str, float]) -> None:
    args = {"duration": 1000.0, "target": T, "overlap": OV, "search": SR, **kwargs}
    with pytest.raises(ValueError):
        plan_chunks(args["duration"], [], target=args["target"], overlap=args["overlap"], search=args["search"])


# ----------------------------------------------------------------------------- plan_chunks: properties
def _check_partition(plans: list[ChunkPlan], duration: float, *, target: float, overlap: float, search: float,
                     silences: list[tuple[float, float]] | None = None, rng: random.Random | None = None) -> None:
    eps = 5e-6  # plan times are rounded to whole microseconds
    search_eff = min(search, target / 4.0)
    assert plans, "at least one chunk"
    assert [p.index for p in plans] == list(range(len(plans)))
    # keep ranges: contiguous, non-overlapping, covering [0, duration] (last one open-ended)
    assert plans[0].keep_start == 0.0
    assert plans[-1].keep_end == duration + 1.0
    for a, b in zip(plans, plans[1:]):
        assert a.keep_end == b.keep_start
    for p in plans:
        assert p.keep_start < p.keep_end
        # audio range = keep range widened by the overlap, clamped to the media
        assert p.start == pytest.approx(max(0.0, p.keep_start - overlap), abs=eps)
        assert p.end == pytest.approx(min(duration, p.keep_end + overlap), abs=eps)
        assert 0.0 <= p.start < p.end <= duration
        assert p.start <= p.keep_start and p.end >= min(p.keep_end, duration)
    if duration <= target * 1.1:
        assert len(plans) == 1
    if len(plans) == 1:
        assert (plans[0].start, plans[0].end) == (0.0, duration)
        assert duration <= max(target * 1.1, target + 2 * search_eff) + eps
    else:
        for p in plans:
            assert p.end - p.start <= target + 2 * overlap + 2 * search + eps, p
        # no tiny final chunk
        assert duration - plans[-1].keep_start >= min(MIN_TAIL_SECONDS, search_eff) - eps
    # every cut is either an ideal multiple of the target or the midpoint of a provided silence
    # (silences are first clamped to [0, duration])
    mids = [(max(0.0, s) + min(duration, e)) / 2.0 for s, e in (silences or [])
            if math.isfinite(s) and math.isfinite(e) and min(duration, e) > max(0.0, s)]
    for cut in _cuts(plans):
        k = round(cut / target)
        assert abs(cut - k * target) <= search_eff + eps
        assert abs(cut - k * target) <= eps or any(abs(cut - m) <= eps for m in mids), cut
    # any source time belongs to exactly one keep range, and that chunk's audio contains it
    if rng is not None:
        probes = [0.0, duration] + [rng.uniform(0.0, duration) for _ in range(20)] + _cuts(plans)
        for t in probes:
            owners = [p for p in plans if p.keep_start <= t < p.keep_end]
            assert len(owners) == 1, (t, owners)
            assert owners[0].start <= t <= owners[0].end


def _random_silences(rng: random.Random, duration: float) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for _ in range(rng.randint(0, 120)):
        start = rng.uniform(-10.0, duration + 10.0)
        length = rng.choice([0.0, rng.uniform(0.05, 0.6), rng.uniform(0.3, 6.0)])
        out.append((start, start + length) if rng.random() > 0.02 else (start + length, start))
    return out


def test_plan_chunks_partition_property_random() -> None:
    rng = random.Random(20261006)
    for _ in range(2500):
        target = rng.choice([600.0, rng.uniform(1.0, 60.0), rng.uniform(20.0, 1200.0)])
        duration = rng.choice([
            rng.uniform(0.01, target * 1.2),
            rng.uniform(target * 1.05, target * 4.5),
            rng.uniform(1.0, 9000.0),
            float(rng.randint(1, 8)) * target,  # exact multiples
            float(rng.randint(1, 8)) * target + rng.choice([0.5, 29.9, 30.0, 30.1]),
        ])
        overlap = rng.choice([0.0, 3.0, rng.uniform(0.0, 15.0)])
        search = rng.choice([0.0, 60.0, rng.uniform(0.0, 150.0)])
        silences = _random_silences(rng, duration)
        plans = plan_chunks(duration, silences, target=target, overlap=overlap, search=search)
        _check_partition(plans, duration, target=target, overlap=overlap, search=search,
                         silences=silences, rng=rng)


# ============================================================================= real FFmpeg
@pytest.mark.ffmpeg
@requires_ffmpeg
def test_detect_silences_finds_known_gaps(settings: Settings, tmp_path: Path) -> None:
    wav = _make_tone_wav(settings, tmp_path / "tones.wav")
    assert wav_duration(wav) == pytest.approx(TONE_DURATION, abs=1e-3)
    _assert_ranges_close(detect_silences(wav, settings=settings), TONE_GAPS, tol=0.05)


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_detect_silences_parameters_are_applied(settings: Settings, tmp_path: Path) -> None:
    wav = _make_tone_wav(settings, tmp_path / "tones.wav")
    # only the 2 s gap is at least 1.5 s long
    _assert_ranges_close(detect_silences(wav, settings=settings, min_silence=1.5), [(10.0, 12.0)], tol=0.05)
    # the sine tones peak around -18 dBFS, so with a -10 dB threshold everything is "silence"
    _assert_ranges_close(detect_silences(wav, settings=settings, noise_db=-10.0), [(0.0, TONE_DURATION)],
                         tol=0.05)


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_detect_silences_leading_and_trailing(settings: Settings, tmp_path: Path) -> None:
    wav = _make_tone_wav(settings, tmp_path / "edges.wav", [("gap", 1.5), ("tone", 2.0), ("gap", 1.25)])
    _assert_ranges_close(detect_silences(wav, settings=settings), [(0.0, 1.5), (3.5, 4.75)], tol=0.05)


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_detect_silences_many_pauses_beyond_stderr_tail(settings: Settings, tmp_path: Path) -> None:
    # 40 pauses -> 80 log lines, more than the 60-line stderr tail kept by procs.run.
    wav = tmp_path / "pulses.wav"
    _ffmpeg(settings, "-f", "lavfi", "-i",
            "aevalsrc=exprs='if(lt(mod(t\\,0.8)\\,0.3)\\,0.25*sin(2*PI*440*t)\\,0)':s=16000:d=32",
            "-c:a", "pcm_s16le", str(wav))
    silences = detect_silences(wav, settings=settings)
    assert len(silences) == 40
    _assert_ranges_close(silences, [(0.8 * k + 0.3, 0.8 * (k + 1)) for k in range(40)], tol=0.05)


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_mean_volume_db(settings: Settings, tmp_path: Path) -> None:
    tones = _make_tone_wav(settings, tmp_path / "tones.wav")
    loud = mean_volume_db(tones, settings=settings)
    assert loud is not None and -30.0 < loud < -15.0
    silent = _make_tone_wav(settings, tmp_path / "silent.wav", [("gap", 2.0)])
    quiet = mean_volume_db(silent, settings=settings)
    assert quiet is not None and quiet <= -90.0


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_extract_analysis_audio_from_video(settings: Settings, tmp_path: Path) -> None:
    src_dir, out_dir = tmp_path / "src", tmp_path / "out"
    src_dir.mkdir()
    src = _make_video_with_layout(settings, src_dir / "in.mp4", [("tone", 2.0), ("gap", 1.0), ("tone", 2.0)])
    wav = out_dir / "analysis.wav"
    duration = extract_analysis_audio(src, wav, settings=settings)
    assert _wav_params(wav) == (1, 2, 16000)  # mono, 16-bit, 16 kHz
    assert duration == wav_duration(wav)
    assert duration == pytest.approx(5.0, abs=0.05)
    # the analysis timeline is the source timeline
    _assert_ranges_close(detect_silences(wav, settings=settings), [(2.0, 3.0)], tol=0.05)
    assert _leftovers(out_dir, {"analysis.wav"}) == []  # atomic write left no temp files


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_extract_aligns_late_starting_audio_to_source_time(settings: Settings, tmp_path: Path) -> None:
    src = tmp_path / "late.mp4"
    _ffmpeg(settings, "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=10:duration=3",
            "-itsoffset", "0.5", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=2.5",
            "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", str(src))
    wav = tmp_path / "late.wav"
    duration = extract_analysis_audio(src, wav, settings=settings)
    assert duration == pytest.approx(3.0, abs=0.05)
    silences = detect_silences(wav, settings=settings, min_silence=0.1)
    _assert_ranges_close(silences, [(0.0, 0.5)], tol=0.05)


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_extract_uses_first_audio_stream(settings: Settings, tmp_path: Path) -> None:
    src = tmp_path / "two_tracks.mkv"
    _ffmpeg(settings, "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=10:duration=2",
            "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo:d=2",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=2",
            "-map", "0:v", "-map", "1:a", "-map", "2:a", "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "flac", str(src))
    wav = tmp_path / "a.wav"
    extract_analysis_audio(src, wav, settings=settings)
    vol = mean_volume_db(wav, settings=settings)
    assert vol is not None and vol <= -90.0  # the silent first track, not the tone


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_extract_audio_only_input(settings: Settings, tmp_path: Path) -> None:
    src = tmp_path / "voice.m4a"
    _ffmpeg(settings, "-f", "lavfi", "-i", "sine=frequency=300:sample_rate=48000:duration=2", "-c:a", "aac",
            str(src))
    duration = extract_analysis_audio(src, tmp_path / "a.wav", settings=settings)
    assert duration == pytest.approx(2.0, abs=0.05)


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_extract_video_without_audio_raises_no_speech(settings: Settings, tmp_path: Path) -> None:
    src = tmp_path / "silent.mp4"
    _ffmpeg(settings, "-f", "lavfi", "-i", "testsrc2=size=160x120:rate=10:duration=1",
            "-c:v", "libx264", "-preset", "ultrafast", str(src))
    out_dir = tmp_path / "out"
    with pytest.raises(NoSpeechError) as exc:
        extract_analysis_audio(src, out_dir / "a.wav", settings=settings)
    assert exc.value.user_message == NO_AUDIO_MESSAGE
    assert not (out_dir / "a.wav").exists()


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_extract_corrupt_input_raises_media_error(settings: Settings, tmp_path: Path) -> None:
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(random.Random(99).randbytes(40_000))
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    with pytest.raises(MediaError) as exc:
        extract_analysis_audio(junk, out_dir / "a.wav", settings=settings)
    assert str(tmp_path) not in exc.value.user_message
    assert _leftovers(out_dir, set()) == []


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_extract_respects_cancel(settings: Settings, tmp_path: Path) -> None:
    src = _make_video_with_layout(settings, tmp_path / "in.mp4", [("tone", 1.0)])
    token = CancelToken()
    token.cancel()
    out_dir = tmp_path / "out"
    with pytest.raises(CancelledError):
        extract_analysis_audio(src, out_dir / "a.wav", settings=settings, cancel=token)
    assert not (out_dir / "a.wav").exists()


@pytest.fixture
def tone_wav(settings: Settings, tmp_path: Path) -> Path:
    """25 s analysis WAV with known gaps (users carry ``requires_ffmpeg``, evaluated before setup)."""
    return _make_tone_wav(settings, tmp_path / "analysis.wav")


_CODEC = {"flac": "flac", "wav": "pcm_s16le", "mp3": "mp3"}


@pytest.mark.ffmpeg
@requires_ffmpeg
@pytest.mark.parametrize("fmt", ["flac", "wav", "mp3"])
@pytest.mark.parametrize(("start", "end"), [(0.0, 7.3), (3.21, 11.7), (12.345, 25.0), (24.5, 25.0)])
def test_encode_chunk_exact_duration(settings: Settings, tone_wav: Path, tmp_path: Path, fmt: str,
                                     start: float, end: float) -> None:
    s = _with_format(settings, fmt)
    out_dir = tmp_path / "chunks"
    out = out_dir / f"chunk_000{chunk_extension(s)}"
    encode_chunk(tone_wav, out, start, end, settings=s)
    info = probe(out, settings=s)
    # FFmpeg 7.1 (Debian trixie container) reports MP3 container durations including ~80 ms of
    # encoder padding; the audio content itself stays aligned (see the alignment test below).
    tolerance = 0.1 if fmt == "mp3" else 0.03
    assert info.duration == pytest.approx(end - start, abs=tolerance)
    assert info.has_audio and not info.has_video
    assert (info.audio_codec, info.audio_channels, info.audio_sample_rate) == (_CODEC[fmt], 1, 16000)
    assert _leftovers(out_dir, {out.name}) == []


@pytest.mark.ffmpeg
@requires_ffmpeg
@pytest.mark.parametrize("fmt", ["flac", "wav", "mp3"])
def test_encode_chunk_content_is_aligned(settings: Settings, tone_wav: Path, tmp_path: Path, fmt: str) -> None:
    s = _with_format(settings, fmt)
    out = tmp_path / f"c{chunk_extension(s)}"
    encode_chunk(tone_wav, out, 3.0, 13.0, settings=s)
    back = tmp_path / "back.wav"
    _ffmpeg(settings, "-i", str(out), "-c:a", "pcm_s16le", str(back))
    # source gaps [4, 5] and [10, 12] appear at chunk-relative [1, 2] and [7, 9]
    _assert_ranges_close(detect_silences(back, settings=settings), [(1.0, 2.0), (7.0, 9.0)], tol=0.05)


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_plan_and_encode_round_trip(settings: Settings, tone_wav: Path, tmp_path: Path) -> None:
    silences = detect_silences(tone_wav, settings=settings)
    duration = wav_duration(tone_wav)
    plans = plan_chunks(duration, silences, target=10.0, overlap=0.5, search=2.5)
    # ideal cuts 10 s and 20 s move to the middle of the nearby pauses (11.0 and 18.4)
    assert _cuts(plans) == pytest.approx([11.0, 18.4], abs=0.05)
    for plan in plans:
        out = tmp_path / f"chunk_{plan.index:03d}{chunk_extension(settings)}"
        encode_chunk(tone_wav, out, plan.start, plan.end, settings=settings)
        assert probe(out, settings=settings).duration == pytest.approx(plan.end - plan.start, abs=0.03)


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_encode_chunk_overwrites_existing_file(settings: Settings, tone_wav: Path, tmp_path: Path) -> None:
    out = tmp_path / "chunk.flac"
    out.write_bytes(b"stale")
    encode_chunk(tone_wav, out, 1.0, 3.0, settings=settings)
    assert probe(out, settings=settings).duration == pytest.approx(2.0, abs=0.03)


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_encode_chunk_over_upload_limit_is_media_error(settings: Settings, tone_wav: Path, tmp_path: Path,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(audio, "MAX_CHUNK_BYTES", 1000)  # deterministic stand-in for a > 24.5 MB chunk
    out_dir = tmp_path / "chunks"
    with pytest.raises(MediaError) as exc:
        encode_chunk(tone_wav, out_dir / "chunk.flac", 0.0, 10.0, settings=settings)
    assert "25 MB" in exc.value.user_message
    assert str(tmp_path) not in exc.value.user_message
    assert _leftovers(out_dir, set()) == []


@pytest.mark.parametrize(("start", "end"), [(-1.0, 2.0), (5.0, 5.0), (5.0, 4.0), (math.nan, 3.0), (0.0, math.inf)])
def test_encode_chunk_rejects_bad_ranges(settings: Settings, tmp_path: Path, start: float, end: float) -> None:
    with pytest.raises(ValueError):
        encode_chunk(tmp_path / "in.wav", tmp_path / "out.flac", start, end, settings=settings)
    assert not (tmp_path / "out.flac").exists()


# ----------------------------------------------------------------------------- chunk content identity
def _pcm_digest(frames: bytes, rate: int = 16000, channels: int = 1, width: int = 2) -> str:
    import hashlib
    return hashlib.sha256(f"pcm:{rate}:{channels}:{width}:".encode("ascii") + frames).hexdigest()


@pytest.mark.ffmpeg
@requires_ffmpeg
def test_pcm_sha256_identifies_exactly_the_samples_a_chunk_sends(settings: Settings, tone_wav: Path,
                                                                 tmp_path: Path) -> None:
    s = _with_format(settings, "wav")
    out = tmp_path / "chunk.wav"
    encode_chunk(tone_wav, out, 3.21, 11.7, settings=s)
    with wave.open(str(out), "rb") as fh:
        sent = fh.readframes(fh.getnframes())
    assert pcm_sha256(tone_wav, 3.21, 11.7) == _pcm_digest(sent)
    # same content → same key (a re-extracted, byte-identical WAV); other ranges → other keys
    copy = tmp_path / "copy.wav"
    copy.write_bytes(tone_wav.read_bytes())
    assert pcm_sha256(copy, 3.21, 11.7) == pcm_sha256(tone_wav, 3.21, 11.7)
    assert pcm_sha256(tone_wav, 3.25, 11.7) != pcm_sha256(tone_wav, 3.21, 11.7)
    # a range running past the end hashes the samples that exist (as encode_chunk sends them)
    with wave.open(str(tone_wav), "rb") as fh:
        fh.setpos(round(24.0 * fh.getframerate()))
        tail = fh.readframes(fh.getnframes())
    assert pcm_sha256(tone_wav, 24.0, 30.0) == _pcm_digest(tail)


@pytest.mark.parametrize(("start", "end"), [(-1.0, 2.0), (5.0, 5.0), (math.nan, 3.0)])
def test_pcm_sha256_rejects_bad_ranges(tmp_path: Path, start: float, end: float) -> None:
    with pytest.raises(ValueError):
        pcm_sha256(tmp_path / "in.wav", start, end)


def test_pcm_sha256_unreadable_wav_is_media_error(tmp_path: Path) -> None:
    junk = tmp_path / "junk.wav"
    junk.write_bytes(b"not a wav file at all" * 10)
    with pytest.raises(MediaError):
        pcm_sha256(junk, 0.0, 1.0)
