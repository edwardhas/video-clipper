"""Analysis audio: extraction, pause detection, chunk planning and chunk encoding.

All times in this module are **source seconds** (``src`` timeline): the analysis WAV
is aligned so that sample 0 is source time 0 — the container ``start_time``, the same
origin ``-ss`` seeking and the renderer use (a late-starting audio stream is padded
with silence, encoder-delay samples with negative timestamps are trimmed), so a chunk
that starts at ``plan.start`` in the WAV starts at ``plan.start`` on the source. The
extraction keeps the demuxer's absolute timestamps (``-copyts``) and shifts them by the
container start itself: without that, the FFmpeg CLI re-bases MPEG-TS/PS timestamps to the
first *audio* packet, so a late audio start was never padded and every transcript time was
early by the audio delay.

Pipeline usage::

    duration = extract_analysis_audio(src, wav, settings=s, cancel=c)
    silences = detect_silences(wav, settings=s, cancel=c)
    plans = plan_chunks(duration, silences, target=s.transcribe_chunk_seconds,
                        overlap=s.transcribe_overlap_seconds)
    for plan in plans:
        encode_chunk(wav, out_dir / f"chunk_{plan.index:03d}{chunk_extension(s)}",
                     plan.start, plan.end, settings=s, cancel=c)
        content_key = pcm_sha256(wav, plan.start, plan.end)   # transcription cache identity
"""

from __future__ import annotations

import hashlib
import math
import re
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..config import Settings
from ..errors import MediaError, NoSpeechError
from ..procs import CancelToken, run
from ..storage import atomic_path
from .probe import probe

# Part of the audio stage key (re-extraction is local). Cached chunk transcripts are keyed by the
# chunk's audio content (pcm_sha256), so a bump that leaves the audio unchanged never re-bills.
AUDIO_VERSION = 2

SAMPLE_RATE = 16000
# OpenAI's transcription upload limit is 25 MB; stay safely below it.
MAX_CHUNK_BYTES = 24_500_000
# A final chunk shorter than this is merged into the previous one (see plan_chunks).
MIN_TAIL_SECONDS = 30.0
# Lowest value reported by mean_volume_db (16-bit digital silence reads about -91 dB).
MIN_VOLUME_DB = -150.0

NO_AUDIO_MESSAGE = "This video has no audio track, so there is no speech to analyse."
EMPTY_AUDIO_MESSAGE = "The audio track of this video is empty, so there is no speech to analyse."

_EXTENSIONS = {"flac": ".flac", "wav": ".wav", "mp3": ".mp3"}
_CODEC_ARGS = {
    "flac": ["-c:a", "flac", "-sample_fmt", "s16", "-f", "flac"],
    "wav": ["-c:a", "pcm_s16le", "-f", "wav"],
    "mp3": ["-c:a", "libmp3lame", "-b:a", "64k", "-f", "mp3"],
}
_SILENCE_RE = re.compile(r"silence_(start|end):\s*(\S+)")
_MEAN_VOLUME_RE = re.compile(r"mean_volume:\s*(\S+)\s*dB")


@dataclass(frozen=True)
class ChunkPlan:
    """One transcription chunk, in source seconds.

    ``start``/``end`` bound the audio sent to the provider (including overlap);
    words whose midpoint lies in ``[keep_start, keep_end)`` belong to this chunk.
    """

    index: int
    start: float
    end: float
    keep_start: float
    keep_end: float


# ----------------------------------------------------------------------------- helpers
def chunk_extension(settings: Settings) -> str:
    """File extension for transcription chunks (``.flac``, ``.wav`` or ``.mp3``)."""
    return _EXTENSIONS[settings.transcribe_audio_format]


def wav_duration(wav: Path) -> float:
    """Exact duration of a PCM WAV from its sample count."""
    try:
        with wave.open(str(wav), "rb") as fh:
            rate = fh.getframerate()
            frames = fh.getnframes()
    except (wave.Error, EOFError, OSError) as exc:
        raise MediaError("The extracted audio could not be read.", detail=f"wav header: {exc}") from exc
    if rate <= 0:
        raise MediaError("The extracted audio could not be read.", detail="wav header: zero sample rate")
    return frames / float(rate)


def pcm_sha256(wav: Path, start: float, end: float) -> str:
    """sha256 of the PCM samples of ``[start, end)`` source seconds of a WAV (plus its sample format).

    Identifies the audio a chunk sends for transcription by content, so a cached result is
    reused exactly when the audio is the same (whatever version of the extraction made it).
    """
    if not (math.isfinite(start) and math.isfinite(end)) or start < 0 or end <= start:
        raise ValueError("chunk range must satisfy 0 <= start < end")
    try:
        with wave.open(str(wav), "rb") as fh:
            rate, channels, width = fh.getframerate(), fh.getnchannels(), fh.getsampwidth()
            total = fh.getnframes()
            first = min(total, round(start * rate))
            remaining = min(total, round(end * rate)) - first
            digest = hashlib.sha256(f"pcm:{rate}:{channels}:{width}:".encode("ascii"))
            fh.setpos(first)
            while remaining > 0:
                block = fh.readframes(min(remaining, 1 << 20))
                if not block:
                    break
                digest.update(block)
                remaining -= len(block) // (channels * width)
    except (wave.Error, EOFError, OSError) as exc:
        raise MediaError("The extracted audio could not be read.", detail=f"wav read: {exc}") from exc
    return digest.hexdigest()


def _ffmpeg_base(settings: Settings) -> list[str]:
    return [settings.ffmpeg, "-hide_banner", "-nostdin"]


def _fmt_seconds(value: float) -> str:
    return f"{value:.6f}"


# ----------------------------------------------------------------------------- extraction
def extract_analysis_audio(src: Path, out_wav: Path, *, settings: Settings,
                           cancel: CancelToken | None = None) -> float:
    """Write the first audio stream of ``src`` as 16 kHz mono s16 WAV; return its duration.

    Raises ``NoSpeechError`` when the source has no (or an empty) audio track and
    ``MediaError`` when it cannot be decoded.
    """
    media = probe(src, settings=settings, cancel=cancel)
    if not media.has_audio:
        raise NoSpeechError(NO_AUDIO_MESSAGE)
    args = [
        *_ffmpeg_base(settings), "-v", "error", "-y",
        "-copyts", "-i", str(src),
        "-map", "0:a:0", "-vn", "-sn", "-dn", "-map_metadata", "-1",
        # Absolute timestamps shifted so the container start is 0, then pad a late audio
        # start / trim negative-pts priming so WAV time == source time.
        "-af", f"asetpts=PTS-({media.start_time:.6f})/TB,aresample=async=1:first_pts=0",
        "-ac", "1", "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le",
        "-fflags", "+bitexact", "-f", "wav",
    ]
    with atomic_path(out_wav) as tmp:
        run([*args, str(tmp)], cancel=cancel, timeout=600 + media.duration * 0.5,
            capture_stdout=False, error_message="Could not extract the audio track from this video.")
        duration = wav_duration(tmp)
        if duration <= 0:
            raise NoSpeechError(EMPTY_AUDIO_MESSAGE)
    return duration


# ----------------------------------------------------------------------------- analysis
def detect_silences(wav: Path, *, settings: Settings, cancel: CancelToken | None = None,
                    noise_db: float = -35.0, min_silence: float = 0.35) -> list[tuple[float, float]]:
    """Return sorted ``(src_start, src_end)`` pauses found by ffmpeg ``silencedetect``.

    A silence still open at the end of the file is closed at the file duration.
    """
    duration = wav_duration(wav)
    lines: list[str] = []

    def on_line(line: str) -> None:
        if "silence_" in line:
            lines.append(line)

    run(
        [*_ffmpeg_base(settings), "-nostats", "-v", "info", "-i", str(wav),
         "-af", f"silencedetect=noise={noise_db:g}dB:d={min_silence:g}", "-vn", "-sn", "-f", "null", "-"],
        cancel=cancel, timeout=600 + duration * 0.25, capture_stdout=False,
        stderr_line_cb=on_line, error_message="Could not analyse pauses in the audio.",
    )
    return parse_silencedetect(lines, duration)


def parse_silencedetect(lines: Iterable[str], duration: float) -> list[tuple[float, float]]:
    """Pair ``silence_start``/``silence_end`` log lines into clamped ``(start, end)`` ranges."""
    silences: list[tuple[float, float]] = []
    open_start: float | None = None
    for line in lines:
        for kind, token in _SILENCE_RE.findall(line):
            try:
                value = float(token)
            except ValueError:
                continue
            if not math.isfinite(value):
                continue
            if kind == "start":
                if open_start is None:
                    open_start = value
            elif open_start is not None:
                silences.append((open_start, value))
                open_start = None
    if open_start is not None:
        silences.append((open_start, duration))
    clamped = [(max(0.0, s), min(duration, e)) for s, e in silences]
    return sorted((s, e) for s, e in clamped if e > s)


def mean_volume_db(wav: Path, *, settings: Settings, cancel: CancelToken | None = None) -> float | None:
    """Mean volume in dBFS via ``volumedetect`` (about -91 for digital silence); None if unknown."""
    values: list[float] = []

    def on_line(line: str) -> None:
        match = _MEAN_VOLUME_RE.search(line)
        if match:
            try:
                values.append(float(match.group(1)))
            except ValueError:
                pass

    run(
        [*_ffmpeg_base(settings), "-nostats", "-v", "info", "-i", str(wav),
         "-af", "volumedetect", "-vn", "-sn", "-f", "null", "-"],
        cancel=cancel, timeout=600 + wav_duration(wav) * 0.25, capture_stdout=False,
        stderr_line_cb=on_line, error_message="Could not measure the audio volume.",
    )
    if not values or math.isnan(values[-1]):
        return None
    return max(values[-1], MIN_VOLUME_DB)


# ----------------------------------------------------------------------------- planning (pure)
def plan_chunks(duration: float, silences: list[tuple[float, float]], *, target: float, overlap: float,
                search: float = 60.0) -> list[ChunkPlan]:
    """Split ``[0, duration]`` into transcription chunks cut at pauses near multiples of ``target``.

    * ``duration <= 1.1 * target`` → a single chunk.
    * Otherwise the ideal cuts are ``k * target``. Among silences whose midpoint lies
      within ``±search`` of an ideal cut, the longest wins (ties → closest to the ideal);
      the cut is its midpoint, or the ideal itself when no silence qualifies.
      ``search`` is capped at ``target / 4`` so neighbouring windows never overlap.
    * A final piece shorter than ``min(30 s, capped search)`` is merged into the previous chunk,
      and cut windows never leave less than that for the final chunk, so every chunk's
      keep range is at most ``target + 2 * search`` long.
    * Keep ranges partition the timeline: the first starts at 0, consecutive ranges share
      endpoints, the last ends at ``duration + 1`` (open end). Chunk audio is the keep range
      widened by ``overlap`` on both sides, clamped to ``[0, duration]``. Cut times are
      rounded to whole microseconds.

    Raises ``ValueError`` for non-finite/non-positive ``duration``, ``target < 1`` or a
    negative ``overlap``/``search``.
    """
    _validate_plan_args(duration, target, overlap, search)
    if duration <= target * 1.1:
        return [ChunkPlan(index=0, start=0.0, end=duration, keep_start=0.0, keep_end=duration + 1.0)]

    search_eff = min(search, target / 4.0)
    min_tail = min(MIN_TAIL_SECONDS, search_eff)
    usable = _clamp_silences(silences, duration)

    n_chunks = math.ceil(duration / target)
    remainder = duration - (n_chunks - 1) * target
    if n_chunks > 1 and (remainder < min_tail or remainder <= 0.0):  # `<= 0`: float edge, empty tail
        n_chunks -= 1

    cuts: list[float] = []
    for k in range(1, n_chunks):
        ideal = k * target
        cuts.append(_choose_cut(ideal, usable, lo=ideal - search_eff,
                                hi=min(ideal + search_eff, duration - min_tail)))
    return _build_plans(duration, cuts, overlap)


def _validate_plan_args(duration: float, target: float, overlap: float, search: float) -> None:
    for name, value in (("duration", duration), ("target", target), ("overlap", overlap), ("search", search)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
    if duration <= 0:
        raise ValueError("duration must be positive")
    if target < 1.0:
        raise ValueError("target must be at least 1 second")
    if overlap < 0 or search < 0:
        raise ValueError("overlap and search must not be negative")


def _clamp_silences(silences: Iterable[tuple[float, float]], duration: float) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for start, end in silences:
        if not (math.isfinite(start) and math.isfinite(end)):
            continue
        start, end = max(0.0, start), min(duration, end)
        if end > start:
            out.append((start, end))
    return out


def _choose_cut(ideal: float, silences: list[tuple[float, float]], *, lo: float, hi: float) -> float:
    """Midpoint of the longest silence centred in ``[lo, hi]`` (ties → closest to ideal), else ideal.

    Cuts are rounded to whole microseconds so that rounding the overlapped audio range in
    :func:`_build_plans` can never shrink it inside its keep range.
    """
    best: tuple[float, float, float] | None = None
    for start, end in silences:
        mid = (start + end) / 2.0
        if lo <= mid <= hi:
            key = (-(end - start), abs(mid - ideal), mid)
            if best is None or key < best:
                best = key
    return round(best[2] if best is not None else ideal, 6)


def _build_plans(duration: float, cuts: list[float], overlap: float) -> list[ChunkPlan]:
    keep_bounds = [0.0, *cuts, duration + 1.0]
    plans: list[ChunkPlan] = []
    for index in range(len(keep_bounds) - 1):
        keep_start, keep_end = keep_bounds[index], keep_bounds[index + 1]
        plans.append(ChunkPlan(
            index=index,
            start=max(0.0, round(keep_start - overlap, 6)),
            end=min(duration, round(keep_end + overlap, 6)),
            keep_start=keep_start,
            keep_end=keep_end,
        ))
    return plans


# ----------------------------------------------------------------------------- chunk encoding
def encode_chunk(wav: Path, out_path: Path, start: float, end: float, *, settings: Settings,
                 cancel: CancelToken | None = None) -> None:
    """Encode ``[start, end)`` source seconds of the analysis WAV for upload.

    Input seeking on PCM WAV is sample-accurate. The container follows
    ``settings.transcribe_audio_format`` (16 kHz mono). The file is written atomically
    and rejected with ``MediaError`` when it exceeds the upload limit.
    """
    if not (math.isfinite(start) and math.isfinite(end)) or start < 0 or end <= start:
        raise ValueError("chunk range must satisfy 0 <= start < end")
    length = end - start
    args = [
        *_ffmpeg_base(settings), "-v", "error", "-y",
        "-ss", _fmt_seconds(start), "-t", _fmt_seconds(length), "-i", str(wav),
        "-map", "0:a:0", "-vn", "-map_metadata", "-1",
        "-ac", "1", "-ar", str(SAMPLE_RATE),
        "-fflags", "+bitexact", "-flags:a", "+bitexact",
        *_CODEC_ARGS[settings.transcribe_audio_format],
    ]
    with atomic_path(out_path) as tmp:
        run([*args, str(tmp)], cancel=cancel, timeout=300 + length * 0.5, capture_stdout=False,
            error_message="Could not prepare an audio chunk for transcription.")
        size = tmp.stat().st_size
        if size == 0:
            raise MediaError("Could not prepare an audio chunk for transcription.",
                             detail="encoded chunk is empty")
        if size > MAX_CHUNK_BYTES:
            raise MediaError(
                "An audio chunk is larger than the 25 MB transcription upload limit. Use FLAC chunks "
                "(CLIPPER_TRANSCRIBE_AUDIO_FORMAT=flac) or a shorter CLIPPER_TRANSCRIBE_CHUNK_SECONDS.",
                detail=f"chunk size {size} bytes",
            )
