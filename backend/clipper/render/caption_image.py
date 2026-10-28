"""Caption track as transparent PNG images + an ffconcat timeline.

FFmpeg builds here may lack libass/freetype (no ``subtitles``/``drawtext``), so captions
are drawn with Pillow using the bundled OFL Montserrat font. Every distinct visual state
(a cue, or a cue with one spoken word highlighted) becomes one full-width RGBA PNG "band";
an ffconcat file lists the states with exact durations (transparent PNG for gaps; each
entry carries ``option framerate 1000`` so switch times keep 1 ms precision). The
renderer feeds it to the concat demuxer and overlays it at ``(x, y)``.

All times here are clip-relative seconds (``rel_*``).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ..config import ASSETS_DIR
from ..models import CaptionStyle, Cue

FONT_PATH = ASSETS_DIR / "fonts" / "Montserrat-ExtraBold.ttf"
BASE_WIDTH = 1080
BASE_HEIGHT = 1920
SIZE_PX = {"S": 58, "M": 70, "L": 84}
POSITION_CENTER = {"top": 0.22, "middle": 0.50, "bottom": 0.70}
SIDE_MARGIN = 90          # px at 1080 wide
TOP_SAFE = 160            # px at 1920 high
BOTTOM_SAFE = 340         # px at 1920 high (TikTok/Reels/Shorts UI)
STROKE = 6
LINE_SPACING = 1.12       # baseline-to-baseline distance as a multiple of the font size

RGBA = tuple[int, int, int, int]
WHITE: RGBA = (255, 255, 255, 255)
BLACK: RGBA = (0, 0, 0, 255)
AMBER: RGBA = (0xF2, 0xA3, 0x3A, 255)
YELLOW: RGBA = (0xFF, 0xE1, 0x4D, 255)
BOX_FILL: RGBA = (0x59, 0x1E, 0x4A, round(0.88 * 255))
TRANSPARENT: RGBA = (0, 0, 0, 0)

BLANK_NAME = "blank.png"
CONCAT_NAME = "captions.ffconcat"
# Every concat entry gets a 1 ms image2 time base. Without it the concat demuxer inherits
# image2's default 1/25 s time base and rounds every state switch to a 40 ms grid. The
# option must follow *every* file line: later files' timestamps are not rescaled to the
# first file's time base.
_FRAMERATE_OPTION = "framerate 1000"

# (size factor, max lines, allow breaking inside a word) tried in order.
_FIT_STEPS: tuple[tuple[float, int, bool], ...] = (
    *((f, 2, False) for f in (1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7)),
    (0.7, 3, True),
    *((f, 6, True) for f in (0.65, 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3)),
)


@dataclass(frozen=True)
class CaptionTrack:
    concat_path: Path      # ffconcat file (PNG paths inside are relative to it)
    x: int                 # overlay position of the band on the output frame
    y: int
    w: int                 # band size (all PNGs share it)
    h: int
    frames: int            # number of distinct PNG images written (incl. the transparent one)


@dataclass(frozen=True)
class _Palette:
    text: RGBA
    highlight: RGBA
    stroke: RGBA | None
    box: RGBA | None


PALETTES: dict[str, _Palette] = {
    "white": _Palette(text=WHITE, highlight=AMBER, stroke=BLACK, box=None),
    "yellow": _Palette(text=YELLOW, highlight=WHITE, stroke=BLACK, box=None),
    "boxed": _Palette(text=WHITE, highlight=AMBER, stroke=None, box=BOX_FILL),
}

Piece = tuple[str, int]          # (text drawn, index of the word it belongs to)
Lines = tuple[tuple[Piece, ...], ...]


@dataclass(frozen=True)
class _Geometry:
    width: int
    height: int
    scale: float           # width / 1080
    vscale: float          # height / 1920
    palette: _Palette
    base_px: int

    @property
    def stroke(self) -> int:
        return max(1, round(STROKE * self.scale)) if self.palette.stroke else 0

    def box_pad(self, px: int) -> tuple[int, int]:
        """(horizontal, vertical) padding of the rounded box around the text block."""
        if self.palette.box is None:
            return 0, 0
        return round(px * 0.42), round(px * 0.24)

    def max_text_width(self, px: int) -> float:
        return self.width - 2 * round(SIDE_MARGIN * self.scale) - 2 * self.stroke - 2 * self.box_pad(px)[0]


@dataclass(frozen=True)
class _Layout:
    px: int
    lines: Lines

    @property
    def n_lines(self) -> int:
        return len(self.lines)


@lru_cache(maxsize=64)
def _font(px: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_PATH), px)


def _split_long(token: str, font: ImageFont.FreeTypeFont, max_w: float) -> list[str]:
    """Break a single over-wide token into pieces that each fit ``max_w``."""
    pieces: list[str] = []
    cur = ""
    for ch in token:
        if cur and font.getlength(cur + ch) > max_w:
            pieces.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        pieces.append(cur)
    return pieces


def _wrap(tokens: list[str], font: ImageFont.FreeTypeFont, max_w: float, hard_break: bool) -> Lines | None:
    """Greedy wrap by measured width; None when a token is too wide and breaking is not allowed."""
    space = font.getlength(" ")
    lines: list[tuple[Piece, ...]] = []
    cur: list[Piece] = []
    cur_w = 0.0
    for idx, tok in enumerate(tokens):
        if not tok:
            continue
        pieces = [tok]
        if font.getlength(tok) > max_w:
            if not hard_break:
                return None
            pieces = _split_long(tok, font, max_w)
        for piece in pieces:
            pw = font.getlength(piece)
            if cur and cur_w + space + pw <= max_w:
                cur.append((piece, idx))
                cur_w += space + pw
            else:
                if cur:
                    lines.append(tuple(cur))
                cur, cur_w = [(piece, idx)], pw
    if cur:
        lines.append(tuple(cur))
    return tuple(lines)


def _layout(tokens: list[str], geo: _Geometry) -> _Layout:
    last: _Layout | None = None
    for factor, max_lines, hard in _FIT_STEPS:
        px = max(8, round(geo.base_px * factor))
        lines = _wrap(tokens, _font(px), geo.max_text_width(px), hard)
        if lines is None:
            continue
        last = _Layout(px=px, lines=lines)
        if len(lines) <= max_lines:
            return last
    assert last is not None  # the final steps always allow hard breaks
    return last


def _block_height(layout: _Layout, geo: _Geometry) -> int:
    """Height of the text block incl. stroke and box padding."""
    ascent, descent = _font(layout.px).getmetrics()
    line_gap = round(layout.px * LINE_SPACING)
    pad_v = geo.box_pad(layout.px)[1]
    return ascent + descent + (layout.n_lines - 1) * line_gap + 2 * geo.stroke + 2 * pad_v


def _line_width(line: tuple[Piece, ...], font: ImageFont.FreeTypeFont) -> float:
    return sum(font.getlength(p) for p, _ in line) + font.getlength(" ") * (len(line) - 1)


def _draw_state(layout: _Layout, highlight_idx: int | None, geo: _Geometry, band_h: int) -> Image.Image:
    font = _font(layout.px)
    pal = geo.palette
    img = Image.new("RGBA", (geo.width, band_h), TRANSPARENT)
    draw = ImageDraw.Draw(img)
    ascent, descent = font.getmetrics()
    line_gap = round(layout.px * LINE_SPACING)
    block_h = _block_height(layout, geo)
    top = (band_h - block_h) // 2
    pad_h, pad_v = geo.box_pad(layout.px)
    widths = [_line_width(line, font) for line in layout.lines]
    if pal.box is not None:
        box_w = max(widths) + 2 * pad_h
        x0 = (geo.width - box_w) / 2
        draw.rounded_rectangle((x0, top, x0 + box_w, top + block_h), radius=round(layout.px * 0.28), fill=pal.box)
    space = font.getlength(" ")
    for li, (line, lw) in enumerate(zip(layout.lines, widths)):
        x = (geo.width - lw) / 2
        baseline = top + geo.stroke + pad_v + ascent + li * line_gap
        for piece, idx in line:
            fill = pal.highlight if idx == highlight_idx else pal.text
            draw.text((x, baseline), piece, font=font, fill=fill, anchor="ls",
                      stroke_width=geo.stroke, stroke_fill=pal.stroke)
            x += font.getlength(piece) + space
    return img


def _display(texts: list[str], style: CaptionStyle) -> list[str]:
    raw = [" ".join(t.split()) for t in texts]
    return [t.upper() for t in raw] if style.uppercase else raw


def _tokens(cue: Cue, style: CaptionStyle) -> list[str]:
    """Display tokens; index k matches ``cue.words[k]`` when the cue has word timings."""
    return _display([w.text for w in cue.words] if cue.words else cue.text.split(), style)


def _geometry(style: CaptionStyle, width: int, height: int) -> _Geometry:
    scale = width / BASE_WIDTH
    return _Geometry(width=width, height=height, scale=scale, vscale=height / BASE_HEIGHT,
                     palette=PALETTES[style.color], base_px=max(8, round(SIZE_PX[style.size] * scale)))


def lines_needed(texts: list[str], style: CaptionStyle, *, width: int = BASE_WIDTH) -> int:
    """Lines ``texts`` occupy at the style's full font size (greedy wrap by measured width).

    Cue grouping uses this so a normal cue fits its line budget without shrinking the font.
    """
    geo = _geometry(style, width, round(BASE_HEIGHT * width / BASE_WIDTH))
    lines = _wrap(_display(texts, style), _font(geo.base_px), geo.max_text_width(geo.base_px), hard_break=True)
    return len(lines or ())


def _band_top(band_h: int, geo: _Geometry, position: str) -> int:
    centre = geo.height * POSITION_CENTER[position]
    lo = round(TOP_SAFE * geo.vscale)
    hi = geo.height - round(BOTTOM_SAFE * geo.vscale) - band_h
    top = round(centre - band_h / 2)
    return max(lo, min(hi, top)) if hi >= lo else lo


class _StateWriter:
    """Writes each distinct visual state once and returns its file name."""

    def __init__(self, out_dir: Path, geo: _Geometry, band_h: int) -> None:
        self.out_dir, self.geo, self.band_h = out_dir, geo, band_h
        self.names: dict[tuple[object, ...], str] = {}

    def blank(self) -> str:
        key: tuple[object, ...] = ("blank",)
        if key not in self.names:
            Image.new("RGBA", (self.geo.width, self.band_h), TRANSPARENT).save(self.out_dir / BLANK_NAME)
            self.names[key] = BLANK_NAME
        return self.names[key]

    def state(self, layout: _Layout, highlight_idx: int | None) -> str:
        key = (layout.px, layout.lines, highlight_idx)
        if key not in self.names:
            name = f"cap_{len(self.names):05d}.png"
            _draw_state(layout, highlight_idx, self.geo, self.band_h).save(self.out_dir / name, compress_level=3)
            self.names[key] = name
        return self.names[key]


def _us(t: float) -> int:
    return int(round(t * 1_000_000))


def _cue_segments(cue: Cue, layout: _Layout, start: float, end: float, highlight: bool,
                  writer: _StateWriter) -> list[tuple[int, int, str]]:
    """(start_us, end_us, file) states covering [start, end) for one cue."""
    a, b = _us(start), _us(end)
    if not highlight:
        return [(a, b, writer.state(layout, None))]
    bounds = [a]
    for w in cue.words[1:]:                        # word k is lit from its start until word k+1 starts
        bounds.append(min(b, max(bounds[-1], _us(w.rel_start))))
    bounds.append(b)
    segs: list[tuple[int, int, str]] = []
    for k in range(len(cue.words)):
        s, e = bounds[k], bounds[k + 1]
        if e > s:
            segs.append((s, e, writer.state(layout, k)))
    return segs


def _timeline(cues: list[tuple[Cue, _Layout]], style: CaptionStyle, duration: float,
              writer: _StateWriter) -> list[tuple[int, int, str]]:
    total = _us(duration)
    segs: list[tuple[int, int, str]] = []
    t = 0
    for cue, layout in cues:
        start = max(_us(cue.rel_start), t)
        end = min(_us(cue.rel_end), total)
        if end - start < 1000:            # under 1 ms (or overlapped away): nothing to show
            continue
        if start > t:
            segs.append((t, start, writer.blank()))
        highlight = cue.highlight and style.highlight and bool(cue.words)
        segs.extend(_cue_segments(cue, layout, start / 1e6, end / 1e6, highlight, writer))
        t = end
    if t < total:
        segs.append((t, total, writer.blank()))
    merged: list[tuple[int, int, str]] = []
    for s, e, name in segs:
        if merged and merged[-1][2] == name and merged[-1][1] == s:
            merged[-1] = (merged[-1][0], e, name)
        else:
            merged.append((s, e, name))
    return merged


def _write_concat(path: Path, segs: list[tuple[int, int, str]], blank: str) -> None:
    lines = ["ffconcat version 1.0"]
    for s, e, name in segs:
        lines.append(f"file '{name}'")
        lines.append(f"option {_FRAMERATE_OPTION}")
        lines.append(f"duration {(e - s) / 1e6:.6f}")
    # Concat demuxer quirk: the last entry's duration is only honoured when a file follows.
    # The trailing entry starts at t = duration (outside the clip) and is transparent.
    lines.append(f"file '{blank}'")
    lines.append(f"option {_FRAMERATE_OPTION}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_caption_track(cues: list[Cue], style: CaptionStyle, out_dir: Path, *, duration: float,
                         width: int = 1080, height: int = 1920) -> CaptionTrack | None:
    """Render caption PNG states + ffconcat timeline; None when there is nothing to show."""
    if duration <= 0:
        return None
    geo = _geometry(style, width, height)
    laid: list[tuple[Cue, _Layout]] = []
    for cue in sorted(cues, key=lambda c: (c.rel_start, c.rel_end)):
        tokens = _tokens(cue, style)
        if any(tokens) and cue.rel_end > cue.rel_start and cue.rel_start < duration:
            laid.append((cue, _layout(tokens, geo)))
    if not laid:
        return None
    band_h = max(_block_height(layout, geo) for _, layout in laid) + 2 * max(2, round(4 * geo.scale))
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = _StateWriter(out_dir, geo, band_h)
    segs = _timeline(laid, style, duration, writer)
    if all(name == BLANK_NAME for _, _, name in segs):
        return None
    concat = out_dir / CONCAT_NAME
    _write_concat(concat, segs, writer.blank())
    return CaptionTrack(concat_path=concat, x=0, y=_band_top(band_h, geo, style.position), w=width, h=band_h,
                        frames=len(writer.names))
