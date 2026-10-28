/**
 * Pure trim helpers for the clip editor. All values are source seconds (src_*).
 * Typed values and nudges are kept to tenths of a second (rounded inwards when the
 * nearest tenth would break a length limit). Picking a word rounds outwards to a tenth
 * so the whole word stays inside the clip, unless that tenth would reach into the
 * neighbouring word; then the cut goes into the silence between the two words.
 */

export interface TrimRange {
  src_start: number
  src_end: number
}

export interface TrimLimits {
  /** Source media duration (upper bound for src_end). */
  sourceDuration: number
  /** Shortest allowed clip in seconds. */
  minLength: number
  /** Longest allowed clip in seconds. */
  maxLength: number
}

export interface WordTiming {
  id: string
  start: number
  end: number
}

/** The API's clip length bounds (backend/clipper/models.py MIN_CLIP_SECONDS / MAX_CLIP_SECONDS). */
export const MIN_CLIP_SECONDS = 3
export const MAX_CLIP_SECONDS = 180

/** Trim limits for the clip editor: the same length bounds the API enforces on save. */
export function clipTrimLimits(sourceDuration: number): TrimLimits {
  return { sourceDuration, minLength: MIN_CLIP_SECONDS, maxLength: MAX_CLIP_SECONDS }
}

const EPS = 1e-6
/** Step past a zero-length neighbour's midpoint (the backend's BOUNDARY_EDGE, 1 ms). */
const EDGE = 1e-3

export function roundTenth(sec: number): number {
  return Math.round(sec * 10) / 10
}

export function floorTenth(sec: number): number {
  return Math.floor(sec * 10 + EPS) / 10
}

export function ceilTenth(sec: number): number {
  return Math.ceil(sec * 10 - EPS) / 10
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v))
}

function floorHundredth(sec: number): number {
  return Math.floor(sec * 100 + EPS) / 100
}

function ceilHundredth(sec: number): number {
  return Math.ceil(sec * 100 - EPS) / 100
}

/** Drop float noise (e.g. 10.160000000001) without moving the value in any way that matters. */
function clean(sec: number): number {
  return Math.round(sec * 1e6) / 1e6
}

function midpoint(word: WordTiming): number {
  return (word.start + word.end) / 2
}

/**
 * Clamp `v` to [lo, hi]; with `snap`, round to the nearest tenth but round inwards
 * when the nearest tenth would fall outside the bounds (the API checks lengths exactly).
 */
function within(v: number, lo: number, hi: number, snap: boolean): number {
  const top = Math.max(lo, hi)
  const c = clamp(v, lo, top)
  if (!snap) return c
  const r = roundTenth(c)
  if (r >= lo - EPS && r <= top + EPS) return r
  const inward = r < lo ? ceilTenth(lo) : floorTenth(top)
  return inward >= lo - EPS && inward <= top + EPS ? inward : c
}

function placeStart(range: TrimRange, srcStart: number, limits: TrimLimits, snap: boolean): TrimRange {
  if (!Number.isFinite(srcStart)) return range
  const lo = Math.max(0, range.src_end - limits.maxLength)
  const hi = range.src_end - limits.minLength
  return { src_start: Math.max(0, within(srcStart, lo, hi, snap)), src_end: range.src_end }
}

function placeEnd(range: TrimRange, srcEnd: number, limits: TrimLimits, snap: boolean): TrimRange {
  if (!Number.isFinite(srcEnd)) return range
  const lo = range.src_start + limits.minLength
  const hi = range.src_start + limits.maxLength
  const next = within(srcEnd, Math.min(lo, hi), hi, snap)
  // Capped after snapping: the source end is a valid end even when it is not on the tenth grid.
  return { src_start: range.src_start, src_end: Math.min(limits.sourceDuration, next) }
}

/** Move the start (to a tenth), keeping the clip within [minLength, maxLength] and the source bounds. */
export function withStart(range: TrimRange, srcStart: number, limits: TrimLimits): TrimRange {
  return placeStart(range, srcStart, limits, true)
}

/** Move the end (to a tenth), keeping the clip within [minLength, maxLength] and the source bounds. */
export function withEnd(range: TrimRange, srcEnd: number, limits: TrimLimits): TrimRange {
  return placeEnd(range, srcEnd, limits, true)
}

/**
 * Where "Start here" puts the start: the word's start rounded down to a tenth, so the word
 * is not clipped. When that tenth would reach back into the previous word, the start goes
 * into the silence between the two words instead, so the previous word stays out of the
 * clip: out of its captions (midpoint rule, half-open [start, end)) and its audio. Only a
 * zero-length previous word touching the picked word (its midpoint is on the boundary) moves
 * the start 1 ms past that midpoint, as the backend does. The picked word always stays in.
 */
export function wordStartPoint(word: WordTiming, prev?: WordTiming | null): number {
  const tenth = floorTenth(word.start)
  let s = tenth
  // A zero-length previous word on the tenth has its midpoint on it: [start, end) would include it.
  if (prev && (tenth < prev.end - EPS || tenth <= midpoint(prev))) {
    s = clean(Math.min(word.start, ceilHundredth(prev.end)))
    // Zero-length (or overlapping) previous word: step 1 ms past its midpoint (capped below at the
    // picked word's midpoint, so identical midpoints keep the picked word).
    if (s <= midpoint(prev)) s = clean(midpoint(prev) + EDGE)
  }
  return Math.min(s, midpoint(word))
}

/**
 * Where "End here" puts the end: the mirror of {@link wordStartPoint} with the next word. The end
 * is exclusive, so a next word whose midpoint is on the end is already out, and a zero-length
 * picked word needs the end at least 1 ms past its midpoint to stay in.
 */
export function wordEndPoint(word: WordTiming, next?: WordTiming | null): number {
  const keep = clean(midpoint(word) + EDGE)
  let e = ceilTenth(word.end)
  if (e <= midpoint(word)) e = ceilTenth(keep) // zero-length word on a tenth
  if (next && (e > next.start + EPS || e > midpoint(next))) {
    e = clean(Math.max(word.end, floorHundredth(next.start), keep))
    // Overlapping (or zero-length, same-time) next word: end on its midpoint, which keeps it out,
    // unless that would drop the picked word as well (identical midpoints cannot be separated).
    if (e > midpoint(next)) e = midpoint(next) > midpoint(word) ? midpoint(next) : keep
  }
  return e
}

/** The words just before and after `word` in transcript order (for "Start here" / "End here"). */
export function neighbours<T extends WordTiming>(words: T[], word: WordTiming): { prev: T | null; next: T | null } {
  const i = words.findIndex((w) => w.id === word.id)
  if (i < 0) return { prev: null, next: null }
  return { prev: words[i - 1] ?? null, next: words[i + 1] ?? null }
}

/** Start the clip at a word (see {@link wordStartPoint}). */
export function startAtWord(
  range: TrimRange,
  word: WordTiming,
  limits: TrimLimits,
  prev?: WordTiming | null,
): TrimRange {
  return placeStart(range, wordStartPoint(word, prev), limits, false)
}

/** End the clip at a word (see {@link wordEndPoint}). */
export function endAtWord(range: TrimRange, word: WordTiming, limits: TrimLimits, next?: WordTiming | null): TrimRange {
  return placeEnd(range, wordEndPoint(word, next), limits, false)
}

/** Why "Start here" is not possible for this word, or null when it is. */
export function startProblem(
  range: TrimRange,
  word: WordTiming,
  limits: TrimLimits,
  prev?: WordTiming | null,
): string | null {
  const start = wordStartPoint(word, prev)
  if (start >= range.src_end - EPS) return 'This word is after the current end. Move the end first.'
  if (range.src_end - start < limits.minLength - EPS) {
    return `The clip would be shorter than ${limits.minLength} seconds. Move the end first.`
  }
  if (range.src_end - start > limits.maxLength + EPS) {
    return `The clip would be longer than ${limits.maxLength} seconds.`
  }
  return null
}

/** Why "End here" is not possible for this word, or null when it is. */
export function endProblem(
  range: TrimRange,
  word: WordTiming,
  limits: TrimLimits,
  next?: WordTiming | null,
): string | null {
  const end = wordEndPoint(word, next)
  if (end <= range.src_start + EPS) return 'This word is before the current start. Move the start first.'
  if (end - range.src_start < limits.minLength - EPS) {
    return `The clip would be shorter than ${limits.minLength} seconds. Move the start first.`
  }
  if (end - range.src_start > limits.maxLength + EPS) {
    return `The clip would be longer than ${limits.maxLength} seconds.`
  }
  return null
}

/** A word belongs to the clip when its midpoint is inside [src_start, src_end) (same rule as the backend). */
export function wordInRange(word: WordTiming, range: TrimRange): boolean {
  const mid = (word.start + word.end) / 2
  return mid >= range.src_start && mid < range.src_end
}

export function sameTime(a: number, b: number): boolean {
  return Math.abs(a - b) < 1e-3
}

/** Position of `t` inside [lo, hi] as a 0..100 percentage (clamped), for drawing the trim strip. */
export function percentOf(t: number, lo: number, hi: number): number {
  if (hi <= lo) return 0
  return clamp(((t - lo) / (hi - lo)) * 100, 0, 100)
}
