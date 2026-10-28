import assert from 'node:assert/strict'
import { test } from 'node:test'
import {
  MAX_CLIP_SECONDS,
  MIN_CLIP_SECONDS,
  ceilTenth,
  clipTrimLimits,
  endAtWord,
  endProblem,
  floorTenth,
  percentOf,
  startAtWord,
  startProblem,
  withEnd,
  withStart,
  wordEndPoint,
  wordInRange,
  wordStartPoint,
  type TrimLimits,
  type WordTiming,
} from '../src/lib/trim.ts'

const limits: TrimLimits = { sourceDuration: 600, minLength: 1, maxLength: 180 }

test('withStart rounds to tenths and keeps at least minLength before the end', () => {
  assert.deepEqual(withStart({ src_start: 10, src_end: 40 }, 12.34, limits), { src_start: 12.3, src_end: 40 })
  assert.deepEqual(withStart({ src_start: 10, src_end: 40 }, 45, limits), { src_start: 39, src_end: 40 })
  assert.deepEqual(withStart({ src_start: 10, src_end: 40 }, -5, limits), { src_start: 0, src_end: 40 })
})

test('withStart never makes the clip longer than maxLength', () => {
  assert.deepEqual(withStart({ src_start: 300, src_end: 400 }, 100, limits), { src_start: 220, src_end: 400 })
})

test('withEnd clamps to the source duration and maxLength', () => {
  assert.deepEqual(withEnd({ src_start: 500, src_end: 560 }, 700, limits), { src_start: 500, src_end: 600 })
  assert.deepEqual(withEnd({ src_start: 10, src_end: 40 }, 500, limits), { src_start: 10, src_end: 190 })
  assert.deepEqual(withEnd({ src_start: 10, src_end: 40 }, 5, limits), { src_start: 10, src_end: 11 })
})

test('nudging by 0.1 s does not accumulate float error', () => {
  let r = { src_start: 10, src_end: 40 }
  for (let i = 0; i < 7; i++) r = withEnd(r, r.src_end + 0.1, limits)
  assert.equal(r.src_end, 40.7)
})

test('unreadable input leaves the range unchanged', () => {
  const r = { src_start: 10, src_end: 40 }
  assert.equal(withStart(r, Number.NaN, limits), r)
  assert.equal(withEnd(r, Number.POSITIVE_INFINITY, limits), r)
})

test('picking words rounds outwards so the word stays whole', () => {
  const word = { id: 'w1', start: 12.37, end: 12.81 }
  assert.equal(floorTenth(12.37), 12.3)
  assert.equal(ceilTenth(12.81), 12.9)
  assert.equal(ceilTenth(12.8), 12.8)
  assert.deepEqual(startAtWord({ src_start: 5, src_end: 30 }, word, limits), { src_start: 12.3, src_end: 30 })
  assert.deepEqual(endAtWord({ src_start: 5, src_end: 30 }, word, limits), { src_start: 5, src_end: 12.9 })
})

test('start/end problems explain impossible picks', () => {
  const range = { src_start: 10, src_end: 20 }
  assert.match(startProblem(range, { id: 'a', start: 25, end: 25.5 }, limits) ?? '', /after the current end/)
  assert.equal(startProblem(range, { id: 'b', start: 12, end: 12.4 }, limits), null)
  assert.match(endProblem(range, { id: 'c', start: 5, end: 5.4 }, limits) ?? '', /before the current start/)
  assert.match(endProblem(range, { id: 'd', start: 300, end: 300.5 }, limits) ?? '', /longer than 180 seconds/)
  assert.equal(endProblem(range, { id: 'e', start: 30, end: 30.5 }, limits), null)
})

test('wordInRange uses the word midpoint and a half-open range', () => {
  const range = { src_start: 10, src_end: 20 }
  assert.equal(wordInRange({ id: 'a', start: 9.8, end: 10.4 }, range), true)
  assert.equal(wordInRange({ id: 'b', start: 9.5, end: 10.3 }, range), false)
  assert.equal(wordInRange({ id: 'c', start: 19.8, end: 20.2 }, range), false)
})

test('percentOf clamps to 0..100', () => {
  assert.equal(percentOf(15, 10, 20), 50)
  assert.equal(percentOf(5, 10, 20), 0)
  assert.equal(percentOf(25, 10, 20), 100)
  assert.equal(percentOf(5, 10, 10), 0)
})

// ---------------------------------------------------------------- regressions

// Finding 41: "Start here"/"End here" rounded outwards to a tenth without looking at the neighbouring
// word, so a short contiguous neighbour (midpoint inside the rounding margin) joined the clip.
test('"Start here" does not pull a contiguous previous word into the clip', () => {
  const a = { id: 'a', start: 10.06, end: 10.16 }
  const point = { id: 'point', start: 10.16, end: 10.5 }
  const range = { src_start: 5, src_end: 30 }
  assert.equal(startProblem(range, point, limits, a), null)
  const r = startAtWord(range, point, limits, a)
  assert.equal(wordInRange(a, r), false)
  assert.equal(wordInRange(point, r), true)
  // inside the (empty) silence gap: after the previous word's audio, never after the picked word starts
  assert.ok(r.src_start >= a.end - 1e-9 && r.src_start <= point.start + 1e-9, String(r.src_start))
  assert.equal(r.src_end, 30)
})

test('"End here" does not pull a contiguous next word into the clip', () => {
  const the = { id: 'the', start: 20.0, end: 20.33 }
  const nx = { id: 'nx', start: 20.33, end: 20.41 }
  const range = { src_start: 5, src_end: 30 }
  assert.equal(endProblem(range, the, limits, nx), null)
  const r = endAtWord(range, the, limits, nx)
  assert.equal(wordInRange(nx, r), false)
  assert.equal(wordInRange(the, r), true)
  assert.ok(r.src_end >= the.end - 1e-9 && r.src_end <= nx.start + 1e-9, String(r.src_end))
})

test('word picks snap into a short silence gap instead of the neighbouring word', () => {
  // gap 10.13..10.16 is shorter than the tenth grid: the tenth (10.1) would cut into "a"
  const a = { id: 'a', start: 10.0, end: 10.13 }
  const point = { id: 'point', start: 10.16, end: 10.5 }
  const r = startAtWord({ src_start: 5, src_end: 30 }, point, limits, a)
  assert.ok(r.src_start >= a.end && r.src_start <= point.start, String(r.src_start))
  assert.equal(r.src_start, 10.13)
  // the end side, mirrored: tenth 20.4 would reach into "nx" (starts 20.37)
  const the = { id: 'the', start: 20.0, end: 20.33 }
  const nx = { id: 'nx', start: 20.37, end: 20.6 }
  const e = endAtWord({ src_start: 5, src_end: 30 }, the, limits, nx)
  assert.equal(e.src_end, 20.37)
  assert.equal(wordInRange(nx, e), false)
})

test('word picks keep the outward tenth when it stays inside the silence', () => {
  const prev = { id: 'p', start: 11.5, end: 12.0 }
  const word = { id: 'w1', start: 12.37, end: 12.81 }
  const next = { id: 'n', start: 13.2, end: 13.5 }
  assert.deepEqual(startAtWord({ src_start: 5, src_end: 30 }, word, limits, prev), { src_start: 12.3, src_end: 30 })
  assert.deepEqual(endAtWord({ src_start: 5, src_end: 30 }, word, limits, next), { src_start: 5, src_end: 12.9 })
})

test('word picks with overlapping neighbour timings still exclude the neighbour', () => {
  const prev = { id: 'p', start: 10.0, end: 10.4 }
  const word = { id: 'w', start: 10.15, end: 10.6 } // overlaps prev; prev midpoint 10.2 is after the word start
  const r = startAtWord({ src_start: 5, src_end: 30 }, word, limits, prev)
  assert.equal(wordInRange(prev, r), false)
  assert.equal(wordInRange(word, r), true)
  const next = { id: 'n', start: 10.3, end: 10.7 } // overlaps the word; next midpoint 10.5 is before the word end
  const e = endAtWord({ src_start: 5, src_end: 30 }, word, limits, next)
  assert.equal(wordInRange(next, e), false)
  assert.equal(wordInRange(word, e), true)
})

test('the start/end problem checks use the same snapped point as the action', () => {
  const real = clipTrimLimits(600)
  const a = { id: 'a', start: 10.06, end: 10.16 }
  const point = { id: 'point', start: 10.16, end: 10.5 }
  // The tenth (10.1) would leave 3.03 s, but the snapped start (10.16) leaves only 2.97 s.
  assert.equal(startProblem({ src_start: 5, src_end: 13.13 }, point, real), null)
  assert.match(startProblem({ src_start: 5, src_end: 13.13 }, point, real, a) ?? '', /shorter than 3 seconds/)
  assert.equal(startProblem({ src_start: 5, src_end: 13.2 }, point, real, a), null)
  // Mirrored: the tenth (20.4) would leave 3.04 s, the snapped end (20.33) only 2.97 s.
  const the = { id: 'the', start: 20.0, end: 20.33 }
  const nx = { id: 'nx', start: 20.33, end: 20.41 }
  assert.equal(endProblem({ src_start: 17.36, src_end: 30 }, the, real), null)
  assert.match(endProblem({ src_start: 17.36, src_end: 30 }, the, real, nx) ?? '', /shorter than 3 seconds/)
})

// Finding 42: clamping to maxLength and then rounding to the nearest tenth could land outside the
// bound (180.03 s), which the API rejects. Rounding now goes inward when it would leave the bounds.
test('a max-length trim never exceeds the limit after tenth rounding', () => {
  const real = clipTrimLimits(3600)
  const s = withStart({ src_start: 830, src_end: 1000.33 }, 800, real)
  assert.equal(s.src_start, 820.4)
  assert.ok(s.src_end - s.src_start <= 180 + 1e-9, String(s.src_end - s.src_start))
  const e = withEnd({ src_start: 830.07, src_end: 1000 }, 1100, real)
  assert.equal(e.src_end, 1010)
  assert.ok(e.src_end - e.src_start <= 180 + 1e-9, String(e.src_end - e.src_start))
})

test('a min-length trim never drops below the limit after tenth rounding', () => {
  const real = clipTrimLimits(3600)
  // hi = 10.37 - 3 = 7.37 would round up to 7.4 (2.97 s)
  const s = withStart({ src_start: 1, src_end: 10.37 }, 9, real)
  assert.equal(s.src_start, 7.3)
  assert.ok(s.src_end - s.src_start >= 3 - 1e-9)
  // lo = 20.04 + 3 = 23.04 would round down to 23.0 (2.96 s)
  const e = withEnd({ src_start: 20.04, src_end: 40 }, 21, real)
  assert.equal(e.src_end, 23.1)
  assert.ok(e.src_end - e.src_start >= 3 - 1e-9)
})

test('ending at the very end of the source is still possible', () => {
  const real = clipTrimLimits(48.07)
  assert.equal(withEnd({ src_start: 40, src_end: 45 }, 999, real).src_end, 48.07)
})

// Findings 44/45: the editor allowed 1-3 s clips that the API (MIN_CLIP_SECONDS = 3) rejects.
test('the editor uses the API clip length limits (3 s to 180 s)', () => {
  assert.equal(MIN_CLIP_SECONDS, 3)
  assert.equal(MAX_CLIP_SECONDS, 180)
  const real = clipTrimLimits(600)
  assert.deepEqual(real, { sourceDuration: 600, minLength: 3, maxLength: 180 })
  assert.deepEqual(withEnd({ src_start: 20, src_end: 40 }, 21.5, real), { src_start: 20, src_end: 23 })
  assert.deepEqual(withStart({ src_start: 10, src_end: 30 }, 29, real), { src_start: 27, src_end: 30 })
  assert.match(endProblem({ src_start: 20, src_end: 40 }, { id: 'w', start: 21.6, end: 22 }, real) ?? '', /shorter than 3 seconds/)
})

// Re-review finding 11: the overlap fallback also fired next to a zero-length word that touches the
// picked word (Whisper output has them after the monotonic clean-up) and then cut a quarter of the
// picked word away. Membership is half-open [start, end), like the backend; a zero-length neighbour
// is kept out with a 1 ms nudge past its midpoint (the backend's BOUNDARY_EDGE).
test('"End here" next to a zero-length next word keeps the picked word whole', () => {
  const a = { id: 'a', start: 20.0, end: 20.53 }
  const b = { id: 'b', start: 20.53, end: 20.53 }
  const r = endAtWord({ src_start: 5, src_end: 30 }, a, limits, b)
  assert.equal(r.src_end, 20.53) // was 20.3975, 0.13 s into "a"
  assert.equal(wordInRange(a, r), true)
  assert.equal(wordInRange(b, r), false)
})

test('"Start here" after a zero-length previous word cuts at most 1 ms into the picked word', () => {
  const a = { id: 'a', start: 10.16, end: 10.16 }
  const b = { id: 'b', start: 10.16, end: 10.56 }
  const r = startAtWord({ src_start: 5, src_end: 30 }, b, limits, a)
  assert.equal(r.src_start, 10.161) // was 10.26, 0.1 s into "b"
  assert.equal(wordInRange(a, r), false)
  assert.equal(wordInRange(b, r), true)
  // with a 20 ms gap the start stays in the gap and "b" is not clipped at all (was 10.27)
  const c = { id: 'c', start: 10.18, end: 10.58 }
  assert.equal(wordStartPoint(c, a), 10.161)
})

test('a zero-length picked word stays in the clip', () => {
  const range = { src_start: 5, src_end: 30 }
  // touching the next word: the end used to land on its own midpoint and drop it
  const w = { id: 'w', start: 20.93, end: 20.93 }
  const n = { id: 'n', start: 20.93, end: 21.4 }
  const e = endAtWord(range, w, limits, n)
  assert.equal(wordInRange(w, e), true)
  assert.equal(wordInRange(n, e), false)
  // both zero-length at the same time: no cut can separate them; the picked word wins
  const z = { id: 'z', start: 20.93, end: 20.93 }
  assert.equal(wordInRange(w, endAtWord(range, w, limits, z)), true)
  // on a tenth: ceilTenth(10.9) is 10.9, which is its midpoint (excluded by the half-open end)
  const t = { id: 't', start: 10.9, end: 10.9 }
  assert.equal(wordEndPoint(t), 11)
  assert.equal(wordEndPoint(t, { id: 'u', start: 11.3, end: 11.6 }), 11)
  // float noise just below a tenth: floorTenth rounds up to 612.4, past the word's midpoint
  const f = { id: 'f', start: 612.3999999999, end: 612.3999999999 }
  assert.equal(wordInRange(f, startAtWord({ src_start: 600, src_end: 630 }, f, limits)), true)
})

test('"Start here" keeps a zero-length previous word sitting on the tenth out', () => {
  const p = { id: 'p', start: 10.2, end: 10.2 }
  const w = { id: 'w', start: 10.2, end: 10.77 }
  const r = startAtWord({ src_start: 5, src_end: 30 }, w, limits, p)
  assert.equal(wordInRange(p, r), false) // the tenth 10.2 is its midpoint, which [start, end) includes
  assert.equal(wordInRange(w, r), true)
  assert.equal(r.src_start, 10.201)
})

test('word picks on monotonic timings with zero-length words: picked word in, neighbour out, <= 1 ms cut', () => {
  // Deterministic LCG; 10 ms grid, ~30% zero-length words, ~50% touching (as after _enforce_monotonic).
  let seed = 12345
  const rand = () => ((seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648)
  const step = () => (rand() < 0.5 ? 0 : Math.floor(rand() * 30) / 100)
  const len = () => (rand() < 0.3 ? 0 : Math.floor(1 + rand() * 60) / 100)
  const mid = (w: WordTiming) => (w.start + w.end) / 2
  const tol = 1e-6
  for (let i = 0; i < 20000; i++) {
    let t = Math.round((10 + rand() * 50) * 100) / 100
    const ws: WordTiming[] = []
    for (const id of ['p', 'w', 'n']) {
      const start = Math.round((t + (ws.length ? step() : 0)) * 100) / 100
      const end = Math.round((start + len()) * 100) / 100
      ws.push({ id, start, end })
      t = end
    }
    const [p, w, n] = ws
    const label = JSON.stringify(ws)
    const s = wordStartPoint(w, p)
    assert.ok(mid(w) >= s, `start drops the picked word: ${s} ${label}`)
    if (mid(p) < mid(w)) assert.ok(mid(p) < s, `start pulls in the previous word: ${s} ${label}`)
    assert.ok(s <= w.start + 1e-3 + tol, `start cuts into the picked word: ${s} ${label}`)
    const e = wordEndPoint(w, n)
    assert.ok(mid(w) < e, `end drops the picked word: ${e} ${label}`)
    if (mid(n) > mid(w)) assert.ok(mid(n) >= e, `end pulls in the next word: ${e} ${label}`)
    assert.ok(e >= w.end - tol, `end cuts into the picked word: ${e} ${label}`)
  }
})
