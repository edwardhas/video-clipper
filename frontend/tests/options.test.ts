import assert from 'node:assert/strict'
import { test } from 'node:test'
import type { LimitsInfo } from '../src/api/types.ts'
import { DURATION_PRESETS, boundsFrom, presetFor, validateOptions } from '../src/lib/options.ts'

const limits: LimitsInfo = {
  max_source_minutes: 180,
  max_upload_mb: 4096,
  max_clips: 8,
  min_seconds_floor: 5,
  max_seconds_cap: 120,
}
const ok = { clip_count: 5, min_seconds: 20, max_seconds: 60, captions_enabled: true }

test('presets match the brief', () => {
  assert.deepEqual(
    DURATION_PRESETS.map((p) => [p.label, p.min, p.max]),
    [
      ['Short', 15, 30],
      ['Default', 20, 60],
      ['Long', 45, 90],
    ],
  )
  assert.equal(presetFor(45, 90)?.id, 'long')
  assert.equal(presetFor(30, 70), null)
})

test('server limits tighten the schema bounds', () => {
  const b = boundsFrom(limits)
  assert.equal(b.clipMax, 8)
  assert.equal(b.maxCap, 120)
  assert.equal(boundsFrom(null).clipMax, 10)
})

test('validateOptions accepts valid values and explains invalid ones', () => {
  assert.equal(validateOptions(ok, limits), null)
  assert.match(validateOptions({ ...ok, clip_count: 9 }, limits) ?? '', /between 1 and 8 clips/)
  assert.match(validateOptions({ ...ok, clip_count: 2.5 }, limits) ?? '', /clips/)
  assert.match(validateOptions({ ...ok, min_seconds: 2 }, limits) ?? '', /minimum length/)
  assert.match(validateOptions({ ...ok, max_seconds: 150 }, limits) ?? '', /maximum length must be between 10 and 120/)
  assert.match(validateOptions({ ...ok, min_seconds: 40, max_seconds: 42 }, limits) ?? '', /at least 5 seconds longer/)
  assert.match(validateOptions({ ...ok, min_seconds: Number.NaN }, limits) ?? '', /Enter a minimum and maximum/)
})
