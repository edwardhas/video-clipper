import assert from 'node:assert/strict'
import { test } from 'node:test'
import {
  fmtBytes,
  fmtClock,
  fmtDuration,
  fmtMegabytes,
  fmtMinutes,
  fmtPrecise,
  fmtRange,
  fmtScore,
  mbToBytes,
  plural,
} from '../src/lib/format.ts'

test('fmtClock pads minutes and seconds and adds hours when needed', () => {
  assert.equal(fmtClock(0), '00:00')
  assert.equal(fmtClock(65.9), '01:05')
  assert.equal(fmtClock(3600 + 61), '1:01:01')
  assert.equal(fmtClock(-3), '00:00')
})

test('fmtPrecise shows tenths', () => {
  assert.equal(fmtPrecise(12.34), '00:12.3')
  assert.equal(fmtPrecise(59.96), '01:00.0')
  assert.equal(fmtPrecise(3725.25), '1:02:05.3')
})

test('fmtRange joins source times with an en dash', () => {
  assert.equal(fmtRange(61, 125), '01:01–02:05')
})

test('fmtDuration reads naturally', () => {
  assert.equal(fmtDuration(null), '—')
  assert.equal(fmtDuration(4.25), '4.3 s')
  assert.equal(fmtDuration(5), '5 s')
  assert.equal(fmtDuration(42.4), '42 s')
  assert.equal(fmtDuration(185), '3 min 05 s')
  assert.equal(fmtDuration(120), '2 min')
  assert.equal(fmtDuration(3840), '1 h 04 min')
})

test('byte sizes use binary units like the server limits', () => {
  assert.equal(fmtBytes(512), '512 bytes')
  assert.equal(fmtBytes(1536), '1.5 KB')
  assert.equal(fmtBytes(mbToBytes(4096)), '4.0 GB')
  assert.equal(fmtMegabytes(4096), '4 GB')
  assert.equal(fmtMegabytes(1536), '1.5 GB')
  assert.equal(fmtMegabytes(500), '500 MB')
  assert.equal(mbToBytes(1), 1048576)
})

test('fmtMinutes prefers whole hours', () => {
  assert.equal(fmtMinutes(180), '3 hours')
  assert.equal(fmtMinutes(60), '1 hour')
  assert.equal(fmtMinutes(90), '90 minutes')
})

test('fmtScore clamps and rounds to /100', () => {
  assert.equal(fmtScore(77.6), '78/100')
  assert.equal(fmtScore(130), '100/100')
  assert.equal(fmtScore(-2), '0/100')
})

test('plural', () => {
  assert.equal(plural(1, 'clip'), '1 clip')
  assert.equal(plural(3, 'clip'), '3 clips')
})
