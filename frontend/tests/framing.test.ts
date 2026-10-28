import assert from 'node:assert/strict'
import { test } from 'node:test'
import { cropWidthFraction, cropWindow } from '../src/lib/framing.ts'

test('9:16 crop width for a 16:9 source', () => {
  const w = cropWidthFraction(1920, 1080)
  assert.ok(Math.abs(w - (1080 * 9) / 16 / 1920) < 1e-9)
})

test('vertical sources need no crop', () => {
  assert.equal(cropWidthFraction(1080, 1920), 1)
  assert.equal(cropWidthFraction(0, 0), 1)
})

test('crop window is clamped inside the frame', () => {
  const left = cropWindow(1920, 1080, 0)
  assert.equal(left.left, 0)
  const right = cropWindow(1920, 1080, 1)
  assert.ok(Math.abs(right.left + right.width - 1) < 1e-9)
  const mid = cropWindow(1920, 1080, 0.5)
  assert.ok(Math.abs(mid.left + mid.width / 2 - 0.5) < 1e-9)
})
