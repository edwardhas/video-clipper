import assert from 'node:assert/strict'
import { test } from 'node:test'
import { diffDraft, draftFromEdit, isDirty, rangeDirty } from '../src/lib/draft.ts'
import { edit } from './fixtures.ts'

test('an untouched draft has nothing to save', () => {
  const saved = edit()
  const draft = draftFromEdit(saved)
  assert.deepEqual(diffDraft(draft, saved), {})
  assert.equal(isDirty(draft, saved), false)
})

test('draft is a copy, not a reference to the saved edit', () => {
  const saved = edit()
  const draft = draftFromEdit(saved)
  draft.style.size = 'L'
  assert.equal(saved.style.size, 'M')
})

test('range changes send both ends together', () => {
  const saved = edit()
  const draft = { ...draftFromEdit(saved), src_end: 41.5 }
  assert.deepEqual(diffDraft(draft, saved), { src_start: 10, src_end: 41.5 })
  assert.equal(rangeDirty(draft, saved), true)
})

test('only changed groups are included', () => {
  const saved = edit()
  const draft = draftFromEdit(saved)
  draft.style = { ...draft.style, color: 'yellow' }
  draft.framing = { mode: 'crop', crop_x: 0.3 }
  draft.captions_enabled = false
  const body = diffDraft(draft, saved)
  assert.deepEqual(Object.keys(body).sort(), ['captions_enabled', 'framing', 'style'])
  assert.equal(body.style?.color, 'yellow')
  assert.deepEqual(body.framing, { mode: 'crop', crop_x: 0.3 })
  assert.equal(rangeDirty(draft, saved), false)
})

test('framing crop_x null vs number counts as a change', () => {
  const saved = edit({ framing: { mode: 'crop', crop_x: null } })
  const draft = draftFromEdit(saved)
  draft.framing = { mode: 'crop', crop_x: 0.5 }
  assert.equal(isDirty(draft, saved), true)
  draft.framing = { mode: 'crop', crop_x: null }
  assert.equal(isDirty(draft, saved), false)
})
