import assert from 'node:assert/strict'
import { test } from 'node:test'
import type { Cue } from '../src/api/types.ts'
import { editedCueCount, isHiddenCue, restoreTextFor } from '../src/lib/captions.ts'

function cue(overrides: Partial<Cue> = {}): Cue {
  return {
    id: 'w1',
    first_word_id: 'w1',
    last_word_id: 'w2',
    text: 'Hello there.',
    words: [],
    rel_start: 0,
    rel_end: 1,
    highlight: true,
    edited: false,
    ...overrides,
  }
}

const words = [
  { id: 'w1', text: 'Hello', start: 10, end: 10.3 },
  { id: 'w2', text: 'there.', start: 10.4, end: 10.8 },
  { id: 'w3', text: 'Their', start: 11, end: 11.3 },
  { id: 'w4', text: 'going.', start: 11.4, end: 11.8 },
]

// Finding 17: the API now lists a hidden caption (text "" and edited=true, include_hidden) so it
// can be restored on its own; the panel shows it as a "Hidden caption" row and counts it as edited.
test('a hidden caption is recognised and counted as edited', () => {
  const hidden = cue({ text: '', edited: true, highlight: false })
  assert.equal(isHiddenCue(hidden), true)
  assert.equal(isHiddenCue(cue({ text: '   ' })), true)
  assert.equal(isHiddenCue(cue()), false)
  const corrected = cue({ id: 'w3', first_word_id: 'w3', last_word_id: 'w4', text: "They're going.", edited: true })
  assert.equal(editedCueCount([hidden, corrected, cue({ id: 'w9' })]), 2)
  // even if a server ever omitted the flag, an empty caption is still an edit
  assert.equal(editedCueCount([cue({ text: '', edited: false })]), 1)
})

test('editing a hidden caption starts from the spoken words so Enter restores it', () => {
  const hidden = cue({ text: '', edited: true, highlight: false })
  assert.equal(restoreTextFor(hidden, words), 'Hello there.')
  // a visible caption starts from its current text
  assert.equal(restoreTextFor(cue({ text: "They're going." , first_word_id: 'w3', last_word_id: 'w4' }), words), "They're going.")
  // words that are no longer listed: nothing to prefill, the owner types the text
  assert.equal(restoreTextFor(cue({ text: '', first_word_id: 'x1', last_word_id: 'x2' }), words), '')
  assert.equal(restoreTextFor(cue({ text: '', first_word_id: 'w4', last_word_id: 'w3' }), words), '')
})
