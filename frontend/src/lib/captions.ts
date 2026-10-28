import type { Cue, TranscriptWordOut } from '../api/types'

/**
 * A caption the owner hid by saving empty text. The API still lists it (text "" and
 * edited=true) so it can be shown as a "Hidden caption" row and restored on its own.
 */
export function isHiddenCue(cue: Pick<Cue, 'text'>): boolean {
  return !cue.text.trim()
}

/** Captions the owner changed, hidden ones included (the "N edited." count). */
export function editedCueCount(cues: Pick<Cue, 'text' | 'edited'>[]): number {
  return cues.filter((c) => c.edited || isHiddenCue(c)).length
}

/** The transcript's own words for a caption's word range ('' when they are not listed). */
function spokenText(cue: Pick<Cue, 'first_word_id' | 'last_word_id'>, words: Pick<TranscriptWordOut, 'id' | 'text'>[]): string {
  const i = words.findIndex((w) => w.id === cue.first_word_id)
  const j = words.findIndex((w) => w.id === cue.last_word_id)
  if (i < 0 || j < i) return ''
  return words
    .slice(i, j + 1)
    .map((w) => w.text.trim())
    .filter(Boolean)
    .join(' ')
}

/**
 * Starting text when editing a caption: its current text, or for a hidden caption the
 * spoken words, so saving it unchanged shows the caption again with its real word timings.
 */
export function restoreTextFor(
  cue: Pick<Cue, 'text' | 'first_word_id' | 'last_word_id'>,
  words: Pick<TranscriptWordOut, 'id' | 'text'>[],
): string {
  return isHiddenCue(cue) ? spokenText(cue, words) : cue.text
}
