/**
 * Local (unsaved) editor state for a clip and the minimal PATCH body that
 * applies it. Caption text corrections are not part of the draft: they are
 * saved immediately through POST /api/clips/:id/captions.
 */
import type { CaptionStyle, ClipEdit, ClipEditIn, FramingChoice } from '../api/types'

export interface EditDraft {
  src_start: number
  src_end: number
  captions_enabled: boolean
  style: CaptionStyle
  framing: FramingChoice
}

export function draftFromEdit(edit: ClipEdit): EditDraft {
  return {
    src_start: edit.src_start,
    src_end: edit.src_end,
    captions_enabled: edit.captions_enabled,
    style: { ...edit.style },
    framing: { ...edit.framing },
  }
}

const TIME_EPS = 1e-3

function sameStyle(a: CaptionStyle, b: CaptionStyle): boolean {
  return (
    a.size === b.size &&
    a.color === b.color &&
    a.position === b.position &&
    a.highlight === b.highlight &&
    a.uppercase === b.uppercase
  )
}

function sameFraming(a: FramingChoice, b: FramingChoice): boolean {
  if (a.mode !== b.mode) return false
  if (a.crop_x == null || b.crop_x == null) return a.crop_x == null && b.crop_x == null
  return Math.abs(a.crop_x - b.crop_x) < 1e-4
}

/** Only the fields that differ from the saved edit (empty object = nothing to save). */
export function diffDraft(draft: EditDraft, saved: ClipEdit): ClipEditIn {
  const body: ClipEditIn = {}
  const rangeChanged =
    Math.abs(draft.src_start - saved.src_start) > TIME_EPS || Math.abs(draft.src_end - saved.src_end) > TIME_EPS
  if (rangeChanged) {
    // Send both ends together so the server always validates a consistent range.
    body.src_start = draft.src_start
    body.src_end = draft.src_end
  }
  if (draft.captions_enabled !== saved.captions_enabled) body.captions_enabled = draft.captions_enabled
  if (!sameStyle(draft.style, saved.style)) body.style = { ...draft.style }
  if (!sameFraming(draft.framing, saved.framing)) body.framing = { ...draft.framing }
  return body
}

export function isDirty(draft: EditDraft, saved: ClipEdit): boolean {
  return Object.keys(diffDraft(draft, saved)).length > 0
}

export function rangeDirty(draft: EditDraft, saved: ClipEdit): boolean {
  const d = diffDraft(draft, saved)
  return d.src_start !== undefined || d.src_end !== undefined
}
