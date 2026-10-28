<script setup lang="ts">
/**
 * Caption switch, style presets and per-cue text corrections. Text corrections
 * are saved immediately (POST /api/clips/:id/captions); the switch and style are
 * part of the editor draft and saved with "Save changes".
 */
import { computed, nextTick, ref, useId } from 'vue'
import { ApiError, api } from '../../api/client'
import type { CaptionStyle, ClipDetailOut, Cue, TranscriptWordOut } from '../../api/types'
import { editedCueCount, isHiddenCue, restoreTextFor } from '../../lib/captions'
import { fmtPrecise } from '../../lib/format'
import { notify } from '../../stores/notify'
import ActionButton from '../ActionButton.vue'
import CaptionStyleFields from '../CaptionStyleFields.vue'

const captionsEnabled = defineModel<boolean>('captionsEnabled', { required: true })
const captionStyle = defineModel<CaptionStyle>('captionStyle', { required: true })
const props = defineProps<{
  clipId: string
  cues: Cue[]
  /** Transcript words around the clip, to restore a hidden caption's spoken text. */
  words: TranscriptWordOut[]
  hasOverrides: boolean
  rangeDirty: boolean
}>()
const emit = defineEmits<{ updated: [clip: ClipDetailOut] }>()

const uid = useId()
const editingId = ref<string | null>(null)
const editText = ref('')
const savingId = ref<string | null>(null)
const resetting = ref(false)
const cueError = ref<string | null>(null)
// Hidden captions (empty text) are listed by the API too, and count as edited.
const editedCount = computed(() => editedCueCount(props.cues))

async function startEdit(cue: Cue) {
  editingId.value = cue.id
  // A hidden caption starts from its spoken words: saving them shows it again.
  editText.value = restoreTextFor(cue, props.words)
  cueError.value = null
  await nextTick()
  document.getElementById(`${uid}-cue-${cue.id}`)?.focus()
}

function cancelEdit() {
  editingId.value = null
  cueError.value = null
}

async function saveCue(cue: Cue) {
  const text = editText.value.replace(/\s+/g, ' ').trim()
  if (text === cue.text.trim()) {
    cancelEdit()
    return
  }
  savingId.value = cue.id
  cueError.value = null
  try {
    const clip = await api.editCaption(props.clipId, {
      first_word_id: cue.first_word_id,
      last_word_id: cue.last_word_id,
      text,
    })
    emit('updated', clip)
    editingId.value = null
    const message = !text
      ? 'Caption hidden. Re-render to apply.'
      : isHiddenCue(cue)
        ? 'Caption shown again. Re-render to see it in the video.'
        : 'Caption text saved. Re-render to see it in the video.'
    notify(message, 'success')
  } catch (err) {
    cueError.value = err instanceof ApiError ? err.message : 'Could not save the caption.'
  } finally {
    savingId.value = null
  }
}

async function resetEdits() {
  resetting.value = true
  try {
    const clip = await api.resetCaptions(props.clipId)
    emit('updated', clip)
    editingId.value = null
    notify('Caption text restored to the transcript. Re-render to apply.', 'success')
  } catch (err) {
    notify(err instanceof ApiError ? err.message : 'Could not reset the caption edits.', 'error')
  } finally {
    resetting.value = false
  }
}

function onKey(event: KeyboardEvent, cue: Cue) {
  if (event.key === 'Escape') {
    event.preventDefault()
    cancelEdit()
  } else if (event.key === 'Enter' && !event.shiftKey) {
    event.preventDefault()
    void saveCue(cue)
  }
}
</script>

<template>
  <section class="captions stack" :aria-labelledby="`${uid}-h`">
    <h2 :id="`${uid}-h`" class="visually-hidden">Captions</h2>

    <label class="switch">
      <input v-model="captionsEnabled" type="checkbox" role="switch" />
      <span class="track" aria-hidden="true"></span>
      Show captions in the video
    </label>
    <p v-if="!captionsEnabled" class="field-hint">
      Captions are off, so no text will be burned into this clip.
    </p>

    <div class="sub">
      <h3 class="sub-title">Style</h3>
      <CaptionStyleFields v-model="captionStyle" :disabled="!captionsEnabled" />
    </div>

    <div class="sub">
      <div class="row-between">
        <h3 class="sub-title">Caption text</h3>
        <ActionButton
          size="sm"
          :loading="resetting"
          loading-label="Resetting…"
          :disabled="!hasOverrides"
          reason="No caption text has been changed"
          @click="resetEdits"
        >
          Reset caption edits
        </ActionButton>
      </div>
      <p class="field-hint">
        Fix a misheard word by editing its caption. Edits are saved right away and appear in the video after a re-render.
        <template v-if="editedCount"> {{ editedCount }} edited.</template>
      </p>
      <p v-if="rangeDirty" class="notice notice-warn cue-note">
        These captions match the saved trim. Save your changes to update the list for the new start and end.
      </p>

      <p v-if="!cues.length" class="muted small empty-cues">
        No speech falls inside this clip, so there are no captions.
      </p>
      <ol v-else class="cues">
        <li v-for="cue in cues" :key="cue.id" class="cue" :class="{ edited: cue.edited || isHiddenCue(cue) }">
          <span class="cue-time num">{{ fmtPrecise(cue.rel_start) }}</span>
          <div v-if="editingId === cue.id" class="cue-edit">
            <label class="visually-hidden" :for="`${uid}-cue-${cue.id}`">Caption text</label>
            <textarea
              :id="`${uid}-cue-${cue.id}`"
              v-model="editText"
              class="input"
              rows="2"
              maxlength="300"
              :aria-describedby="`${uid}-cue-hint`"
              @keydown="onKey($event, cue)"
            ></textarea>
            <p :id="`${uid}-cue-hint`" class="field-hint">
              <template v-if="isHiddenCue(cue)">
                Enter saves, Escape cancels. This caption is hidden; save its words to show it again. Changing the
                number of words turns off word highlighting for this caption, because new words have no timing.
              </template>
              <template v-else>
                Enter saves, Escape cancels. Leave it empty to hide this caption. Changing the number of words turns off
                word highlighting for this caption, because new words have no timing.
              </template>
            </p>
            <p v-if="cueError" class="field-error" role="alert">{{ cueError }}</p>
            <div class="row">
              <ActionButton
                size="sm"
                variant="primary"
                :loading="savingId === cue.id"
                loading-label="Saving…"
                @click="saveCue(cue)"
              >
                Save caption
              </ActionButton>
              <ActionButton size="sm" variant="quiet" :disabled="savingId === cue.id" @click="cancelEdit">
                Cancel
              </ActionButton>
            </div>
          </div>
          <template v-else>
            <div class="cue-main">
              <p class="cue-text" :class="{ hidden: isHiddenCue(cue) }">
                {{ isHiddenCue(cue) ? 'Hidden caption' : cue.text }}
              </p>
              <p v-if="isHiddenCue(cue)" class="cue-flags">
                <span class="badge badge-amber">Edited</span>
                <span class="xsmall muted">Not shown in the video. Edit it to show it again.</span>
              </p>
              <p v-else-if="cue.edited" class="cue-flags">
                <span class="badge badge-amber">Edited</span>
                <span v-if="!cue.highlight" class="xsmall muted">Word highlight is off for this caption.</span>
              </p>
            </div>
            <button
              type="button"
              class="btn btn-quiet btn-sm"
              :aria-label="`Edit caption at ${fmtPrecise(cue.rel_start)}`"
              :disabled="editingId !== null"
              :title="editingId !== null ? 'Finish the caption you are editing first' : undefined"
              @click="startEdit(cue)"
            >
              Edit
            </button>
          </template>
        </li>
      </ol>
    </div>
  </section>
</template>

<style scoped>
.sub {
  padding-top: var(--s4);
  border-top: 1px solid var(--line);
}

.sub-title {
  font-size: var(--fs-md);
  margin-bottom: var(--s3);
}

.row-between .sub-title {
  margin-bottom: 0;
}

.row-between {
  margin-bottom: var(--s2);
}

.cue-note {
  margin-top: var(--s3);
}

.empty-cues {
  margin-top: var(--s3);
}

.cues {
  list-style: none;
  margin: var(--s3) 0 0;
  padding: 0;
  border: 1px solid var(--line);
  border-radius: var(--r-control);
  max-height: 420px;
  overflow-y: auto;
}

.cue {
  display: grid;
  grid-template-columns: 64px minmax(0, 1fr) auto;
  align-items: start;
  gap: var(--s2) var(--s3);
  padding: var(--s2) var(--s3);
}

.cue + .cue {
  border-top: 1px solid var(--line);
}

.cue.edited {
  background: #fffaf2;
}

.cue-time {
  padding-top: 6px;
  font-size: var(--fs-xs);
  color: var(--ink-soft);
}

.cue-main {
  padding-top: 4px;
  min-width: 0;
}

.cue-text {
  font-size: var(--fs-sm);
}

.cue-text.hidden {
  color: var(--ink-soft);
  font-style: italic;
}

.cue-flags {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: var(--s2);
  margin-top: var(--s1);
}

.cue-edit {
  grid-column: 2 / -1;
  display: flex;
  flex-direction: column;
  gap: var(--s2);
}

@media (max-width: 420px) {
  .cue {
    grid-template-columns: 1fr auto;
  }

  .cue-time {
    grid-column: 1 / -1;
    padding-top: 0;
  }

  .cue-edit {
    grid-column: 1 / -1;
  }
}
</style>
