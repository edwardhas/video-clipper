<script setup lang="ts">
/** Re-run moment selection on the existing transcript with new count/length options. */
import { computed, nextTick, ref, watch } from 'vue'
import { ApiError, api } from '../api/client'
import type { LimitsInfo, ProjectOptions } from '../api/types'
import { validateOptions, type ClipOptionValues } from '../lib/options'
import { notify } from '../stores/notify'
import ActionButton from './ActionButton.vue'
import ClipOptionsFields from './ClipOptionsFields.vue'

const props = defineProps<{
  open: boolean
  projectId: string
  options: ProjectOptions
  hasClips: boolean
  limits: LimitsInfo | null
}>()
const emit = defineEmits<{ close: []; started: [] }>()

const dialog = ref<HTMLDialogElement | null>(null)
const values = ref<ClipOptionValues>(fromOptions(props.options))
const busy = ref(false)
const error = ref<string | null>(null)
const problem = computed(() => validateOptions(values.value, props.limits))

function fromOptions(o: ProjectOptions): ClipOptionValues {
  return {
    clip_count: o.clip_count,
    min_seconds: o.min_seconds,
    max_seconds: o.max_seconds,
    captions_enabled: o.captions_enabled,
  }
}

watch(
  () => props.open,
  async (open) => {
    await nextTick()
    const el = dialog.value
    if (!el) return
    if (open && !el.open) {
      values.value = fromOptions(props.options)
      error.value = null
      el.showModal()
    } else if (!open && el.open) {
      el.close()
    }
  },
  { immediate: true },
)

async function submit() {
  if (problem.value || busy.value) return
  busy.value = true
  error.value = null
  try {
    await api.reselectProject(props.projectId, {
      options: { ...props.options, ...values.value, language: props.options.language },
    })
    notify('Selection is running again with the new options.', 'success')
    emit('started')
    emit('close')
  } catch (err) {
    error.value = err instanceof ApiError ? err.message : 'Could not start the selection.'
  } finally {
    busy.value = false
  }
}
</script>

<template>
  <dialog ref="dialog" class="modal" aria-labelledby="reselect-title" @cancel.prevent="emit('close')">
    <form v-if="open" @submit.prevent="submit">
      <div class="modal-body stack">
        <h2 id="reselect-title">Find different clips</h2>
        <p class="muted small">
          The moment selection runs again on the existing transcript, so the speech is not transcribed again.
          <template v-if="hasClips">
            If new clips are found they replace the current ones, including any edits you made. If none are
            found, your current clips are kept.
          </template>
        </p>
        <ClipOptionsFields v-model="values" :limits="limits" :show-captions="false" :disabled="busy" />
        <p v-if="problem" class="field-error" role="alert">{{ problem }}</p>
        <div v-if="error" class="notice notice-error" role="alert">
          <p>{{ error }}</p>
        </div>
      </div>
      <div class="modal-actions">
        <ActionButton :disabled="busy" @click="emit('close')">Cancel</ActionButton>
        <ActionButton
          type="submit"
          variant="primary"
          :loading="busy"
          loading-label="Starting…"
          :disabled="!!problem"
          :reason="problem"
        >
          Run selection again
        </ActionButton>
      </div>
    </form>
  </dialog>
</template>
