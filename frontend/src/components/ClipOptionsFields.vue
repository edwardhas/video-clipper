<script setup lang="ts">
/**
 * Number of clips, clip length (presets or custom range) and optionally the
 * captions switch. Used by the import page, the defaults form and the
 * re-run selection dialog.
 */
import { computed, ref, useId, watch } from 'vue'
import type { LimitsInfo } from '../api/types'
import { DURATION_PRESETS, boundsFrom, presetFor, type ClipOptionValues } from '../lib/options'

const model = defineModel<ClipOptionValues>({ required: true })
const props = withDefaults(
  defineProps<{ limits?: LimitsInfo | null; showCaptions?: boolean; disabled?: boolean }>(),
  { limits: null, showCaptions: true, disabled: false },
)

const uid = useId()
const bounds = computed(() => boundsFrom(props.limits))
const custom = ref(presetFor(model.value.min_seconds, model.value.max_seconds) === null)

// Values loaded later (e.g. saved defaults) decide whether "Custom" is shown.
watch(
  () => [model.value.min_seconds, model.value.max_seconds] as const,
  ([min, max]) => {
    if (presetFor(min, max) === null) custom.value = true
  },
)

const selectedPreset = computed(() =>
  custom.value ? 'custom' : (presetFor(model.value.min_seconds, model.value.max_seconds)?.id ?? 'custom'),
)

function patch(values: Partial<ClipOptionValues>) {
  model.value = { ...model.value, ...values }
}

function choosePreset(id: string) {
  const preset = DURATION_PRESETS.find((p) => p.id === id)
  if (!preset) {
    custom.value = true
    return
  }
  custom.value = false
  patch({ min_seconds: preset.min, max_seconds: preset.max })
}

function setCount(n: number) {
  if (!Number.isFinite(n)) return
  patch({ clip_count: Math.min(bounds.value.clipMax, Math.max(bounds.value.clipMin, Math.round(n))) })
}

function readNumber(event: Event): number {
  return Number.parseFloat((event.target as HTMLInputElement).value)
}
</script>

<template>
  <div class="clip-options">
    <div class="field">
      <label class="field-label" :for="`${uid}-count`">Number of clips</label>
      <div class="stepper">
        <button
          type="button"
          class="btn icon-btn"
          :disabled="disabled || model.clip_count <= bounds.clipMin"
          :title="model.clip_count <= bounds.clipMin ? `At least ${bounds.clipMin} clip` : 'One fewer clip'"
          aria-label="One fewer clip"
          @click="setCount(model.clip_count - 1)"
        >
          <span aria-hidden="true" class="stepper-sign">−</span>
        </button>
        <input
          :id="`${uid}-count`"
          class="input input-num stepper-input"
          type="number"
          inputmode="numeric"
          :min="bounds.clipMin"
          :max="bounds.clipMax"
          step="1"
          :value="model.clip_count"
          :disabled="disabled"
          :aria-describedby="`${uid}-count-hint`"
          @change="setCount(readNumber($event))"
        />
        <button
          type="button"
          class="btn icon-btn"
          :disabled="disabled || model.clip_count >= bounds.clipMax"
          :title="model.clip_count >= bounds.clipMax ? `At most ${bounds.clipMax} clips` : 'One more clip'"
          aria-label="One more clip"
          @click="setCount(model.clip_count + 1)"
        >
          <span aria-hidden="true" class="stepper-sign">+</span>
        </button>
      </div>
      <p :id="`${uid}-count-hint`" class="field-hint">
        Up to {{ bounds.clipMax }}. You may get fewer when the video has fewer strong moments.
      </p>
    </div>

    <fieldset class="field fieldset" :disabled="disabled">
      <legend class="field-label">Clip length</legend>
      <div class="chips">
        <label v-for="p in DURATION_PRESETS" :key="p.id" class="chip">
          <input
            type="radio"
            :name="`${uid}-preset`"
            :value="p.id"
            :checked="selectedPreset === p.id"
            @change="choosePreset(p.id)"
          />
          <span>
            {{ p.label }}
            <small>{{ p.min }}–{{ p.max }} s</small>
          </span>
        </label>
        <label class="chip">
          <input
            type="radio"
            :name="`${uid}-preset`"
            value="custom"
            :checked="selectedPreset === 'custom'"
            @change="choosePreset('custom')"
          />
          <span>
            Custom
            <small v-if="selectedPreset === 'custom'">{{ model.min_seconds }}–{{ model.max_seconds }} s</small>
            <small v-else>Set your own</small>
          </span>
        </label>
      </div>
      <div v-if="selectedPreset === 'custom'" class="custom-range">
        <div class="field">
          <label class="field-label" :for="`${uid}-min`">Shortest (seconds)</label>
          <input
            :id="`${uid}-min`"
            class="input input-num"
            type="number"
            inputmode="decimal"
            :min="bounds.minFloor"
            :max="bounds.minCeil"
            step="1"
            :value="model.min_seconds"
            @change="patch({ min_seconds: readNumber($event) })"
          />
        </div>
        <div class="field">
          <label class="field-label" :for="`${uid}-max`">Longest (seconds)</label>
          <input
            :id="`${uid}-max`"
            class="input input-num"
            type="number"
            inputmode="decimal"
            :min="bounds.maxFloor"
            :max="bounds.maxCap"
            step="1"
            :value="model.max_seconds"
            @change="patch({ max_seconds: readNumber($event) })"
          />
        </div>
      </div>
    </fieldset>

    <div v-if="showCaptions" class="field">
      <span class="field-label" :id="`${uid}-cap-label`">Captions</span>
      <label class="switch">
        <input
          type="checkbox"
          role="switch"
          :checked="model.captions_enabled"
          :disabled="disabled"
          :aria-labelledby="`${uid}-cap-label`"
          :aria-describedby="`${uid}-cap-state`"
          @change="patch({ captions_enabled: ($event.target as HTMLInputElement).checked })"
        />
        <span class="track" aria-hidden="true"></span>
        <span :id="`${uid}-cap-state`">{{ model.captions_enabled ? 'Burned into the video' : 'Off' }}</span>
      </label>
    </div>
  </div>
</template>

<style scoped>
.clip-options {
  display: flex;
  flex-direction: column;
  gap: var(--s5);
}

.fieldset {
  margin: 0;
  padding: 0;
  border: 0;
  gap: var(--s2);
}

.fieldset legend {
  padding: 0;
  margin-bottom: var(--s2);
}

.stepper {
  display: flex;
  align-items: stretch;
  gap: var(--s2);
  max-width: 220px;
}

.stepper-input {
  text-align: center;
  min-width: 0;
  flex: 1;
}

.stepper-sign {
  font-size: var(--fs-lg);
  line-height: 1;
}

.custom-range {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 180px));
  gap: var(--s3);
  margin-top: var(--s3);
}

@media (max-width: 400px) {
  .custom-range {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
</style>
