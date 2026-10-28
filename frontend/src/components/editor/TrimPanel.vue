<script setup lang="ts">
/**
 * Trim the clip with numeric start/end (source seconds), nudge buttons and the
 * transcript word list ("Start here" / "End here"). Changes stay local until saved.
 */
import { computed, nextTick, onMounted, ref, useId } from 'vue'
import type { TranscriptWordOut } from '../../api/types'
import { fmtDuration, fmtPrecise, fmtRange } from '../../lib/format'
import {
  endAtWord,
  endProblem,
  neighbours,
  sameTime,
  startAtWord,
  startProblem,
  withEnd,
  withStart,
  wordInRange,
  type TrimLimits,
  type TrimRange,
} from '../../lib/trim'
import ActionButton from '../ActionButton.vue'
import TrimStrip from './TrimStrip.vue'

const range = defineModel<TrimRange>({ required: true })
const props = defineProps<{
  words: TranscriptWordOut[]
  limits: TrimLimits
  contextStart: number
  contextEnd: number
  aiSrcStart: number
  aiSrcEnd: number
}>()

const uid = useId()
const NUDGES = [-1, -0.1, 0.1, 1]
const selectedWord = ref<TranscriptWordOut | null>(null)
const wordBox = ref<HTMLElement | null>(null)

const length = computed(() => range.value.src_end - range.value.src_start)
const atAi = computed(
  () => sameTime(range.value.src_start, props.aiSrcStart) && sameTime(range.value.src_end, props.aiSrcEnd),
)
// The words either side of the selected one: "Start here" / "End here" must not pull them into the clip.
const around = computed(() =>
  selectedWord.value ? neighbours(props.words, selectedWord.value) : { prev: null, next: null },
)
const startBlock = computed(() =>
  selectedWord.value ? startProblem(range.value, selectedWord.value, props.limits, around.value.prev) : null,
)
const endBlock = computed(() =>
  selectedWord.value ? endProblem(range.value, selectedWord.value, props.limits, around.value.next) : null,
)

function nudgeLabel(delta: number): string {
  const sign = delta < 0 ? '−' : '+'
  return `${sign}${Math.abs(delta)} s`
}

function nudgeAria(which: 'start' | 'end', delta: number): string {
  const amount = Math.abs(delta) === 1 ? '1 second' : `${Math.abs(delta)} seconds`
  return `Move the ${which} ${amount} ${delta < 0 ? 'earlier' : 'later'}`
}

function setStart(v: number) {
  range.value = withStart(range.value, v, props.limits)
}

function setEnd(v: number) {
  range.value = withEnd(range.value, v, props.limits)
}

function onNumber(which: 'start' | 'end', event: Event) {
  const input = event.target as HTMLInputElement
  const v = Number.parseFloat(input.value)
  if (which === 'start') setStart(v)
  else setEnd(v)
  // Reflect clamping (or revert an unreadable value) in the field.
  input.value = String(which === 'start' ? range.value.src_start : range.value.src_end)
}

function pick(word: TranscriptWordOut) {
  selectedWord.value = selectedWord.value?.id === word.id ? null : word
}

function startHere() {
  if (!selectedWord.value || startBlock.value) return
  range.value = startAtWord(range.value, selectedWord.value, props.limits, around.value.prev)
  selectedWord.value = null
}

function endHere() {
  if (!selectedWord.value || endBlock.value) return
  range.value = endAtWord(range.value, selectedWord.value, props.limits, around.value.next)
  selectedWord.value = null
}

function resetToAi() {
  range.value = { src_start: props.aiSrcStart, src_end: props.aiSrcEnd }
}

// Bring the first word of the clip into view inside the scrolling word list.
onMounted(async () => {
  await nextTick()
  const box = wordBox.value
  const first = box?.querySelector<HTMLElement>('.word.in')
  if (box && first) box.scrollTop = Math.max(0, first.offsetTop - box.offsetTop - 24)
})
</script>

<template>
  <section class="trim stack" :aria-labelledby="`${uid}-h`">
    <h2 :id="`${uid}-h`" class="visually-hidden">Trim</h2>

    <TrimStrip
      :context-start="contextStart"
      :context-end="contextEnd"
      :src-start="range.src_start"
      :src-end="range.src_end"
      :ai-src-start="aiSrcStart"
      :ai-src-end="aiSrcEnd"
    />

    <div class="trim-rows">
      <div v-for="which in (['start', 'end'] as const)" :key="which" class="trim-row">
        <label class="field-label trim-label" :for="`${uid}-${which}`">{{ which === 'start' ? 'Start' : 'End' }}</label>
        <div class="trim-input">
          <input
            :id="`${uid}-${which}`"
            class="input input-num"
            type="number"
            inputmode="decimal"
            step="0.1"
            min="0"
            :max="limits.sourceDuration"
            :value="which === 'start' ? range.src_start : range.src_end"
            :aria-describedby="`${uid}-${which}-clock`"
            @change="onNumber(which, $event)"
          />
          <span class="unit" aria-hidden="true">s</span>
        </div>
        <span :id="`${uid}-${which}-clock`" class="clock num">{{
          fmtPrecise(which === 'start' ? range.src_start : range.src_end)
        }}</span>
        <div class="nudges" role="group" :aria-label="`Nudge ${which}`">
          <button
            v-for="d in NUDGES"
            :key="d"
            type="button"
            class="btn btn-sm nudge num"
            :aria-label="nudgeAria(which, d)"
            @click="which === 'start' ? setStart(range.src_start + d) : setEnd(range.src_end + d)"
          >
            {{ nudgeLabel(d) }}
          </button>
        </div>
      </div>
    </div>

    <div class="trim-summary">
      <p class="small summary-text">
        <strong class="num">Length {{ fmtDuration(length) }}</strong>
        <span class="muted num">AI selection {{ fmtRange(aiSrcStart, aiSrcEnd) }}</span>
      </p>
      <ActionButton
        size="sm"
        :disabled="atAi"
        reason="The clip already matches the AI selection"
        @click="resetToAi"
      >
        Reset to AI selection
      </ActionButton>
    </div>

    <div class="transcript">
      <div class="transcript-head">
        <h3 class="field-label">Transcript</h3>
        <p class="field-hint">Highlighted words are in the clip. Select a word to start or end the clip there.</p>
      </div>
      <div v-if="selectedWord" class="word-actions" role="group" aria-label="Selected word">
        <p class="small">
          “<strong>{{ selectedWord.text }}</strong>” at <span class="num">{{ fmtPrecise(selectedWord.start) }}</span>
        </p>
        <div class="row">
          <ActionButton size="sm" variant="primary" :disabled="!!startBlock" :reason="startBlock" @click="startHere">
            Start here
          </ActionButton>
          <ActionButton size="sm" variant="primary" :disabled="!!endBlock" :reason="endBlock" @click="endHere">
            End here
          </ActionButton>
          <button type="button" class="btn btn-quiet btn-sm" @click="selectedWord = null">Close</button>
        </div>
      </div>
      <div ref="wordBox" class="words">
        <p v-if="!words.length" class="muted small">No transcript words are available around this clip.</p>
        <button
          v-for="w in words"
          :key="w.id"
          type="button"
          class="word"
          :class="{ in: wordInRange(w, range), picked: selectedWord?.id === w.id }"
          :aria-pressed="selectedWord?.id === w.id"
          :title="fmtPrecise(w.start)"
          @click="pick(w)"
        >
          {{ w.text }}
        </button>
      </div>
    </div>
  </section>
</template>

<style scoped>
.trim-rows {
  display: flex;
  flex-direction: column;
  gap: var(--s3);
}

.trim-row {
  display: grid;
  grid-template-columns: 44px 120px auto 1fr;
  align-items: center;
  gap: var(--s2) var(--s3);
}

.trim-input {
  position: relative;
}

.trim-input .input {
  padding-right: 26px;
}

.unit {
  position: absolute;
  right: 10px;
  top: 50%;
  transform: translateY(-50%);
  color: var(--ink-soft);
  font-size: var(--fs-sm);
  pointer-events: none;
}

.clock {
  font-size: var(--fs-sm);
  font-weight: 650;
  color: var(--brand-900);
}

.nudges {
  display: flex;
  flex-wrap: wrap;
  gap: var(--s1);
  justify-content: flex-end;
}

.nudge {
  min-width: 52px;
  padding: var(--s1) var(--s2);
}

.trim-summary {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: var(--s2);
  padding: var(--s3);
  border-radius: var(--r-control);
  background: var(--mist);
}

.summary-text {
  display: flex;
  flex-wrap: wrap;
  gap: var(--s1) var(--s4);
}

.transcript-head {
  margin-bottom: var(--s2);
}

.word-actions {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: var(--s2);
  padding: var(--s2) var(--s3);
  margin-bottom: var(--s2);
  border-radius: var(--r-control);
  background: var(--brand-100);
  border: 1px solid var(--brand-150);
}

.words {
  max-height: 300px;
  overflow-y: auto;
  padding: var(--s3);
  border: 1px solid var(--line);
  border-radius: var(--r-control);
  background: var(--white);
  line-height: 2;
}

.word {
  display: inline;
  padding: 2px 3px;
  margin: 0 1px;
  border: 0;
  border-radius: 3px;
  background: transparent;
  color: var(--ink-soft);
  font: inherit;
  font-size: var(--fs-md);
  cursor: pointer;
}

.word:hover {
  background: var(--brand-100);
  color: var(--brand-900);
}

.word.in {
  background: var(--amber-100);
  color: var(--ink);
  box-shadow: inset 0 -2px 0 var(--amber);
}

.word.picked {
  background: var(--brand);
  color: var(--white);
  box-shadow: none;
}

.word:focus-visible {
  outline-offset: 1px;
}

@media (max-width: 720px) {
  .trim-row {
    grid-template-columns: 44px minmax(0, 1fr) auto;
  }

  .nudges {
    grid-column: 1 / -1;
    justify-content: flex-start;
  }

  .nudge {
    flex: 1;
  }
}
</style>
