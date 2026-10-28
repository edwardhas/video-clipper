<script setup lang="ts">
/**
 * Read-only picture of the trim: the transcript context window, the current
 * range (amber) and the AI's original range (outlined). All times are source seconds.
 */
import { computed } from 'vue'
import { fmtPrecise } from '../../lib/format'
import { percentOf } from '../../lib/trim'

const props = defineProps<{
  contextStart: number
  contextEnd: number
  srcStart: number
  srcEnd: number
  aiSrcStart: number
  aiSrcEnd: number
}>()

const lo = computed(() => Math.min(props.contextStart, props.srcStart, props.aiSrcStart))
const hi = computed(() => Math.max(props.contextEnd, props.srcEnd, props.aiSrcEnd))

function span(a: number, b: number) {
  const left = percentOf(a, lo.value, hi.value)
  const right = percentOf(b, lo.value, hi.value)
  return { left: `${left}%`, width: `${Math.max(0.6, right - left)}%` }
}

const current = computed(() => span(props.srcStart, props.srcEnd))
const ai = computed(() => span(props.aiSrcStart, props.aiSrcEnd))
const summary = computed(
  () =>
    `Clip runs from ${fmtPrecise(props.srcStart)} to ${fmtPrecise(props.srcEnd)}. ` +
    `The AI chose ${fmtPrecise(props.aiSrcStart)} to ${fmtPrecise(props.aiSrcEnd)}.`,
)
</script>

<template>
  <figure class="strip" :aria-label="summary" role="img">
    <div class="strip-track">
      <div class="strip-ai" :style="ai"></div>
      <div class="strip-range" :style="current"></div>
    </div>
    <figcaption class="strip-scale num" aria-hidden="true">
      <span>{{ fmtPrecise(lo) }}</span>
      <span class="strip-legend">
        <span class="key key-range"></span>Your clip
        <span class="key key-ai"></span>AI selection
      </span>
      <span>{{ fmtPrecise(hi) }}</span>
    </figcaption>
  </figure>
</template>

<style scoped>
.strip {
  margin: 0;
}

.strip-track {
  position: relative;
  height: 28px;
  border-radius: var(--r-control);
  background: repeating-linear-gradient(90deg, var(--brand-100) 0 6px, var(--brand-150) 6px 7px);
  border: 1px solid var(--line);
  overflow: hidden;
}

.strip-range,
.strip-ai {
  position: absolute;
  top: 0;
  bottom: 0;
}

.strip-range {
  background: var(--amber);
  border-left: 3px solid var(--brand-900);
  border-right: 3px solid var(--brand-900);
  opacity: 0.92;
  top: 5px;
  bottom: 5px;
  border-radius: 3px;
}

.strip-ai {
  border: 2px dashed var(--brand);
  border-radius: 4px;
  top: 1px;
  bottom: 1px;
}

.strip-scale {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: var(--s2);
  margin-top: var(--s1);
  font-size: var(--fs-xs);
  color: var(--ink-soft);
}

.strip-legend {
  display: inline-flex;
  align-items: center;
  gap: var(--s1);
  flex-wrap: wrap;
  justify-content: center;
}

.key {
  display: inline-block;
  width: 14px;
  height: 10px;
  border-radius: 2px;
  margin-left: var(--s2);
}

.key-range {
  background: var(--amber);
  border: 1px solid var(--brand-900);
}

.key-ai {
  border: 2px dashed var(--brand);
}
</style>
