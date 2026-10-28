<script setup lang="ts">
/** "78/100" editorial score with an explanatory tooltip (hover or keyboard focus). */
import { computed, useId } from 'vue'
import { fmtScore } from '../lib/format'

const props = defineProps<{ score: number }>()
const uid = useId()
const label = computed(() => fmtScore(props.score))
const TIP = 'Editorial score — a heuristic, not a prediction of views'
</script>

<template>
  <span class="score" tabindex="0" :aria-describedby="`${uid}-tip`">
    <span class="score-value num"><span class="visually-hidden">Score </span>{{ label }}</span>
    <span :id="`${uid}-tip`" role="tooltip" class="score-tip">{{ TIP }}</span>
  </span>
</template>

<style scoped>
.score {
  position: relative;
  display: inline-flex;
  border-radius: 4px;
  cursor: help;
}

.score-value {
  display: inline-block;
  padding: 2px var(--s2);
  border-radius: 4px;
  background: var(--amber);
  color: var(--brand-900);
  font-size: var(--fs-xs);
  font-weight: 800;
  line-height: 1.5;
  white-space: nowrap;
}

.score-tip {
  position: absolute;
  z-index: 10;
  right: 0;
  bottom: calc(100% + 8px);
  width: max-content;
  max-width: min(240px, 70vw);
  padding: var(--s2) var(--s3);
  border-radius: var(--r-control);
  background: var(--brand-900);
  color: var(--white);
  font-size: var(--fs-xs);
  font-weight: 500;
  line-height: 1.4;
  white-space: normal;
  box-shadow: 0 6px 18px rgba(42, 12, 35, 0.25);
  opacity: 0;
  visibility: hidden;
  transition: opacity 120ms ease;
  pointer-events: none;
}

.score:hover .score-tip,
.score:focus-visible .score-tip {
  opacity: 1;
  visibility: visible;
}
</style>
