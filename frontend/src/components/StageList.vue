<script setup lang="ts">
/**
 * The eight analysis stages with their real state. Step counts come only from
 * the job's progress_current/progress_total; nothing is estimated.
 */
import { computed } from 'vue'
import type { JobOut, StageOut, StageState } from '../api/types'
import { fmtDuration } from '../lib/format'
import { normalizedStages } from '../lib/status'
import AppIcon from './AppIcon.vue'

const props = defineProps<{ stages: StageOut[]; job: JobOut | null }>()

const rows = computed(() => normalizedStages(props.stages))
const jobActive = computed(() => props.job?.state === 'running' || props.job?.state === 'queued')

const STATE_TEXT: Record<StageState, string> = {
  pending: 'Not started',
  running: 'In progress',
  done: 'Done',
  failed: 'Failed',
  skipped: 'Skipped',
}

function isCurrent(stage: StageOut): boolean {
  return stage.state === 'running' && jobActive.value
}

function detailFor(stage: StageOut): string | null {
  if (isCurrent(stage) && props.job?.stage === stage.key && props.job.stage_detail) return props.job.stage_detail
  return stage.detail
}

function stepFor(stage: StageOut): string | null {
  const job = props.job
  if (!job || !isCurrent(stage) || job.stage !== stage.key) return null
  if (job.progress_current == null || job.progress_total == null || job.progress_total <= 0) return null
  return `Step ${job.progress_current} of ${job.progress_total}`
}
</script>

<template>
  <ol class="stages">
    <li v-for="stage in rows" :key="stage.key" class="stage" :class="`stage-${stage.state}`">
      <span class="stage-icon" aria-hidden="true">
        <span v-if="isCurrent(stage)" class="spinner"></span>
        <AppIcon v-else-if="stage.state === 'done'" name="check" :size="16" />
        <AppIcon v-else-if="stage.state === 'failed'" name="x" :size="16" />
        <AppIcon v-else-if="stage.state === 'skipped'" name="minus" :size="16" />
      </span>
      <div class="stage-body">
        <div class="stage-line">
          <span class="stage-label">{{ stage.label }}</span>
          <span class="visually-hidden">: {{ STATE_TEXT[stage.state] }}</span>
          <span v-if="stepFor(stage)" class="stage-step num">{{ stepFor(stage) }}</span>
          <span v-if="stage.state === 'done' && stage.duration_s != null" class="stage-time num">
            {{ fmtDuration(stage.duration_s) }}
          </span>
          <span v-if="stage.state === 'skipped'" class="stage-time">Skipped</span>
        </div>
        <p v-if="detailFor(stage)" class="stage-detail">{{ detailFor(stage) }}</p>
      </div>
    </li>
  </ol>
</template>

<style scoped>
.stages {
  list-style: none;
  margin: 0;
  padding: 0;
  counter-reset: none;
}

.stage {
  position: relative;
  display: flex;
  gap: var(--s3);
  padding: var(--s2) 0;
}

/* vertical connector between stage markers */
.stage:not(:last-child)::before {
  content: '';
  position: absolute;
  left: 11px;
  top: 34px;
  bottom: -6px;
  width: 2px;
  background: var(--line);
}

.stage-done:not(:last-child)::before {
  background: var(--brand-300);
}

.stage-icon {
  position: relative;
  z-index: 1;
  display: grid;
  place-items: center;
  flex: none;
  width: 24px;
  height: 24px;
  margin-top: 1px;
  border-radius: 50%;
  border: 2px solid var(--line-strong);
  background: var(--white);
  color: var(--ink-soft);
}

.stage-done .stage-icon {
  border-color: var(--brand);
  background: var(--brand);
  color: var(--white);
}

.stage-running .stage-icon {
  border-color: var(--amber);
  background: var(--amber-100);
  color: var(--brand);
}

.stage-running .stage-icon .spinner {
  width: 12px;
  height: 12px;
}

.stage-failed .stage-icon {
  border-color: var(--danger);
  background: var(--danger);
  color: var(--white);
}

.stage-skipped .stage-icon {
  border-style: dashed;
}

.stage-body {
  min-width: 0;
  flex: 1;
  padding-top: 2px;
}

.stage-line {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: var(--s1) var(--s3);
}

.stage-label {
  font-weight: 650;
  color: var(--brand-900);
}

.stage-pending .stage-label {
  color: var(--ink-soft);
  font-weight: 550;
}

.stage-step {
  padding: 0 var(--s2);
  border-radius: 4px;
  background: var(--amber);
  color: var(--brand-900);
  font-size: var(--fs-xs);
  font-weight: 750;
}

.stage-time {
  font-size: var(--fs-xs);
  color: var(--ink-soft);
}

.stage-detail {
  margin-top: 2px;
  font-size: var(--fs-sm);
  color: var(--ink-soft);
}

.stage-failed .stage-detail {
  color: var(--danger);
}
</style>
