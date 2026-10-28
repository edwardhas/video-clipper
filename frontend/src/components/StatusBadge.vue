<script setup lang="ts">
import { computed } from 'vue'
import type { ProjectStatus } from '../api/types'
import { PROJECT_STATUS } from '../lib/status'

const props = defineProps<{ status: ProjectStatus }>()
const meta = computed(() => PROJECT_STATUS[props.status])
const toneClass = computed(() => {
  switch (meta.value.tone) {
    case 'ok':
      return 'badge-ok'
    case 'danger':
      return 'badge-danger'
    case 'active':
      return 'badge-active'
    case 'amber':
      return 'badge-amber'
    default:
      return 'badge-muted'
  }
})
</script>

<template>
  <span class="badge" :class="toneClass">
    <span v-if="status === 'running' || status === 'queued'" class="spinner status-spin" aria-hidden="true"></span>
    <span v-else class="status-dot" aria-hidden="true"></span>
    {{ meta.label }}
  </span>
</template>

<style scoped>
.status-spin {
  width: 10px;
  height: 10px;
  border-width: 2px;
}

.status-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: currentColor;
}

.status-spin[class] {
  animation-duration: 1.2s;
}
</style>
