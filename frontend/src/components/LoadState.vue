<script setup lang="ts">
/** Loading placeholder or a load error with a retry button. */
import AppIcon from './AppIcon.vue'

defineProps<{ loading: boolean; error?: string | null; what?: string }>()
defineEmits<{ retry: [] }>()
</script>

<template>
  <div v-if="loading" class="load-state" role="status">
    <span class="spinner" aria-hidden="true"></span>
    <span>Loading {{ what ?? '' }}…</span>
  </div>
  <div v-else-if="error" class="notice notice-error load-error" role="alert">
    <AppIcon name="alert" />
    <div class="notice-body">
      <p>
        <strong>Could not load {{ what ?? 'this page' }}.</strong>
      </p>
      <p>{{ error }}</p>
      <p>
        <button type="button" class="btn btn-sm" @click="$emit('retry')">
          <AppIcon name="refresh" :size="16" /> Try again
        </button>
      </p>
    </div>
  </div>
</template>

<style scoped>
.load-state {
  display: flex;
  align-items: center;
  gap: var(--s3);
  padding: var(--s6) 0;
  color: var(--ink-soft);
  font-weight: 600;
}

.load-error {
  margin: var(--s4) 0;
}
</style>
