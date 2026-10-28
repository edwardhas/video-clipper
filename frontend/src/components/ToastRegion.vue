<script setup lang="ts">
import { useToasts } from '../stores/notify'
import AppIcon from './AppIcon.vue'

const { toasts, dismiss } = useToasts()
</script>

<template>
  <div class="toasts" aria-live="polite" aria-relevant="additions" role="status">
    <div v-for="t in toasts" :key="t.id" class="toast" :class="`toast-${t.kind}`">
      <AppIcon :name="t.kind === 'error' ? 'alert' : t.kind === 'success' ? 'check' : 'info'" />
      <p>{{ t.text }}</p>
      <button type="button" class="toast-close on-dark" aria-label="Dismiss message" @click="dismiss(t.id)">
        <AppIcon name="x" :size="16" />
      </button>
    </div>
  </div>
</template>

<style scoped>
.toasts {
  position: fixed;
  z-index: 50;
  right: var(--s4);
  bottom: var(--s4);
  display: flex;
  flex-direction: column;
  gap: var(--s2);
  width: min(380px, calc(100vw - 32px));
  pointer-events: none;
}

.toast {
  pointer-events: auto;
  display: flex;
  align-items: flex-start;
  gap: var(--s3);
  padding: var(--s3) var(--s3) var(--s3) var(--s4);
  border-radius: 10px;
  background: var(--brand-900);
  color: var(--white);
  font-size: var(--fs-sm);
  box-shadow: 0 10px 30px rgba(42, 12, 35, 0.3);
}

.toast > svg {
  flex: none;
  margin-top: 2px;
}

.toast p {
  flex: 1;
}

.toast-success > svg {
  color: #8fe0b4;
}

.toast-error {
  background: #6e1519;
}

.toast-info > svg {
  color: var(--amber);
}

.toast-close {
  flex: none;
  display: grid;
  place-items: center;
  width: 28px;
  height: 28px;
  border: 0;
  border-radius: 6px;
  background: transparent;
  color: inherit;
  cursor: pointer;
}

.toast-close:hover {
  background: rgba(255, 255, 255, 0.12);
}
</style>
