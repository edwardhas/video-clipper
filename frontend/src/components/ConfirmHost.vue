<script setup lang="ts">
import { nextTick, ref, watch } from 'vue'
import { settleConfirm, useConfirmState } from '../stores/confirm'
import ActionButton from './ActionButton.vue'

const { pending } = useConfirmState()
const dialog = ref<HTMLDialogElement | null>(null)
let returnFocus: HTMLElement | null = null

watch(pending, async (p) => {
  await nextTick()
  const el = dialog.value
  if (!el) return
  if (p && !el.open) {
    returnFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null
    el.showModal()
  } else if (!p && el.open) {
    el.close()
  }
})

function answer(ok: boolean) {
  settleConfirm(ok)
  returnFocus?.focus()
  returnFocus = null
}
</script>

<template>
  <dialog
    ref="dialog"
    class="modal"
    aria-labelledby="confirm-title"
    aria-describedby="confirm-message"
    @cancel.prevent="answer(false)"
  >
    <template v-if="pending">
      <div class="modal-body">
        <h2 id="confirm-title">{{ pending.title }}</h2>
        <p id="confirm-message" class="muted">{{ pending.message }}</p>
      </div>
      <div class="modal-actions">
        <ActionButton variant="secondary" @click="answer(false)">{{ pending.cancelLabel ?? 'Cancel' }}</ActionButton>
        <ActionButton :variant="pending.danger ? 'danger-solid' : 'primary'" @click="answer(true)">
          {{ pending.confirmLabel }}
        </ActionButton>
      </div>
    </template>
  </dialog>
</template>
