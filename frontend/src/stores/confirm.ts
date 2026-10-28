import { readonly, ref } from 'vue'

export interface ConfirmRequest {
  title: string
  message: string
  confirmLabel: string
  cancelLabel?: string
  danger?: boolean
}

interface Pending extends ConfirmRequest {
  resolve: (ok: boolean) => void
}

const pending = ref<Pending | null>(null)

/** Ask the user to confirm an action in a modal dialog. Resolves true on confirm. */
export function confirmAction(req: ConfirmRequest): Promise<boolean> {
  pending.value?.resolve(false)
  return new Promise((resolve) => {
    pending.value = { ...req, resolve }
  })
}

export function settleConfirm(ok: boolean): void {
  const p = pending.value
  pending.value = null
  p?.resolve(ok)
}

export function useConfirmState() {
  return { pending: readonly(pending) }
}
