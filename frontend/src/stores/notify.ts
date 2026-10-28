import { readonly, ref } from 'vue'

export type ToastKind = 'info' | 'success' | 'error'

export interface Toast {
  id: number
  kind: ToastKind
  text: string
}

const toasts = ref<Toast[]>([])
let nextId = 1

/** Show a short message; also announced to screen readers via the live region. */
export function notify(text: string, kind: ToastKind = 'info', ms = 5000): void {
  const id = nextId++
  toasts.value = [...toasts.value.slice(-3), { id, kind, text }]
  window.setTimeout(() => dismiss(id), kind === 'error' ? Math.max(ms, 8000) : ms)
}

export function dismiss(id: number): void {
  toasts.value = toasts.value.filter((t) => t.id !== id)
}

export function useToasts() {
  return { toasts: readonly(toasts), dismiss }
}
