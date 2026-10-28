import { readonly, ref } from 'vue'
import { ApiError, api } from '../api/client'
import type { SystemOut } from '../api/types'

const system = ref<SystemOut | null>(null)
const systemError = ref<string | null>(null)
let inFlight: Promise<SystemOut | null> | null = null

/** Fetch GET /api/system (deduplicated). Returns the cached value on failure. */
export function loadSystem(): Promise<SystemOut | null> {
  if (!inFlight) {
    inFlight = api
      .system()
      .then((s) => {
        system.value = s
        systemError.value = null
        return s
      })
      .catch((err: unknown) => {
        systemError.value = err instanceof ApiError ? err.message : 'Could not load system information.'
        return system.value
      })
      .finally(() => {
        inFlight = null
      })
  }
  return inFlight
}

export function useSystem() {
  return { system: readonly(system), systemError: readonly(systemError), loadSystem }
}
