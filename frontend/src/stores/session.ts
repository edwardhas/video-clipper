import { readonly, ref } from 'vue'
import { ApiError, onSessionChange, refreshSession } from '../api/client'
import type { SessionOut } from '../api/types'

const session = ref<SessionOut | null>(null)
const sessionError = ref<string | null>(null)

onSessionChange((s) => {
  session.value = s
  sessionError.value = null
})

/** Load the session once (or again when `force`); never throws. */
export async function loadSession(force = false): Promise<SessionOut | null> {
  if (session.value && !force) return session.value
  try {
    return await refreshSession()
  } catch (err) {
    sessionError.value = err instanceof ApiError ? err.message : 'Cannot reach Local Clipper.'
    return null
  }
}

/** Called on a 401 so the router guard sends the user to the login page. */
export function markSignedOut(): void {
  if (session.value) session.value = { ...session.value, authenticated: false, csrf_token: null }
}

export function useSession() {
  return { session: readonly(session), sessionError: readonly(sessionError) }
}
