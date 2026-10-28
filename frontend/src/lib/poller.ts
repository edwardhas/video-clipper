import { onBeforeUnmount } from 'vue'

/**
 * Repeatedly run `task` while it returns true, waiting `intervalMs` between runs.
 * Never overlaps runs; `kick()` runs immediately (e.g. after a user action).
 * Stops automatically when the owning component unmounts.
 */
export function usePoller(task: () => Promise<boolean>, intervalMs: number) {
  let timer: number | undefined
  let running = false
  let rerun = false
  let disposed = false

  const clear = () => {
    if (timer !== undefined) window.clearTimeout(timer)
    timer = undefined
  }

  async function run(): Promise<void> {
    if (disposed) return
    if (running) {
      rerun = true
      return
    }
    clear()
    running = true
    let again = false
    try {
      again = await task()
    } catch {
      again = false
    } finally {
      running = false
    }
    if (disposed) return
    if (rerun) {
      rerun = false
      void run()
      return
    }
    if (again) timer = window.setTimeout(() => void run(), intervalMs)
  }

  const stop = () => {
    disposed = true
    clear()
  }
  onBeforeUnmount(stop)

  return { kick: () => run(), stop }
}
