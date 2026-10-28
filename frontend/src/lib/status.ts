import type { ClipRenderOut, JobOut, ProjectOut, ProjectStatus, RenderState, StageKey, StageOut } from '../api/types'

export const STAGE_ORDER: { key: StageKey; label: string }[] = [
  { key: 'import', label: 'Import source' },
  { key: 'probe', label: 'Inspect media' },
  { key: 'audio', label: 'Extract audio' },
  { key: 'transcribe', label: 'Transcribe speech' },
  { key: 'candidates', label: 'Find moments' },
  { key: 'review', label: 'Review & rank' },
  { key: 'framing', label: 'Frame clips' },
  { key: 'render', label: 'Render clips' },
]

/** All 8 stages in order, filling in any the server has not reported yet. */
export function normalizedStages(stages: StageOut[]): StageOut[] {
  const byKey = new Map(stages.map((s) => [s.key, s]))
  return STAGE_ORDER.map(
    (def) => byKey.get(def.key) ?? { key: def.key, label: def.label, state: 'pending', detail: null, duration_s: null },
  )
}

export interface StatusMeta {
  label: string
  tone: 'ok' | 'danger' | 'active' | 'muted' | 'amber'
}

export const PROJECT_STATUS: Record<ProjectStatus, StatusMeta> = {
  queued: { label: 'Queued', tone: 'active' },
  running: { label: 'Processing', tone: 'active' },
  ready: { label: 'Ready', tone: 'ok' },
  no_clips: { label: 'No clips found', tone: 'muted' },
  failed: { label: 'Failed', tone: 'danger' },
  cancelled: { label: 'Cancelled', tone: 'muted' },
  interrupted: { label: 'Interrupted', tone: 'amber' },
}

export function isProjectActive(status: ProjectStatus): boolean {
  return status === 'queued' || status === 'running'
}

export function isRenderActive(render: Pick<ClipRenderOut, 'state'>): boolean {
  return render.state === 'queued' || render.state === 'rendering'
}

function isJobActive(job: Pick<JobOut, 'state'> | null | undefined): boolean {
  return job?.state === 'queued' || job?.state === 'running'
}

/**
 * Whether a clip can go into the ZIP download. The export only includes finished, current
 * renders (backend routes_media EXPORT_STATES = ('ready',)); a stale, failed or re-rendering
 * clip still has its previous video but would be left out of the ZIP.
 */
export function isClipExportable(render: Pick<ClipRenderOut, 'state' | 'video_url'>): boolean {
  return render.state === 'ready' && !!render.video_url
}

/**
 * Any queued or running work for the project: its analysis or a clip render. The server
 * blocks delete, cleanup and reselect for both, and the project's cancel stops both.
 */
export function projectHasActiveWork(p: ProjectOut): boolean {
  return projectNeedsPolling(p) || p.actions.can_cancel
}

/** A cancel was requested for work that has not stopped yet (analysis or a clip render). */
export function projectCancelling(p: ProjectOut): boolean {
  if (isJobActive(p.job) && p.job?.cancel_requested) return true
  return p.clips.some((c) => isJobActive(c.job) && !!c.job?.cancel_requested)
}

/** Keep polling the project while analysis runs or any clip render is pending. */
export function projectNeedsPolling(p: ProjectOut): boolean {
  if (isProjectActive(p.status)) return true
  if (isJobActive(p.job)) return true
  return p.clips.some((c) => isRenderActive(c.render))
}

export const RENDER_STATE_LABEL: Record<RenderState, string> = {
  none: 'Not rendered',
  queued: 'Waiting to render',
  rendering: 'Rendering',
  ready: 'Ready',
  failed: 'Render failed',
  stale: 'Edited since last render',
}
