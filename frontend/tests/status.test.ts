import assert from 'node:assert/strict'
import { test } from 'node:test'
import {
  STAGE_ORDER,
  isClipExportable,
  normalizedStages,
  projectCancelling,
  projectHasActiveWork,
  projectNeedsPolling,
} from '../src/lib/status.ts'
import { clip, job, project } from './fixtures.ts'

test('all eight stages are listed in pipeline order', () => {
  assert.deepEqual(
    STAGE_ORDER.map((s) => s.key),
    ['import', 'probe', 'audio', 'transcribe', 'candidates', 'review', 'framing', 'render'],
  )
  const rows = normalizedStages([
    { key: 'probe', label: 'Inspect media', state: 'done', detail: null, duration_s: 1.2 },
  ])
  assert.equal(rows.length, 8)
  assert.equal(rows[0].state, 'pending')
  assert.equal(rows[1].state, 'done')
})

test('polling continues while the analysis or any render is active', () => {
  assert.equal(projectNeedsPolling(project()), false)
  assert.equal(projectNeedsPolling(project({ status: 'running' })), true)
  assert.equal(projectNeedsPolling(project({ status: 'queued' })), true)
  assert.equal(projectNeedsPolling(project({ status: 'ready', job: job({ state: 'running' }) })), true)
  const rendering = clip({ render: { ...clip().render, state: 'rendering' } })
  assert.equal(projectNeedsPolling(project({ clips: [rendering] })), true)
  const queued = clip({ render: { ...clip().render, state: 'queued' } })
  assert.equal(projectNeedsPolling(project({ clips: [queued] })), true)
})

test('polling stops for finished, failed and interrupted projects', () => {
  for (const status of ['ready', 'no_clips', 'failed', 'cancelled', 'interrupted'] as const) {
    assert.equal(projectNeedsPolling(project({ status, job: job({ state: 'failed' }) })), false, status)
  }
})

// Finding 23: the ZIP export only includes clips whose render_state is 'ready' (routes_media
// EXPORT_STATES); a stale, failed or re-rendering clip still has its old video but is dropped.
test('only ready clips with a video can be selected for the ZIP', () => {
  const base = clip().render
  assert.equal(isClipExportable(base), true)
  for (const state of ['stale', 'failed', 'queued', 'rendering'] as const) {
    assert.equal(isClipExportable({ ...base, state, stale: state === 'stale' }), false, state)
  }
  assert.equal(isClipExportable({ ...base, state: 'none', video_url: null }), false)
  assert.equal(isClipExportable({ ...base, video_url: null }), false)
})

// Finding 24: a clip re-render leaves the project 'ready' but still blocks delete/cleanup/reselect;
// the project-level Cancel (which also cancels render jobs) must be offered for it.
test('a queued clip render counts as active work that can be cancelled', () => {
  assert.equal(projectHasActiveWork(project()), false)
  assert.equal(projectHasActiveWork(project({ status: 'running' })), true)
  const queued = clip({ render: { ...clip().render, state: 'queued' }, job: job({ kind: 'render', state: 'queued' }) })
  const p = project({
    status: 'ready',
    clips: [queued],
    actions: { can_cancel: true, can_resume: false, can_reselect: false, can_delete: false },
  })
  assert.equal(projectHasActiveWork(p), true)
  // reported by the server even before the clip list shows it
  assert.equal(
    projectHasActiveWork(project({ actions: { can_cancel: true, can_resume: false, can_reselect: false, can_delete: false } })),
    true,
  )
})

test('cancelling is shown only for work that is still active', () => {
  // an old, finished analysis job that was cancelled must not keep the button spinning
  const oldCancelled = job({ state: 'cancelled', cancel_requested: true })
  assert.equal(projectCancelling(project({ job: oldCancelled })), false)
  assert.equal(projectCancelling(project({ status: 'running', job: job({ state: 'running', cancel_requested: true }) })), true)
  const stopping = clip({
    render: { ...clip().render, state: 'rendering' },
    job: job({ kind: 'render', state: 'running', cancel_requested: true }),
  })
  assert.equal(projectCancelling(project({ job: oldCancelled, clips: [stopping] })), true)
  assert.equal(projectHasActiveWork(project({ clips: [stopping] })), true)
})
