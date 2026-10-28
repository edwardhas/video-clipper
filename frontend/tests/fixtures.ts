import type { ClipEdit, ClipOut, JobOut, ProjectOut } from '../src/api/types.ts'

export function edit(overrides: Partial<ClipEdit> = {}): ClipEdit {
  return {
    src_start: 10,
    src_end: 40,
    captions_enabled: true,
    style: { size: 'M', color: 'white', position: 'bottom', highlight: true, uppercase: false },
    word_overrides: {},
    cue_overrides: [],
    framing: { mode: 'auto', crop_x: null },
    ...overrides,
  }
}

export function job(overrides: Partial<JobOut> = {}): JobOut {
  return {
    id: 'j1',
    kind: 'analyze',
    state: 'succeeded',
    stage: null,
    stage_detail: null,
    progress_current: null,
    progress_total: null,
    cancel_requested: false,
    error_code: null,
    error_message: null,
    retryable: null,
    attempts: 1,
    created_at: 1,
    started_at: 1,
    finished_at: 2,
    ...overrides,
  }
}

export function clip(overrides: Partial<ClipOut> = {}): ClipOut {
  return {
    id: 'c1',
    project_id: 'p1',
    rank: 1,
    title: 'A clip',
    reason: 'Strong opening',
    score: 78,
    score_detail: {},
    src_start: 10,
    src_end: 40,
    duration: 30,
    ai_src_start: 10,
    ai_src_end: 40,
    hook: '',
    topic: '',
    edit: edit(),
    render: {
      state: 'ready',
      error: null,
      video_url: '/media/clips/c1/video.mp4?v=h',
      download_url: '/media/clips/c1/download.mp4',
      srt_url: '/media/clips/c1/captions.srt',
      thumb_url: null,
      render_hash: 'h',
      info: null,
      stale: false,
    },
    framing: null,
    job: null,
    ...overrides,
  }
}

export function project(overrides: Partial<ProjectOut> = {}): ProjectOut {
  return {
    id: 'p1',
    title: 'Talk',
    source_kind: 'upload',
    source_url: null,
    status: 'ready',
    status_detail: null,
    created_at: 1,
    updated_at: 2,
    duration: 600,
    clip_count: 1,
    thumb_url: null,
    options: { clip_count: 5, min_seconds: 20, max_seconds: 60, captions_enabled: true, language: 'en' },
    media: null,
    stages: [],
    job: job(),
    clips: [clip()],
    usage: { requests: 0, input_tokens: 0, output_tokens: 0, audio_seconds: 0, estimated_cost_usd: null, cost_note: '' },
    actions: { can_cancel: false, can_resume: false, can_reselect: true, can_delete: true },
    ...overrides,
  }
}
