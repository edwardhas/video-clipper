<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { RouterLink, onBeforeRouteLeave, useRouter } from 'vue-router'
import { ApiError, api, mediaUrls } from '../api/client'
import type { ClipDetailOut, JobOut } from '../api/types'
import ActionButton from '../components/ActionButton.vue'
import AppIcon from '../components/AppIcon.vue'
import LoadState from '../components/LoadState.vue'
import ScoreBadge from '../components/ScoreBadge.vue'
import CaptionsPanel from '../components/editor/CaptionsPanel.vue'
import FramingPanel from '../components/editor/FramingPanel.vue'
import TrimPanel from '../components/editor/TrimPanel.vue'
import { diffDraft, draftFromEdit, rangeDirty, type EditDraft } from '../lib/draft'
import { fmtBytes, fmtDuration, fmtRange } from '../lib/format'
import { usePoller } from '../lib/poller'
import { isRenderActive } from '../lib/status'
import { clipTrimLimits, type TrimLimits, type TrimRange } from '../lib/trim'
import { confirmAction } from '../stores/confirm'
import { notify } from '../stores/notify'
import { loadSystem, useSystem } from '../stores/system'

const props = defineProps<{ id: string }>()
const router = useRouter()
const { system } = useSystem()

type Tab = 'trim' | 'captions' | 'framing'
const TABS: { id: Tab; label: string }[] = [
  { id: 'trim', label: 'Trim' },
  { id: 'captions', label: 'Captions' },
  { id: 'framing', label: 'Framing' },
]

const clip = ref<ClipDetailOut | null>(null)
const draft = ref<EditDraft | null>(null)
const loading = ref(true)
const loadError = ref<string | null>(null)
const notFound = ref(false)
const saving = ref(false)
const rendering = ref(false)
const restoring = ref(false)
const tab = ref<Tab>('trim')
const statusMessage = ref('')

const patch = computed(() => (clip.value && draft.value ? diffDraft(draft.value, clip.value.edit) : {}))
const dirty = computed(() => Object.keys(patch.value).length > 0)
const trimDirty = computed(() => !!clip.value && !!draft.value && rangeDirty(draft.value, clip.value.edit))
const render = computed(() => clip.value?.render ?? null)
const renderJob = computed<JobOut | null>(() => clip.value?.job ?? null)
const renderActive = computed(() => {
  if (!clip.value) return false
  const jobActive = renderJob.value?.state === 'queued' || renderJob.value?.state === 'running'
  return isRenderActive(clip.value.render) || jobActive
})
const busy = computed(() => saving.value || rendering.value || restoring.value)
const hasCaptionEdits = computed(() => {
  const c = clip.value
  if (!c) return false
  return (
    Object.keys(c.edit.word_overrides).length > 0 || c.edit.cue_overrides.length > 0 || c.cues.some((q) => q.edited)
  )
})

// The same 3 s to 3 min bounds the API enforces on save (routes_clips.validate_range), so the
// trim controls cannot produce a range that Save then rejects.
const trimLimits = computed<TrimLimits>(() => clipTrimLimits(clip.value?.source_duration ?? 0))

const range = computed<TrimRange>({
  get: () => ({ src_start: draft.value?.src_start ?? 0, src_end: draft.value?.src_end ?? 0 }),
  set: (r) => {
    if (draft.value) draft.value = { ...draft.value, src_start: r.src_start, src_end: r.src_end }
  },
})

const frameUrl = computed(() => {
  if (!clip.value) return ''
  const mid = (clip.value.src_start + clip.value.src_end) / 2
  return mediaUrls.projectFrame(clip.value.project_id, Math.round(mid * 2) / 2)
})

const renderReason = computed<string | null>(() => {
  if (!clip.value) return 'Loading'
  if (renderActive.value) return 'A render is already in progress.'
  if (!dirty.value && clip.value.render.state === 'ready' && !clip.value.render.stale) {
    return 'The video is already up to date. Change something first.'
  }
  return null
})

const renderStateText = computed(() => {
  const r = render.value
  if (!r) return ''
  const job = renderJob.value
  if (r.state === 'queued' || job?.state === 'queued') {
    return system.value && !system.value.worker.online
      ? 'Waiting to render. The background worker is not running, so the render starts when it is back.'
      : 'Waiting to render…'
  }
  if (r.state === 'rendering' || job?.state === 'running') {
    const step =
      job?.progress_current != null && job.progress_total ? ` Step ${job.progress_current} of ${job.progress_total}.` : ''
    return `Rendering…${step}${job?.stage_detail ? ` ${job.stage_detail}` : ''}`
  }
  if (r.state === 'failed') return r.error || job?.error_message || 'The render did not finish.'
  if (r.state === 'none') return 'This clip has not been rendered yet.'
  if (r.state === 'stale') return ''
  return r.info ? `Rendered ${r.info.width}×${r.info.height}, ${fmtDuration(r.info.duration)}, ${fmtBytes(r.info.size_bytes)}.` : 'Rendered.'
})

// Bumped by every change the user makes on the server, so a poll that was already in flight
// cannot overwrite the newer state with an older response.
let mutationSeq = 0

/** Apply the response of a user action (save, render, caption edit, restore). */
function applyMutation(next: ClipDetailOut, opts: { resetDraft?: boolean } = {}) {
  mutationSeq += 1
  applyServer(next, opts)
}

/** Take a fresh server copy; keep the user's unsaved draft unless there is none. */
function applyServer(next: ClipDetailOut, opts: { resetDraft?: boolean } = {}) {
  const prev = clip.value
  const keepDraft = !opts.resetDraft && draft.value && prev && dirty.value
  clip.value = next
  if (!keepDraft) draft.value = draftFromEdit(next.edit)
  if (prev) announceRenderChange(prev, next)
}

function announceRenderChange(prev: ClipDetailOut, next: ClipDetailOut) {
  const was = prev.render.state
  const now = next.render.state
  if (was === now) return
  if (now === 'ready' && (was === 'queued' || was === 'rendering')) {
    statusMessage.value = 'Render finished. The preview shows the new version.'
    notify('Render finished.', 'success')
  } else if (now === 'failed') {
    statusMessage.value = 'The render failed.'
  } else if (now === 'rendering') {
    statusMessage.value = 'Rendering started.'
  }
}

function needsPolling(c: ClipDetailOut): boolean {
  const jobActive = c.job?.state === 'queued' || c.job?.state === 'running'
  return isRenderActive(c.render) || jobActive
}

const poller = usePoller(async () => {
  const seq = mutationSeq
  try {
    const next = await api.getClip(props.id)
    if (seq !== mutationSeq) return true // superseded by a user action; poll again
    applyServer(next)
    return needsPolling(next)
  } catch (err) {
    if (err instanceof ApiError && err.status === 404) {
      notFound.value = true
      return false
    }
    // Transient error (e.g. worker restarting the API): keep trying.
    return true
  }
}, 1500)

async function load() {
  loading.value = true
  loadError.value = null
  notFound.value = false
  try {
    const c = await api.getClip(props.id)
    applyServer(c, { resetDraft: true })
    if (needsPolling(c)) void poller.kick()
  } catch (err) {
    if (err instanceof ApiError && err.status === 404) notFound.value = true
    else loadError.value = err instanceof ApiError ? err.message : 'Something went wrong.'
  } finally {
    loading.value = false
  }
}

async function save(): Promise<boolean> {
  if (!clip.value || !dirty.value) return true
  saving.value = true
  try {
    const next = await api.patchClip(clip.value.id, patch.value)
    applyMutation(next, { resetDraft: true })
    return true
  } catch (err) {
    notify(err instanceof ApiError ? err.message : 'Could not save the changes.', 'error')
    return false
  } finally {
    saving.value = false
  }
}

async function saveOnly() {
  if (await save()) notify('Changes saved. Re-render to see them in the video.', 'success')
}

async function rerender() {
  if (!clip.value || renderReason.value) return
  rendering.value = true
  try {
    if (dirty.value && !(await save())) return
    const next = await api.renderClip(clip.value.id)
    applyMutation(next, { resetDraft: true })
    statusMessage.value = needsPolling(next) ? 'Render queued.' : 'The video is already up to date.'
    void poller.kick()
  } catch (err) {
    notify(err instanceof ApiError ? err.message : 'Could not start the render.', 'error')
  } finally {
    rendering.value = false
  }
}

function discard() {
  if (clip.value) draft.value = draftFromEdit(clip.value.edit)
}

async function restoreOriginal() {
  if (!clip.value) return
  const ok = await confirmAction({
    title: 'Restore the AI’s original edit?',
    message:
      'This brings back the clip as the AI first made it: its original start and end, caption text, style and framing. Your changes to this clip are discarded.',
    confirmLabel: 'Restore original',
    danger: true,
  })
  if (!ok) return
  restoring.value = true
  try {
    const next = await api.resetClip(clip.value.id)
    applyMutation(next, { resetDraft: true })
    notify('Original edit restored. Re-render to update the video.', 'success')
  } catch (err) {
    notify(err instanceof ApiError ? err.message : 'Could not restore the original edit.', 'error')
  } finally {
    restoring.value = false
  }
}

function onCaptionsUpdated(next: ClipDetailOut) {
  applyMutation(next)
}

function onTabKey(event: KeyboardEvent) {
  const idx = TABS.findIndex((t) => t.id === tab.value)
  let nextIdx = idx
  if (event.key === 'ArrowRight') nextIdx = (idx + 1) % TABS.length
  else if (event.key === 'ArrowLeft') nextIdx = (idx - 1 + TABS.length) % TABS.length
  else if (event.key === 'Home') nextIdx = 0
  else if (event.key === 'End') nextIdx = TABS.length - 1
  else return
  event.preventDefault()
  tab.value = TABS[nextIdx].id
  document.getElementById(`tab-${tab.value}`)?.focus()
}

// Unsaved changes: ask before leaving the page or closing the tab.
onBeforeRouteLeave(async () => {
  if (!dirty.value) return true
  return confirmAction({
    title: 'Leave without saving?',
    message: 'Your trim, caption style or framing changes have not been saved.',
    confirmLabel: 'Leave without saving',
    cancelLabel: 'Stay here',
    danger: true,
  })
})

function beforeUnload(event: BeforeUnloadEvent) {
  if (dirty.value) {
    event.preventDefault()
    event.returnValue = ''
  }
}

onMounted(() => {
  window.addEventListener('beforeunload', beforeUnload)
  void loadSystem()
  void load()
})
onBeforeUnmount(() => window.removeEventListener('beforeunload', beforeUnload))

watch(
  () => clip.value?.title,
  (title) => {
    if (title) document.title = `${title} – Local Clipper`
  },
)

function goBack() {
  if (clip.value) void router.push(`/projects/${clip.value.project_id}`)
  else void router.push('/projects')
}
</script>

<template>
  <div class="page editor-page">
    <a href="#" class="back-link" @click.prevent="goBack">
      <AppIcon name="chevron-left" :size="16" /> Back to project
    </a>

    <LoadState v-if="loading || loadError" :loading="loading" :error="loadError" what="this clip" @retry="load" />

    <div v-else-if="notFound" class="panel empty">
      <h1>This clip no longer exists</h1>
      <p class="muted">It may have been deleted, or replaced when the moments were selected again.</p>
      <RouterLink to="/projects" class="btn btn-primary">See all projects</RouterLink>
    </div>

    <template v-else-if="clip && draft">
      <header class="editor-head">
        <div class="editor-title">
          <h1>{{ clip.title }}</h1>
          <div class="editor-meta">
            <ScoreBadge :score="clip.score" />
            <span class="num"><span class="muted">Length</span> {{ fmtDuration(clip.duration) }}</span>
            <span class="num"><span class="muted">From the source</span> {{ fmtRange(clip.src_start, clip.src_end) }}</span>
          </div>
          <p v-if="clip.reason" class="editor-reason">{{ clip.reason }}</p>
        </div>
        <ActionButton size="sm" variant="quiet" :loading="restoring" loading-label="Restoring…" :disabled="busy" @click="restoreOriginal">
          <AppIcon name="refresh" :size="16" /> Restore original edit
        </ActionButton>
      </header>

      <div class="editor-grid">
        <!-- The rendered result: the visual focus of the page. -->
        <section class="stage on-dark" aria-label="Rendered preview">
          <div class="stage-video">
            <video
              v-if="render?.video_url"
              :key="render.video_url"
              class="preview"
              :src="render.video_url"
              :poster="render.thumb_url ?? undefined"
              controls
              preload="metadata"
              playsinline
              :aria-label="`Rendered preview of ${clip.title}`"
            ></video>
            <div v-else class="preview preview-empty">
              <span v-if="renderActive" class="spinner" aria-hidden="true"></span>
              <AppIcon v-else name="film" :size="28" />
              <p>{{ renderActive ? 'The first render is on its way.' : 'No render yet.' }}</p>
            </div>
          </div>

          <p v-if="render?.stale && render.video_url" class="stale-banner">
            <AppIcon name="info" :size="16" />
            Preview shows the last render — click Re-render to apply your changes
          </p>

          <div v-if="renderStateText" class="stage-status" :class="{ failed: render?.state === 'failed' }">
            <span v-if="renderActive" class="spinner" aria-hidden="true"></span>
            <AppIcon v-else-if="render?.state === 'failed'" name="alert" :size="16" />
            <AppIcon v-else-if="render?.state === 'ready'" name="check" :size="16" />
            <p>{{ renderStateText }}</p>
          </div>
          <p class="visually-hidden" aria-live="polite">{{ statusMessage }}</p>

          <div class="stage-actions">
            <ActionButton
              variant="accent"
              :loading="rendering"
              :loading-label="dirty ? 'Saving…' : 'Starting…'"
              :disabled="!!renderReason || saving || restoring"
              :reason="renderReason"
              @click="rerender"
            >
              <AppIcon name="refresh" :size="16" />
              {{ dirty ? 'Save and re-render' : 'Re-render' }}
            </ActionButton>
            <a v-if="render?.download_url" class="btn on-dark-btn" :href="render.download_url" download>
              <AppIcon name="download" :size="16" /> MP4
            </a>
            <a v-if="render?.srt_url" class="btn on-dark-btn" :href="render.srt_url" download>
              <AppIcon name="captions" :size="16" /> SRT
            </a>
          </div>
        </section>

        <section class="controls" aria-label="Edit clip">
          <div class="tabs" role="tablist" aria-label="Editing tools" @keydown="onTabKey">
            <button
              v-for="t in TABS"
              :id="`tab-${t.id}`"
              :key="t.id"
              type="button"
              role="tab"
              class="tab"
              :aria-selected="tab === t.id"
              :aria-controls="`panel-${t.id}`"
              :tabindex="tab === t.id ? 0 : -1"
              @click="tab = t.id"
            >
              <AppIcon :name="t.id === 'trim' ? 'trim' : t.id === 'captions' ? 'captions' : 'frame'" :size="16" />
              {{ t.label }}
            </button>
          </div>

          <div id="panel-trim" v-show="tab === 'trim'" role="tabpanel" aria-labelledby="tab-trim" class="tab-panel">
            <TrimPanel
              v-model="range"
              :words="clip.words"
              :limits="trimLimits"
              :context-start="clip.context_start"
              :context-end="clip.context_end"
              :ai-src-start="clip.ai_src_start"
              :ai-src-end="clip.ai_src_end"
            />
          </div>
          <div id="panel-captions" v-show="tab === 'captions'" role="tabpanel" aria-labelledby="tab-captions" class="tab-panel">
            <CaptionsPanel
              v-model:captions-enabled="draft.captions_enabled"
              v-model:caption-style="draft.style"
              :clip-id="clip.id"
              :cues="clip.cues"
              :words="clip.words"
              :has-overrides="hasCaptionEdits"
              :range-dirty="trimDirty"
              @updated="onCaptionsUpdated"
            />
          </div>
          <div id="panel-framing" v-show="tab === 'framing'" role="tabpanel" aria-labelledby="tab-framing" class="tab-panel">
            <FramingPanel
              v-model="draft.framing"
              :frame-url="frameUrl"
              :source-width="clip.source_width"
              :source-height="clip.source_height"
              :auto="clip.framing"
            />
          </div>

          <div class="save-bar" :class="{ dirty }">
            <p class="small" aria-live="polite">
              <template v-if="dirty"><strong>Unsaved changes.</strong> Save them, then re-render to update the video.</template>
              <template v-else>All changes saved.</template>
            </p>
            <div class="row">
              <ActionButton variant="quiet" size="sm" :disabled="!dirty || busy" reason="Nothing to discard" @click="discard">
                Discard
              </ActionButton>
              <ActionButton
                variant="primary"
                :loading="saving"
                loading-label="Saving…"
                :disabled="!dirty || rendering || restoring"
                reason="There are no unsaved changes"
                @click="saveOnly"
              >
                Save changes
              </ActionButton>
            </div>
          </div>
        </section>
      </div>
    </template>
  </div>
</template>

<style scoped>
.editor-page {
  padding-top: var(--s5);
}

.empty {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: var(--s3);
}

.editor-head {
  display: flex;
  flex-wrap: wrap;
  align-items: flex-start;
  justify-content: space-between;
  gap: var(--s3);
  margin-bottom: var(--s5);
}

.editor-title {
  min-width: 0;
  flex: 1 1 420px;
}

.editor-meta {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: var(--s2) var(--s4);
  margin-top: var(--s2);
  font-size: var(--fs-sm);
  font-weight: 600;
}

.editor-reason {
  margin-top: var(--s2);
  max-width: 70ch;
  color: var(--ink-soft);
  font-size: var(--fs-sm);
}

.editor-grid {
  display: grid;
  grid-template-columns: minmax(300px, 5fr) minmax(0, 7fr);
  gap: var(--s5);
  align-items: start;
}

/* ---------------------------------------------------------- stage */
.stage {
  position: sticky;
  top: calc(var(--header-h) + var(--s4));
  display: flex;
  flex-direction: column;
  gap: var(--s3);
  padding: var(--s4);
  border-radius: var(--r-media);
  background: var(--brand-900);
  color: #f3e6ee;
}

.stage-video {
  display: grid;
  place-items: center;
}

.preview {
  width: 100%;
  max-width: 420px;
  max-height: calc(100vh - var(--header-h) - 230px);
  min-height: 320px;
  aspect-ratio: 9 / 16;
  border-radius: 10px;
  background: #000;
  object-fit: contain;
}

.preview-empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: var(--s2);
  background: #3b1231;
  color: #e6d3df;
  font-weight: 600;
  text-align: center;
  padding: var(--s4);
}

.stale-banner {
  display: flex;
  align-items: flex-start;
  gap: var(--s2);
  padding: var(--s2) var(--s3);
  border-radius: var(--r-control);
  background: var(--amber);
  color: var(--brand-900);
  font-size: var(--fs-sm);
  font-weight: 650;
}

.stale-banner svg {
  flex: none;
  margin-top: 2px;
}

.stage-status {
  display: flex;
  align-items: flex-start;
  gap: var(--s2);
  font-size: var(--fs-sm);
  min-height: 20px;
}

.stage-status svg,
.stage-status .spinner {
  flex: none;
  margin-top: 2px;
}

.stage-status.failed {
  color: #ffd0d2;
}

.stage-actions {
  display: flex;
  flex-wrap: wrap;
  gap: var(--s2);
}

.on-dark-btn {
  --btn-bg: transparent;
  --btn-fg: var(--white);
  --btn-border: rgba(255, 255, 255, 0.35);
  --btn-bg-hover: rgba(255, 255, 255, 0.1);
  --btn-bg-active: rgba(255, 255, 255, 0.18);
}

/* ---------------------------------------------------------- controls */
.controls {
  min-width: 0;
  background: var(--white);
  border: 1px solid var(--line);
  border-radius: var(--r-panel);
}

.tabs {
  display: flex;
  gap: var(--s1);
  padding: var(--s2) var(--s3) 0;
  border-bottom: 1px solid var(--line);
  overflow-x: auto;
}

.tab {
  position: relative;
  display: inline-flex;
  align-items: center;
  gap: var(--s2);
  min-height: 44px;
  padding: var(--s2) var(--s4);
  border: 0;
  border-radius: var(--r-control) var(--r-control) 0 0;
  background: transparent;
  color: var(--ink-soft);
  font: inherit;
  font-size: var(--fs-sm);
  font-weight: 650;
  cursor: pointer;
}

.tab:hover {
  color: var(--brand-900);
  background: var(--brand-100);
}

.tab[aria-selected='true'] {
  color: var(--brand);
}

.tab[aria-selected='true']::after {
  content: '';
  position: absolute;
  left: var(--s3);
  right: var(--s3);
  bottom: -1px;
  height: 3px;
  border-radius: 3px 3px 0 0;
  background: var(--amber);
}

.tab-panel {
  padding: var(--s5);
}

.save-bar {
  position: sticky;
  bottom: 0;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: var(--s2) var(--s3);
  padding: var(--s3) var(--s5);
  border-top: 1px solid var(--line);
  background: #fbf8fa;
  border-radius: 0 0 var(--r-panel) var(--r-panel);
}

.save-bar.dirty {
  background: var(--amber-100);
  border-top-color: var(--amber-200);
}

@media (max-width: 900px) {
  .editor-grid {
    grid-template-columns: minmax(0, 1fr);
  }

  .stage {
    position: static;
  }

  .preview {
    max-height: 70vh;
    min-height: 0;
  }
}

@media (max-width: 640px) {
  .tab-panel {
    padding: var(--s4);
  }

  .save-bar {
    padding: var(--s3) var(--s4);
  }

  .stage {
    padding: var(--s3);
    margin: 0 calc(-1 * var(--s2));
  }
}
</style>
