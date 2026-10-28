<script setup lang="ts">
import { computed, onMounted, ref, watch } from 'vue'
import { RouterLink, useRouter } from 'vue-router'
import { ApiError, api, mediaUrls } from '../api/client'
import type { ProjectOut, ProjectStatus } from '../api/types'
import ActionButton from '../components/ActionButton.vue'
import AppIcon from '../components/AppIcon.vue'
import ClipCard from '../components/ClipCard.vue'
import LoadState from '../components/LoadState.vue'
import ReselectDialog from '../components/ReselectDialog.vue'
import StageList from '../components/StageList.vue'
import StatusBadge from '../components/StatusBadge.vue'
import { fmtDateTime, fmtDuration, fmtRelative, plural } from '../lib/format'
import { usePoller } from '../lib/poller'
import {
  PROJECT_STATUS,
  isClipExportable,
  isProjectActive,
  projectCancelling,
  projectHasActiveWork,
  projectNeedsPolling,
} from '../lib/status'
import { confirmAction } from '../stores/confirm'
import { notify } from '../stores/notify'
import { loadSystem, useSystem } from '../stores/system'

const props = defineProps<{ id: string }>()
const router = useRouter()
const { system } = useSystem()

const project = ref<ProjectOut | null>(null)
const loading = ref(true)
const loadError = ref<string | null>(null)
const notFound = ref(false)
const acting = ref<null | 'cancel' | 'resume' | 'delete' | 'cleanup'>(null)
const reselectOpen = ref(false)
const selected = ref<Set<string>>(new Set())
const includeSrt = ref(true)
const liveMessage = ref('')

const active = computed(() => !!project.value && isProjectActive(project.value.status))
// Analysis or a clip render: both block delete/cleanup/reselect and are stopped by Cancel.
const hasActiveWork = computed(() => !!project.value && projectHasActiveWork(project.value))
const cancelling = computed(() => !!project.value && projectCancelling(project.value))
const job = computed(() => project.value?.job ?? null)
// Only clips the ZIP export includes (render state 'ready'); a stale or re-rendering clip still
// has its previous video but the export would leave it out.
const readyClips = computed(() => project.value?.clips.filter((c) => isClipExportable(c.render)) ?? [])
const selectedIds = computed(() => readyClips.value.filter((c) => selected.value.has(c.id)).map((c) => c.id))
const allSelected = computed(() => readyClips.value.length > 0 && selectedIds.value.length === readyClips.value.length)
const zipUrl = computed(() =>
  project.value && selectedIds.value.length
    ? mediaUrls.exportZip(project.value.id, selectedIds.value, includeSrt.value)
    : null,
)
const showStagesOpen = computed(() => {
  const s = project.value?.status
  return s === 'queued' || s === 'running' || s === 'failed' || s === 'interrupted' || s === 'cancelled'
})
const resumeLabel = computed(() => (project.value?.status === 'failed' ? 'Try again' : 'Resume'))
const problemText = computed(() => {
  const p = project.value
  if (!p) return null
  if (p.status === 'failed') return p.status_detail || p.job?.error_message || 'Processing stopped because of an error.'
  if (p.status === 'interrupted') {
    return p.status_detail || 'Processing was interrupted (the app or worker stopped). Resume to continue where it left off.'
  }
  if (p.status === 'cancelled') return p.status_detail || 'Processing was cancelled.'
  return null
})
const workerOffline = computed(() => !!system.value && !system.value.worker.online)

function setSelected(id: string, on: boolean) {
  const next = new Set(selected.value)
  if (on) next.add(id)
  else next.delete(id)
  selected.value = next
}

function toggleAll() {
  selected.value = allSelected.value ? new Set() : new Set(readyClips.value.map((c) => c.id))
}

let lastStatus: ProjectStatus | null = null
function apply(next: ProjectOut) {
  if (lastStatus && lastStatus !== next.status) {
    liveMessage.value = `Status: ${PROJECT_STATUS[next.status].label}.`
    if (next.status === 'ready') notify(`${plural(next.clips.length, 'clip')} ready.`, 'success')
  }
  lastStatus = next.status
  project.value = next
  // Drop clips that can no longer be exported (e.g. edited or re-rendering) from the selection.
  const exportable = new Set(next.clips.filter((c) => isClipExportable(c.render)).map((c) => c.id))
  if ([...selected.value].some((id) => !exportable.has(id))) {
    selected.value = new Set([...selected.value].filter((id) => exportable.has(id)))
  }
}

const poller = usePoller(async () => {
  try {
    const next = await api.getProject(props.id)
    apply(next)
    return projectNeedsPolling(next)
  } catch (err) {
    if (err instanceof ApiError && err.status === 404) {
      notFound.value = true
      return false
    }
    return true
  }
}, 1500)

async function load() {
  loading.value = true
  loadError.value = null
  notFound.value = false
  try {
    const p = await api.getProject(props.id)
    apply(p)
    if (projectNeedsPolling(p)) void poller.kick()
  } catch (err) {
    if (err instanceof ApiError && err.status === 404) notFound.value = true
    else loadError.value = err instanceof ApiError ? err.message : 'Something went wrong.'
  } finally {
    loading.value = false
  }
}

async function runAction(kind: 'cancel' | 'resume' | 'cleanup', fn: () => Promise<unknown>, done: string) {
  acting.value = kind
  try {
    await fn()
    notify(done, 'success')
    await poller.kick()
  } catch (err) {
    notify(err instanceof ApiError ? err.message : 'That did not work. Try again.', 'error')
  } finally {
    acting.value = null
  }
}

function cancel() {
  void runAction('cancel', () => api.cancelProject(props.id), 'Cancelling. Work stops at the next safe point.')
}

function resume() {
  void runAction('resume', () => api.resumeProject(props.id), 'Processing restarted. Finished steps are reused.')
}

async function cleanup() {
  const ok = await confirmAction({
    title: 'Free up disk space?',
    message:
      'This removes the project’s working files (such as extracted audio and analysis frames). The source video and the rendered clips stay.',
    confirmLabel: 'Remove working files',
  })
  if (ok) void runAction('cleanup', () => api.cleanupProject(props.id), 'Working files removed.')
}

async function remove() {
  if (!project.value) return
  const ok = await confirmAction({
    title: 'Delete this project?',
    message: `“${project.value.title}” its source video and all of its clips will be deleted from Local Clipper’s storage. This cannot be undone.`,
    confirmLabel: 'Delete project',
    danger: true,
  })
  if (!ok) return
  acting.value = 'delete'
  try {
    await api.deleteProject(props.id)
    notify('Project deleted.', 'success')
    await router.push('/projects')
  } catch (err) {
    notify(err instanceof ApiError ? err.message : 'Could not delete the project.', 'error')
  } finally {
    acting.value = null
  }
}

onMounted(() => {
  void loadSystem()
  void load()
})

watch(
  () => project.value?.title,
  (title) => {
    if (title) document.title = `${title} – Local Clipper`
  },
)

const sourceLabel = computed(() => (project.value?.source_kind === 'youtube' ? 'YouTube' : 'Uploaded file'))
const cleanupReason = computed(() =>
  project.value?.actions.can_delete ? null : 'Available when processing has stopped.',
)
</script>

<template>
  <div class="page">
    <RouterLink to="/projects" class="back-link"><AppIcon name="chevron-left" :size="16" /> All projects</RouterLink>

    <LoadState v-if="loading || loadError" :loading="loading" :error="loadError" what="this project" @retry="load" />

    <div v-else-if="notFound" class="panel empty">
      <h1>This project no longer exists</h1>
      <p class="muted">It may have been deleted.</p>
      <RouterLink to="/" class="btn btn-primary">Start a new project</RouterLink>
    </div>

    <template v-else-if="project">
      <header class="project-head">
        <div class="project-title">
          <div class="title-row">
            <h1>{{ project.title }}</h1>
            <StatusBadge :status="project.status" />
          </div>
          <dl class="project-meta">
            <div>
              <dt>Source</dt>
              <dd>
                <a v-if="project.source_url" :href="project.source_url" target="_blank" rel="noopener noreferrer">
                  {{ sourceLabel }}
                </a>
                <template v-else>{{ sourceLabel }}</template>
              </dd>
            </div>
            <div>
              <dt>Length</dt>
              <dd class="num">{{ fmtDuration(project.duration) }}</dd>
            </div>
            <div>
              <dt>Created</dt>
              <dd>
                <time :datetime="new Date(project.created_at * 1000).toISOString()" :title="fmtDateTime(project.created_at)">
                  {{ fmtRelative(project.created_at) }}
                </time>
              </dd>
            </div>
            <div>
              <dt>Clips</dt>
              <dd class="num">{{ project.clips.length }}</dd>
            </div>
          </dl>
        </div>
        <div class="head-actions">
          <ActionButton
            v-if="project.actions.can_reselect || project.status === 'ready' || project.status === 'no_clips'"
            :disabled="!project.actions.can_reselect"
            reason="Available once the speech has been transcribed and nothing else is running."
            @click="reselectOpen = true"
          >
            <AppIcon name="scissors" :size="16" /> Find different clips
          </ActionButton>
          <ActionButton
            variant="quiet"
            :loading="acting === 'cleanup'"
            loading-label="Removing…"
            :disabled="!!cleanupReason || acting !== null"
            :reason="cleanupReason"
            @click="cleanup"
          >
            Free up space
          </ActionButton>
          <ActionButton
            variant="danger"
            :loading="acting === 'delete'"
            loading-label="Deleting…"
            :disabled="!project.actions.can_delete || acting !== null"
            reason="Stop processing before deleting the project."
            @click="remove"
          >
            <AppIcon name="trash" :size="16" /> Delete
          </ActionButton>
        </div>
      </header>

      <p class="visually-hidden" aria-live="polite">{{ liveMessage }}</p>

      <!-- Processing status -->
      <section class="panel status-panel" :class="{ open: showStagesOpen }" aria-labelledby="processing-h">
        <div class="row-between">
          <h2 id="processing-h" class="panel-title">Processing</h2>
          <div class="row">
            <ActionButton
              v-if="hasActiveWork"
              variant="danger"
              size="sm"
              :loading="acting === 'cancel' || cancelling"
              :loading-label="'Cancelling…'"
              :disabled="!project.actions.can_cancel"
              reason="Cancellation has already been requested."
              @click="cancel"
            >
              <AppIcon name="stop" :size="14" /> Cancel
            </ActionButton>
            <ActionButton
              v-if="project.actions.can_resume"
              variant="primary"
              size="sm"
              :loading="acting === 'resume'"
              loading-label="Starting…"
              @click="resume"
            >
              <AppIcon name="refresh" :size="14" /> {{ resumeLabel }}
            </ActionButton>
          </div>
        </div>

        <div v-if="hasActiveWork && workerOffline" class="notice notice-warn status-note" role="status">
          <AppIcon name="alert" />
          <p>
            The background worker is not running, so nothing will move forward. Restart Local Clipper with its start
            script to continue.
          </p>
        </div>
        <div v-if="problemText" class="notice status-note" :class="project.status === 'failed' ? 'notice-error' : 'notice-warn'" role="status">
          <AppIcon name="alert" />
          <div class="notice-body">
            <p>{{ problemText }}</p>
            <p v-if="project.actions.can_resume" class="small muted">
              {{ resumeLabel }} reuses every step that already finished.
            </p>
          </div>
        </div>
        <p v-if="project.status === 'queued'" class="small muted status-note">
          Waiting for the worker to pick this up. Projects are processed one at a time.
        </p>

        <StageList v-if="showStagesOpen" :stages="project.stages" :job="job" />
        <details v-else class="stages-details">
          <summary>Show processing steps</summary>
          <StageList :stages="project.stages" :job="job" />
          <p v-if="project.usage.requests > 0" class="usage small muted">
            OpenAI usage: {{ plural(project.usage.requests, 'request') }},
            {{ project.usage.input_tokens.toLocaleString() }} input and
            {{ project.usage.output_tokens.toLocaleString() }} output tokens,
            {{ fmtDuration(project.usage.audio_seconds) }} of audio transcribed.
            <template v-if="project.usage.estimated_cost_usd != null">
              Estimated cost ${{ project.usage.estimated_cost_usd.toFixed(2) }}.
            </template>
            {{ project.usage.cost_note }}
          </p>
        </details>
      </section>

      <!-- Results -->
      <section v-if="project.clips.length" class="results" aria-labelledby="results-h">
        <div class="results-head">
          <div>
            <h2 id="results-h">Clips</h2>
            <p class="small muted">Ranked best first. Open a clip to trim it, fix captions or change the framing.</p>
            <p v-if="project.status === 'ready' && project.status_detail" class="small results-note">
              {{ project.status_detail }}
            </p>
          </div>
          <div class="toolbar">
            <label class="check">
              <input
                type="checkbox"
                :checked="allSelected"
                :indeterminate="selectedIds.length > 0 && !allSelected"
                :disabled="!readyClips.length"
                @change="toggleAll"
              />
              Select all
            </label>
            <label class="check">
              <input v-model="includeSrt" type="checkbox" />
              Include SRT
            </label>
            <a v-if="zipUrl" :href="zipUrl" class="btn btn-primary" download>
              <AppIcon name="download" :size="16" /> Download selected (ZIP)
              <span class="count num">{{ selectedIds.length }}</span>
            </a>
            <button
              v-else
              type="button"
              class="btn btn-primary"
              aria-disabled="true"
              title="Select at least one up-to-date rendered clip first"
            >
              <AppIcon name="download" :size="16" /> Download selected (ZIP)
            </button>
          </div>
        </div>
        <div class="clip-grid">
          <ClipCard
            v-for="c in project.clips"
            :key="c.id"
            :clip="c"
            :selected="selected.has(c.id)"
            @update:selected="setSelected(c.id, $event)"
          />
        </div>
      </section>

      <section v-else-if="project.status === 'no_clips'" class="panel empty-results" aria-labelledby="noclips-h">
        <h2 id="noclips-h">No clips were made from this video</h2>
        <p>{{ project.status_detail || 'No moments were strong enough to stand on their own as clips.' }}</p>
        <p class="small muted">
          Videos where one or more people talk clearly work best. You can try different clip lengths with “Find
          different clips” when the transcript is available.
        </p>
      </section>

      <section v-else-if="active" class="panel empty-results">
        <p class="muted">Clips appear here as soon as they are ready.</p>
      </section>
    </template>

    <ReselectDialog
      v-if="project"
      :open="reselectOpen"
      :project-id="project.id"
      :options="project.options"
      :has-clips="project.clips.length > 0"
      :limits="system?.limits ?? null"
      @close="reselectOpen = false"
      @started="poller.kick()"
    />
  </div>
</template>

<style scoped>
.empty {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: var(--s3);
}

.project-head {
  display: flex;
  flex-wrap: wrap;
  align-items: flex-start;
  justify-content: space-between;
  gap: var(--s4);
  margin-bottom: var(--s5);
}

.project-title {
  min-width: 0;
  flex: 1 1 420px;
}

.title-row {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: var(--s2) var(--s3);
}

.project-meta {
  display: flex;
  flex-wrap: wrap;
  gap: var(--s2) var(--s5);
  margin: var(--s3) 0 0;
  font-size: var(--fs-sm);
}

.project-meta dt {
  font-size: var(--fs-xs);
  color: var(--ink-soft);
}

.project-meta dd {
  margin: 0;
  font-weight: 650;
  color: var(--brand-900);
}

.head-actions {
  display: flex;
  flex-wrap: wrap;
  gap: var(--s2);
}

.status-panel {
  margin-bottom: var(--s6);
}

.status-panel:not(.open) {
  padding-top: var(--s4);
  padding-bottom: var(--s4);
}

.status-panel:not(.open) .panel-title {
  margin-bottom: 0;
}

.status-note {
  margin: var(--s3) 0;
}

.stages-details summary {
  margin-top: var(--s2);
  cursor: pointer;
  font-size: var(--fs-sm);
  font-weight: 650;
  color: var(--brand);
  border-radius: 4px;
  width: fit-content;
}

.stages-details[open] summary {
  margin-bottom: var(--s3);
}

.usage {
  margin-top: var(--s3);
  max-width: 75ch;
}

.results-head {
  display: flex;
  flex-wrap: wrap;
  align-items: flex-end;
  justify-content: space-between;
  gap: var(--s3) var(--s5);
  margin-bottom: var(--s4);
}

.results-note {
  margin-top: var(--s1);
  color: #6a3a00;
}

.toolbar {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: var(--s3) var(--s4);
}

.count {
  min-width: 22px;
  padding: 0 6px;
  border-radius: var(--r-pill);
  background: var(--amber);
  color: var(--brand-900);
  font-size: var(--fs-xs);
  font-weight: 800;
  text-align: center;
}

.clip-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(250px, 1fr));
  gap: var(--s4);
}

.empty-results {
  max-width: 70ch;
}

.empty-results h2 {
  margin-bottom: var(--s2);
}

.empty-results > * + * {
  margin-top: var(--s2);
}

@media (max-width: 400px) {
  .clip-grid {
    grid-template-columns: minmax(0, 1fr);
  }

  .toolbar .btn {
    width: 100%;
  }
}
</style>
