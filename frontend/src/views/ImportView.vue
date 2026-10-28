<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref, useId } from 'vue'
import { RouterLink, useRouter } from 'vue-router'
import { ApiError, UploadAbortedError, api, uploadFile, type UploadHandle } from '../api/client'
import type { ProjectCreateIn, ProjectOptions } from '../api/types'
import ActionButton from '../components/ActionButton.vue'
import AppIcon from '../components/AppIcon.vue'
import ClipOptionsFields from '../components/ClipOptionsFields.vue'
import OpenAINotice from '../components/OpenAINotice.vue'
import { ACCEPT_ATTR, fileProblem } from '../lib/files'
import { fmtBytes, fmtMegabytes, fmtMinutes } from '../lib/format'
import { validateOptions, type ClipOptionValues } from '../lib/options'
import { youtubeUrlProblem } from '../lib/youtube'
import { notify } from '../stores/notify'
import { loadSystem, useSystem } from '../stores/system'

type SourceMode = 'youtube' | 'file'
type Phase = 'idle' | 'uploading' | 'creating'

const router = useRouter()
const uid = useId()
const { system, systemError } = useSystem()

const mode = ref<SourceMode>('youtube')
const url = ref('')
const showUrlProblem = ref(false)
const file = ref<File | null>(null)
const dragging = ref(false)
const fileInput = ref<HTMLInputElement | null>(null)
const options = ref<ClipOptionValues>({ clip_count: 5, min_seconds: 20, max_seconds: 60, captions_enabled: true })
const phase = ref<Phase>('idle')
const loaded = ref(0)
const total = ref(0)
const submitError = ref<string | null>(null)
const systemLoading = ref(true)

let upload: UploadHandle | null = null
// Reuse a finished upload if creating the project failed and the user tries again.
let finishedUpload: { key: string; id: string } | null = null

const limits = computed(() => system.value?.limits ?? null)
const storage = computed(() => system.value?.storage ?? null)
const openaiMissing = computed(() => system.value?.openai.configured === false)
const usableBytes = computed(() => (storage.value ? storage.value.free_bytes - storage.value.reserve_bytes : null))

const urlProblem = computed(() => youtubeUrlProblem(url.value))
const currentFileProblem = computed(() =>
  file.value
    ? fileProblem(file.value, { maxUploadMb: limits.value?.max_upload_mb ?? null, usableBytes: usableBytes.value })
    : null,
)
const optionsProblem = computed(() => validateOptions(options.value, limits.value))

const blockReason = computed<string | null>(() => {
  if (openaiMissing.value) return 'Add your OpenAI API key first (see the setup notice above).'
  if (mode.value === 'youtube' && urlProblem.value) return urlProblem.value
  if (mode.value === 'file') {
    if (!file.value) return 'Choose a video file first.'
    if (currentFileProblem.value) return currentFileProblem.value
  }
  return optionsProblem.value
})

const percent = computed(() => (total.value > 0 ? Math.min(100, Math.floor((loaded.value / total.value) * 100)) : 0))

function fileKey(f: File): string {
  return `${f.name}:${f.size}:${f.lastModified}`
}

function chooseFile(f: File | null | undefined) {
  if (!f) return
  file.value = f
  mode.value = 'file'
  submitError.value = null
}

function onPick(event: Event) {
  const input = event.target as HTMLInputElement
  chooseFile(input.files?.[0])
  input.value = ''
}

function onDragOver(event: DragEvent) {
  if (phase.value !== 'idle') return
  if (!event.dataTransfer?.types.includes('Files')) return
  event.preventDefault()
  event.dataTransfer.dropEffect = 'copy'
  dragging.value = true
}

function onDrop(event: DragEvent) {
  dragging.value = false
  if (phase.value !== 'idle') return
  const f = event.dataTransfer?.files?.[0]
  if (!f) return
  event.preventDefault()
  chooseFile(f)
}

function clearFile() {
  file.value = null
  submitError.value = null
}

function projectOptions(): ProjectOptions {
  return { ...options.value, language: 'en' }
}

async function uploadSelected(f: File): Promise<string> {
  const key = fileKey(f)
  if (finishedUpload?.key === key) return finishedUpload.id
  phase.value = 'uploading'
  loaded.value = 0
  total.value = f.size
  upload = uploadFile(f, (l, t) => {
    loaded.value = l
    total.value = t
  })
  try {
    const result = await upload.promise
    finishedUpload = { key, id: result.id }
    return result.id
  } finally {
    upload = null
  }
}

async function generate() {
  showUrlProblem.value = true
  if (blockReason.value || phase.value !== 'idle') return
  submitError.value = null
  try {
    let body: ProjectCreateIn
    if (mode.value === 'youtube') {
      body = { source: { kind: 'youtube', url: url.value.trim() }, options: projectOptions() }
    } else {
      const uploadId = await uploadSelected(file.value as File)
      body = { source: { kind: 'upload', upload_id: uploadId }, options: projectOptions() }
    }
    phase.value = 'creating'
    const project = await api.createProject(body)
    finishedUpload = null
    await router.push(`/projects/${project.id}`)
  } catch (err) {
    if (err instanceof UploadAbortedError) {
      notify('Upload cancelled.', 'info')
    } else {
      submitError.value = err instanceof ApiError ? err.message : 'Something went wrong. Try again.'
    }
  } finally {
    phase.value = 'idle'
  }
}

function cancelUpload() {
  upload?.abort()
}

// A file dropped next to the drop zone must not make the browser navigate away to it.
function swallowFileDrop(event: DragEvent) {
  if (event.dataTransfer?.types.includes('Files')) event.preventDefault()
}

onMounted(async () => {
  window.addEventListener('dragover', swallowFileDrop)
  window.addEventListener('drop', swallowFileDrop)
  await Promise.all([
    loadSystem(),
    api
      .getDefaults()
      .then((d) => {
        options.value = {
          clip_count: d.clip_count,
          min_seconds: d.min_seconds,
          max_seconds: d.max_seconds,
          captions_enabled: d.captions_enabled,
        }
      })
      .catch(() => undefined),
  ])
  systemLoading.value = false
})

onBeforeUnmount(() => {
  window.removeEventListener('dragover', swallowFileDrop)
  window.removeEventListener('drop', swallowFileDrop)
  upload?.abort()
})
</script>

<template>
  <div class="page">
    <div class="intro">
      <h1>Turn a long video into short clips</h1>
      <p>
        Paste a YouTube link or add a video file. Local Clipper finds the strongest spoken moments and cuts them into
        vertical 9:16 clips with captions, ready to review and edit.
      </p>
    </div>

    <OpenAINotice v-if="openaiMissing" class="setup-notice" />

    <form class="import-grid" novalidate @submit.prevent="generate">
      <div class="import-main">
        <section
          class="panel source-panel"
          :class="{ dragging }"
          aria-labelledby="source-h"
          @dragover="onDragOver"
          @dragleave.self="dragging = false"
          @drop="onDrop"
        >
          <h2 id="source-h" class="panel-title">Video</h2>
          <div class="seg source-switch" role="radiogroup" aria-label="Where the video comes from">
            <label class="seg-opt">
              <input v-model="mode" type="radio" :name="`${uid}-src`" value="youtube" :disabled="phase !== 'idle'" />
              <span><AppIcon name="link" :size="16" /> YouTube link</span>
            </label>
            <label class="seg-opt">
              <input v-model="mode" type="radio" :name="`${uid}-src`" value="file" :disabled="phase !== 'idle'" />
              <span><AppIcon name="upload" :size="16" /> Video file</span>
            </label>
          </div>

          <div v-if="mode === 'youtube'" class="field source-field">
            <label class="field-label" :for="`${uid}-url`">YouTube link</label>
            <input
              :id="`${uid}-url`"
              v-model="url"
              class="input"
              type="url"
              inputmode="url"
              autocomplete="off"
              spellcheck="false"
              placeholder="https://www.youtube.com/watch?v=…"
              :disabled="phase !== 'idle'"
              :aria-invalid="showUrlProblem && !!urlProblem ? 'true' : undefined"
              :aria-describedby="`${uid}-url-msg`"
              @blur="showUrlProblem = url.trim().length > 0"
            />
            <p v-if="showUrlProblem && urlProblem" :id="`${uid}-url-msg`" class="field-error">{{ urlProblem }}</p>
            <p v-else :id="`${uid}-url-msg`" class="field-hint">
              A single public video. Private, members-only and live videos cannot be imported.
            </p>
          </div>

          <div v-else class="source-field">
            <div v-if="file" class="file-card" :class="{ bad: !!currentFileProblem }">
              <AppIcon name="film" :size="22" />
              <div class="file-info">
                <p class="file-name">{{ file.name }}</p>
                <p class="small muted num">{{ fmtBytes(file.size) }}</p>
              </div>
              <div class="row">
                <button type="button" class="btn btn-sm" :disabled="phase !== 'idle'" @click="fileInput?.click()">
                  Change
                </button>
                <button
                  type="button"
                  class="btn btn-quiet btn-sm icon-btn"
                  :disabled="phase !== 'idle'"
                  aria-label="Remove file"
                  title="Remove file"
                  @click="clearFile"
                >
                  <AppIcon name="x" :size="16" />
                </button>
              </div>
            </div>
            <label v-else class="drop-zone" :class="{ dragging }">
              <input
                ref="fileInput"
                type="file"
                class="visually-hidden"
                :accept="ACCEPT_ATTR"
                :disabled="phase !== 'idle'"
                @change="onPick"
              />
              <AppIcon name="upload" :size="28" />
              <span class="drop-title">Drop a video here, or <u>choose a file</u></span>
              <span class="small muted">
                MP4, MOV, MKV, WebM and other common formats<template v-if="limits">
                  up to {{ fmtMegabytes(limits.max_upload_mb) }}</template
                >. The file is copied into Local Clipper; your original is not changed.
              </span>
            </label>
            <!-- Kept for "Change" when a file is already chosen. -->
            <input
              v-if="file"
              ref="fileInput"
              type="file"
              class="visually-hidden"
              tabindex="-1"
              aria-hidden="true"
              :accept="ACCEPT_ATTR"
              @change="onPick"
            />
            <p v-if="currentFileProblem" class="field-error file-error" role="alert">{{ currentFileProblem }}</p>
          </div>
        </section>

        <section class="panel" aria-labelledby="options-h">
          <h2 id="options-h" class="panel-title">Clips</h2>
          <ClipOptionsFields v-model="options" :limits="limits" :disabled="phase !== 'idle'" />
        </section>
      </div>

      <aside class="import-side" aria-labelledby="limits-h">
        <section class="panel side-panel">
          <h2 id="limits-h" class="panel-title">Before you start</h2>

          <p v-if="systemLoading" class="small muted">Checking limits…</p>
          <p v-else-if="!system" class="small field-error">
            {{ systemError || 'The limits could not be loaded.' }}
          </p>
          <template v-else>
            <dl class="limits">
              <div>
                <dt>Longest video</dt>
                <dd class="num">{{ fmtMinutes(system.limits.max_source_minutes) }}</dd>
              </div>
              <div>
                <dt>Largest upload</dt>
                <dd class="num">{{ fmtMegabytes(system.limits.max_upload_mb) }}</dd>
              </div>
              <div>
                <dt>Free disk space</dt>
                <dd class="num">{{ fmtBytes(system.storage.free_bytes) }}</dd>
              </div>
            </dl>
            <p class="small muted">
              {{ fmtBytes(system.storage.reserve_bytes) }} is kept free for processing. Long videos need room for the
              download, audio and rendered clips.
            </p>
            <div class="sent">
              <h3 class="field-label">Sent to OpenAI</h3>
              <ul class="small">
                <li v-for="line in system.openai.data_sent" :key="line">{{ line }}</li>
              </ul>
              <RouterLink to="/settings" class="small">Details and models in Settings</RouterLink>
            </div>
            <div v-if="!system.worker.online" class="notice notice-warn" role="status">
              <AppIcon name="alert" />
              <p>
                The background worker is not running. You can still start a project, but processing begins only when
                Local Clipper is restarted.
              </p>
            </div>
          </template>

          <div class="go">
            <div v-if="phase === 'uploading'" class="upload-progress" role="group" aria-label="Upload progress">
              <div class="row-between small">
                <span class="file-name">Uploading {{ file?.name }}</span>
                <span class="num">{{ percent }}%</span>
              </div>
              <progress :value="loaded" :max="total || 1" :aria-valuetext="`${fmtBytes(loaded)} of ${fmtBytes(total)}`">
                {{ percent }}%
              </progress>
              <p class="small muted num" aria-live="polite">{{ fmtBytes(loaded) }} of {{ fmtBytes(total) }}</p>
              <button type="button" class="btn btn-danger btn-block" @click="cancelUpload">
                <AppIcon name="x" :size="16" /> Cancel upload
              </button>
            </div>
            <ActionButton
              v-else
              type="submit"
              variant="primary"
              size="lg"
              block
              :loading="phase === 'creating'"
              loading-label="Creating project…"
              :disabled="!!blockReason"
              :reason="blockReason"
            >
              <AppIcon name="scissors" :size="18" /> Generate clips
            </ActionButton>
            <p v-if="blockReason && phase === 'idle'" class="small muted go-hint">{{ blockReason }}</p>
            <div v-if="submitError" class="notice notice-error" role="alert">
              <AppIcon name="alert" />
              <p>{{ submitError }}</p>
            </div>
          </div>
        </section>
      </aside>
    </form>
  </div>
</template>

<style scoped>
.intro {
  max-width: 68ch;
  margin-bottom: var(--s5);
}

.intro h1 {
  font-size: var(--fs-3xl);
  letter-spacing: -0.02em;
}

.intro p {
  margin-top: var(--s3);
  color: var(--ink-soft);
  font-size: var(--fs-lg);
}

.setup-notice {
  margin-bottom: var(--s5);
}

.import-grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 340px;
  gap: var(--s5);
  align-items: start;
}

.import-main {
  display: flex;
  flex-direction: column;
  gap: var(--s5);
  min-width: 0;
}

.source-panel.dragging {
  border-color: var(--amber);
  box-shadow: 0 0 0 3px var(--amber-200);
}

.source-switch {
  margin-bottom: var(--s4);
}

.source-field {
  margin-top: var(--s1);
}

.drop-zone {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--s2);
  padding: var(--s6) var(--s4);
  border: 2px dashed var(--line-strong);
  border-radius: var(--r-panel);
  background: var(--mist);
  color: var(--brand);
  text-align: center;
  cursor: pointer;
}

.drop-zone:hover,
.drop-zone.dragging {
  border-color: var(--brand);
  background: var(--brand-100);
}

.drop-zone:focus-within {
  outline: 2px solid var(--brand-900);
  outline-offset: 2px;
  box-shadow: 0 0 0 5px var(--amber);
}

.drop-title {
  font-weight: 700;
  color: var(--brand-900);
}

.file-card {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: var(--s3);
  padding: var(--s3) var(--s4);
  border: 1px solid var(--line-strong);
  border-radius: var(--r-control);
  background: var(--mist);
  color: var(--brand);
}

.file-card.bad {
  border-color: var(--danger);
}

.file-info {
  flex: 1;
  min-width: 0;
}

.file-name {
  font-weight: 650;
  color: var(--brand-900);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  min-width: 0;
}

.file-error {
  margin-top: var(--s2);
}

.import-side {
  position: sticky;
  top: calc(var(--header-h) + var(--s4));
}

.side-panel {
  display: flex;
  flex-direction: column;
  gap: var(--s4);
  border-top: 4px solid var(--brand);
}

.side-panel .panel-title {
  margin-bottom: 0;
}

.limits {
  display: grid;
  gap: var(--s2);
  margin: 0;
}

.limits div {
  display: flex;
  justify-content: space-between;
  gap: var(--s3);
  padding-bottom: var(--s2);
  border-bottom: 1px solid var(--line);
  font-size: var(--fs-sm);
}

.limits dt {
  color: var(--ink-soft);
}

.limits dd {
  margin: 0;
  font-weight: 700;
  color: var(--brand-900);
}

.sent ul {
  margin: var(--s2) 0;
  padding-left: var(--s5);
  color: var(--ink);
}

.sent li + li {
  margin-top: var(--s1);
}

.go {
  display: flex;
  flex-direction: column;
  gap: var(--s3);
  padding-top: var(--s2);
}

.go-hint {
  text-align: center;
}

.upload-progress {
  display: flex;
  flex-direction: column;
  gap: var(--s2);
}

.upload-progress .file-name {
  flex: 1;
}

progress {
  width: 100%;
  height: 10px;
  appearance: none;
  border: 0;
  border-radius: var(--r-pill);
  background: var(--brand-100);
  overflow: hidden;
}

progress::-webkit-progress-bar {
  background: var(--brand-100);
  border-radius: var(--r-pill);
}

progress::-webkit-progress-value {
  background: var(--brand);
  border-radius: var(--r-pill);
}

progress::-moz-progress-bar {
  background: var(--brand);
  border-radius: var(--r-pill);
}

@media (max-width: 960px) {
  .import-grid {
    grid-template-columns: minmax(0, 1fr);
  }

  .import-side {
    position: static;
  }
}

@media (max-width: 640px) {
  .intro h1 {
    font-size: var(--fs-2xl);
  }

  .intro p {
    font-size: var(--fs-md);
  }
}
</style>
