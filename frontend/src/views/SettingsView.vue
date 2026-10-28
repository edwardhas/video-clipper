<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useRouter } from 'vue-router'
import { ApiError, api } from '../api/client'
import type { CaptionStyle, DefaultsModel } from '../api/types'
import ActionButton from '../components/ActionButton.vue'
import AppIcon from '../components/AppIcon.vue'
import CaptionStyleFields from '../components/CaptionStyleFields.vue'
import ClipOptionsFields from '../components/ClipOptionsFields.vue'
import LoadState from '../components/LoadState.vue'
import { fmtBytes, fmtDuration } from '../lib/format'
import { validateOptions, type ClipOptionValues } from '../lib/options'
import { notify } from '../stores/notify'
import { useSession } from '../stores/session'
import { loadSystem, useSystem } from '../stores/system'

const router = useRouter()
const { session } = useSession()
const { system, systemError } = useSystem()

const saved = ref<DefaultsModel | null>(null)
const options = ref<ClipOptionValues | null>(null)
const style = ref<CaptionStyle | null>(null)
const loading = ref(true)
const loadError = ref<string | null>(null)
const saving = ref(false)
const signingOut = ref(false)
const systemLoading = ref(true)

const problem = computed(() => (options.value ? validateOptions(options.value, system.value?.limits) : null))
const dirty = computed(() => {
  const s = saved.value
  const o = options.value
  const st = style.value
  if (!s || !o || !st) return false
  return (
    s.clip_count !== o.clip_count ||
    s.min_seconds !== o.min_seconds ||
    s.max_seconds !== o.max_seconds ||
    s.captions_enabled !== o.captions_enabled ||
    JSON.stringify(s.caption_style) !== JSON.stringify(st)
  )
})
const saveReason = computed(() => problem.value ?? (dirty.value ? null : 'There are no changes to save.'))

function applyDefaults(d: DefaultsModel) {
  saved.value = d
  options.value = {
    clip_count: d.clip_count,
    min_seconds: d.min_seconds,
    max_seconds: d.max_seconds,
    captions_enabled: d.captions_enabled,
  }
  style.value = { ...d.caption_style }
}

async function loadDefaults() {
  loading.value = true
  loadError.value = null
  try {
    applyDefaults(await api.getDefaults())
  } catch (err) {
    loadError.value = err instanceof ApiError ? err.message : 'Something went wrong.'
  } finally {
    loading.value = false
  }
}

async function save() {
  if (!options.value || !style.value || saveReason.value) return
  saving.value = true
  try {
    const body: DefaultsModel = { ...options.value, caption_style: { ...style.value } }
    applyDefaults(await api.saveDefaults(body))
    notify('Defaults saved. New projects will use them.', 'success')
  } catch (err) {
    notify(err instanceof ApiError ? err.message : 'Could not save the defaults.', 'error')
  } finally {
    saving.value = false
  }
}

async function signOut() {
  signingOut.value = true
  try {
    await api.logout()
    await router.replace({ name: 'login' })
  } catch (err) {
    notify(err instanceof ApiError ? err.message : 'Could not sign out.', 'error')
  } finally {
    signingOut.value = false
  }
}

async function refreshSystem() {
  systemLoading.value = true
  await loadSystem()
  systemLoading.value = false
}

onMounted(() => {
  void loadDefaults()
  void refreshSystem()
})

const heartbeat = computed(() => {
  const age = system.value?.worker.last_heartbeat_age_s
  return age == null ? null : `last seen ${fmtDuration(age)} ago`
})
</script>

<template>
  <div class="page settings-page">
    <div class="page-head">
      <div>
        <h1>Settings</h1>
        <p>Defaults for new projects, and what this installation uses.</p>
      </div>
      <ActionButton
        v-if="session?.auth_required && session.authenticated"
        :loading="signingOut"
        loading-label="Signing out…"
        @click="signOut"
      >
        Sign out<template v-if="session.username"> ({{ session.username }})</template>
      </ActionButton>
    </div>

    <div class="settings-grid">
      <section class="panel" aria-labelledby="defaults-h">
        <h2 id="defaults-h" class="panel-title">Defaults for new projects</h2>
        <LoadState v-if="loading || loadError" :loading="loading" :error="loadError" what="the defaults" @retry="loadDefaults" />
        <form v-else-if="options && style" class="stack defaults-form" @submit.prevent="save">
          <ClipOptionsFields v-model="options" :limits="system?.limits ?? null" :disabled="saving" />
          <div class="sub">
            <h3 class="sub-title">Caption style</h3>
            <CaptionStyleFields v-model="style" :disabled="saving" />
          </div>
          <p v-if="problem" class="field-error" role="alert">{{ problem }}</p>
          <div class="row">
            <ActionButton
              type="submit"
              variant="primary"
              :loading="saving"
              loading-label="Saving…"
              :disabled="!!saveReason"
              :reason="saveReason"
            >
              Save defaults
            </ActionButton>
            <ActionButton
              variant="quiet"
              :disabled="!dirty || saving"
              reason="Nothing to undo"
              @click="saved && applyDefaults(saved)"
            >
              Undo changes
            </ActionButton>
          </div>
          <p class="small muted">Per-project clean-up and deletion are on each project’s page.</p>
        </form>
      </section>

      <div class="system-col">
        <LoadState
          v-if="systemLoading || (!system && systemError)"
          :loading="systemLoading"
          :error="systemError"
          what="system information"
          @retry="refreshSystem"
        />
        <template v-else-if="system">
          <section class="panel" aria-labelledby="openai-h">
            <h2 id="openai-h" class="panel-title">OpenAI</h2>
            <p class="key-state" :class="system.openai.configured ? 'ok' : 'missing'">
              <AppIcon :name="system.openai.configured ? 'check' : 'alert'" :size="16" />
              <template v-if="system.openai.configured">API key configured</template>
              <template v-else>No API key configured</template>
            </p>
            <p v-if="!system.openai.configured" class="small">
              Add <code>OPENAI_API_KEY</code> to the <code>.env</code> file in the project folder, then restart Local
              Clipper. The key is never entered or shown in this page.
            </p>
            <h3 class="sub-title">What is sent to OpenAI</h3>
            <ul class="sent small">
              <li v-for="line in system.openai.data_sent" :key="line">{{ line }}</li>
            </ul>
            <p class="small muted">{{ system.openai.transcription_note }}</p>
            <h3 class="sub-title">Models in use</h3>
            <dl class="facts">
              <dt>Transcription</dt>
              <dd><code>{{ system.openai.transcribe_model }}</code></dd>
              <dt>Finding moments</dt>
              <dd><code>{{ system.openai.select_model }}</code></dd>
              <dt>Reviewing and ranking</dt>
              <dd><code>{{ system.openai.review_model }}</code></dd>
            </dl>
          </section>

          <section class="panel" aria-labelledby="storage-h">
            <h2 id="storage-h" class="panel-title">Storage</h2>
            <dl class="facts">
              <dt>Location</dt>
              <dd>{{ system.storage.location_label }}</dd>
              <dt>Folder</dt>
              <dd><code class="path">{{ system.storage.data_dir_display }}</code></dd>
              <dt>Free space</dt>
              <dd class="num">{{ fmtBytes(system.storage.free_bytes) }}</dd>
              <dt>Used by Local Clipper</dt>
              <dd class="num">{{ fmtBytes(system.storage.used_bytes) }}</dd>
              <dt>Kept free</dt>
              <dd class="num">{{ fmtBytes(system.storage.reserve_bytes) }} reserved for processing</dd>
            </dl>
          </section>

          <section class="panel" aria-labelledby="system-h">
            <h2 id="system-h" class="panel-title">This installation</h2>
            <dl class="facts">
              <dt>Mode</dt>
              <dd>{{ system.mode === 'local' ? 'Local (this computer)' : 'Hosted (server)' }}</dd>
              <dt>Version</dt>
              <dd class="num">{{ system.version }}</dd>
              <dt>Background worker</dt>
              <dd>
                <span class="badge" :class="system.worker.online ? 'badge-ok' : 'badge-danger'">
                  {{ system.worker.online ? 'Online' : 'Offline' }}
                </span>
                <span v-if="heartbeat" class="xsmall muted heartbeat">{{ heartbeat }}</span>
              </dd>
              <dt>FFmpeg</dt>
              <dd>{{ system.tools.ffmpeg_version ?? 'Not found' }}</dd>
              <dt>Video encoder</dt>
              <dd>{{ system.tools.encoder ?? 'Not checked yet (the worker checks it when it starts)' }}</dd>
              <dt>yt-dlp</dt>
              <dd>{{ system.tools.ytdlp_version ?? 'Not found' }}</dd>
              <dt>JavaScript runtime</dt>
              <dd>{{ system.tools.js_runtime ?? 'Not found (needed for YouTube downloads)' }}</dd>
              <dt>Face detector</dt>
              <dd>{{ system.tools.face_detector ?? 'Not available (clips use a centred crop or the full frame)' }}</dd>
            </dl>
          </section>
        </template>
      </div>
    </div>
  </div>
</template>

<style scoped>
.settings-grid {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
  gap: var(--s5);
  align-items: start;
}

.system-col {
  display: flex;
  flex-direction: column;
  gap: var(--s5);
  min-width: 0;
}

.sub {
  padding-top: var(--s4);
  border-top: 1px solid var(--line);
}

.sub-title {
  font-size: var(--fs-md);
  margin: var(--s4) 0 var(--s2);
}

.sub > .sub-title {
  margin-top: 0;
  margin-bottom: var(--s3);
}

.key-state {
  display: inline-flex;
  align-items: center;
  gap: var(--s2);
  padding: var(--s1) var(--s3);
  border-radius: var(--r-control);
  font-weight: 700;
  font-size: var(--fs-sm);
  margin-bottom: var(--s2);
}

.key-state.ok {
  background: var(--ok-bg);
  color: var(--ok);
}

.key-state.missing {
  background: var(--amber-100);
  color: #6a3a00;
}

.sent {
  margin: 0 0 var(--s2);
  padding-left: var(--s5);
}

.sent li + li {
  margin-top: var(--s1);
}

.path {
  word-break: break-all;
}

.heartbeat {
  margin-left: var(--s2);
}

@media (max-width: 960px) {
  .settings-grid {
    grid-template-columns: minmax(0, 1fr);
  }
}
</style>
