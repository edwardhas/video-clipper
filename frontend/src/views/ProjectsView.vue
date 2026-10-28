<script setup lang="ts">
import { onMounted, ref } from 'vue'
import { RouterLink } from 'vue-router'
import { ApiError, api } from '../api/client'
import type { ProjectSummaryOut } from '../api/types'
import AppIcon from '../components/AppIcon.vue'
import LoadState from '../components/LoadState.vue'
import StatusBadge from '../components/StatusBadge.vue'
import { fmtDateTime, fmtDuration, fmtRelative, plural } from '../lib/format'
import { usePoller } from '../lib/poller'
import { isProjectActive } from '../lib/status'

const projects = ref<ProjectSummaryOut[]>([])
const loading = ref(true)
const loadError = ref<string | null>(null)
const brokenThumbs = ref<Set<string>>(new Set())

function anyActive(list: ProjectSummaryOut[]): boolean {
  return list.some((p) => isProjectActive(p.status))
}

// Refresh while something is processing so status badges stay current.
const poller = usePoller(async () => {
  try {
    projects.value = await api.listProjects()
  } catch {
    return true
  }
  return anyActive(projects.value)
}, 3000)

async function load() {
  loading.value = true
  loadError.value = null
  try {
    projects.value = await api.listProjects()
    if (anyActive(projects.value)) void poller.kick()
  } catch (err) {
    loadError.value = err instanceof ApiError ? err.message : 'Something went wrong.'
  } finally {
    loading.value = false
  }
}

function thumbFailed(id: string) {
  brokenThumbs.value = new Set(brokenThumbs.value).add(id)
}

onMounted(load)
</script>

<template>
  <div class="page">
    <div class="page-head">
      <div>
        <h1>Projects</h1>
        <p>Every video you have imported, newest first.</p>
      </div>
      <RouterLink to="/" class="btn btn-primary"><AppIcon name="scissors" :size="16" /> New project</RouterLink>
    </div>

    <LoadState v-if="loading || loadError" :loading="loading" :error="loadError" what="your projects" @retry="load" />

    <div v-else-if="!projects.length" class="panel empty">
      <AppIcon name="film" :size="32" />
      <h2>No projects yet</h2>
      <p class="muted">Start with a YouTube link or a video file. Your projects and their clips will be listed here.</p>
      <RouterLink to="/" class="btn btn-primary">Make your first clips</RouterLink>
    </div>

    <ul v-else class="project-list">
      <li v-for="p in projects" :key="p.id">
        <RouterLink :to="`/projects/${p.id}`" class="project-row">
          <span class="thumb">
            <img
              v-if="p.thumb_url && !brokenThumbs.has(p.id)"
              :src="p.thumb_url"
              alt=""
              loading="lazy"
              decoding="async"
              @error="thumbFailed(p.id)"
            />
            <AppIcon v-else name="film" :size="22" />
            <span v-if="p.duration != null" class="thumb-time num">{{ fmtDuration(p.duration) }}</span>
          </span>
          <span class="row-main">
            <span class="row-title">{{ p.title }}</span>
            <span class="row-meta">
              <span>
                <AppIcon :name="p.source_kind === 'youtube' ? 'link' : 'upload'" :size="14" />
                {{ p.source_kind === 'youtube' ? 'YouTube' : 'Uploaded file' }}
              </span>
              <span class="num">{{ plural(p.clip_count, 'clip') }}</span>
              <time :datetime="new Date(p.created_at * 1000).toISOString()" :title="fmtDateTime(p.created_at)">
                {{ fmtRelative(p.created_at) }}
              </time>
            </span>
            <span v-if="p.status_detail && p.status !== 'ready' && p.status !== 'running'" class="row-detail">
              {{ p.status_detail }}
            </span>
          </span>
          <span class="row-status">
            <StatusBadge :status="p.status" />
          </span>
        </RouterLink>
      </li>
    </ul>
  </div>
</template>

<style scoped>
.empty {
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: var(--s3);
  color: var(--brand);
}

.project-list {
  list-style: none;
  margin: 0;
  padding: 0;
  background: var(--white);
  border: 1px solid var(--line);
  border-radius: var(--r-panel);
  overflow: hidden;
}

.project-list li + li {
  border-top: 1px solid var(--line);
}

.project-row {
  display: grid;
  grid-template-columns: 148px minmax(0, 1fr) auto;
  align-items: center;
  gap: var(--s4);
  padding: var(--s3) var(--s4);
  color: inherit;
  text-decoration: none;
}

.project-row:hover {
  background: #fcf8fa;
  color: inherit;
}

.project-row:hover .row-title {
  text-decoration: underline;
  text-decoration-color: var(--amber);
  text-decoration-thickness: 2px;
}

.project-row:focus-visible {
  outline-offset: -4px;
  box-shadow: inset 0 0 0 4px var(--amber);
}

.thumb {
  position: relative;
  display: grid;
  place-items: center;
  aspect-ratio: 16 / 9;
  border-radius: 8px;
  overflow: hidden;
  background: var(--brand-900);
  color: var(--brand-300);
}

.thumb img {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.thumb-time {
  position: absolute;
  right: 4px;
  bottom: 4px;
  padding: 0 5px;
  border-radius: 3px;
  background: rgba(42, 12, 35, 0.85);
  color: var(--white);
  font-size: 0.75rem;
  font-weight: 650;
}

.row-main {
  display: flex;
  flex-direction: column;
  gap: var(--s1);
  min-width: 0;
}

.row-title {
  font-weight: 700;
  color: var(--brand-900);
  font-size: var(--fs-md);
  line-height: 1.3;
}

.row-meta {
  display: flex;
  flex-wrap: wrap;
  gap: var(--s1) var(--s4);
  font-size: var(--fs-sm);
  color: var(--ink-soft);
}

.row-meta > span {
  display: inline-flex;
  align-items: center;
  gap: var(--s1);
}

.row-detail {
  font-size: var(--fs-xs);
  color: var(--ink-soft);
  display: -webkit-box;
  -webkit-line-clamp: 2;
  line-clamp: 2;
  -webkit-box-orient: vertical;
  overflow: hidden;
}

@media (max-width: 640px) {
  .project-row {
    grid-template-columns: 96px minmax(0, 1fr);
    gap: var(--s3);
  }

  .row-status {
    grid-column: 2;
    grid-row: 2;
  }

  .thumb {
    grid-row: span 2;
    align-self: start;
  }
}
</style>
