<script setup lang="ts">
import { computed, useId } from 'vue'
import { RouterLink } from 'vue-router'
import type { ClipOut } from '../api/types'
import { fmtDuration, fmtRange } from '../lib/format'
import { isClipExportable, isRenderActive } from '../lib/status'
import AppIcon from './AppIcon.vue'
import ScoreBadge from './ScoreBadge.vue'

const props = defineProps<{ clip: ClipOut }>()
const selected = defineModel<boolean>('selected', { default: false })

const uid = useId()
const render = computed(() => props.clip.render)
const hasVideo = computed(() => !!render.value.video_url)
// The ZIP export only includes 'ready' renders, not a stale or re-rendering clip's old video.
const exportable = computed(() => isClipExportable(render.value))
const selectTitle = computed(() => {
  if (exportable.value) return 'Include in ZIP download'
  if (hasVideo.value) return 'Re-render this clip to include it in the ZIP download'
  return 'Available once the clip is rendered'
})
const active = computed(() => isRenderActive(render.value))
const renderLabel = computed(() => {
  if (render.value.state === 'queued') return 'Waiting to render'
  if (render.value.state === 'rendering') {
    const job = props.clip.job
    if (job?.progress_current != null && job.progress_total) {
      return `Rendering, step ${job.progress_current} of ${job.progress_total}`
    }
    return 'Rendering'
  }
  return ''
})
</script>

<template>
  <article class="clip-card" :aria-labelledby="`${uid}-title`">
    <div class="media-well">
      <video
        v-if="hasVideo"
        :key="render.video_url ?? ''"
        class="clip-video"
        :src="render.video_url ?? undefined"
        :poster="render.thumb_url ?? undefined"
        controls
        preload="metadata"
        playsinline
        :aria-label="`Preview of ${clip.title}`"
      ></video>
      <div v-else class="media-placeholder">
        <template v-if="active">
          <span class="spinner" aria-hidden="true"></span>
          <p>{{ renderLabel }}</p>
        </template>
        <template v-else-if="render.state === 'failed'">
          <AppIcon name="alert" :size="22" />
          <p>Render failed</p>
        </template>
        <template v-else>
          <AppIcon name="film" :size="22" />
          <p>Not rendered yet</p>
        </template>
      </div>
      <p v-if="hasVideo && active" class="media-chip" role="status">
        <span class="spinner" aria-hidden="true"></span>{{ renderLabel }}
      </p>
      <span class="rank num" aria-hidden="true">{{ clip.rank }}</span>
    </div>

    <div class="clip-body">
      <div class="clip-top">
        <label class="check select-check" :title="selectTitle">
          <input v-model="selected" type="checkbox" :disabled="!exportable" />
          <span class="visually-hidden">Select “{{ clip.title }}” for the ZIP download</span>
          <span aria-hidden="true">Select</span>
        </label>
        <ScoreBadge :score="clip.score" />
      </div>

      <h3 :id="`${uid}-title`" class="clip-title">{{ clip.title }}</h3>
      <p v-if="clip.reason" class="clip-reason">{{ clip.reason }}</p>

      <dl class="clip-meta">
        <div>
          <dt>Length</dt>
          <dd class="num">{{ fmtDuration(clip.duration) }}</dd>
        </div>
        <div>
          <dt>From the source</dt>
          <dd class="num">{{ fmtRange(clip.src_start, clip.src_end) }}</dd>
        </div>
      </dl>

      <p v-if="render.stale" class="clip-flag">
        <AppIcon name="pencil" :size="14" /> Edited since the last render. Open the editor to re-render.
      </p>
      <p v-else-if="render.state === 'failed'" class="clip-flag clip-flag-error">
        <AppIcon name="alert" :size="14" /> {{ render.error || 'The render did not finish. Open the editor to try again.' }}
      </p>

      <div class="clip-actions">
        <RouterLink :to="`/clips/${clip.id}`" class="btn btn-primary btn-sm">
          <AppIcon name="pencil" :size="16" /> Edit
        </RouterLink>
        <a v-if="render.download_url" :href="render.download_url" class="btn btn-sm" download>
          <AppIcon name="download" :size="16" /> Download MP4
        </a>
        <button v-else type="button" class="btn btn-sm" aria-disabled="true" title="Available once the clip is rendered">
          <AppIcon name="download" :size="16" /> Download MP4
        </button>
        <a v-if="render.srt_url" :href="render.srt_url" class="btn btn-quiet btn-sm" download>
          <AppIcon name="captions" :size="16" /> SRT
        </a>
      </div>
    </div>
  </article>
</template>

<style scoped>
.clip-card {
  display: flex;
  flex-direction: column;
  min-width: 0;
  background: var(--white);
  border: 1px solid var(--line);
  border-radius: var(--r-media);
  overflow: hidden;
}

.media-well {
  position: relative;
  display: grid;
  place-items: center;
  background: var(--brand-900);
  padding: var(--s3);
}

.clip-video,
.media-placeholder {
  width: 100%;
  max-width: 300px;
  aspect-ratio: 9 / 16;
  max-height: 72vh;
  border-radius: 8px;
  background: #000;
}

.clip-video {
  object-fit: contain;
}

.media-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: var(--s2);
  color: #e6d3df;
  background: #3b1231;
  font-size: var(--fs-sm);
  font-weight: 600;
  text-align: center;
  padding: var(--s4);
}

.media-chip {
  position: absolute;
  left: 50%;
  top: var(--s5);
  transform: translateX(-50%);
  display: inline-flex;
  align-items: center;
  gap: var(--s2);
  padding: var(--s1) var(--s3);
  border-radius: var(--r-pill);
  background: var(--amber);
  color: var(--brand-900);
  font-size: var(--fs-xs);
  font-weight: 700;
  white-space: nowrap;
  pointer-events: none;
}

.media-chip .spinner {
  width: 12px;
  height: 12px;
}

.rank {
  position: absolute;
  top: var(--s3);
  left: var(--s3);
  min-width: 28px;
  height: 28px;
  display: grid;
  place-items: center;
  padding: 0 var(--s2);
  border-radius: 0 0 8px 0;
  background: var(--amber);
  color: var(--brand-900);
  font-weight: 800;
  font-size: var(--fs-sm);
}

.clip-body {
  display: flex;
  flex-direction: column;
  gap: var(--s2);
  padding: var(--s3) var(--s4) var(--s4);
  flex: 1;
}

.clip-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--s2);
}

.select-check {
  font-size: var(--fs-xs);
  color: var(--ink-soft);
  min-height: 32px;
}

.clip-title {
  font-size: var(--fs-md);
  line-height: 1.3;
}

.clip-reason {
  font-size: var(--fs-sm);
  color: var(--ink-soft);
}

.clip-meta {
  display: flex;
  flex-wrap: wrap;
  gap: var(--s1) var(--s4);
  margin: 0;
  font-size: var(--fs-xs);
}

.clip-meta div {
  display: flex;
  gap: var(--s1);
}

.clip-meta dt {
  color: var(--ink-soft);
}

.clip-meta dt::after {
  content: ':';
}

.clip-meta dd {
  margin: 0;
  font-weight: 650;
  color: var(--brand-900);
}

.clip-flag {
  display: flex;
  align-items: flex-start;
  gap: var(--s2);
  padding: var(--s2);
  border-radius: var(--r-control);
  background: var(--amber-100);
  font-size: var(--fs-xs);
  font-weight: 600;
  color: #6a3a00;
}

.clip-flag svg {
  flex: none;
  margin-top: 2px;
}

.clip-flag-error {
  background: var(--danger-bg);
  color: var(--danger);
}

.clip-actions {
  display: flex;
  flex-wrap: wrap;
  gap: var(--s2);
  margin-top: auto;
  padding-top: var(--s2);
}
</style>
