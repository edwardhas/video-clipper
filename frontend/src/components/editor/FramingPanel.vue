<script setup lang="ts">
/**
 * Framing choice: Auto (face analysis), Face crop (9:16 crop, optional fixed
 * horizontal position) or Full frame (whole picture on a blurred background).
 * The preview is a real source frame from the middle of the clip.
 */
import { computed, ref, useId, watch } from 'vue'
import type { FramingChoice, FramingMode, FramingSummary } from '../../api/types'
import { cropWindow } from '../../lib/framing'
import AppIcon from '../AppIcon.vue'

const framing = defineModel<FramingChoice>({ required: true })
const props = defineProps<{
  frameUrl: string
  sourceWidth: number
  sourceHeight: number
  auto: FramingSummary | null
}>()

const uid = useId()
const frameFailed = ref(false)
const frameLoaded = ref(false)
watch(
  () => props.frameUrl,
  () => {
    frameFailed.value = false
    frameLoaded.value = false
  },
)

const MODES: { value: FramingMode; label: string; hint: string }[] = [
  { value: 'auto', label: 'Auto', hint: 'Let the face analysis decide.' },
  { value: 'crop', label: 'Face crop', hint: 'Fill the vertical frame with a 9:16 crop.' },
  { value: 'fit', label: 'Full frame', hint: 'Show the whole picture on a blurred background.' },
]

const FRAME_MAX_H = 340
// The frame box keeps the source aspect ratio exactly so the crop overlay lines up with the picture.
const frameStyle = computed(() => {
  const w = props.sourceWidth > 0 ? props.sourceWidth : 16
  const h = props.sourceHeight > 0 ? props.sourceHeight : 9
  return { aspectRatio: `${w} / ${h}`, maxWidth: `${Math.round((FRAME_MAX_H * w) / h)}px` }
})
const sliderValue = computed(() => framing.value.crop_x ?? 0.5)
const windowBox = computed(() => cropWindow(props.sourceWidth, props.sourceHeight, sliderValue.value))
const cropStyle = computed(() => ({
  left: `${windowBox.value.left * 100}%`,
  width: `${windowBox.value.width * 100}%`,
}))
const autoLabel = computed(() => {
  if (!props.auto) return null
  return props.auto.layout === 'crop' ? 'Face crop' : 'Full frame'
})
const sourceIsVertical = computed(() => windowBox.value.width >= 0.999)

function setMode(mode: FramingMode) {
  framing.value = { mode, crop_x: mode === 'crop' ? framing.value.crop_x : null }
}

function onSlide(event: Event) {
  const v = Number.parseFloat((event.target as HTMLInputElement).value)
  if (Number.isFinite(v)) framing.value = { mode: 'crop', crop_x: Math.min(1, Math.max(0, v)) }
}

function followFace() {
  framing.value = { mode: 'crop', crop_x: null }
}

function positionText(v: number): string {
  const pct = Math.round(v * 100)
  if (pct === 50) return 'Centre'
  return pct < 50 ? `${50 - pct}% left of centre` : `${pct - 50}% right of centre`
}
</script>

<template>
  <section class="framing stack" :aria-labelledby="`${uid}-h`">
    <h2 :id="`${uid}-h`" class="visually-hidden">Framing</h2>

    <fieldset class="modes">
      <legend class="field-label">How the video fills the vertical frame</legend>
      <label v-for="m in MODES" :key="m.value" class="mode" :class="{ on: framing.mode === m.value }">
        <input
          type="radio"
          :name="`${uid}-mode`"
          :value="m.value"
          :checked="framing.mode === m.value"
          @change="setMode(m.value)"
        />
        <span class="mode-label">{{ m.label }}</span>
        <span class="mode-hint">{{ m.hint }}</span>
      </label>
    </fieldset>

    <div v-if="framing.mode === 'auto'" class="notice">
      <AppIcon name="info" />
      <div class="notice-body">
        <p v-if="auto">
          <strong>Auto chose {{ autoLabel }}.</strong> {{ auto.reason }}
        </p>
        <p v-else>The face analysis runs with the next render and picks a crop or the full frame.</p>
      </div>
    </div>

    <div class="frame-wrap">
      <div class="frame" :class="{ fit: framing.mode === 'fit' }" :style="frameStyle">
        <img
          v-show="!frameFailed"
          :src="frameUrl"
          alt=""
          class="frame-img"
          decoding="async"
          @load="frameLoaded = true"
          @error="frameFailed = true"
        />
        <p v-if="frameFailed" class="frame-missing">The source frame preview is not available.</p>
        <p v-else-if="!frameLoaded" class="frame-missing"><span class="spinner" aria-hidden="true"></span></p>
        <template v-if="framing.mode === 'crop' && !frameFailed">
          <div class="crop-shade" aria-hidden="true">
            <div class="crop-box" :class="{ floating: framing.crop_x == null }" :style="cropStyle">
              <span class="crop-tag">9:16</span>
            </div>
          </div>
        </template>
      </div>
      <div v-if="framing.mode === 'fit'" class="fit-mock" aria-hidden="true">
        <img v-if="!frameFailed" :src="frameUrl" alt="" class="fit-bg" />
        <img v-if="!frameFailed" :src="frameUrl" alt="" class="fit-fg" />
      </div>
    </div>
    <p class="field-hint frame-caption">
      <template v-if="framing.mode === 'crop'">
        A frame from the middle of the clip. The highlighted area is what appears in the vertical video.
      </template>
      <template v-else-if="framing.mode === 'fit'">
        The whole frame is placed in the middle of the vertical video, with a blurred copy filling the space around it.
      </template>
      <template v-else>A frame from the middle of the clip.</template>
    </p>

    <div v-if="framing.mode === 'crop'" class="position stack-sm">
      <div class="row-between">
        <label class="field-label" :for="`${uid}-x`">Horizontal position</label>
        <span class="small num">{{ framing.crop_x == null ? 'Follows the detected face' : positionText(sliderValue) }}</span>
      </div>
      <input
        :id="`${uid}-x`"
        type="range"
        min="0"
        max="1"
        step="0.01"
        :value="sliderValue"
        :disabled="sourceIsVertical"
        :aria-valuetext="framing.crop_x == null ? 'Follows the detected face' : positionText(sliderValue)"
        @input="onSlide"
      />
      <p v-if="sourceIsVertical" class="field-hint">This video is already vertical, so there is nothing to move.</p>
      <div v-else class="row-between">
        <p class="field-hint">
          {{
            framing.crop_x == null
              ? 'The crop follows the detected face (centred when no face is found). Move the slider to fix the position.'
              : 'The crop stays at this position for the whole clip.'
          }}
        </p>
        <button
          type="button"
          class="btn btn-sm"
          :aria-disabled="framing.crop_x == null ? 'true' : undefined"
          :title="framing.crop_x == null ? 'Already following the detected face' : undefined"
          @click="framing.crop_x != null && followFace()"
        >
          Follow the face
        </button>
      </div>
    </div>
  </section>
</template>

<style scoped>
.modes {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: var(--s2);
  margin: 0;
  padding: 0;
  border: 0;
  min-width: 0;
}

.modes legend {
  padding: 0;
  margin-bottom: var(--s2);
}

.mode {
  position: relative;
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: var(--s3);
  border: 1px solid var(--line-strong);
  border-radius: var(--r-control);
  background: var(--white);
  cursor: pointer;
}

.mode input {
  position: absolute;
  opacity: 0;
  inset: 0;
  margin: 0;
  cursor: pointer;
}

.mode:hover {
  border-color: var(--brand-300);
  background: var(--brand-100);
}

.mode.on {
  border-color: var(--brand);
  box-shadow: inset 0 0 0 1px var(--brand);
  background: var(--brand-100);
}

.mode:has(input:focus-visible) {
  outline: 2px solid var(--brand-900);
  outline-offset: 2px;
  box-shadow: 0 0 0 4px var(--amber);
}

.mode-label {
  font-weight: 700;
  color: var(--brand-900);
  font-size: var(--fs-sm);
}

.mode-hint {
  font-size: var(--fs-xs);
  color: var(--ink-soft);
}

.frame-wrap {
  display: flex;
  gap: var(--s3);
  align-items: stretch;
}

.frame {
  position: relative;
  flex: 1;
  min-width: 0;
  border-radius: var(--r-control);
  overflow: hidden;
  background: var(--brand-900);
}

.frame-img {
  width: 100%;
  height: 100%;
  object-fit: contain;
}

.frame-missing {
  position: absolute;
  inset: 0;
  display: grid;
  place-items: center;
  padding: var(--s4);
  color: #e6d3df;
  font-size: var(--fs-sm);
  text-align: center;
}

.crop-shade {
  position: absolute;
  inset: 0;
}

.crop-box {
  position: absolute;
  top: 0;
  bottom: 0;
  border: 3px solid var(--amber);
  box-shadow: 0 0 0 100vmax rgba(42, 12, 35, 0.6);
  transition: left 80ms linear;
}

.crop-box.floating {
  border-style: dashed;
}

.crop-tag {
  position: absolute;
  top: var(--s1);
  left: var(--s1);
  padding: 0 var(--s1);
  border-radius: 3px;
  background: var(--amber);
  color: var(--brand-900);
  font-size: var(--fs-xs);
  font-weight: 800;
}

.fit-mock {
  position: relative;
  flex: none;
  width: 110px;
  aspect-ratio: 9 / 16;
  border-radius: var(--r-control);
  overflow: hidden;
  background: #000;
  align-self: center;
  border: 2px solid var(--amber);
}

.fit-bg {
  position: absolute;
  inset: -12px;
  width: calc(100% + 24px);
  height: calc(100% + 24px);
  max-width: none;
  object-fit: cover;
  filter: blur(8px) brightness(0.8);
}

.fit-fg {
  position: absolute;
  top: 50%;
  left: 0;
  width: 100%;
  transform: translateY(-50%);
}

.frame-caption {
  margin-top: var(--s2);
}

@media (max-width: 520px) {
  .modes {
    grid-template-columns: 1fr;
  }

  .fit-mock {
    width: 84px;
  }
}

@media (prefers-reduced-motion: reduce) {
  .crop-box {
    transition: none;
  }
}
</style>
