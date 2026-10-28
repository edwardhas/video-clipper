<script setup lang="ts">
/** Caption style presets: size, colour, position, word highlight and uppercase. */
import { useId } from 'vue'
import type { CaptionColor, CaptionPosition, CaptionSize, CaptionStyle } from '../api/types'

const model = defineModel<CaptionStyle>({ required: true })
withDefaults(defineProps<{ disabled?: boolean }>(), { disabled: false })

const uid = useId()

const SIZES: { value: CaptionSize; label: string }[] = [
  { value: 'S', label: 'Small' },
  { value: 'M', label: 'Medium' },
  { value: 'L', label: 'Large' },
]
const COLORS: { value: CaptionColor; label: string }[] = [
  { value: 'white', label: 'White' },
  { value: 'yellow', label: 'Yellow' },
  { value: 'boxed', label: 'Boxed' },
]
const POSITIONS: { value: CaptionPosition; label: string }[] = [
  { value: 'top', label: 'Top' },
  { value: 'middle', label: 'Middle' },
  { value: 'bottom', label: 'Bottom' },
]

function patch(values: Partial<CaptionStyle>) {
  model.value = { ...model.value, ...values }
}

function checked(event: Event): boolean {
  return (event.target as HTMLInputElement).checked
}
</script>

<template>
  <div class="caption-style">
    <fieldset class="group" :disabled="disabled">
      <legend class="field-label">Size</legend>
      <div class="seg">
        <label v-for="s in SIZES" :key="s.value" class="seg-opt">
          <input
            type="radio"
            :name="`${uid}-size`"
            :value="s.value"
            :checked="model.size === s.value"
            @change="patch({ size: s.value })"
          />
          <span>{{ s.label }}</span>
        </label>
      </div>
    </fieldset>

    <fieldset class="group" :disabled="disabled">
      <legend class="field-label">Colour</legend>
      <div class="swatches">
        <label v-for="c in COLORS" :key="c.value" class="swatch">
          <input
            type="radio"
            :name="`${uid}-color`"
            :value="c.value"
            :checked="model.color === c.value"
            @change="patch({ color: c.value })"
          />
          <span class="swatch-tile" :class="`swatch-${c.value}`" aria-hidden="true">
            <span class="swatch-text">Aa</span>
          </span>
          <span class="swatch-label">{{ c.label }}</span>
        </label>
      </div>
    </fieldset>

    <fieldset class="group" :disabled="disabled">
      <legend class="field-label">Position</legend>
      <div class="seg">
        <label v-for="p in POSITIONS" :key="p.value" class="seg-opt">
          <input
            type="radio"
            :name="`${uid}-pos`"
            :value="p.value"
            :checked="model.position === p.value"
            @change="patch({ position: p.value })"
          />
          <span>
            <svg class="pos-icon" viewBox="0 0 12 20" aria-hidden="true" focusable="false">
              <rect x="0.75" y="0.75" width="10.5" height="18.5" rx="2" fill="none" stroke="currentColor" stroke-width="1.5" />
              <rect
                x="2.5"
                :y="p.value === 'top' ? 4 : p.value === 'middle' ? 9 : 14"
                width="7"
                height="2.2"
                rx="1"
                fill="currentColor"
              />
            </svg>
            {{ p.label }}
          </span>
        </label>
      </div>
    </fieldset>

    <div class="toggles">
      <label class="switch">
        <input
          type="checkbox"
          role="switch"
          :checked="model.highlight"
          :disabled="disabled"
          @change="patch({ highlight: checked($event) })"
        />
        <span class="track" aria-hidden="true"></span>
        Highlight the spoken word
      </label>
      <label class="switch">
        <input
          type="checkbox"
          role="switch"
          :checked="model.uppercase"
          :disabled="disabled"
          @change="patch({ uppercase: checked($event) })"
        />
        <span class="track" aria-hidden="true"></span>
        Uppercase text
      </label>
    </div>
  </div>
</template>

<style scoped>
.caption-style {
  display: flex;
  flex-direction: column;
  gap: var(--s4);
}

.group {
  margin: 0;
  padding: 0;
  border: 0;
  min-width: 0;
}

.group legend {
  padding: 0;
  margin-bottom: var(--s2);
}

.pos-icon {
  width: 10px;
  height: 16px;
}

.swatches {
  display: flex;
  flex-wrap: wrap;
  gap: var(--s3);
}

.swatch {
  position: relative;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--s1);
  cursor: pointer;
}

.swatch input {
  position: absolute;
  opacity: 0;
  inset: 0;
  margin: 0;
  cursor: pointer;
}

.swatch-tile {
  display: grid;
  place-items: center;
  width: 72px;
  height: 52px;
  border-radius: var(--r-control);
  background:
    linear-gradient(160deg, rgba(255, 255, 255, 0.12), transparent 60%),
    linear-gradient(200deg, #6d4a63, #2a0c23);
  border: 2px solid transparent;
  box-shadow: 0 0 0 1px var(--line-strong);
}

.swatch-text {
  font-size: 1.25rem;
  font-weight: 900;
  line-height: 1;
  color: #fff;
  -webkit-text-stroke: 1px #000;
  paint-order: stroke fill;
  text-shadow: 0 1px 0 #000;
}

.swatch-yellow .swatch-text {
  color: #ffd84d;
}

.swatch-boxed .swatch-text {
  -webkit-text-stroke: 0;
  text-shadow: none;
  padding: 3px 7px;
  border-radius: 4px;
  background: #000;
}

.swatch-label {
  font-size: var(--fs-xs);
  font-weight: 650;
  color: var(--brand-900);
}

.swatch:hover .swatch-tile {
  box-shadow: 0 0 0 1px var(--brand-300);
}

.swatch input:checked ~ .swatch-tile {
  border-color: var(--amber);
  box-shadow: 0 0 0 1px var(--brand-900);
}

.swatch input:checked ~ .swatch-label {
  color: var(--brand);
  text-decoration: underline;
  text-decoration-thickness: 2px;
  text-decoration-color: var(--amber);
  text-underline-offset: 3px;
}

.swatch input:focus-visible ~ .swatch-tile {
  outline: 2px solid var(--brand-900);
  outline-offset: 2px;
}

.swatch input:disabled ~ .swatch-tile {
  opacity: 0.5;
}

.toggles {
  display: flex;
  flex-direction: column;
  gap: var(--s3);
}
</style>
