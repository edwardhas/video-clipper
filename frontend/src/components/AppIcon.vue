<script setup lang="ts">
import { computed } from 'vue'

export type IconName =
  | 'check'
  | 'x'
  | 'alert'
  | 'info'
  | 'circle'
  | 'minus'
  | 'download'
  | 'pencil'
  | 'scissors'
  | 'upload'
  | 'link'
  | 'refresh'
  | 'stop'
  | 'trash'
  | 'chevron-left'
  | 'film'
  | 'captions'
  | 'frame'
  | 'trim'

const props = withDefaults(defineProps<{ name: IconName; size?: number; label?: string }>(), {
  size: 18,
  label: undefined,
})

// 24×24 stroke paths (round caps), drawn for this app.
const PATHS: Record<IconName, string> = {
  check: 'M5 12.5l4.5 4.5L19 7.5',
  x: 'M6 6l12 12M18 6L6 18',
  alert: 'M12 4l9 16H3L12 4zM12 10v4.5M12 17.5v.01',
  info: 'M12 21a9 9 0 100-18 9 9 0 000 18zM12 11v6M12 7.5v.01',
  circle: 'M12 20a8 8 0 100-16 8 8 0 000 16z',
  minus: 'M6 12h12',
  download: 'M12 4v11M7 10.5l5 5 5-5M5 20h14',
  pencil: 'M4 20h4L19 9l-4-4L4 16v4zM13.5 6.5l4 4',
  scissors:
    'M8.5 17.5a3 3 0 11-4.24 0 3 3 0 014.24 0zM19.74 17.5a3 3 0 11-4.24 0 3 3 0 014.24 0zM8.5 17.5L18 4.5M15.5 17.5L6 4.5',
  upload: 'M12 16V5M7 9.5l5-5 5 5M5 20h14',
  link: 'M10 14a4.5 4.5 0 006.4 0l3-3a4.5 4.5 0 00-6.4-6.4L11.5 6M14 10a4.5 4.5 0 00-6.4 0l-3 3a4.5 4.5 0 006.4 6.4l1.5-1.4',
  refresh: 'M19.5 12a7.5 7.5 0 11-2.2-5.3M19.5 4.5v4h-4',
  stop: 'M7 7h10v10H7z',
  trash: 'M4.5 7h15M9.5 7V4.5h5V7M6.5 7l1 13h9l1-13M10 11v5.5M14 11v5.5',
  'chevron-left': 'M14.5 6L8.5 12l6 6',
  film: 'M4 5h16v14H4zM8 5v14M16 5v14M4 9h4M4 15h4M16 9h4M16 15h4',
  captions: 'M3.5 6h17v12h-17zM7 11h4M13 11h4M7 14.5h7',
  frame: 'M9 3.5h6v17H9zM3.5 7v10M20.5 7v10',
  trim: 'M7 4v16M17 4v16M7 12h10M4 8h3M17 16h3',
}

const d = computed(() => PATHS[props.name])
</script>

<template>
  <svg
    :width="size"
    :height="size"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    stroke-width="2"
    stroke-linecap="round"
    stroke-linejoin="round"
    :role="label ? 'img' : undefined"
    :aria-label="label"
    :aria-hidden="label ? undefined : 'true'"
    focusable="false"
  >
    <path :d="d" />
  </svg>
</template>
