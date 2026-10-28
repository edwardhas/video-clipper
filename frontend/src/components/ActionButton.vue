<script setup lang="ts">
/**
 * Button with real loading and disabled states. A disabled button stays
 * focusable (aria-disabled) so its `reason` tooltip is reachable by keyboard
 * and screen readers; clicks are swallowed while disabled or busy.
 */
import { computed } from 'vue'

const props = withDefaults(
  defineProps<{
    variant?: 'primary' | 'secondary' | 'accent' | 'danger' | 'danger-solid' | 'quiet'
    size?: 'sm' | 'md' | 'lg'
    disabled?: boolean
    reason?: string | null
    loading?: boolean
    loadingLabel?: string
    type?: 'button' | 'submit'
    block?: boolean
    title?: string
  }>(),
  {
    variant: 'secondary',
    size: 'md',
    disabled: false,
    reason: null,
    loading: false,
    loadingLabel: undefined,
    type: 'button',
    block: false,
    title: undefined,
  },
)

const emit = defineEmits<{ click: [event: MouseEvent] }>()

const inactive = computed(() => props.disabled || props.loading)
const classes = computed(() => [
  'btn',
  props.variant !== 'secondary' ? `btn-${props.variant}` : '',
  props.size !== 'md' ? `btn-${props.size}` : '',
  props.block ? 'btn-block' : '',
])
const tooltip = computed(() => (props.disabled && props.reason ? props.reason : props.title))

function onClick(event: MouseEvent) {
  if (inactive.value) {
    event.preventDefault()
    event.stopPropagation()
    return
  }
  emit('click', event)
}
</script>

<template>
  <button
    :type="type"
    :class="classes"
    :aria-disabled="disabled ? 'true' : undefined"
    :aria-busy="loading ? 'true' : undefined"
    :title="tooltip"
    @click="onClick"
  >
    <span v-if="loading" class="spinner" aria-hidden="true"></span>
    <template v-if="loading && loadingLabel">{{ loadingLabel }}</template>
    <slot v-else />
    <span v-if="disabled && reason" class="visually-hidden">({{ reason }})</span>
  </button>
</template>
