import type { LimitsInfo } from '../api/types'

export interface DurationPreset {
  id: 'short' | 'default' | 'long'
  label: string
  min: number
  max: number
}

export const DURATION_PRESETS: DurationPreset[] = [
  { id: 'short', label: 'Short', min: 15, max: 30 },
  { id: 'default', label: 'Default', min: 20, max: 60 },
  { id: 'long', label: 'Long', min: 45, max: 90 },
]

export interface ClipOptionValues {
  clip_count: number
  min_seconds: number
  max_seconds: number
  captions_enabled: boolean
}

/** Fallback bounds mirroring the backend schema when /api/system is unavailable. */
export const SCHEMA_BOUNDS = {
  clipMin: 1,
  clipMax: 10,
  minFloor: 5,
  minCeil: 170,
  maxFloor: 10,
  maxCap: 180,
  minGap: 5,
}

export function boundsFrom(limits: LimitsInfo | null | undefined) {
  return {
    ...SCHEMA_BOUNDS,
    clipMax: Math.min(SCHEMA_BOUNDS.clipMax, Math.max(1, limits?.max_clips ?? SCHEMA_BOUNDS.clipMax)),
    minFloor: Math.max(SCHEMA_BOUNDS.minFloor, limits?.min_seconds_floor ?? SCHEMA_BOUNDS.minFloor),
    maxCap: Math.min(SCHEMA_BOUNDS.maxCap, limits?.max_seconds_cap ?? SCHEMA_BOUNDS.maxCap),
  }
}

export function presetFor(min: number, max: number): DurationPreset | null {
  return DURATION_PRESETS.find((p) => p.min === min && p.max === max) ?? null
}

/** Plain-language problem with the options, or null when they are valid. */
export function validateOptions(v: ClipOptionValues, limits: LimitsInfo | null | undefined): string | null {
  const b = boundsFrom(limits)
  if (!Number.isInteger(v.clip_count) || v.clip_count < b.clipMin || v.clip_count > b.clipMax) {
    return `Choose between ${b.clipMin} and ${b.clipMax} clips.`
  }
  if (!Number.isFinite(v.min_seconds) || !Number.isFinite(v.max_seconds)) {
    return 'Enter a minimum and maximum clip length in seconds.'
  }
  if (v.min_seconds < b.minFloor || v.min_seconds > b.minCeil) {
    return `The minimum length must be between ${b.minFloor} and ${b.minCeil} seconds.`
  }
  if (v.max_seconds < b.maxFloor || v.max_seconds > b.maxCap) {
    return `The maximum length must be between ${b.maxFloor} and ${b.maxCap} seconds.`
  }
  if (v.max_seconds - v.min_seconds < b.minGap) {
    return `The maximum length must be at least ${b.minGap} seconds longer than the minimum.`
  }
  return null
}
