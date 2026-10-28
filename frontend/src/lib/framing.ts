/** Geometry for the 9:16 crop window drawn over a source frame (all values are fractions of width). */
export interface CropWindow {
  /** crop window width as a fraction of the source width (1 = whole width) */
  width: number
  /** left edge as a fraction of the source width */
  left: number
}

export const VERTICAL_RATIO = 9 / 16

export function cropWidthFraction(srcW: number, srcH: number): number {
  if (srcW <= 0 || srcH <= 0) return 1
  return Math.min(1, (srcH * VERTICAL_RATIO) / srcW)
}

export function cropWindow(srcW: number, srcH: number, centreX: number): CropWindow {
  const width = cropWidthFraction(srcW, srcH)
  const left = Math.min(1 - width, Math.max(0, centreX - width / 2))
  return { width, left }
}

export function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v))
}
