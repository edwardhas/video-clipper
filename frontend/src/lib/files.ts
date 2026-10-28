import { fmtBytes, fmtMegabytes, mbToBytes } from './format'

/** Mirrors ALLOWED_EXTENSIONS in backend/clipper/api/routes_uploads.py (the server re-checks). */
export const VIDEO_EXTENSIONS = [
  'mp4', 'mov', 'm4v', 'mkv', 'webm', 'avi', 'mpg', 'mpeg', 'ts', 'mts', 'm2ts', 'wmv', 'flv', '3gp', 'ogv',
] as const

export const ACCEPT_ATTR = ['video/*', ...VIDEO_EXTENSIONS.map((e) => `.${e}`)].join(',')

export function fileExtension(name: string): string {
  const dot = name.lastIndexOf('.')
  return dot > 0 ? name.slice(dot + 1).toLowerCase() : ''
}

export interface FileLimits {
  maxUploadMb: number | null
  /** Bytes the app may still use (free minus the reserve kept for processing). */
  usableBytes: number | null
}

/** Plain-language reason the file cannot be uploaded, or null when it looks fine. */
export function fileProblem(file: { name: string; size: number }, limits: FileLimits): string | null {
  const ext = fileExtension(file.name)
  if (!(VIDEO_EXTENSIONS as readonly string[]).includes(ext)) {
    return `This file type is not supported. Choose a video file (${VIDEO_EXTENSIONS.join(', ')}).`
  }
  if (file.size === 0) return 'This file is empty.'
  if (limits.maxUploadMb != null && file.size > mbToBytes(limits.maxUploadMb)) {
    return `This file is ${fmtBytes(file.size)}, larger than the ${fmtMegabytes(limits.maxUploadMb)} upload limit.`
  }
  if (limits.usableBytes != null && file.size > limits.usableBytes) {
    return `There is not enough free disk space for this file (${fmtBytes(Math.max(0, limits.usableBytes))} available).`
  }
  return null
}
