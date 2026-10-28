/**
 * Thin fetch wrapper for the Local Clipper API.
 *
 * - Relative URLs only (the app is always served same-origin).
 * - Mutating requests carry the session-bound X-CSRF-Token, fetched from
 *   GET /api/auth/session and cached; a 403 CSRF rejection refreshes it once.
 * - 401 responses trigger the registered unauthorized handler (login redirect).
 * - Errors are normalised to ApiError with a plain-language message.
 */
import type {
  CaptionEditIn,
  ClipDetailOut,
  ClipEditIn,
  DefaultsModel,
  LoginIn,
  ProjectCreateIn,
  ProjectOut,
  ProjectSummaryOut,
  ReselectIn,
  SessionOut,
  SystemOut,
  UploadOut,
} from './types'

export class ApiError extends Error {
  readonly status: number
  readonly code: string | null

  constructor(status: number, message: string, code: string | null = null) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.code = code
  }

  get isCsrf(): boolean {
    return this.status === 403 && /csrf/i.test(this.code ?? '')
  }

  get isNetwork(): boolean {
    return this.status === 0
  }
}

export class UploadAbortedError extends Error {
  constructor() {
    super('Upload cancelled.')
    this.name = 'UploadAbortedError'
  }
}

type Method = 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE'

const NETWORK_MESSAGE =
  'Cannot reach Local Clipper. Check that it is still running, then try again.'

let csrfToken: string | null = null
let sessionInFlight: Promise<SessionOut> | null = null
let unauthorizedHandler: (() => void) | null = null
const sessionListeners = new Set<(s: SessionOut) => void>()

/** Register what happens on a 401 (normally: router redirect to /login?next=). */
export function setUnauthorizedHandler(fn: () => void): void {
  unauthorizedHandler = fn
}

/** Be told whenever a fresh session object arrives (login, logout, refresh). */
export function onSessionChange(fn: (s: SessionOut) => void): () => void {
  sessionListeners.add(fn)
  return () => sessionListeners.delete(fn)
}

function publishSession(s: SessionOut): void {
  csrfToken = s.csrf_token
  for (const fn of sessionListeners) fn(s)
}

function assertRelative(path: string): void {
  if (!path.startsWith('/') || path.startsWith('//')) {
    throw new Error(`API paths must be relative to this site: ${path}`)
  }
}

function fallbackMessage(status: number): string {
  if (status === 401) return 'Please sign in to continue.'
  if (status === 403) return 'This action was not allowed. Reload the page and try again.'
  if (status === 404) return 'This item no longer exists.'
  if (status === 409) return 'This action is not possible right now.'
  if (status === 413) return 'The file is larger than the upload limit.'
  if (status === 429) return 'Too many attempts. Wait a moment and try again.'
  if (status >= 500) return 'Local Clipper hit an internal error. Try again in a moment.'
  return `The request failed (HTTP ${status}).`
}

/** Turn an error body ({detail, code} or FastAPI validation errors) into an ApiError. */
export function errorFromBody(status: number, bodyText: string): ApiError {
  let detail: unknown
  let code: string | null = null
  try {
    const parsed: unknown = JSON.parse(bodyText)
    if (parsed && typeof parsed === 'object') {
      const obj = parsed as Record<string, unknown>
      detail = obj.detail
      if (typeof obj.code === 'string') code = obj.code
    }
  } catch {
    detail = undefined
  }
  let message: string
  if (typeof detail === 'string' && detail.trim()) {
    message = detail
  } else if (Array.isArray(detail)) {
    // FastAPI request-validation errors: [{loc, msg, type}, ...]
    const parts = detail
      .map((d) => (d && typeof d === 'object' ? (d as Record<string, unknown>).msg : null))
      .filter((m): m is string => typeof m === 'string')
    message = parts.length ? `Some values were not accepted: ${parts.join('; ')}` : fallbackMessage(status)
    code = code ?? 'validation_error'
  } else {
    message = fallbackMessage(status)
  }
  return new ApiError(status, message, code)
}

function handleUnauthorized(path: string): void {
  if (path.startsWith('/api/auth/')) return
  if (unauthorizedHandler) unauthorizedHandler()
}

/** GET /api/auth/session (deduplicated); refreshes the cached CSRF token. */
export function refreshSession(): Promise<SessionOut> {
  if (!sessionInFlight) {
    sessionInFlight = rawRequest<SessionOut>('GET', '/api/auth/session')
      .then((s) => {
        publishSession(s)
        return s
      })
      .finally(() => {
        sessionInFlight = null
      })
  }
  return sessionInFlight
}

async function ensureCsrfToken(): Promise<string | null> {
  if (csrfToken) return csrfToken
  try {
    await refreshSession()
  } catch {
    // Fall through: the server will reject the request with a clear error.
  }
  return csrfToken
}

async function rawRequest<T>(
  method: Method,
  path: string,
  body?: unknown,
  csrf?: string | null,
  signal?: AbortSignal,
): Promise<T> {
  assertRelative(path)
  const headers: Record<string, string> = { Accept: 'application/json' }
  if (body !== undefined) headers['Content-Type'] = 'application/json'
  if (csrf) headers['X-CSRF-Token'] = csrf
  let res: Response
  try {
    res = await fetch(path, {
      method,
      headers,
      body: body === undefined ? undefined : JSON.stringify(body),
      credentials: 'same-origin',
      cache: 'no-store',
      signal,
    })
  } catch (err) {
    if (err instanceof DOMException && err.name === 'AbortError') throw err
    throw new ApiError(0, NETWORK_MESSAGE, 'network_error')
  }
  const text = await res.text()
  if (!res.ok) throw errorFromBody(res.status, text)
  if (!text) return undefined as T
  try {
    return JSON.parse(text) as T
  } catch {
    throw new ApiError(res.status, 'The server sent a response this page could not read.', 'bad_response')
  }
}

async function request<T>(method: Method, path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  const mutating = method !== 'GET'
  for (let attempt = 0; ; attempt++) {
    const token = mutating ? await ensureCsrfToken() : null
    try {
      return await rawRequest<T>(method, path, body, token, signal)
    } catch (err) {
      if (err instanceof ApiError) {
        if (mutating && attempt === 0 && err.isCsrf) {
          csrfToken = null
          await refreshSession().catch(() => undefined)
          continue
        }
        if (err.status === 401) handleUnauthorized(path)
      }
      throw err
    }
  }
}

// ---------------------------------------------------------------- uploads
export interface UploadHandle {
  promise: Promise<UploadOut>
  abort: () => void
}

/**
 * Stream a file to POST /api/uploads with real byte progress (XMLHttpRequest).
 * A fresh CSRF token is fetched first so a large body is never sent with a stale one.
 */
export function uploadFile(file: File, onProgress: (loaded: number, total: number) => void): UploadHandle {
  const xhr = new XMLHttpRequest()
  let aborted = false

  const send = (token: string | null) =>
    new Promise<UploadOut>((resolve, reject) => {
      xhr.open('POST', '/api/uploads')
      xhr.setRequestHeader('Content-Type', 'application/octet-stream')
      xhr.setRequestHeader('X-Filename', encodeURIComponent(file.name))
      xhr.setRequestHeader('Accept', 'application/json')
      if (token) xhr.setRequestHeader('X-CSRF-Token', token)
      xhr.upload.onprogress = (e) => onProgress(e.loaded, e.lengthComputable ? e.total : file.size)
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            resolve(JSON.parse(xhr.responseText) as UploadOut)
          } catch {
            reject(new ApiError(xhr.status, 'The server sent a response this page could not read.', 'bad_response'))
          }
          return
        }
        const err = errorFromBody(xhr.status, xhr.responseText)
        if (err.isCsrf) csrfToken = null
        if (err.status === 401) handleUnauthorized('/api/uploads')
        reject(err)
      }
      xhr.onerror = () =>
        reject(new ApiError(0, 'The upload stopped because the connection to Local Clipper was lost.', 'network_error'))
      xhr.onabort = () => reject(new UploadAbortedError())
      xhr.send(file)
    })

  const promise = (async () => {
    let token: string | null = null
    try {
      token = (await refreshSession()).csrf_token
    } catch {
      token = csrfToken
    }
    if (aborted) throw new UploadAbortedError()
    return send(token)
  })()

  return {
    promise,
    abort: () => {
      aborted = true
      xhr.abort()
    },
  }
}

// ---------------------------------------------------------------- endpoints
const enc = encodeURIComponent

export const api = {
  // auth
  session: () => refreshSession(),
  async login(body: LoginIn): Promise<SessionOut> {
    const s = await request<SessionOut>('POST', '/api/auth/login', body)
    // The session (and its CSRF token) changes on login; re-read it to be sure.
    if (s && typeof s === 'object' && 'authenticated' in s) publishSession(s)
    return refreshSession()
  },
  async logout(): Promise<SessionOut> {
    await request<unknown>('POST', '/api/auth/logout')
    csrfToken = null
    return refreshSession()
  },

  // system & defaults
  system: () => request<SystemOut>('GET', '/api/system'),
  getDefaults: () => request<DefaultsModel>('GET', '/api/settings'),
  saveDefaults: (body: DefaultsModel) => request<DefaultsModel>('PUT', '/api/settings', body),

  // projects
  listProjects: () => request<ProjectSummaryOut[]>('GET', '/api/projects'),
  createProject: (body: ProjectCreateIn) => request<ProjectOut>('POST', '/api/projects', body),
  getProject: (id: string, signal?: AbortSignal) =>
    request<ProjectOut>('GET', `/api/projects/${enc(id)}`, undefined, signal),
  deleteProject: (id: string) => request<unknown>('DELETE', `/api/projects/${enc(id)}`),
  cancelProject: (id: string) => request<unknown>('POST', `/api/projects/${enc(id)}/cancel`),
  resumeProject: (id: string) => request<unknown>('POST', `/api/projects/${enc(id)}/resume`),
  reselectProject: (id: string, body: ReselectIn) =>
    request<unknown>('POST', `/api/projects/${enc(id)}/reselect`, body),
  cleanupProject: (id: string) => request<unknown>('POST', `/api/projects/${enc(id)}/cleanup`),

  // clips
  getClip: (id: string, signal?: AbortSignal) =>
    request<ClipDetailOut>('GET', `/api/clips/${enc(id)}`, undefined, signal),
  patchClip: (id: string, body: ClipEditIn) => request<ClipDetailOut>('PATCH', `/api/clips/${enc(id)}`, body),
  editCaption: (id: string, body: CaptionEditIn) =>
    request<ClipDetailOut>('POST', `/api/clips/${enc(id)}/captions`, body),
  resetCaptions: (id: string) => request<ClipDetailOut>('POST', `/api/clips/${enc(id)}/captions/reset`),
  resetClip: (id: string) => request<ClipDetailOut>('POST', `/api/clips/${enc(id)}/reset`),
  renderClip: (id: string) => request<ClipDetailOut>('POST', `/api/clips/${enc(id)}/render`),
  deleteClip: (id: string) => request<unknown>('DELETE', `/api/clips/${enc(id)}`),
}

// ---------------------------------------------------------------- media URLs (relative)
export const mediaUrls = {
  projectFrame: (projectId: string, srcT: number) =>
    `/media/projects/${enc(projectId)}/frame.jpg?t=${Math.max(0, srcT).toFixed(1)}`,
  exportZip: (projectId: string, clipIds: string[], includeSrt: boolean) => {
    // Always explicit: the API treats a missing `srt` as 1 (include SRT files).
    const params = new URLSearchParams({ clips: clipIds.join(','), srt: includeSrt ? '1' : '0' })
    return `/media/projects/${enc(projectId)}/export.zip?${params.toString()}`
  },
}
