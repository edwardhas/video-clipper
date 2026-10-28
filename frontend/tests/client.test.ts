import assert from 'node:assert/strict'
import { afterEach, test } from 'node:test'

type Call = { url: string; init: RequestInit }
type Handler = (url: string, init: RequestInit, n: number) => Response | Promise<Response>

let counter = 0
const realFetch = globalThis.fetch

/** A fresh copy of the client module (its CSRF cache is module state). */
async function freshClient() {
  counter += 1
  return import(`../src/api/client.ts?case=${counter}`)
}

function json(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } })
}

function session(token: string | null, extra: Record<string, unknown> = {}) {
  return {
    authenticated: true,
    auth_required: false,
    mode: 'local',
    username: null,
    csrf_token: token,
    expires_at: null,
    ...extra,
  }
}

function installFetch(handler: Handler): Call[] {
  const calls: Call[] = []
  globalThis.fetch = (async (input: RequestInfo | URL, init: RequestInit = {}) => {
    const url = String(input)
    calls.push({ url, init })
    return handler(url, init, calls.length)
  }) as typeof fetch
  return calls
}

function header(call: Call, name: string): string | undefined {
  return (call.init.headers as Record<string, string> | undefined)?.[name]
}

afterEach(() => {
  globalThis.fetch = realFetch
})

test('GET uses the relative path, same-origin credentials and no CSRF header', async () => {
  const { api } = await freshClient()
  const calls = installFetch(() => json(200, []))
  const out = await api.listProjects()
  assert.deepEqual(out, [])
  assert.equal(calls.length, 1)
  assert.equal(calls[0].url, '/api/projects')
  assert.equal(calls[0].init.method, 'GET')
  assert.equal(calls[0].init.credentials, 'same-origin')
  assert.equal(header(calls[0], 'X-CSRF-Token'), undefined)
})

test('mutating requests fetch the CSRF token once and reuse it', async () => {
  const { api } = await freshClient()
  const calls = installFetch((url) => (url === '/api/auth/session' ? json(200, session('tok1')) : json(200, {})))
  await api.cancelProject('p1')
  await api.resumeProject('p1')
  assert.deepEqual(
    calls.map((c) => c.url),
    ['/api/auth/session', '/api/projects/p1/cancel', '/api/projects/p1/resume'],
  )
  assert.equal(header(calls[1], 'X-CSRF-Token'), 'tok1')
  assert.equal(header(calls[2], 'X-CSRF-Token'), 'tok1')
  assert.equal(calls[1].init.method, 'POST')
})

test('JSON bodies are sent with a JSON content type', async () => {
  const { api } = await freshClient()
  const calls = installFetch((url) => (url === '/api/auth/session' ? json(200, session('t')) : json(200, {})))
  await api.patchClip('c 1', { src_start: 1.5, src_end: 9 })
  const patch = calls[1]
  assert.equal(patch.url, '/api/clips/c%201')
  assert.equal(patch.init.method, 'PATCH')
  assert.equal(header(patch, 'Content-Type'), 'application/json')
  assert.deepEqual(JSON.parse(String(patch.init.body)), { src_start: 1.5, src_end: 9 })
})

test('a 403 CSRF rejection refreshes the token and retries exactly once', async () => {
  const { api } = await freshClient()
  let sessions = 0
  const calls = installFetch((url, init) => {
    if (url === '/api/auth/session') {
      sessions += 1
      return json(200, session(`tok${sessions}`))
    }
    const token = (init.headers as Record<string, string>)['X-CSRF-Token']
    return token === 'tok2' ? json(200, { ok: true }) : json(403, { detail: 'stale', code: 'csrf_failed' })
  })
  await api.deleteProject('p1')
  assert.deepEqual(
    calls.map((c) => c.url),
    ['/api/auth/session', '/api/projects/p1', '/api/auth/session', '/api/projects/p1'],
  )
})

test('a second CSRF rejection is reported, not retried forever', async () => {
  const { api, ApiError } = await freshClient()
  const calls = installFetch((url) =>
    url === '/api/auth/session' ? json(200, session('x')) : json(403, { detail: 'Token out of date.', code: 'csrf_failed' }),
  )
  await assert.rejects(api.deleteProject('p1'), (err: unknown) => {
    assert.ok(err instanceof ApiError)
    assert.equal((err as InstanceType<typeof ApiError>).message, 'Token out of date.')
    return true
  })
  assert.equal(calls.filter((c) => c.url === '/api/projects/p1').length, 2)
})

test('other 403 errors are not retried', async () => {
  const { api } = await freshClient()
  const calls = installFetch((url) =>
    url === '/api/auth/session' ? json(200, session('x')) : json(403, { detail: 'Origin not allowed.', code: 'bad_origin' }),
  )
  await assert.rejects(api.deleteProject('p1'), /Origin not allowed/)
  assert.equal(calls.filter((c) => c.url === '/api/projects/p1').length, 1)
})

test('401 calls the unauthorized handler, except for the auth endpoints', async () => {
  const { api, setUnauthorizedHandler } = await freshClient()
  let redirects = 0
  setUnauthorizedHandler(() => {
    redirects += 1
  })
  installFetch((url) => {
    if (url === '/api/auth/session') return json(200, session(null, { authenticated: false, auth_required: true }))
    if (url === '/api/auth/login') return json(401, { detail: 'Wrong username or password.', code: 'invalid_credentials' })
    return json(401, { detail: 'Please sign in to continue.', code: 'auth_required' })
  })
  await assert.rejects(api.listProjects(), /Please sign in/)
  assert.equal(redirects, 1)
  await assert.rejects(api.login({ username: 'a', password: 'b' }), /Wrong username or password/)
  assert.equal(redirects, 1)
})

test('login and logout refresh the cached session token', async () => {
  const { api, onSessionChange } = await freshClient()
  const seen: (string | null)[] = []
  onSessionChange((s: { csrf_token: string | null }) => seen.push(s.csrf_token))
  let phase = 'out'
  const calls = installFetch((url) => {
    if (url === '/api/auth/session') {
      return json(200, phase === 'in' ? session('after', { auth_required: true }) : session(null, { authenticated: false, auth_required: true }))
    }
    if (url === '/api/auth/login') {
      phase = 'in'
      return json(200, session('after', { auth_required: true }))
    }
    if (url === '/api/auth/logout') {
      phase = 'out'
      return json(200, session(null, { authenticated: false, auth_required: true }))
    }
    return json(200, {})
  })
  const s = await api.login({ username: 'owner', password: 'pw' })
  assert.equal(s.authenticated, true)
  await api.cancelProject('p1')
  const cancel = calls.find((c) => c.url === '/api/projects/p1/cancel')
  assert.ok(cancel)
  assert.equal(header(cancel, 'X-CSRF-Token'), 'after')
  await api.logout()
  assert.equal(seen.at(-1), null)
})

test('error bodies become plain-language ApiErrors', async () => {
  const { errorFromBody } = await freshClient()
  const e1 = errorFromBody(409, JSON.stringify({ detail: 'Already running.', code: 'conflict' }))
  assert.equal(e1.message, 'Already running.')
  assert.equal(e1.code, 'conflict')
  assert.equal(e1.status, 409)
  const e2 = errorFromBody(422, JSON.stringify({ detail: [{ loc: ['body', 'x'], msg: 'Field required', type: 'missing' }] }))
  assert.match(e2.message, /Field required/)
  assert.equal(e2.code, 'validation_error')
  const e3 = errorFromBody(502, '<html>Bad gateway</html>')
  assert.match(e3.message, /internal error/)
  assert.ok(!e3.message.includes('<html>'))
  const e4 = errorFromBody(413, '')
  assert.match(e4.message, /upload limit/)
})

test('network failures are reported without technical detail', async () => {
  const { api, ApiError } = await freshClient()
  globalThis.fetch = (async () => {
    throw new TypeError('fetch failed: ECONNREFUSED 127.0.0.1')
  }) as typeof fetch
  await assert.rejects(api.system(), (err: unknown) => {
    assert.ok(err instanceof ApiError)
    const e = err as InstanceType<typeof ApiError>
    assert.equal(e.status, 0)
    assert.ok(!e.message.includes('127.0.0.1'))
    return true
  })
})

test('media URLs are relative and encoded', async () => {
  const { mediaUrls } = await freshClient()
  assert.equal(mediaUrls.projectFrame('p1', 12.345), '/media/projects/p1/frame.jpg?t=12.3')
  assert.equal(mediaUrls.projectFrame('p1', -4), '/media/projects/p1/frame.jpg?t=0.0')
  const zip = mediaUrls.exportZip('p1', ['a', 'b'], true)
  assert.ok(zip.startsWith('/media/projects/p1/export.zip?'))
  const params = new URLSearchParams(zip.split('?')[1])
  assert.equal(params.get('clips'), 'a,b')
  assert.equal(params.get('srt'), '1')
  // Finding 22: the API defaults a missing `srt` to 1, so "Include SRT" off must send srt=0 explicitly
  // (this used to assert the parameter was left out, which silently kept the SRT files).
  assert.equal(new URLSearchParams(mediaUrls.exportZip('p1', ['a'], false).split('?')[1]).get('srt'), '0')
})

// ------------------------------------------------------------------ uploads (XMLHttpRequest)
class FakeXHR {
  static onSend: ((x: FakeXHR) => void) | null = null
  method = ''
  url = ''
  headers: Record<string, string> = {}
  body: unknown = null
  status = 0
  responseText = ''
  upload: { onprogress: ((e: { loaded: number; total: number; lengthComputable: boolean }) => void) | null } = {
    onprogress: null,
  }
  onload: (() => void) | null = null
  onerror: (() => void) | null = null
  onabort: (() => void) | null = null
  open(method: string, url: string) {
    this.method = method
    this.url = url
  }
  setRequestHeader(k: string, v: string) {
    this.headers[k] = v
  }
  send(body: unknown) {
    this.body = body
    FakeXHR.onSend?.(this)
  }
  abort() {
    this.onabort?.()
  }
}

const realXHR = (globalThis as Record<string, unknown>).XMLHttpRequest

function withFakeXHR() {
  ;(globalThis as Record<string, unknown>).XMLHttpRequest = FakeXHR
}

afterEach(() => {
  ;(globalThis as Record<string, unknown>).XMLHttpRequest = realXHR
  FakeXHR.onSend = null
})

test('uploads stream the raw file with octet-stream, encoded X-Filename, CSRF and byte progress', async () => {
  const { uploadFile } = await freshClient()
  withFakeXHR()
  installFetch(() => json(200, session('up-token')))
  const file = new File([new Uint8Array(1000)], 'My talk ü.mp4', { type: 'video/mp4' })
  let sent: FakeXHR | null = null
  FakeXHR.onSend = (x) => {
    sent = x
    x.upload.onprogress?.({ loaded: 400, total: 1000, lengthComputable: true })
    x.upload.onprogress?.({ loaded: 1000, total: 1000, lengthComputable: true })
    x.status = 201
    x.responseText = JSON.stringify({ id: 'u1', filename: 'My talk ü.mp4', size_bytes: 1000 })
    x.onload?.()
  }
  const progress: [number, number][] = []
  const handle = uploadFile(file, (l: number, t: number) => progress.push([l, t]))
  const out = await handle.promise
  assert.deepEqual(out, { id: 'u1', filename: 'My talk ü.mp4', size_bytes: 1000 })
  assert.ok(sent)
  const x = sent as FakeXHR
  assert.equal(x.method, 'POST')
  assert.equal(x.url, '/api/uploads')
  assert.equal(x.headers['Content-Type'], 'application/octet-stream')
  assert.equal(x.headers['X-Filename'], encodeURIComponent('My talk ü.mp4'))
  assert.equal(x.headers['X-CSRF-Token'], 'up-token')
  assert.equal(x.body, file)
  assert.deepEqual(progress, [
    [400, 1000],
    [1000, 1000],
  ])
})

test('cancelling an upload aborts the request', async () => {
  const { uploadFile, UploadAbortedError } = await freshClient()
  withFakeXHR()
  installFetch(() => json(200, session('t')))
  let xhr: FakeXHR | null = null
  FakeXHR.onSend = (x) => {
    xhr = x
  }
  const handle = uploadFile(new File([new Uint8Array(10)], 'a.mp4'), () => undefined)
  // Wait until the request is actually in flight.
  for (let i = 0; i < 20 && !xhr; i++) await new Promise((r) => setTimeout(r, 1))
  assert.ok(xhr)
  handle.abort()
  await assert.rejects(handle.promise, (err: unknown) => err instanceof UploadAbortedError)
})

test('cancelling before the upload starts never sends the file', async () => {
  const { uploadFile, UploadAbortedError } = await freshClient()
  withFakeXHR()
  installFetch(() => json(200, session('t')))
  let sends = 0
  FakeXHR.onSend = () => {
    sends += 1
  }
  const handle = uploadFile(new File([new Uint8Array(10)], 'a.mp4'), () => undefined)
  handle.abort()
  await assert.rejects(handle.promise, (err: unknown) => err instanceof UploadAbortedError)
  assert.equal(sends, 0)
})

test('upload errors carry the server message', async () => {
  const { uploadFile } = await freshClient()
  withFakeXHR()
  installFetch(() => json(200, session('t')))
  FakeXHR.onSend = (x) => {
    x.status = 413
    x.responseText = JSON.stringify({ detail: 'The file is larger than the 4096 MB upload limit.', code: 'too_large' })
    x.onload?.()
  }
  const handle = uploadFile(new File([new Uint8Array(10)], 'a.mp4'), () => undefined)
  await assert.rejects(handle.promise, /larger than the 4096 MB upload limit/)
})
