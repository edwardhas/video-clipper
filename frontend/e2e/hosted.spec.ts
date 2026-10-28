/**
 * Hosted-mode browser tests (run against the compose stack behind Caddy HTTPS):
 *   E2E_BASE_URL=https://localhost:8443 E2E_PASSWORD=... E2E_UPLOAD=/path/video.mp4 npx playwright test e2e/hosted.spec.ts
 * The instance must use a short CLIPPER_SESSION_TTL_HOURS (e.g. 0.02 = 72 s) for the expiry test.
 */
import fs from 'node:fs'
import path from 'node:path'
import { expect, test, type BrowserContext, type Page } from '@playwright/test'

test.describe.configure({ mode: 'serial' })
test.use({ ignoreHTTPSErrors: true })

const PASSWORD = process.env.E2E_PASSWORD ?? ''
const UPLOAD = process.env.E2E_UPLOAD ?? ''

async function login(page: Page, password = PASSWORD): Promise<void> {
  await page.goto('/')
  await expect(page).toHaveURL(/\/login/)
  await page.getByLabel('Username').fill('owner')
  await page.getByLabel('Password').fill(password)
  await page.getByRole('button', { name: /sign in/i }).click()
}

test.beforeAll(() => {
  expect(PASSWORD, 'E2E_PASSWORD must be set').not.toBe('')
})

test('unauthenticated visitors are sent to the sign-in page; wrong password is refused', async ({ page }) => {
  await page.goto('/projects')
  await expect(page).toHaveURL(/\/login/)
  await expect(page.getByRole('heading', { name: 'Sign in' })).toBeVisible()
  await login(page, 'definitely-not-the-password')
  await expect(page.getByRole('alert').or(page.getByText(/incorrect|wrong|not recognised|invalid/i)).first()).toBeVisible()
  await expect(page).toHaveURL(/\/login/)
})

test('login sets a host-only, Secure, HttpOnly, SameSite=Lax session cookie; authorized use works', async ({ page, context }) => {
  await login(page)
  await expect(page).not.toHaveURL(/\/login/)
  const cookies = await context.cookies()
  const session = cookies.find((c) => c.name === 'clipper_session')!
  expect(session).toBeTruthy()
  expect(session.httpOnly).toBe(true)
  expect(session.secure).toBe(true)
  expect(session.sameSite).toBe('Lax')
  expect(session.domain).toBe('localhost')           // host-only: no leading dot / parent domain
  await page.goto('/settings')
  await expect(page.getByRole('heading', { name: 'Settings', level: 1 })).toBeVisible()
  const sys = await page.request.get('/api/system')
  expect(sys.status()).toBe(200)
  const body = JSON.stringify(await sys.json())
  expect(body).not.toMatch(/sk-|argon2|session_secret/i)
})

test('a background job keeps running after the browser is closed', async ({ browser }) => {
  test.skip(!UPLOAD || !fs.existsSync(UPLOAD), 'E2E_UPLOAD not provided')
  const ctx: BrowserContext = await browser.newContext({ ignoreHTTPSErrors: true })
  const page = await ctx.newPage()
  await login(page)
  await expect(page).not.toHaveURL(/\/login/)
  const base = test.info().project.use.baseURL!
  const session = await (await page.request.get('/api/auth/session')).json()
  const headers = { 'X-CSRF-Token': session.csrf_token, Origin: base }
  const up = await page.request.post('/api/uploads', {
    headers: { ...headers, 'Content-Type': 'application/octet-stream', 'X-Filename': encodeURIComponent(path.basename(UPLOAD)) },
    data: fs.readFileSync(UPLOAD),
  })
  expect(up.ok(), await up.text()).toBe(true)
  const upload = await up.json()
  const created = await page.request.post('/api/projects', {
    headers: { ...headers, 'Content-Type': 'application/json' },
    data: { source: { kind: 'upload', upload_id: upload.id } },
  })
  expect(created.ok(), await created.text()).toBe(true)
  const project = await created.json()
  const stateAtClose = project.job?.state ?? project.status
  await ctx.close()                                   // browser gone while the job is queued/running

  // Observe from a fresh, separately authenticated context.
  const ctx2 = await browser.newContext({ ignoreHTTPSErrors: true })
  const page2 = await ctx2.newPage()
  await login(page2)
  await expect(page2).not.toHaveURL(/\/login/)
  let p = project
  const deadline = Date.now() + 120_000
  while (Date.now() < deadline) {
    p = await (await page2.request.get(`/api/projects/${project.id}`)).json()
    if (['failed', 'ready', 'no_clips'].includes(p.status)) break
    await new Promise((r) => setTimeout(r, 1500))
  }
  test.info().annotations.push({ type: 'job', description: `state at browser close: ${stateAtClose}; final: ${p.status} / ${p.job?.stage}: ${p.job?.error_message}` })
  expect(['queued', 'running']).toContain(stateAtClose)
  const done = Object.fromEntries(p.stages.map((s: { key: string; state: string }) => [s.key, s.state]))
  expect(done.import).toBe('done')
  expect(done.probe).toBe('done')
  expect(done.audio).toBe('done')                     // real work finished with no browser attached
  await page2.goto(`/projects/${project.id}`)
  await expect(page2.getByText(/Extract audio/)).toBeVisible()
  await ctx2.close()
})

test('logout ends the session for the API and media routes', async ({ page, context }) => {
  await login(page)
  await expect(page).not.toHaveURL(/\/login/)
  expect((await page.request.get('/api/projects')).status()).toBe(200)
  await page.goto('/settings')
  await page.getByRole('button', { name: /sign out|log out/i }).click()
  await expect(page).toHaveURL(/\/login/)
  expect((await context.cookies()).find((c) => c.name === 'clipper_session')).toBeFalsy()
  expect((await page.request.get('/api/projects')).status()).toBe(401)
})

test('sessions expire server-side after the configured lifetime', async ({ page }) => {
  test.setTimeout(240_000)
  await login(page)
  await expect(page).not.toHaveURL(/\/login/)
  expect((await page.request.get('/api/projects')).status()).toBe(200)
  await page.waitForTimeout(80_000)                   // CLIPPER_SESSION_TTL_HOURS=0.02 (72 s) on the test instance
  expect((await page.request.get('/api/projects')).status()).toBe(401)
  await page.goto('/projects')
  await expect(page).toHaveURL(/\/login/)
})
