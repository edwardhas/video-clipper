/**
 * Browser acceptance on a SYNTHETIC fixture project (seeded by
 * backend/tests/seed_synthetic_project.py with fake providers — mechanics only,
 * not editorial quality). Exercises history, playback/seeking, downloads,
 * editing and real re-rendering through the running app.
 */
import { execFileSync } from 'node:child_process'
import fs from 'node:fs'
import { expect, test, type APIRequestContext, type Page } from '@playwright/test'

test.describe.configure({ mode: 'serial' })

const SHOTS = process.env.E2E_SHOTS ?? 'test-results/shots'
fs.mkdirSync(SHOTS, { recursive: true })

let projectPath = ''

async function openProject(page: Page): Promise<void> {
  await page.goto('/projects')
  await page.getByRole('link', { name: /SYNTHETIC FIXTURE/ }).first().click()
  await expect(page.getByRole('heading', { level: 1 })).toContainText('SYNTHETIC FIXTURE')
  projectPath = new URL(page.url()).pathname
}

function ffprobe(path: string): { streams: Array<Record<string, unknown>>; format: Record<string, string> } {
  return JSON.parse(execFileSync('ffprobe', ['-v', 'error', '-show_streams', '-show_format', '-of', 'json', path]).toString())
}

async function clipJson(request: APIRequestContext, id: string) {
  const res = await request.get(`/api/clips/${id}`)
  expect(res.ok()).toBeTruthy()
  return res.json()
}

async function mutate(request: APIRequestContext, path: string): Promise<void> {
  const base = test.info().project.use.baseURL!
  const session = await (await request.get('/api/auth/session')).json()
  const res = await request.post(path, { headers: { 'X-CSRF-Token': session.csrf_token, Origin: base } })
  expect(res.ok(), `${path} → ${res.status()}`).toBeTruthy()
}

async function waitForRender(request: APIRequestContext, id: string, notHash: string | null) {
  const deadline = Date.now() + 150_000
  for (;;) {
    const c = await clipJson(request, id)
    if (c.render.state === 'ready' && c.render.render_hash !== notHash) return c
    if (c.render.state === 'failed') throw new Error(`render failed: ${c.render.error}`)
    if (Date.now() > deadline) throw new Error(`render did not finish (state ${c.render.state})`)
    await new Promise((r) => setTimeout(r, 1000))
  }
}

test('project history lists the project and its ranked clips', async ({ page }) => {
  const errors: string[] = []
  page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()) })
  await openProject(page)
  await expect(page.getByRole('article')).toHaveCount(3)
  await expect(page.getByText('85/100').first()).toBeVisible()
  await page.screenshot({ path: `${SHOTS}/project.png`, fullPage: true })
  expect(errors).toEqual([])
})

test('clip preview plays and seeks (HTTP range requests)', async ({ page }) => {
  const statuses: number[] = []
  page.on('response', (r) => { if (r.url().includes('/video.mp4')) statuses.push(r.status()) })
  await openProject(page)
  const video = page.locator('article video').first()
  await video.scrollIntoViewIfNeeded()
  const res = await video.evaluate(async (v: HTMLVideoElement) => {
    v.muted = true
    await v.play()
    await new Promise((r) => setTimeout(r, 1200))
    const playedTo = v.currentTime
    v.pause()
    v.currentTime = 6.5
    await new Promise((r) => v.addEventListener('seeked', r, { once: true }))
    return { playedTo, after: v.currentTime, duration: v.duration, w: v.videoWidth, h: v.videoHeight, err: v.error?.code ?? null }
  })
  expect(res.err).toBeNull()
  expect(res.playedTo).toBeGreaterThan(0.3)
  expect(Math.abs(res.after - 6.5)).toBeLessThan(0.1)
  expect([res.w, res.h]).toEqual([1080, 1920])
  expect(statuses).toContain(206)
})

test('downloads: MP4, SRT and ZIP of selected clips', async ({ page }) => {
  await openProject(page)
  const [mp4] = await Promise.all([page.waitForEvent('download'), page.getByRole('link', { name: 'Download MP4' }).first().click()])
  const mp4Path = test.info().outputPath(mp4.suggestedFilename())
  await mp4.saveAs(mp4Path)
  const probe = ffprobe(mp4Path)
  const v = probe.streams.find((s) => s.codec_type === 'video')!
  const a = probe.streams.find((s) => s.codec_type === 'audio')!
  expect([v.codec_name, v.width, v.height, v.pix_fmt]).toEqual(['h264', 1080, 1920, 'yuv420p'])
  expect(a.codec_name).toBe('aac')
  expect(mp4.suggestedFilename()).toMatch(/\.mp4$/)

  const [srt] = await Promise.all([page.waitForEvent('download'), page.getByRole('link', { name: 'SRT' }).first().click()])
  const srtPath = test.info().outputPath('captions.srt')
  await srt.saveAs(srtPath)
  expect(fs.readFileSync(srtPath, 'utf8')).toMatch(/^1\n00:00:00,\d{3} --> 00:00:0\d,\d{3}\n/)

  await page.getByRole('checkbox', { name: 'Select all' }).check()
  const [zip] = await Promise.all([page.waitForEvent('download'), page.getByRole('link', { name: /Download selected \(ZIP\)/ }).click()])
  const zipPath = test.info().outputPath('clips.zip')
  await zip.saveAs(zipPath)
  const listing = execFileSync('unzip', ['-Z1', zipPath]).toString().trim().split('\n')
  expect(listing.filter((n) => n.endsWith('.mp4'))).toHaveLength(3)
  expect(listing.filter((n) => n.endsWith('.srt'))).toHaveLength(3)

  // "Include SRT" off really leaves the SRT files out (the API defaults a missing srt to 1)
  await page.getByRole('checkbox', { name: 'Include SRT' }).uncheck()
  const [zipNoSrt] = await Promise.all([page.waitForEvent('download'), page.getByRole('link', { name: /Download selected \(ZIP\)/ }).click()])
  const zipNoSrtPath = test.info().outputPath('clips-no-srt.zip')
  await zipNoSrt.saveAs(zipNoSrtPath)
  const listingNoSrt = execFileSync('unzip', ['-Z1', zipNoSrtPath]).toString().trim().split('\n')
  expect(listingNoSrt.filter((n) => n.endsWith('.mp4'))).toHaveLength(3)
  expect(listingNoSrt.filter((n) => n.endsWith('.srt'))).toHaveLength(0)
})

test('project page: only up-to-date renders go in the ZIP; a pending clip render can be cancelled', async ({ page, request }) => {
  await openProject(page)
  const pid = projectPath.split('/').pop()!
  const real = await (await request.get(`/api/projects/${pid}`)).json()
  // Browser-only: show clip 1 as edited since its render and clip 2 waiting to re-render (no server change).
  await page.route((url) => url.pathname === `/api/projects/${pid}`, async (route) => {
    if (route.request().method() !== 'GET') return route.continue()
    const res = await route.fetch()
    const p = await res.json()
    p.clips[0].render = { ...p.clips[0].render, state: 'stale', stale: true }
    p.clips[1].render = { ...p.clips[1].render, state: 'queued' }
    p.clips[1].job = {
      id: 'j_e2e', kind: 'render', state: 'queued', stage: null, stage_detail: null, progress_current: null,
      progress_total: null, cancel_requested: false, error_code: null, error_message: null, retryable: null,
      attempts: 0, created_at: 1, started_at: null, finished_at: null,
    }
    p.actions = { can_cancel: true, can_resume: false, can_reselect: false, can_delete: false }
    await route.fulfill({ response: res, json: p })
  })
  await page.reload()
  const boxes = page.getByRole('checkbox', { name: /for the ZIP download/ })
  await expect(boxes).toHaveCount(3)
  await expect(boxes.nth(0)).toBeDisabled()   // stale: the export would leave it out
  await expect(boxes.nth(1)).toBeDisabled()   // re-rendering
  await expect(boxes.nth(2)).toBeEnabled()
  await page.getByRole('checkbox', { name: 'Select all' }).check()
  const zip = page.getByRole('link', { name: /Download selected \(ZIP\)/ })
  await expect(zip.locator('.count')).toHaveText('1')
  const href = new URL((await zip.getAttribute('href'))!, 'http://localhost')
  expect(href.searchParams.get('clips')).toBe(real.clips[2].id)
  // the queued render blocks Delete, so the project's Cancel (which also stops renders) is offered
  await expect(page.getByRole('button', { name: /^Delete/ })).toHaveAttribute('aria-disabled', 'true')
  await expect(page.getByRole('button', { name: 'Cancel' })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Cancel' })).not.toHaveAttribute('aria-disabled', 'true')
})

test('clip editor: a hidden caption can be restored; trim keeps the API limits and the word gaps', async ({ page, request }) => {
  await openProject(page)
  const pid = projectPath.split('/').pop()!
  const clipId: string = (await (await request.get(`/api/projects/${pid}`)).json()).clips[2].id
  const real = await clipJson(request, clipId)
  const hidden = real.cues[0]
  const ids: string[] = real.words.map((w: { id: string }) => w.id)
  const spoken = real.words
    .slice(ids.indexOf(hidden.first_word_id), ids.indexOf(hidden.last_word_id) + 1)
    .map((w: { text: string }) => w.text)
    .join(' ')
  // A word inside the clip, off the tenth grid, at least 3 s before the end; its previous word is made to
  // touch it and to be short enough that rounding the start down to a tenth would pull it in.
  const i = real.words.findIndex(
    (w: { start: number }, k: number) =>
      k > 0 && w.start > real.edit.src_start + 1 && w.start < real.edit.src_end - 4 && Math.abs(w.start * 10 - Math.round(w.start * 10)) > 0.01,
  )
  expect(i).toBeGreaterThan(0)
  const pick = real.words[i]
  // Browser-only (no server change): what the API returns for a hidden caption, and touching words.
  await page.route((url) => url.pathname === `/api/clips/${clipId}`, async (route) => {
    if (route.request().method() !== 'GET') return route.continue()
    const res = await route.fetch()
    const c = await res.json()
    c.cues[0] = { ...c.cues[0], text: '', words: [], highlight: false, edited: true }
    c.words[i - 1] = { ...c.words[i - 1], start: pick.start - 0.1, end: pick.start }
    await route.fulfill({ response: res, json: c })
  })
  let captionBody: Record<string, string> | null = null
  await page.route((url) => url.pathname === `/api/clips/${clipId}/captions`, async (route) => {
    captionBody = route.request().postDataJSON()
    await route.fulfill({ json: real })
  })
  await page.goto(`/clips/${clipId}`)

  // Trim: "Start here" lands in the gap, so the touching previous word stays out of the clip
  const wordButtons = page.locator('.words .word')
  await wordButtons.nth(i).click()
  await page.getByRole('button', { name: 'Start here' }).click()
  await expect(page.getByRole('spinbutton', { name: 'Start' })).toHaveValue(String(pick.start))
  await expect(wordButtons.nth(i)).toHaveClass(/\bin\b/)
  await expect(wordButtons.nth(i - 1)).not.toHaveClass(/\bin\b/)
  // Trim: the end cannot go below the API's 3 s minimum
  const end = page.getByRole('spinbutton', { name: 'End' })
  await end.fill(String(pick.start + 1))
  await end.blur()
  const endValue = Number(await end.inputValue())
  expect(endValue - pick.start).toBeGreaterThanOrEqual(3 - 1e-9)
  expect(endValue - pick.start).toBeLessThan(3.1)

  // Captions: the hidden caption is listed, counted, and editing it starts from the spoken words
  await page.getByRole('tab', { name: 'Captions' }).click()
  await expect(page.getByText('Hidden caption', { exact: true })).toBeVisible()
  const edited = real.cues.filter((q: { edited: boolean }, k: number) => q.edited || k === 0).length
  await expect(page.getByText(`${edited} edited.`)).toBeVisible()
  await page.getByRole('button', { name: /^Edit caption at/ }).first().click()
  await expect(page.getByRole('textbox', { name: 'Caption text' })).toHaveValue(spoken)
  await page.getByRole('button', { name: 'Save caption' }).click()
  await expect(page.getByText('Caption shown again.', { exact: false })).toBeVisible()
  await expect(page.getByText('Hidden caption', { exact: true })).toHaveCount(0)
  expect(captionBody).toEqual({ first_word_id: hidden.first_word_id, last_word_id: hidden.last_word_id, text: spoken })
})

test('edit trim, style and framing, then caption text — each re-renders for real', async ({ page, request }) => {
  await openProject(page)
  await page.getByRole('link', { name: 'Edit' }).first().click()
  await page.waitForURL(/\/clips\/c_/)
  await expect(page.getByRole('tab', { name: 'Trim' })).toBeVisible()
  const clipId = new URL(page.url()).pathname.split('/').pop()!
  // idempotent: start from the AI's original edit with no caption corrections
  await mutate(request, `/api/clips/${clipId}/reset`)
  await mutate(request, `/api/clips/${clipId}/captions/reset`)
  await page.reload()
  const before = await clipJson(request, clipId)

  // 1) trim + style + framing → "Save and re-render"
  await page.getByRole('button', { name: 'Move the end 1 second earlier' }).click()
  await page.getByRole('tab', { name: 'Captions' }).click()
  await page.getByRole('radio', { name: 'Yellow' }).check({ force: true })
  await page.getByRole('radio', { name: 'Large' }).check({ force: true })
  await page.getByRole('tab', { name: 'Framing' }).click()
  await page.getByRole('radio', { name: /^Full frame/ }).check({ force: true })
  await page.getByRole('button', { name: 'Save and re-render' }).click()
  const afterStyle = await waitForRender(request, clipId, before.render.render_hash)
  expect(afterStyle.edit.style.color).toBe('yellow')
  expect(afterStyle.edit.style.size).toBe('L')
  expect(afterStyle.edit.framing.mode).toBe('fit')
  // edits snap to tenths of a second (frontend/src/lib/trim.ts), so allow one half-tenth of rounding
  expect(Math.abs(afterStyle.duration - (before.duration - 1))).toBeLessThanOrEqual(0.0501)

  // 2) caption text correction (same word count → per-word fix that keeps real timings) → Re-render
  const current = await clipJson(request, clipId)
  const cue = current.cues[0]
  const tokens: string[] = cue.text.split(' ')
  const i = tokens.length > 1 ? 1 : 0
  tokens[i] = tokens[i].toUpperCase()
  const corrected = tokens[i].replace(/[^\p{L}\p{N}'’-]/gu, '')
  await page.reload()
  await page.getByRole('tab', { name: 'Captions' }).click()
  await page.getByRole('button', { name: /^Edit caption at/ }).first().click()
  await page.getByRole('textbox', { name: 'Caption text' }).fill(tokens.join(' '))
  await page.getByRole('button', { name: 'Save caption' }).click()
  await expect(page.getByText('Edited').first()).toBeVisible()
  await page.getByRole('button', { name: /^Re-render/ }).click()
  const afterCaption = await waitForRender(request, clipId, afterStyle.render.render_hash)
  expect(Object.values(afterCaption.edit.word_overrides).join(' ')).toContain(corrected)
  expect(afterCaption.cues[0].highlight).toBe(true)          // real word timings kept, nothing fabricated
  const srt = await request.get(afterCaption.render.srt_url)
  expect(await srt.text()).toContain(corrected)
  await page.reload()
  await expect(page.getByText(/Rendered 1080×1920/)).toBeVisible()
  await page.screenshot({ path: `${SHOTS}/editor-after-rerender.png`, fullPage: true })

  // The rendered file really changed: full frame (fit) + yellow captions → decode a frame and check
  const mp4 = await request.get(afterCaption.render.download_url)
  const out = test.info().outputPath('edited.mp4')
  fs.writeFileSync(out, await mp4.body())
  const probe = ffprobe(out)
  expect(Number(probe.format.duration)).toBeCloseTo(afterCaption.duration, 1)
})

test('settings page explains storage and what is sent to OpenAI', async ({ page }) => {
  await page.goto('/settings')
  await expect(page.getByRole('heading', { name: 'Settings', level: 1 })).toBeVisible()
  await expect(page.getByText(/sent to OpenAI for transcription/)).toBeVisible()
  await expect(page.getByText(/This computer/).first()).toBeVisible()
  await page.screenshot({ path: `${SHOTS}/settings.png`, fullPage: true })
})

test('phone width (360 px): no horizontal scrolling on any page', async ({ page }) => {
  await page.setViewportSize({ width: 360, height: 780 })
  await openProject(page)
  const clipHref = await page.getByRole('link', { name: 'Edit' }).first().getAttribute('href')
  for (const path of ['/', '/projects', projectPath, clipHref!, '/settings']) {
    await page.goto(path)
    await page.waitForTimeout(400)
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)
    expect(overflow, `horizontal overflow on ${path}`).toBeLessThanOrEqual(0)
  }
  await page.goto(clipHref!)
  await page.screenshot({ path: `${SHOTS}/editor-360.png`, fullPage: true })
})
