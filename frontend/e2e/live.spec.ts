/**
 * LIVE acceptance through the real UI (paid OpenAI calls). Opt-in only:
 *   E2E_LIVE=1 E2E_UPLOAD=/path/to/speech-video.webm npx playwright test e2e/live.spec.ts
 * Uploads the file in the browser, generates clips with the default options and waits for the
 * real pipeline (whisper-1 transcription, gpt-6.1-sol selection/review, framing, rendering).
 */
import fs from 'node:fs'
import path from 'node:path'
import { expect, test } from '@playwright/test'

const UPLOAD = process.env.E2E_UPLOAD ?? ''
const SHOTS = process.env.E2E_SHOTS ?? 'test-results/shots'

test('upload → generate → ranked, rendered clips (live OpenAI)', async ({ page, request }) => {
  test.skip(process.env.E2E_LIVE !== '1', 'live test: set E2E_LIVE=1 (makes paid OpenAI requests)')
  test.skip(!UPLOAD || !fs.existsSync(UPLOAD), 'E2E_UPLOAD must point to a video file')
  test.setTimeout(30 * 60_000)
  fs.mkdirSync(SHOTS, { recursive: true })

  await page.goto('/')
  await expect(page.getByText(/OpenAI is not set up/)).toHaveCount(0)
  await page.getByRole('radio', { name: 'Video file' }).check({ force: true })
  await page.locator('input[type=file]').setInputFiles(UPLOAD)
  await expect(page.getByText(path.basename(UPLOAD)).first()).toBeVisible()
  await page.getByRole('button', { name: /^Generate clips/ }).click()

  await page.waitForURL(/\/projects\/p_/, { timeout: 10 * 60_000 })   // upload (real byte progress) then create
  const projectId = new URL(page.url()).pathname.split('/').pop()!
  await page.screenshot({ path: `${SHOTS}/live-processing.png`, fullPage: true })

  let project: any
  const deadline = Date.now() + 28 * 60_000
  const seen = new Set<string>()
  for (;;) {
    project = await (await request.get(`/api/projects/${projectId}`)).json()
    const job = project.job
    if (job?.stage) seen.add(`${job.stage}${job.progress_total ? ` ${job.progress_current}/${job.progress_total}` : ''}`)
    if (['ready', 'no_clips', 'failed', 'cancelled', 'interrupted'].includes(project.status)) break
    if (Date.now() > deadline) throw new Error(`pipeline did not finish: ${project.status} / ${job?.stage}`)
    await page.waitForTimeout(3000)
  }
  test.info().annotations.push({ type: 'stages seen', description: [...seen].join(' → ') })
  test.info().annotations.push({ type: 'result', description: `${project.status}: ${project.status_detail ?? ''}` })
  expect(project.status, project.job?.error_message ?? project.status_detail ?? '').toBe('ready')
  expect(project.clips.length).toBeGreaterThanOrEqual(1)
  for (const clip of project.clips) {
    expect(clip.render.state).toBe('ready')
    expect(clip.duration).toBeGreaterThanOrEqual(3)
    expect(clip.duration).toBeLessThanOrEqual(180)
  }
  await page.reload()
  await expect(page.getByRole('article')).toHaveCount(project.clips.length)
  await page.screenshot({ path: `${SHOTS}/live-results.png`, fullPage: true })
  fs.writeFileSync(test.info().outputPath('project.json'), JSON.stringify(project, null, 1))
})

test('YouTube link → generate → ranked, rendered clips (live OpenAI + yt-dlp)', async ({ page, request }) => {
  const url = process.env.E2E_YOUTUBE ?? ''
  test.skip(process.env.E2E_LIVE !== '1' || !url, 'live test: set E2E_LIVE=1 and E2E_YOUTUBE=<url>')
  test.setTimeout(30 * 60_000)
  fs.mkdirSync(SHOTS, { recursive: true })
  await page.goto('/')
  await page.getByRole('textbox', { name: 'YouTube link' }).fill(url)
  await page.getByRole('button', { name: /^Generate clips/ }).click()
  await page.waitForURL(/\/projects\/p_/, { timeout: 60_000 })
  const projectId = new URL(page.url()).pathname.split('/').pop()!
  let project: any
  const deadline = Date.now() + 28 * 60_000
  for (;;) {
    project = await (await request.get(`/api/projects/${projectId}`)).json()
    if (['ready', 'no_clips', 'failed', 'cancelled', 'interrupted'].includes(project.status)) break
    if (Date.now() > deadline) throw new Error(`pipeline did not finish: ${project.status}`)
    await page.waitForTimeout(3000)
  }
  test.info().annotations.push({ type: 'result', description: `${project.status}: ${project.title} | ${project.status_detail ?? ''}` })
  expect(project.status, project.job?.error_message ?? project.status_detail ?? '').toBe('ready')
  expect(project.source_kind).toBe('youtube')
  expect(project.clips.length).toBeGreaterThanOrEqual(1)
  await page.reload()
  await expect(page.getByRole('article')).toHaveCount(project.clips.length)
  await page.screenshot({ path: `${SHOTS}/live-youtube-results.png`, fullPage: true })
  fs.writeFileSync(test.info().outputPath('project.json'), JSON.stringify(project, null, 1))
})
