/**
 * Run AFTER the app was stopped and started again (scripts/stop.sh + scripts/start.sh):
 * project history, clip edits and rendered media must survive the restart.
 */
import { expect, test } from '@playwright/test'

test('project history, edits and renders survive an application restart', async ({ page, request }) => {
  const health = await request.get('/api/health')
  expect(health.ok()).toBeTruthy()
  await page.goto('/projects')
  await page.getByRole('link', { name: /SYNTHETIC FIXTURE/ }).first().click()
  await expect(page.getByRole('article')).toHaveCount(3)
  await page.getByRole('link', { name: 'Edit' }).first().click()
  await page.waitForURL(/\/clips\/c_/)
  const clipId = new URL(page.url()).pathname.split('/').pop()!
  const clip = await (await request.get(`/api/clips/${clipId}`)).json()
  expect(clip.render.state).toBe('ready')
  expect(clip.edit.style.color).toBe('yellow')            // edit made before the restart
  expect(clip.edit.framing.mode).toBe('fit')
  expect(Object.keys(clip.edit.word_overrides).length).toBeGreaterThan(0)
  const video = page.getByLabel(/Rendered preview of/).locator('video').or(page.locator('video')).first()
  const ok = await video.evaluate(async (v: HTMLVideoElement) => {
    if (v.readyState < 1) await new Promise((r) => v.addEventListener('loadedmetadata', r, { once: true }))
    return { w: v.videoWidth, h: v.videoHeight, err: v.error?.code ?? null }
  })
  expect(ok).toEqual({ w: 1080, h: 1920, err: null })
})
