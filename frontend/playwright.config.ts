import { defineConfig } from '@playwright/test'

/**
 * Browser acceptance tests against a RUNNING Local Clipper instance.
 * E2E_BASE_URL selects the instance (default http://127.0.0.1:8765).
 * See ../VALIDATION.md for how the synthetic and hosted-mode instances are started.
 */
export default defineConfig({
  testDir: './e2e',
  timeout: 180_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  workers: 1,
  reporter: [['list']],
  outputDir: 'test-results',
  use: {
    baseURL: process.env.E2E_BASE_URL ?? 'http://127.0.0.1:8765',
    browserName: 'chromium',
    channel: 'chromium',
    viewport: { width: 1360, height: 1000 },
    acceptDownloads: true,
    screenshot: 'only-on-failure',
    trace: 'retain-on-failure',
  },
})
