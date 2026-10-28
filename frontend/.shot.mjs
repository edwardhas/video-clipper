import { chromium } from '@playwright/test';
const [url, out, w = '1360'] = process.argv.slice(2);
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: Number(w), height: 900 } });
await page.goto(url); await page.waitForTimeout(1500);
const toggle = page.getByText('Show processing steps'); if (await toggle.count()) await toggle.click();
await page.waitForTimeout(300);
await page.screenshot({ path: out, fullPage: true });
await browser.close();
