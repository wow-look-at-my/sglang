// Runs the dag-view canvas vs SVG bench in Google Chrome on this host's GPU
// and prints the results.

import { chromium } from 'playwright-core';
import { pathToFileURL } from 'node:url';
import { resolve } from 'node:path';

const query = process.argv[2] ?? 'auto';
const args = (process.env.CHROME_ARGS ?? '').split(' ').filter(Boolean);
const browser = await chromium.launch({ executablePath: '/usr/bin/google-chrome', headless: true, args });
const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
page.on('pageerror', (e) => console.error('pageerror', String(e)));
page.on('console', (m) => m.type() === 'error' && console.error('console', m.text()));

await page.goto('chrome://gpu');
await page.waitForTimeout(1500);
const status = await page.evaluate(() => {
	const root = document.querySelector('info-view')?.shadowRoot;
	return root?.textContent?.match(/Graphics Feature Status[\s\S]*?(?=Version Information|Driver Bug Workarounds)/)?.[0] ?? 'no feature status';
});
console.log(status.replace(/\n\s*\n/g, '\n'));

await page.goto(pathToFileURL(resolve('dag-bench/index.html')).href + `?${query}`);
await page.waitForFunction(() => window.__results !== undefined, undefined, { timeout: 1_800_000 });
const text = await page.evaluate(() => document.getElementById('out').textContent);
console.log(text);
const renderer = await page.evaluate(() => window.__results.renderer);
await browser.close();
if (/swiftshader|llvmpipe|software/i.test(renderer)) {
	console.error(`::error::Chrome rendered on ${renderer}, not a GPU. These numbers are software numbers.`);
	process.exit(1);
}
