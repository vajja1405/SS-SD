// Captures the box-annotation flow as screenshots on synthetic frames (A/B mode), for the README.
//   PYTHON=python node storyboard.mjs ../docs/images
import { chromium } from '@playwright/test';
import { spawn } from 'node:child_process';
import { mkdirSync } from 'node:fs';
import { join } from 'node:path';

const out = process.argv[2] ?? 'storyboard';
mkdirSync(out, { recursive: true });
const port = 8798;
const server = spawn(process.env.PYTHON ?? 'python3', ['scripts/annotate_boxes.py', '--demo', '--ab', '--port', String(port),
  '--export-dir', 'outputs/storyboard-yolo'], { cwd: '..', stdio: 'ignore' });
const base = `http://127.0.0.1:${port}`;
for (let i = 0; i < 60; i++) {
  try { if ((await fetch(`${base}/api/health`)).ok) break; } catch { /* starting */ }
  await new Promise((r) => setTimeout(r, 500));
}
const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: 1100, height: 760 }, colorScheme: 'light' });
const page = await ctx.newPage();
const shot = (p, name) => p.screenshot({ path: join(out, `boxes-${name}.png`) });
async function frameId(p) {
  const n = Number(((await p.getByTestId('progress').textContent()) ?? '').match(/#(\d+)/)?.[1]);
  return `demo_f${String(n).padStart(3, '0')}`;
}
async function draw(p, b) {
  const r = await p.getByTestId('canvas').boundingBox();
  const s = r.width / 640;
  await p.mouse.move(r.x + b.x1 * s, r.y + b.y1 * s);
  await p.mouse.down();
  await p.mouse.move(r.x + b.x2 * s, r.y + b.y2 * s, { steps: 5 });
  await p.mouse.up();
}
try {
  await page.goto(base);
  await page.getByTestId('canvas').waitFor();
  for (let k = 0; k < 4; k++) {
    const truth = await (await page.request.get(`${base}/api/demo/truth/${await frameId(page)}`)).json();
    await page.keyboard.press('1'); await draw(page, truth[0]);
    await page.keyboard.press('2'); await draw(page, truth[1]);
    if (k === 0) await shot(page, '1-drawn-by-hand');
    await page.keyboard.press('Enter');
    await page.waitForTimeout(300);
  }
  await page.getByTestId('model').filter({ hasText: 'Dashed boxes are pre-labels' }).waitFor();
  await shot(page, '2-kinematics-prelabels');
  await page.keyboard.press('Enter');
  await page.getByTestId('model').filter({ hasText: 'hidden on this frame' }).waitFor();
  await shot(page, '3-hidden-for-measurement');
  const other = await ctx.newPage();                          // same frame open in a second tab
  await other.goto(base);
  await other.getByTestId('canvas').waitFor();
  const truth = await (await page.request.get(`${base}/api/demo/truth/${await frameId(page)}`)).json();
  await page.keyboard.press('1'); await draw(page, truth[0]);
  await page.keyboard.press('2'); await draw(page, truth[1]);
  await page.keyboard.press('Enter');
  await page.waitForTimeout(400);
  await other.keyboard.press('1'); await draw(other, { x1: 60, y1: 60, x2: 130, y2: 110 });
  await other.keyboard.press('Enter');
  await other.getByTestId('conflict').waitFor();
  await shot(other, '4-conflict-between-tabs');
} finally {
  await browser.close();
  server.kill();
}
console.log('saved to', out);
