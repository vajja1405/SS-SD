import { expect, test, type Page } from '@playwright/test';

test.describe.configure({ mode: 'serial' });

async function toScreen(page: Page, x: number, y: number) {
  const r = (await page.getByTestId('canvas').boundingBox())!;
  const s = r.width / 640;
  return { x: r.x + x * s, y: r.y + y * s };
}

async function draw(page: Page, x1: number, y1: number, x2: number, y2: number) {
  const a = await toScreen(page, x1, y1);
  const b = await toScreen(page, x2, y2);
  await page.mouse.move(a.x, a.y);
  await page.mouse.down();
  await page.mouse.move(b.x, b.y, { steps: 5 });
  await page.mouse.up();
}

async function frameId(page: Page) {
  const n = Number(((await page.getByTestId('progress').textContent()) ?? '').match(/#(\d+)/)?.[1]);
  return `demo_f${String(n).padStart(3, '0')}`;
}

test('a save whose response is lost is retried and stored once', async ({ page }) => {
  await page.goto('/');
  const id = await frameId(page);
  await page.keyboard.press('1');
  await draw(page, 100, 100, 180, 160);
  let dropped = false;
  await page.route('**/annotation', async (route) => {
    if (!dropped) {
      dropped = true;
      await route.fetch();                   // the server stores the boxes...
      await route.abort('connectionreset');  // ...the browser never hears back
    } else {
      await route.continue();
    }
  });
  await page.keyboard.press('Enter');
  await expect(page.getByTestId('progress')).toContainText('Frame 2 of 24');
  const saved = await (await page.request.get(`/api/frames/${id}`)).json();
  expect(saved.version).toBe(1);                       // the retry was recognised as the same save
  expect(saved.boxes[0]).toMatchObject({ tool: 'left_tool' });
});

test('unsaved boxes survive a reload', async ({ page }) => {
  await page.goto('/');
  await page.keyboard.press('2');
  await draw(page, 300, 200, 380, 260);
  const before = await page.getByTestId('box-right_tool').textContent();
  await page.reload();
  await expect(page.getByTestId('save-state')).toContainText('Restored boxes you had not saved yet');
  await expect(page.getByTestId('box-right_tool')).toHaveText(before!);
});

test('a second tab that saved first causes a conflict, not a silent overwrite', async ({ browser }) => {
  const a = await (await browser.newContext()).newPage();
  const b = await (await browser.newContext()).newPage();
  await a.goto('/');
  await b.goto('/');
  const id = await frameId(a);
  expect(await frameId(b)).toBe(id);
  await a.keyboard.press('1');
  await draw(a, 50, 50, 120, 110);
  await a.keyboard.press('Enter');
  await expect(a.getByTestId('progress')).not.toContainText(`#${Number(id.slice(-3))} `);
  await b.keyboard.press('1');
  await draw(b, 200, 150, 260, 210);
  await b.keyboard.press('Enter');
  await expect(b.getByTestId('conflict')).toBeVisible();
  await b.getByRole('button', { name: 'Load the saved boxes' }).click();
  const stored = (await (await b.request.get(`/api/frames/${id}`)).json()).boxes[0];
  await expect(b.getByTestId('box-left_tool')).toHaveText([stored.x1, stored.y1, stored.x2, stored.y2].map(Math.round).join(', '));
  await b.keyboard.press('1');
  await draw(b, 200, 150, 260, 210);
  await b.keyboard.press('Enter');                     // now based on the saved version: no conflict
  await expect(b.getByTestId('conflict')).toHaveCount(0);
  const saved = await (await b.request.get(`/api/frames/${id}`)).json();
  expect(saved.version).toBe(2);
  expect(Math.abs(saved.boxes.find((x: { tool: string }) => x.tool === 'left_tool').x1 - 200)).toBeLessThanOrEqual(1);
});
