import { expect, test, type Page } from '@playwright/test';

test.describe.configure({ mode: 'serial' });

type Box = { tool: 'left_tool' | 'right_tool'; x1: number; y1: number; x2: number; y2: number };

async function frameId(page: Page) {
  const text = (await page.getByTestId('progress').textContent()) ?? '';
  const n = Number(text.match(/#(\d+)/)?.[1]);
  return `demo_f${String(n).padStart(3, '0')}`;
}

async function toScreen(page: Page, x: number, y: number) {
  const r = (await page.getByTestId('canvas').boundingBox())!;
  const s = r.width / 640;
  return { x: r.x + x * s, y: r.y + y * s };
}

async function drag(page: Page, from: { x: number; y: number }, to: { x: number; y: number }) {
  const a = await toScreen(page, from.x, from.y);
  const b = await toScreen(page, to.x, to.y);
  await page.mouse.move(a.x, a.y);
  await page.mouse.down();
  await page.mouse.move(b.x, b.y, { steps: 6 });
  await page.mouse.up();
}

async function readBox(page: Page, tool: string) {
  return ((await page.getByTestId(`box-${tool}`).textContent()) ?? '').split(',').map((v) => Number(v.trim()));
}

test('annotate with the mouse, then accept and edit kinematics pre-labels', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByTestId('model')).toContainText('No pre-labels yet');

  // Four frames drawn by hand (keyboard picks the instrument).
  for (let k = 0; k < 4; k++) {
    await expect(page.getByTestId('progress')).toContainText(`Frame ${k + 1} of 24`);
    const truth: Box[] = await (await page.request.get(`/api/demo/truth/${await frameId(page)}`)).json();
    for (const [key, b] of [['1', truth[0]], ['2', truth[1]]] as const) {
      await page.keyboard.press(key);
      await drag(page, { x: b.x1, y: b.y1 }, { x: b.x2, y: b.y2 });
    }
    const left = await readBox(page, 'left_tool');
    expect(Math.abs(left[0] - truth[0].x1)).toBeLessThanOrEqual(2);
    await page.waitForTimeout(350);                 // a person needs to see the item first
    await page.keyboard.press('Enter');
  }

  // Frame 5: pre-labels from the kinematics land on the instruments; Enter accepts them.
  await expect(page.getByTestId('progress')).toContainText('Frame 5 of 24');
  await expect(page.getByTestId('model')).toContainText('Dashed boxes are pre-labels');
  const truth5: Box[] = await (await page.request.get(`/api/demo/truth/${await frameId(page)}`)).json();
  const pre = await readBox(page, 'right_tool');
  expect(Math.abs(pre[0] - truth5[1].x1)).toBeLessThanOrEqual(15);
  await expect(page.getByTestId('boxes')).toContainText('pre-label');
  await page.waitForTimeout(350);                 // a person needs to see the item first
  await page.keyboard.press('Enter');
  await expect(page.getByTestId('acceptance')).toHaveText(/\d+%/);

  // Frame 6: move, resize, delete and redraw.
  await expect(page.getByTestId('progress')).toContainText('Frame 6 of 24');
  const before = await readBox(page, 'left_tool');
  const mid = { x: (before[0] + before[2]) / 2, y: (before[1] + before[3]) / 2 };
  await drag(page, mid, { x: mid.x + 20, y: mid.y });
  const moved = await readBox(page, 'left_tool');
  expect(moved[0] - before[0]).toBeGreaterThan(15);
  await drag(page, { x: moved[2], y: moved[3] }, { x: moved[2] + 30, y: moved[3] + 10 });
  const resized = await readBox(page, 'left_tool');
  expect(resized[2] - moved[2]).toBeGreaterThan(25);
  await page.keyboard.press('Delete');
  await expect(page.getByTestId('box-left_tool')).toHaveCount(0);
  await page.keyboard.press('1');
  await drag(page, { x: before[0], y: before[1] }, { x: before[2], y: before[3] });
  await expect(page.getByTestId('box-left_tool')).toBeVisible();
  await page.getByRole('button', { name: /Save and next/ }).click();
  await expect(page.getByTestId('progress')).toContainText('Frame 7 of 24');

  // Export YOLO labels for everything saved so far.
  await page.getByRole('button', { name: 'Export YOLO labels' }).click();
  await expect(page.getByTestId('export-message')).toContainText('Exported 6 frames');
});
