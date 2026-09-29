import { defineConfig, devices } from '@playwright/test';

// One synthetic demo server per browser, so parallel projects never share annotation state.
// PYTHON must point at an interpreter with the annotation dependencies (fastapi, opencv, numpy).
const python = process.env.PYTHON ?? 'python3';
const browsers = [
  { name: 'chromium', device: devices['Desktop Chrome'], port: 8791 },
  { name: 'firefox', device: devices['Desktop Firefox'], port: 8792 },
  { name: 'webkit', device: devices['Desktop Safari'], port: 8793 },
];

export default defineConfig({
  testDir: 'e2e',
  fullyParallel: false,
  retries: process.env.CI ? 1 : 0,
  reporter: [['list']],
  expect: { timeout: 10_000 },
  use: { trace: 'retain-on-failure', viewport: { width: 1200, height: 900 } },
  projects: browsers.map((b) => ({ name: b.name, use: { ...b.device, viewport: { width: 1200, height: 900 }, baseURL: `http://127.0.0.1:${b.port}` } })),
  webServer: browsers.map((b) => ({
    command: `${python} scripts/annotate_boxes.py --demo --port ${b.port} --export-dir outputs/e2e-yolo-${b.name}`,
    cwd: '..',
    url: `http://127.0.0.1:${b.port}/api/health`,
    reuseExistingServer: false,
    timeout: 60_000,
  })),
});
