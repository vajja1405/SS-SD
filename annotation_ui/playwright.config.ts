import { defineConfig, devices } from '@playwright/test';

// One synthetic demo server per browser, so parallel projects never share annotation state.
// PYTHON must point at an interpreter with the annotation dependencies (fastapi, opencv, numpy).
const python = process.env.PYTHON ?? 'python3';
const browsers = [
  { name: 'chromium', device: devices['Desktop Chrome'], port: 8791 },
  { name: 'firefox', device: devices['Desktop Firefox'], port: 8792 },
  { name: 'webkit', device: devices['Desktop Safari'], port: 8793 },
];
// Each spec file gets its own servers so one test's saves never change another's queue.
const suites = [
  { spec: 'annotate.spec.ts', suffix: '', offset: 0 },
  { spec: 'reliability.spec.ts', suffix: '-reliability', offset: 10 },
];
const runs = suites.flatMap((s) => browsers.map((b) => ({ ...b, spec: s.spec, project: b.name + s.suffix, port: b.port + s.offset })));

export default defineConfig({
  testDir: 'e2e',
  fullyParallel: false,
  retries: process.env.CI ? 1 : 0,
  reporter: [['list']],
  expect: { timeout: 10_000 },
  use: { trace: 'retain-on-failure', viewport: { width: 1200, height: 900 } },
  projects: runs.map((r) => ({ name: r.project, testMatch: r.spec,
    use: { ...r.device, viewport: { width: 1200, height: 900 }, baseURL: `http://127.0.0.1:${r.port}` } })),
  webServer: runs.map((b) => ({
    command: `${python} scripts/annotate_boxes.py --demo --port ${b.port} --export-dir outputs/e2e-yolo-${b.project}`,
    cwd: '..',
    url: `http://127.0.0.1:${b.port}/api/health`,
    reuseExistingServer: false,
    timeout: 60_000,
  })),
});
