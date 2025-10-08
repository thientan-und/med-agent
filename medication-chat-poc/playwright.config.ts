import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests/e2e',
  timeout: 120 * 1000,
  expect: {
    timeout: 20 * 1000,
  },
  retries: process.env.CI ? 1 : 0,
  reporter: [['list'], ['html', { open: 'never' }]],
  use: {
    baseURL: process.env.E2E_BASE_URL || 'http://127.0.0.1:3000',
    trace: 'on-first-retry',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
  },
  webServer: {
    command: 'npm --prefix frontend run dev',
    port: 3000,
    reuseExistingServer: !process.env.CI,
    timeout: 180 * 1000,
    env: {
      ...process.env,
      REQUIRE_DOCTOR_APPROVAL: process.env.REQUIRE_DOCTOR_APPROVAL || 'true',
      BACKEND_URL: process.env.BACKEND_URL || 'http://localhost:8000',
    },
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
});
