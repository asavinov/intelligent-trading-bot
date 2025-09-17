Playwright UI CI helper

This repository includes a Playwright-based UI test at `tools/ui_playwright_test.js` that exercises the dashboard UI and asserts that a UI-launched script completes with the expected return code.

Locally:

1. Start the server (from repo root):

```powershell
python -m uvicorn dashboard.main:app --host 127.0.0.1 --port 8000
```

2. Install Node deps and Playwright:

```powershell
cd tools
npm install
npx playwright install
```

3. Run the test:

```powershell
node ui_playwright_test.js
```

CI notes:
- A GitHub Actions workflow is included at `.github/workflows/playwright-ui.yml` which starts the server, installs dependencies, runs the test, and uploads artifacts (server.log, tools/ui_playwright_result.json, screenshot and `logs/jobs/`).
- If the workflow cannot run due to environment constraints (e.g., network or permissions), proceed with other issues—CI failure should not block progress on unrelated work.