Run the UI Playwright test (PowerShell instructions)

Prerequisites:
- Node.js (v16+ recommended) installed on the machine running the dashboard.
- The dashboard server must be running and accessible at http://127.0.0.1:8000

Install Playwright (one-time):

PowerShell commands:

```powershell
cd C:\intelligent-trading-bot-master\tools
npm init -y
npm install playwright@1.40.0
# Install browser binaries (chromium)
npx playwright install chromium
```

Run the UI test script:

```powershell
cd C:\intelligent-trading-bot-master\tools
node ui_playwright_test.js
```

Output files (in `tools/`):
- `ui_playwright_result.json` — JSON report with console log, network log, API responses and job status polling
- `ui_playwright_screenshot.png` — full-page screenshot after the click

Notes:
- The script attempts to find a script row mentioning 'download_binance' or 'Binance' and clicks the first run button inside that row. If that heuristic fails it falls back to the first run button on the page.
- If you prefer headful mode for debugging, edit `ui_playwright_test.js` and set `headless: false` in `chromium.launch()`.
