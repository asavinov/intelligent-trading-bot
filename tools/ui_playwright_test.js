// Playwright UI test for Intelligent Trading Bot dashboard
// - Opens http://127.0.0.1:8000
// - Navigates to #scripts
// - Finds the script row for download_binance (or Binance) and clicks the "🚀 اجرا" run button
// - Captures network requests/responses for /api/scripts/run and /api/scripts/status
// - Captures console messages and a screenshot
// - Saves report to tools/ui_playwright_result.json and screenshot to tools/ui_playwright_screenshot.png

const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

(async () => {
    // Configurable expected return code for the run job (default 0)
    const expectedReturnCode = process.env.EXPECTED_RETURN_CODE ? parseInt(process.env.EXPECTED_RETURN_CODE, 10) : 0;

    const out = {
        timestamp: new Date().toISOString(),
        url: 'http://127.0.0.1:8000/',
        console: [],
        network: [],
        runResponse: null,
        job_id: null,
        job_statuses: [],
        error: null
    };

    const browser = await chromium.launch({ headless: true });
    const context = await browser.newContext();
    const page = await context.newPage();

    page.on('console', msg => {
        out.console.push({ type: msg.type(), text: msg.text(), location: msg.location() });
    });

    page.on('request', req => {
        out.network.push({ event: 'request', url: req.url(), method: req.method(), headers: req.headers(), postData: req.postData() });
    });
    page.on('response', async res => {
        try {
            const url = res.url();
            const ct = res.headers()['content-type'] || '';
            let body = null;
            if (ct.includes('application/json') || url.includes('/api/')) {
                try { body = await res.json(); } catch (e) { body = await res.text().catch(() => null); }
            }
            out.network.push({ event: 'response', url, status: res.status(), headers: res.headers(), body });
        } catch (e) {
            out.network.push({ event: 'response_error', error: String(e) });
        }
    });

    try {
        // navigate to root page and wait for load (networkidle can hang due to external CDNs)
        await page.goto(out.url, { waitUntil: 'load', timeout: 60000 });
        // navigate to scripts view and wait for the main UI to be ready
        await page.goto(out.url + '#scripts', { waitUntil: 'load', timeout: 60000 });

        // wait for a run button (heuristic: button that contains the 🚀 emoji) to appear
        // this is more robust than waiting for networkidle
        await page.waitForSelector('button:has-text("🚀")', { timeout: 60000 }).catch(() => null);

        // find the script row that mentions Binance or download_binance
        const scriptHandle = await page.evaluateHandle(() => {
            // Try a few heuristics to find the right row
            const candidates = Array.from(document.querySelectorAll('div, li, tr'));
            for (const node of candidates) {
                try {
                    const txt = (node.innerText || '').toLowerCase();
                    if (txt.includes('download_binance') || txt.includes('binance')) {
                        // find a button inside this node
                        const btn = node.querySelector('button');
                        if (btn) return btn;
                    }
                } catch (e) { }
            }
            // fallback: look for any button that contains the run icon text
            const runButtons = Array.from(document.querySelectorAll('button')).filter(b => (b.innerText || '').includes('🚀'));
            if (runButtons.length > 0) return runButtons[0];
            return null;
        });

        if (!scriptHandle) {
            out.error = 'Could not find script run button for download_binance via UI selectors';
            throw new Error(out.error);
        }

        // click the button
        const btn = scriptHandle.asElement();
        if (!btn) {
            out.error = 'Handle found but not an element';
            throw new Error(out.error);
        }

        // click and wait explicitly for the /api/scripts/run response and parse it
        let runResp = null;
        await btn.click();
        try {
            runResp = await page.waitForResponse(resp => resp.url().includes('/api/scripts/run') && (resp.status() === 200 || resp.status() === 201), { timeout: 15000 });
        } catch (e) {
            // fallback: maybe response captured in page.on('response') handlers
            runResp = null;
        }

        if (runResp) {
            try {
                const json = await runResp.json();
                out.runResponse = { url: runResp.url(), status: runResp.status(), body: json };
                if (json && json.job_id) out.job_id = json.job_id;
            } catch (e) {
                try {
                    const text = await runResp.text();
                    out.runResponse = { url: runResp.url(), status: runResp.status(), body: text };
                } catch (ee) {
                    out.runResponse = { url: runResp.url(), status: runResp.status(), body: null };
                }
            }
        }

        // give the client a moment to update
        await page.waitForTimeout(500);

        // locate the last /api/scripts/run response in recorded network events
        const runResponses = out.network.filter(n => n.event === 'response' && n.url && n.url.includes('/api/scripts/run'));
        if (runResponses.length > 0) {
            out.runResponse = runResponses[runResponses.length - 1];
            if (out.runResponse.body && out.runResponse.body.job_id) out.job_id = out.runResponse.body.job_id;
        }

        // If we have a job_id, poll status via API until non-running or timeout
        if (out.job_id) {
            const statusUrl = `http://127.0.0.1:8000/api/scripts/status/${out.job_id}`;
            const start = Date.now();
            while (Date.now() - start < 120000) { // 2 minutes max
                const resp = await context.request.get(statusUrl);
                let json = null;
                try { json = await resp.json(); } catch (e) { json = await resp.text().catch(() => null); }
                out.job_statuses.push({ ts: new Date().toISOString(), status: resp.status(), body: json });
                if (json && json.status && json.status !== 'running' && json.status !== 'started') break;
                await new Promise(r => setTimeout(r, 1000));
            }

            // After polling, perform assertion on final returncode if available
            try {
                const final = out.job_statuses.length > 0 ? out.job_statuses[out.job_statuses.length - 1].body : null;
                let finalReturn = null;
                if (final && typeof final === 'object') {
                    finalReturn = final.returncode !== undefined ? final.returncode : null;
                }
                out.assertion = {
                    expected: expectedReturnCode,
                    actual: finalReturn,
                    passed: finalReturn === expectedReturnCode
                };
                if (!out.assertion.passed) {
                    out.error = out.error || `Assertion failed: expected returncode ${expectedReturnCode}, got ${finalReturn}`;
                }
            } catch (e) {
                out.error = out.error || `Assertion error: ${String(e)}`;
            }
        }

        const screenshotPath = path.join(__dirname, 'ui_playwright_screenshot.png');
        await page.screenshot({ path: screenshotPath, fullPage: true });
        out.screenshot = screenshotPath;

    } catch (err) {
        out.error = String(err);
    } finally {
        const outPath = path.join(__dirname, 'ui_playwright_result.json');
        fs.writeFileSync(outPath, JSON.stringify(out, null, 2), 'utf8');
        console.log('Saved report to', outPath);
        if (out.screenshot) console.log('Saved screenshot to', out.screenshot);
        await browser.close();
        // exit with non-zero if assertion failed
        if (out.assertion && out.assertion.passed === false) {
            console.error('Playwright assertion failed:', out.error);
            process.exit(2);
        }
    }
})();
