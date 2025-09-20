گزارش وضعیت ایشوهای GitHub (خلاصه فارسی)

تاریخ: 2025-09-20
شاخه فعلی: master

خلاصه: وضعیت هر ایشو (Done / Partial / Pending) با شواهد آخرین اجراها به‌روزرسانی شد. اکنون پیش‌نیازهای شروع پایپلاین بسته شده‌اند.

- #1 Investigate and fix UI-launched script failure — CLOSED — Done
  - چرا: اصلاحات سرور/اجرای فرایند جهت غیربافر کردن خروجی، خوانندهٔ پس‌زمینه برای نوشتن stdout/stderr به فایل، و نوشتن متادیتای job انجام شد.
  - فایل‌های مرتبط: `dashboard/api/scripts.py` (unbuffered spawn, background readers, metadata)، `dashboard/frontend/js/dashboard-v2.js` (client fixes)، `tools/ui_playwright_test.js` (reproducer)
  - شاخه/PR: branch `ci/playwright-test`، کامیت‌ها و نظرات بسته شده در issue.

- #4 Persist per-job stdout/stderr into file storage and expose tail via SSE — CLOSED — Done
  - چرا: پیاده‌سازی نوشتن لاگ‌های هر job در `logs/jobs/{job_id}.stdout.log` و `...stderr.log`، SSE tail و endpoint دانلود لاگ انجام شد.
  - فایل‌های مرتبط: `dashboard/api/scripts.py` (log writers, SSE, download zip)، فایل‌های خروجی در `logs/jobs/` تولید شده‌اند.

- #8 Improve ScriptWrapper to persist complete stdout/stderr per job — OPEN — Partial
  - چرا: قابلیت‌های اصلی (file-backed logging و متادیتا اتمیک) پیاده‌سازی شده‌اند، اما موارد تکمیلی مانند چرخش لاگ (rotation)، نگهداری/پِرِیِنس و تست‌های واحد برای سناریوهای کرش هنوز اضافه نشده‌اند.
  - فایل‌های مرتبط: `dashboard/api/scripts.py`

- #9 Add automated UI test to run download_binance via dashboard and assert exit code — OPEN — Partial
  - چرا: تست Playwright ساخته و محلی اجرا شده (`tools/ui_playwright_test.js`) و فایل ورکفلو CI محلی ایجاد و در شاخه `ci/playwright-test` پوش شده است؛ ولی PR/merge برای اجرای CI در GitHub Actions نهایی نشده است.
  - فایل‌های مرتبط: `tools/ui_playwright_test.js`, `.github/workflows/playwright-ui.yml`, `tools/PLAYWRIGHT_CI_README.md`
  - یادداشت: دو مرجع PR/issue مربوط به این کار در repo ثبت شدند (لیست #21 و #22 در Issues/PRs).

- #6 Reproduce UI failure and capture full stdout/stderr — CLOSED — Done
  - چرا: چند اجرای UI از طریق API اجرا شد و لاگ‌ها و متادیتا و env snapshot برای هر job ذخیره شدند. نمونه job_idها:
    - `642f1420-8345-4946-9955-2338dcf7e7d5`, `d55f4878-78cf-4cf2-ab0e-7c0c5d5ade18`, `f7cf9aae-80db-4a5a-88cb-1c2efecbeb5b` (test_print)
    - `42207f6d-9c58-4e30-8980-9f3deee283d6` (download_binance quick 1d)
    - `228d7731-8753-4c18-bbef-86be1e39f0a6` (download_binance 1h sample)
  - لاگ‌ها: `logs/jobs/<job_id>.stdout.log|.stderr.log|.meta.json|.env.json` و همچنین ZIP: `logs/jobs/<job_id>.zip`
  - ابزار اجرای تکراری: `tools/run_ui_jobs.py`

- #5 Add Jobs history view with filters and ability to download logs — OPEN — Partial
  - چرا: endpoint سرور `GET /api/scripts/history` و endpoint دانلود زیپ لاگ‌ها اضافه شده‌اند و یک widget مانیتورینگ در frontend برای نمایش recent jobs پیاده‌سازی شده؛ صفحهٔ کامل history با فیلترها و صفحه‌بندی هنوز تکمیل نشده است.
  - فایل‌های مرتبط: `dashboard/api/scripts.py`, `dashboard/frontend/js/dashboard-v2.js`, `dashboard/frontend/index.html`

- #7 Collect environment and config differences between UI and terminal runs — CLOSED — Done
  - چرا: برای هر job، env snapshot در `logs/jobs/{job_id}.env.json` نوشته می‌شود. ابزار diff اضافه شد:
    - `tools/diff_ui_vs_terminal_env.py` — خروجی: `logs/jobs/env_diff_latest.json`
  - تفاوت کلیدی: `PYTHONPATH` در UI روی ریشه پروژه ست می‌شود؛ در ترمینال معمولاً خالی است.

- #2 Implement pipeline orchestration endpoint (/api/pipeline/run) — OPEN — Done (behind feature gate)
  - چرا: Endpoint های پایپلاین (`POST /api/pipeline/run`, `GET /api/pipeline/status/{id}`, `GET /api/pipeline/stream/{id}`) اضافه شده‌اند و با job runner موجود یکپارچه شده‌اند. اجرای پایپلاین به‌صورت پیش‌فرض توسط فلگ محیطی غیرفعال است.
  - کنترل سروری: متغیر محیطی `DASHBOARD_PIPELINE_ENABLED` باید روی `1/true/on` تنظیم شود.
  - فایل‌های مرتبط: `dashboard/api/pipeline.py` (جدید)، `dashboard/api/scripts.py` (برای orchestration)

- #3 Add "Run full pipeline" button and modal (Frontend) — OPEN — Done (behind feature gate)
  - چرا: مودال «اجرای پایپلاین» در UI اضافه شده، همراه با وضعیت گام‌ها، استریم لاگ‌ها (SSE) و انتخاب کانفیگ/timeout. اما دکمه آغاز تا زمان فعال‌سازی فلگ سمت سرور غیرفعال می‌ماند.
  - کنترل کلاینت/سرور: UI از `GET /api/system/settings` مقدار `pipeline_enabled` را می‌خواند.
  - فایل‌های مرتبط: `dashboard/frontend/index.html`, `dashboard/frontend/js/dashboard-v2.js`, `dashboard/api/system.py`

- #10 (meta) Create Sprint 1 milestone and Project board — OPEN — Pending
  - چرا: مسائل متا وجود دارند (#11, #12)؛ وضعیت دقیق اتوماسیون پروژه بسته به مراحل بعدی مشخص می‌شود.

- #11 (meta) Project board: configure automation and default assignee aminak58 — OPEN — Pending
  - چرا: هنوز تنظیمات اتوماسیون پروژه/قواعد پیش‌فرض کامل نشده‌اند یا لاگ تغییرات ثبت نشده است.

- #12 Milestone: Sprint 1 (due 2025-09-30) — OPEN — Done (وجود مییلست)
  - چرا: مایل‌استون ایجاد شده است (ISSUE شماره 12 موجود است). تکمیل موارد داخل مایل‌استون نیاز به تخصیص ایشوها و به‌روزرسانی وضعیت دارد.

فایل‌ها و شاخه‌های کلیدی که تغییر یافتند یا اضافه شده‌اند:
- شاخه: `ci/playwright-test` (شامل کامیت‌های مرتبط با Playwright و metadata)
- `dashboard/api/scripts.py` — تغییرات مهم برای logging/metadata/streaming/history/download
- `dashboard/frontend/js/dashboard-v2.js` — اضافه شدن لینک دانلود لاگ و widget مانیتورینگ
- `dashboard/frontend/index.html` — container برای widget مانیتورینگ
- `tools/ui_playwright_test.js` — تست Playwright (محلی اجرا شده)
- `.github/workflows/playwright-ui.yml` — workflow محلی در شاخهٔ CI
- `tools/PLAYWRIGHT_CI_README.md`
- Artifacts محلی نمونه: `tools/ui_playwright_result.json`, `tools/ui_playwright_screenshot.png`, و فایل‌های `logs/jobs/{job_id}.*` و `logs/jobs/{job_id}.meta.json` روی دیسک

پیشنهادات سریع برای ادامهٔ کار:
1. بستن رسمی ایشوهای #6 و #7 در GitHub (مدارک در مسیرهای فوق موجود است).
2. ادامهٔ CI: تست UI (Playwright) به‌صورت پیش‌فرض در CI نادیده گرفته می‌شود مگر اینکه `RUN_PLAYWRIGHT_UI_TESTS=1` ست شود؛ بدین‌ترتیب CI تعیین‌پذیر می‌ماند. اسموک سرور برقرار است.
3. تکمیل issue #5: صفحهٔ History با فیلتر/صفحه‌بندی (کم‌ریسک و مستقل از پایپلاین).
4. تکمیل issue #8: تست‌های واحد و سیاست rotation برای لاگ‌ها (ارزش افزوده برای مقیاس‌پذیری، غیرمسدودکننده برای پایپلاین).
5. فعال‌سازی پایپلاین پس از تایید رسمی RFC: با تنظیم `DASHBOARD_PIPELINE_ENABLED=1` قابل استفاده است.

اگر می‌خواهید، می‌توانم همین حالا یک فایل خروجی JSON/CSV از این وضعیت بسازم و یا PR خلاصه برای مرج کردن شاخهٔ `ci/playwright-test` آماده کنم.
