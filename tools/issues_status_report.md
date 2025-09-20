گزارش وضعیت ایشوهای GitHub (خلاصه فارسی)

تاریخ: 2025-09-16
شاخه فعلی: ci/playwright-test

خلاصه: در این گزارش وضعیت هر ایشو (Done / Partial / Pending) و توضیح کوتاه مرتبط با کار انجام شده در مخزن ثبت شده است. فایل‌ها و شاخه‌هایی که تغییر کرده‌اند نیز فهرست شده‌اند.

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

- #6 Reproduce UI failure and capture full stdout/stderr — OPEN — Partial
  - چرا: چند اجرای خودکار/محلی با Playwright انجام شد و artifacts تولید گردید (`tools/ui_playwright_result.json`, `tools/ui_playwright_screenshot.png`) و لاگ‌های job روی دیسک نوشته شده‌اند؛ اما تکمیل مستندسازی و پیوست کامل لاگ‌ها به ایشو هنوز در دستور کار است.

- #5 Add Jobs history view with filters and ability to download logs — OPEN — Partial
  - چرا: endpoint سرور `GET /api/scripts/history` و endpoint دانلود زیپ لاگ‌ها اضافه شده‌اند و یک widget مانیتورینگ در frontend برای نمایش recent jobs پیاده‌سازی شده؛ صفحهٔ کامل history با فیلترها و صفحه‌بندی هنوز تکمیل نشده است.
  - فایل‌های مرتبط: `dashboard/api/scripts.py`, `dashboard/frontend/js/dashboard-v2.js`, `dashboard/frontend/index.html`

- #7 Collect environment and config differences between UI and terminal runs — OPEN — Pending
  - چرا: کار خودکار برای dump محیط (os.environ) یا مقایسهٔ کامل اجراها انجام نشده است؛ پیشنهاد: از رخنه‌های متادیتای job فعلی برای جمع‌آوری env در هر اجرا استفاده شود.

- #2 Implement pipeline orchestration endpoint (/api/pipeline/run) — OPEN — Pending
  - چرا: هنوز endpoint ترتیب‌دهی pipeline اضافه نشده است.

- #3 Add "Run full pipeline" button and modal (Frontend) — OPEN — Pending
  - چرا: UI مربوطه هنوز ساخته نشده است.

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
1. بستن نهایی PR از `ci/playwright-test` و اجرای CI در GitHub تا وضعیت تست‌ها روی رِموت مشاهده شود (اختیاری، در صورت مشکل CI ادامه کار را متوقف نکنید).
2. تکمیل issue #5: پیاده‌سازی صفحهٔ کامل تاریخچه با فیلترها و صفحه‌بندی.
3. تکمیل issue #8: نوشتن unit-testها برای ScriptWrapper و پیاده‌سازی سیاست نگهداری/rotation لاگ.
4. انجام #7 برای یک گزارش کامل env و attach لاگ‌ها به issue #6.

اگر می‌خواهید، می‌توانم همین حالا یک فایل خروجی JSON/CSV از این وضعیت بسازم و یا PR خلاصه برای مرج کردن شاخهٔ `ci/playwright-test` آماده کنم.
