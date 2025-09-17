CI: Playwright UI test + atomic job metadata

خلاصه: اضافه شدن workflow برای اجرای تست UI با Playwright و نوشتن metadata اتمیک برای هر job (logs/jobs/{job_id}.meta.json).

شامل:
- .github/workflows/playwright-ui.yml
- tools/PLAYWRIGHT_CI_README.md
- تغییرات در dashboard/api/scripts.py برای نوشتن metadata اتمیک

مرتبط با ایشوها: #1 و #4

توضیحات: این PR یک ورکفلو CI اضافه می‌کند که سرور را بالا می‌آورد، Playwright را نصب و تست UI را اجرا می‌کند و artifacts را آپلود می‌کند. اگر اجرای CI با خطا مواجه شد، ادامه کارهای دیگر متوقف نخواهد شد و موارد بعدی پیگیری می‌شوند.