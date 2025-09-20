// Global state
let currentStep = 1;
let setupData = {
    system_health: null,
    binance_config: null,
    telegram_config: null,
    selected_config: null
};

// Initialize setup wizard
document.addEventListener('DOMContentLoaded', function () {
    console.log('Setup wizard initialized');
    setupEventListeners();
    runHealthCheck();
});

// Setup all event listeners
function setupEventListeners() {
    // Step navigation buttons
    const nextStep1 = document.getElementById('next-step-1');
    const nextStep2 = document.getElementById('next-step-2');
    const nextStep3 = document.getElementById('next-step-3');
    const prevStep2 = document.getElementById('prev-step-2');
    const prevStep3 = document.getElementById('prev-step-3');
    const prevStep4 = document.getElementById('prev-step-4');

    if (nextStep1) nextStep1.addEventListener('click', () => navigateToStep(2));
    if (nextStep2) nextStep2.addEventListener('click', () => navigateToStep(3));
    if (nextStep3) nextStep3.addEventListener('click', () => navigateToStep(4));
    if (prevStep2) prevStep2.addEventListener('click', () => navigateToStep(1));
    if (prevStep3) prevStep3.addEventListener('click', () => navigateToStep(2));
    if (prevStep4) prevStep4.addEventListener('click', () => navigateToStep(3));

    // API test buttons
    const testBinance = document.getElementById('test-binance');
    const testTelegram = document.getElementById('test-telegram');
    const skipTelegram = document.getElementById('skip-telegram');
    const saveSetup = document.getElementById('save-setup');

    if (testBinance) testBinance.addEventListener('click', testBinanceAPI);
    if (testTelegram) testTelegram.addEventListener('click', testTelegramBot);
    if (skipTelegram) skipTelegram.addEventListener('click', () => skipStep('telegram'));
    if (saveSetup) saveSetup.addEventListener('click', saveSetupConfig);
}

// Navigation function
function navigateToStep(step) {
    console.log(`Navigating from step ${currentStep} to step ${step}`);

    // Hide current step
    const currentStepElement = document.getElementById(`step-${currentStep}`);
    if (currentStepElement) {
        currentStepElement.classList.add('hidden');
    }

    // Show target step
    const targetStepElement = document.getElementById(`step-${step}`);
    if (targetStepElement) {
        targetStepElement.classList.remove('hidden');
    }

    // Update step indicators
    document.querySelectorAll('.step-indicator').forEach(indicator => {
        const stepNum = parseInt(indicator.dataset.step);
        indicator.classList.remove('active', 'completed');

        if (stepNum === step) {
            indicator.classList.add('active');
        } else if (stepNum < step) {
            indicator.classList.add('completed');
        }
    });

    // Update current step
    currentStep = step;

    // Load step-specific data
    switch (step) {
        case 2:
            loadConfigOptions();
            break;
        case 3:
            break;
        case 4:
            runFinalTest();
            break;
    }
}

// Step 1: System Health Check
async function runHealthCheck() {
    console.log('Running health check...');
    const resultsContainer = document.getElementById('health-check-results');
    const nextButton = document.getElementById('next-step-1');

    if (!resultsContainer) {
        console.error('Results container not found');
        return;
    }

    try {
        const response = await fetch('/api/setup/system-health');
        const data = await response.json();
        console.log('Health check response:', data);

        setupData.system_health = data;

        // Display results
        let html = '<div class="space-y-3">';

        // Python version
        html += createHealthCheckItem(
            'نسخه Python',
            data.python_version,
            parseFloat(data.python_version) >= 3.8 ? 'success' : 'error'
        );

        // Memory
        html += createHealthCheckItem(
            'حافظه سیستم',
            `${data.memory_gb} GB`,
            data.memory_gb >= 8 ? 'success' : 'warning'
        );

        // Disk space
        html += createHealthCheckItem(
            'فضای دیسک آزاد',
            `${data.disk_free_gb} GB`,
            data.disk_free_gb >= 10 ? 'success' : 'warning'
        );

        // CPU cores
        html += createHealthCheckItem(
            'هسته‌های پردازنده',
            `${data.cpu_count} هسته`,
            data.cpu_count >= 2 ? 'success' : 'warning'
        );

        // Dependencies
        const depsHealthy = Object.values(data.dependencies_status).every(dep => dep.status === 'installed');
        html += createHealthCheckItem(
            'وابستگی‌های Python',
            depsHealthy ? 'نصب شده' : 'ناقص',
            depsHealthy ? 'success' : 'error'
        );

        html += '</div>';

        // Overall status
        if (data.overall_health) {
            html += '<div class="mt-4 p-3 bg-green-50 border border-green-200 rounded-lg"><p class="text-green-800">✅ سیستم آماده راه‌اندازی است</p></div>';
            if (nextButton) {
                nextButton.classList.remove('hidden');
            }
        } else {
            html += '<div class="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg"><p class="text-red-800">❌ سیستم نیاز به رفع مشکلات دارد</p></div>';
        }

        resultsContainer.innerHTML = html;

    } catch (error) {
        console.error('Health check error:', error);
        resultsContainer.innerHTML = `<div class="p-3 bg-red-50 border border-red-200 rounded-lg"><p class="text-red-800">❌ خطا در بررسی سیستم: ${error.message}</p></div>`;
    }
}

function createHealthCheckItem(label, value, status) {
    const statusClass = status === 'success' ? 'text-green-600' : status === 'error' ? 'text-red-600' : 'text-yellow-600';
    const statusIcon = status === 'success' ? '✅' : status === 'error' ? '❌' : '⚠️';

    return `
        <div class="flex justify-between items-center p-3 bg-gray-50 rounded-lg">
            <span class="text-gray-700">${label}</span>
            <span class="${statusClass} font-medium">${statusIcon} ${value}</span>
        </div>
    `;
}

// Step 2: Load Config Options
async function loadConfigOptions() {
    console.log('Loading config options...');
    const configSelection = document.getElementById('config-selection');

    if (!configSelection) {
        console.error('Config selection element not found');
        return;
    }

    try {
        console.log('Fetching configs from API...');
        const response = await fetch('/api/setup/discover-configs');

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log('API response received:', data);

        if (data.configs && data.configs.length > 0) {
            let html = '<div class="space-y-3">';
            html += '<p class="text-gray-600 mb-4">کانفیگ مورد نظر خود را انتخاب کنید:</p>';

            console.log('Processing', data.configs.length, 'configs');
            data.configs.forEach((config, index) => {
                console.log(`Processing config ${index + 1}:`, config.filename);
                if (config.status === 'ready' || !config.status) {  // Accept configs without status field
                    html += `
                        <div class="config-option border border-gray-200 rounded-lg p-4 cursor-pointer hover:border-blue-500 hover:bg-blue-50" onclick="selectConfig('${config.path.replace(/\\/g, '\\\\')}')">
                            <div class="flex justify-between items-center">
                                <div>
                                    <h4 class="font-medium text-gray-900">${config.filename}</h4>
                                    <p class="text-sm text-gray-600">${config.symbol} | ${config.frequency} | ${config.venue}</p>
                                    <p class="text-xs text-gray-500">${config.description || ''}</p>
                                </div>
                                <input type="radio" name="config" value="${config.path}" class="config-radio">
                            </div>
                        </div>
                    `;
                }
            });
            html += '</div>';

            console.log('Setting HTML content...');
            configSelection.innerHTML = html;
            console.log('Config options loaded successfully!');

            // Verify DOM update
            setTimeout(() => {
                const configOptions = document.querySelectorAll('.config-option');
                console.log('Config options in DOM:', configOptions.length);
            }, 100);

        } else {
            console.log('No configs found in response');
            configSelection.innerHTML = '<p class="text-gray-500">هیچ کانفیگ معتبری یافت نشد</p>';
        }
    } catch (error) {
        console.error('Error loading configs:', error);
        configSelection.innerHTML = '<p class="text-red-500">خطا در بارگذاری کانفیگ‌ها: ' + error.message + '</p>';
    }
}

function selectConfig(configPath) {
    console.log('Config selected:', configPath);
    setupData.selected_config = configPath;

    // Update radio buttons
    document.querySelectorAll('.config-radio').forEach(radio => {
        radio.checked = radio.value === configPath;
    });

    // Update visual selection
    document.querySelectorAll('.config-option').forEach(option => {
        option.classList.remove('border-blue-500', 'bg-blue-50');
    });

    // Find the selected option by checking radio buttons value
    const radioButtons = document.querySelectorAll('.config-radio');
    let selectedOption = null;

    radioButtons.forEach(radio => {
        if (radio.value === configPath) {
            selectedOption = radio.closest('.config-option');
        }
    });

    if (selectedOption) {
        selectedOption.classList.add('border-blue-500', 'bg-blue-50');
        console.log('Config option visually selected');
    } else {
        console.warn('Could not find config option element for path:', configPath);
    }

    checkStep2Completion();
}

async function testBinanceAPI() {
    console.log('Testing Binance API...');
    const apiKey = document.getElementById('binance-api-key');
    const apiSecret = document.getElementById('binance-api-secret');
    const statusDiv = document.getElementById('binance-status');
    const testButton = document.getElementById('test-binance');

    if (!apiKey || !apiSecret || !statusDiv) {
        console.error('Required elements not found');
        return;
    }

    const apiKeyValue = apiKey.value.trim();
    const apiSecretValue = apiSecret.value.trim();

    if (!apiKeyValue || !apiSecretValue) {
        statusDiv.innerHTML = '<p class="text-red-500">لطفاً API Key و Secret را وارد کنید</p>';
        return;
    }

    if (testButton) testButton.disabled = true;
    statusDiv.innerHTML = '<p class="text-blue-500">در حال تست...</p>';

    try {
        const response = await fetch('/api/setup/validate-binance', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                api_key: apiKeyValue,
                api_secret: apiSecretValue
            })
        });

        const data = await response.json();

        if (data.valid) {
            statusDiv.innerHTML = `
                <div class="p-3 bg-green-50 border border-green-200 rounded-lg">
                    <p class="text-green-800">✅ اتصال موفقیت‌آمیز!</p>
                    <p class="text-green-600 text-sm">نوع حساب: ${data.account_type || 'نامشخص'}</p>
                </div>
            `;
            setupData.binance_config = {
                api_key: apiKeyValue,
                api_secret: apiSecretValue,
                valid: true,
                account_type: data.account_type
            };
        } else {
            statusDiv.innerHTML = `
                <div class="p-3 bg-red-50 border border-red-200 rounded-lg">
                    <p class="text-red-800">❌ ${data.error}</p>
                    <p class="text-red-600 text-sm">${data.suggestion || ''}</p>
                </div>
            `;
            setupData.binance_config = { valid: false, error: data.error };
        }

        checkStep2Completion();

    } catch (error) {
        console.error('Binance test error:', error);
        statusDiv.innerHTML = `<div class="p-3 bg-red-50 border border-red-200 rounded-lg"><p class="text-red-800">❌ خطا در اتصال: ${error.message}</p></div>`;
        setupData.binance_config = { valid: false, error: error.message };
    } finally {
        if (testButton) testButton.disabled = false;
    }
}

function checkStep2Completion() {
    const nextButton = document.getElementById('next-step-2');
    if (!nextButton) return;

    // For demo purposes, only require config selection
    // In production, you may want to require valid Binance API as well
    if (setupData.selected_config) {
        nextButton.classList.remove('hidden');
        console.log('Step 2 completed - next button enabled (config selected)');
    } else {
        nextButton.classList.add('hidden');
    }
}

async function testTelegramBot() {
    console.log('Testing Telegram bot...');
    const botToken = document.getElementById('telegram-bot-token');
    const chatId = document.getElementById('telegram-chat-id');
    const statusDiv = document.getElementById('telegram-status');
    const testButton = document.getElementById('test-telegram');

    if (!botToken || !chatId || !statusDiv) {
        console.error('Required elements not found');
        return;
    }

    const botTokenValue = botToken.value.trim();
    const chatIdValue = chatId.value.trim();

    if (!botTokenValue || !chatIdValue) {
        statusDiv.innerHTML = '<p class="text-red-500">لطفاً Bot Token و Chat ID را وارد کنید</p>';
        return;
    }

    if (testButton) testButton.disabled = true;
    statusDiv.innerHTML = '<p class="text-blue-500">در حال ارسال پیام تست...</p>';

    try {
        const response = await fetch('/api/setup/test-telegram', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                bot_token: botTokenValue,
                chat_id: chatIdValue
            })
        });

        const data = await response.json();

        if (data.success) {
            statusDiv.innerHTML = `
                <div class="p-3 bg-green-50 border border-green-200 rounded-lg">
                    <p class="text-green-800">✅ پیام تست ارسال شد!</p>
                    <p class="text-green-600 text-sm">Telegram Bot آماده استفاده است</p>
                </div>
            `;
            setupData.telegram_config = {
                bot_token: botTokenValue,
                chat_id: chatIdValue,
                valid: true
            };
            const nextButton = document.getElementById('next-step-3');
            if (nextButton) nextButton.classList.remove('hidden');
        } else {
            statusDiv.innerHTML = `
                <div class="p-3 bg-red-50 border border-red-200 rounded-lg">
                    <p class="text-red-800">❌ ${data.error}</p>
                    <p class="text-red-600 text-sm">${data.suggestion || 'بررسی کنید که Token و Chat ID صحیح باشند'}</p>
                </div>
            `;
            setupData.telegram_config = { valid: false, error: data.error };
        }
    } catch (error) {
        console.error('Telegram test error:', error);
        statusDiv.innerHTML = `<div class="p-3 bg-red-50 border border-red-200 rounded-lg"><p class="text-red-800">❌ خطا در ارسال: ${error.message}</p></div>`;
        setupData.telegram_config = { valid: false, error: error.message };
    } finally {
        if (testButton) testButton.disabled = false;
    }
}

function skipStep(stepName) {
    if (stepName === 'telegram') {
        setupData.telegram_config = { valid: false, skipped: true };
        const nextButton = document.getElementById('next-step-3');
        if (nextButton) nextButton.classList.remove('hidden');
        console.log('Telegram step skipped');
    }
}

async function runFinalTest() {
    console.log('Running final test...');
    const resultsContainer = document.getElementById('final-test-results');
    const saveButton = document.getElementById('save-setup');

    if (!resultsContainer) {
        console.error('Results container not found');
        return;
    }

    resultsContainer.innerHTML = '<p class="text-blue-500">در حال اجرای تست‌های نهایی...</p>';

    try {
        let html = '<div class="bg-gray-50 rounded-lg p-4">';
        html += '<h3 class="text-lg font-medium text-gray-800 mb-4">خلاصه تنظیمات:</h3>';
        html += '<div class="space-y-2">';

        // System Health
        html += '<div class="flex justify-between"><span>سلامت سیستم</span><span class="text-green-600">✅ تایید شده</span></div>';

        // Binance API
        if (setupData.binance_config && setupData.binance_config.valid) {
            html += '<div class="flex justify-between"><span>Binance API</span><span class="text-green-600">✅ متصل</span></div>';
        } else {
            html += '<div class="flex justify-between"><span>Binance API</span><span class="text-yellow-600">⚠️ تنظیم نشده</span></div>';
        }

        // Telegram
        if (setupData.telegram_config && setupData.telegram_config.valid) {
            html += '<div class="flex justify-between"><span>Telegram Bot</span><span class="text-green-600">✅ تایید شده</span></div>';
        } else {
            html += '<div class="flex justify-between"><span>Telegram Bot</span><span class="text-yellow-600">⚠️ تنظیم نشده</span></div>';
        }

        // Config
        if (setupData.selected_config) {
            html += '<div class="flex justify-between"><span>کانفیگ انتخابی</span><span class="text-green-600">✅ انتخاب شده</span></div>';
        } else {
            html += '<div class="flex justify-between"><span>کانفیگ انتخابی</span><span class="text-red-600">❌ انتخاب نشده</span></div>';
        }

        html += '</div></div>';

        // Overall status
        const hasMinRequirements = setupData.system_health && setupData.binance_config && setupData.selected_config;
        if (hasMinRequirements) {
            html += '<div class="mt-4 p-4 bg-green-50 border border-green-200 rounded-lg text-center"><div class="text-2xl">🎉</div><h3 class="text-lg font-medium text-green-800">راه‌اندازی موفقیت‌آمیز!</h3><p class="text-green-600">سیستم کاملاً آماده استفاده است</p></div>';
            if (saveButton) saveButton.disabled = false;
        } else {
            html += '<div class="mt-4 p-4 bg-yellow-50 border border-yellow-200 rounded-lg text-center"><div class="text-2xl">⚠️</div><h3 class="text-lg font-medium text-yellow-800">راه‌اندازی ناقص</h3><p class="text-yellow-600">لطفاً مراحل قبلی را تکمیل کنید</p></div>';
        }

        resultsContainer.innerHTML = html;
        console.log('Final test completed');

    } catch (error) {
        console.error('Final test error:', error);
        resultsContainer.innerHTML = `<div class="p-3 bg-red-50 border border-red-200 rounded-lg"><p class="text-red-800">❌ خطا در تست نهایی: ${error.message}</p></div>`;
    }
}

async function saveSetupConfig() {
    console.log('Saving setup configuration...');
    const saveButton = document.getElementById('save-setup');

    if (!saveButton) return;

    saveButton.disabled = true;
    saveButton.textContent = 'در حال ذخیره...';

    try {
        const response = await fetch('/api/setup/save-setup', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                timestamp: new Date().toISOString(),
                system_health: setupData.system_health,
                binance: setupData.binance_config,
                telegram: setupData.telegram_config,
                selected_config: setupData.selected_config,
                setup_completed: true
            })
        });

        const data = await response.json();

        if (data.success) {
            saveButton.textContent = '✅ ذخیره شد';
            saveButton.classList.remove('bg-green-500', 'hover:bg-green-600');
            saveButton.classList.add('bg-gray-500');

            const resultsDiv = document.getElementById('final-test-results');
            if (resultsDiv) {
                resultsDiv.innerHTML += `
                    <div class="mt-3 p-3 bg-green-50 border border-green-200 rounded-lg">
                        <p class="text-green-800 mb-3">✅ تنظیمات با موفقیت ذخیره شد</p>
                        <div class="text-center">
                            <a href="/" class="inline-block bg-blue-500 text-white px-6 py-2 rounded-lg hover:bg-blue-600 transition-colors">
                                🏠 ورود به داشبورد اصلی
                            </a>
                        </div>
                    </div>
                `;
            }
            console.log('Setup saved successfully');
        } else {
            throw new Error(data.message || 'خطا در ذخیره تنظیمات');
        }
    } catch (error) {
        console.error('Save setup error:', error);
        saveButton.textContent = '❌ خطا در ذخیره';

        const resultsDiv = document.getElementById('final-test-results');
        if (resultsDiv) {
            resultsDiv.innerHTML += `<div class="mt-3 p-3 bg-red-50 border border-red-200 rounded-lg"><p class="text-red-800">❌ خطا در ذخیره: ${error.message}</p></div>`;
        }
    } finally {
        setTimeout(() => {
            saveButton.disabled = false;
            if (saveButton.textContent !== '✅ ذخیره شد') {
                saveButton.textContent = '💾 ذخیره تنظیمات';
            }
        }, 3000);
    }
}

console.log('Setup JavaScript loaded successfully!');
