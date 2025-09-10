// Global variables
let autoScroll = true;
let resourceMonitorInterval;

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', function () {
    initializeDashboard();
    setupEventListeners();
});

// Main initialization function
async function initializeDashboard() {
    console.log('Initializing dashboard...');
    addLogMessage('شروع راه‌اندازی داشبورد...', 'info');

    try {
        // Load initial data
        await refreshSystemStatus();
        await loadActiveScripts();
        await loadAvailableScripts();
        await loadConfigs();

        // Start resource monitoring
        startResourceMonitoring();

        // Show dashboard section by default
        showSection('dashboard');

        addLogMessage('داشبورد با موفقیت راه‌اندازی شد', 'success');
    } catch (error) {
        console.error('Error initializing dashboard:', error);
        addLogMessage(`خطا در راه‌اندازی داشبورد: ${error.message}`, 'error');
    }
}

// Setup event listeners
function setupEventListeners() {
    // Navigation links
    document.querySelectorAll('.nav-link').forEach(link => {
        link.addEventListener('click', function (e) {
            e.preventDefault();
            const section = this.getAttribute('href').substring(1);
            showSection(section);
        });
    });

    // Quick action buttons
    const downloadBtn = document.getElementById('download-data-btn');
    if (downloadBtn) {
        downloadBtn.addEventListener('click', downloadData);
    }

    const analysisBtn = document.getElementById('run-analysis-btn');
    if (analysisBtn) {
        analysisBtn.addEventListener('click', runFullAnalysis);
    }

    // System status refresh button
    const refreshBtn = document.getElementById('refresh-status-btn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', refreshSystemStatus);
    }

    // Log controls
    const clearLogsBtn = document.getElementById('clear-logs-btn');
    if (clearLogsBtn) {
        clearLogsBtn.addEventListener('click', clearLogs);
    }

    const toggleScrollBtn = document.getElementById('toggle-scroll-btn');
    if (toggleScrollBtn) {
        toggleScrollBtn.addEventListener('click', toggleAutoScroll);
    }
}

// Section management
function showSection(sectionName) {
    // Hide all sections
    document.querySelectorAll('.section').forEach(section => {
        section.classList.add('hidden');
    });

    // Show selected section
    const targetSection = document.getElementById(sectionName + '-section');
    if (targetSection) {
        targetSection.classList.remove('hidden');
    }

    // Update active nav link
    document.querySelectorAll('.nav-link').forEach(link => {
        link.classList.remove('bg-blue-100', 'text-blue-700');
        link.classList.add('text-gray-600', 'hover:bg-gray-100');
    });

    const activeLink = document.querySelector(`[href="#${sectionName}"]`);
    if (activeLink) {
        activeLink.classList.add('bg-blue-100', 'text-blue-700');
        activeLink.classList.remove('text-gray-600', 'hover:bg-gray-100');
    }
}

// System status functions
async function refreshSystemStatus() {
    addLogMessage('بروزرسانی وضعیت سیستم...', 'info');

    try {
        const response = await fetch('/api/system/health');
        const data = await response.json();

        updateSystemStatusUI(data);
        updateConnectionStatus(data.components);
        addLogMessage('وضعیت سیستم بروزرسانی شد', 'success');
    } catch (error) {
        console.error('Error refreshing system status:', error);
        updateSystemStatusUI({ overall_status: 'error' });
        addLogMessage(`خطا در بروزرسانی وضعیت: ${error.message}`, 'error');
    }
}

function updateSystemStatusUI(healthData) {
    const statusElement = document.getElementById('system-status');
    const statusText = document.getElementById('status-text');

    if (!statusElement || !statusText) return;

    if (healthData.overall_status === 'healthy') {
        statusElement.className = 'flex items-center space-x-2 text-green-600';
        statusText.textContent = 'سیستم سالم';
    } else if (healthData.overall_status === 'degraded') {
        statusElement.className = 'flex items-center space-x-2 text-yellow-600';
        statusText.textContent = 'سیستم با محدودیت';
    } else {
        statusElement.className = 'flex items-center space-x-2 text-red-600';
        statusText.textContent = 'سیستم با مشکل';
    }
}

function updateConnectionStatus(components) {
    const connectionsElement = document.getElementById('connections-status');
    if (!connectionsElement) return;

    if (components && components.length > 0) {
        let html = '';
        components.forEach(component => {
            const statusIcon = component.status === 'healthy' ? '✅' :
                component.status === 'degraded' ? '⚠️' : '❌';
            html += `<div>${statusIcon} ${component.name}</div>`;
        });
        connectionsElement.innerHTML = html;
    } else {
        connectionsElement.innerHTML = 'اطلاعاتی یافت نشد';
    }
}

// Resource monitoring
function startResourceMonitoring() {
    addLogMessage('شروع مانیتورینگ منابع سیستم...', 'info');

    updateResourceMetrics();
    resourceMonitorInterval = setInterval(updateResourceMetrics, 5000);
}

async function updateResourceMetrics() {
    try {
        const response = await fetch('/api/system/resources');
        const data = await response.json();

        // New API structure: data.cpu.percent, data.memory.*, data.disk.*
        updateCPUUsage(data.cpu ? data.cpu.percent : 0);
        updateMemoryUsage(data.memory || {});
        updateDiskUsage(data.disk || {});
    } catch (error) {
        console.error('Error updating resource metrics:', error);
    }
}

function updateCPUUsage(cpuPercent) {
    const cpuPercentElement = document.getElementById('cpu-percent');
    const cpuBarElement = document.getElementById('cpu-bar');

    if (cpuPercentElement) {
        cpuPercentElement.textContent = `${Math.round(cpuPercent)}%`;
    }

    if (cpuBarElement) {
        cpuBarElement.style.width = `${Math.round(cpuPercent)}%`;
    }
}

function updateMemoryUsage(memory) {
    const memoryPercentElement = document.getElementById('memory-percent');
    const memoryBarElement = document.getElementById('memory-bar');
    const memoryDetailsElement = document.getElementById('memory-details');

    if (memoryPercentElement && memory.percent) {
        memoryPercentElement.textContent = `${Math.round(memory.percent)}%`;
    }

    if (memoryBarElement && memory.percent) {
        memoryBarElement.style.width = `${Math.round(memory.percent)}%`;
    }

    if (memoryDetailsElement && memory.used_gb && memory.total_gb) {
        memoryDetailsElement.textContent = `${memory.used_gb} GB / ${memory.total_gb} GB`;
    }
}

function updateDiskUsage(disk) {
    const diskPercentElement = document.getElementById('disk-percent');
    const diskBarElement = document.getElementById('disk-bar');
    const diskDetailsElement = document.getElementById('disk-details');

    if (diskPercentElement && disk.percent) {
        diskPercentElement.textContent = `${Math.round(disk.percent)}%`;
    }

    if (diskBarElement && disk.percent) {
        diskBarElement.style.width = `${Math.round(disk.percent)}%`;
    }

    if (diskDetailsElement && disk.free_gb) {
        diskDetailsElement.textContent = `${disk.free_gb} GB آزاد`;
    }
}

// Script management functions - CRITICAL FOR SYSTEM OPERATION
async function loadAvailableScripts() {
    addLogMessage('بارگذاری لیست اسکریپت‌های موجود...', 'info');

    try {
        const response = await fetch('/api/scripts/list');
        const data = await response.json();
        const scripts = data.scripts || [];

        updateAvailableScriptsList(scripts);
        addLogMessage(`${scripts.length} اسکریپت موجود یافت شد`, 'info');
    } catch (error) {
        console.error('Error loading available scripts:', error);
        addLogMessage(`خطا در بارگذاری اسکریپت‌های موجود: ${error.message}`, 'error');
    }
}

async function loadActiveScripts() {
    addLogMessage('بارگذاری لیست اسکریپت‌های فعال...', 'info');

    try {
        const response = await fetch('/api/scripts/active');
        const scripts = await response.json();

        updateActiveScriptsList(scripts);
        addLogMessage(`${scripts.length} اسکریپت فعال یافت شد`, 'info');
    } catch (error) {
        console.error('Error loading active scripts:', error);
        addLogMessage(`خطا در بارگذاری اسکریپت‌ها: ${error.message}`, 'error');
    }
}

function updateActiveScriptsList(scripts) {
    const scriptsContainer = document.getElementById('active-scripts');
    if (!scriptsContainer) return;

    if (scripts.length === 0) {
        scriptsContainer.innerHTML = '<div class="text-gray-500">هیچ اسکریپت فعالی یافت نشد</div>';
        return;
    }

    let html = '';
    scripts.forEach(script => {
        const statusBadge = getStatusBadge(script.status);
        html += `
            <div class="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div>
                    <div class="font-medium">${script.name}</div>
                    <div class="text-sm text-gray-500">شروع: ${script.start_time}</div>
                </div>
                <div class="flex items-center space-x-2">
                    ${statusBadge}
                    <button onclick="stopScript('${script.job_id}')" 
                            class="px-3 py-1 bg-red-500 text-white text-sm rounded hover:bg-red-600">
                        توقف
                    </button>
                </div>
            </div>
        `;
    });

    scriptsContainer.innerHTML = html;
}

function updateAvailableScriptsList(scripts) {
    const scriptsContainer = document.getElementById('scripts-list');
    if (!scriptsContainer) return;

    if (scripts.length === 0) {
        scriptsContainer.innerHTML = '<div class="text-gray-500">هیچ اسکریپت موجودی یافت نشد</div>';
        return;
    }

    let html = '';
    scripts.forEach(script => {
        html += `
            <div class="flex items-center justify-between p-4 border rounded-lg">
                <div class="flex-1">
                    <div class="font-medium text-lg">${script.name}</div>
                    <div class="text-sm text-gray-600 mt-1">${script.description || 'توضیحی موجود نیست'}</div>
                    <div class="text-xs text-gray-400 mt-1">فایل: ${script.file}</div>
                </div>
                <div class="flex items-center space-x-2">
                    <select id="config-${script.name}" class="px-3 py-2 border rounded text-sm">
                        <option value="">انتخاب کانفیگ</option>
                    </select>
                    <button onclick="runScript('${script.name}')" 
                            class="px-4 py-2 bg-blue-500 text-white text-sm rounded hover:bg-blue-600">
                        🚀 اجرا
                    </button>
                </div>
            </div>
        `;
    });

    scriptsContainer.innerHTML = html;

    // Load configs for each script with delay to ensure DOM is ready
    setTimeout(() => {
        loadConfigsForScripts();
    }, 100);
}

function getStatusBadge(status) {
    const badges = {
        'running': '<span class="px-2 py-1 bg-green-100 text-green-800 text-xs rounded-full">در حال اجرا</span>',
        'completed': '<span class="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full">تکمیل شده</span>',
        'failed': '<span class="px-2 py-1 bg-red-100 text-red-800 text-xs rounded-full">ناموفق</span>',
        'stopped': '<span class="px-2 py-1 bg-gray-100 text-gray-800 text-xs rounded-full">متوقف شده</span>'
    };
    return badges[status] || badges['stopped'];
}

// Core Script Runner Functions - HEART OF THE SYSTEM
async function runFullAnalysis() {
    addLogMessage('🔍 شروع تحلیل کامل سیستم...', 'info');

    try {
        const response = await fetch('/api/scripts/run', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                script_name: 'features',
                config_file: 'config-test.json',
                timeout: 300
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP Error: ${response.status}`);
        }

        const result = await response.json();

        addLogMessage(`✅ تحلیل کامل شروع شد - Job ID: ${result.job_id}`, 'success');

        // Start monitoring the script execution
        monitorScriptExecution(result.job_id);

        // Refresh active scripts list
        await loadActiveScripts();
    } catch (error) {
        console.error('Error running full analysis:', error);
        addLogMessage(`❌ خطا در اجرای تحلیل کامل: ${error.message}`, 'error');
    }
}

async function runScript(scriptName) {
    // Get selected config from dropdown
    const configSelect = document.getElementById(`config-${scriptName}`);
    const configFile = configSelect ? configSelect.value : 'config-test.json';

    if (!configFile) {
        addLogMessage('⚠️ لطفاً ابتدا یک کانفیگ انتخاب کنید', 'error');
        return;
    }

    addLogMessage(`🚀 شروع اجرای اسکریپت: ${scriptName} با کانفیگ: ${configFile}`, 'info');

    try {
        const response = await fetch('/api/scripts/run', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                script_name: scriptName,
                config_file: configFile,
                timeout: 300
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP Error: ${response.status}`);
        }

        const result = await response.json();

        addLogMessage(`✅ اسکریپت ${scriptName} شروع شد - Job ID: ${result.job_id}`, 'success');

        // Start monitoring the script execution
        monitorScriptExecution(result.job_id);

        // Refresh active scripts list
        await loadActiveScripts();
    } catch (error) {
        console.error('Error running script:', error);
        addLogMessage(`❌ خطا در اجرای اسکریپت ${scriptName}: ${error.message}`, 'error');
    }
}

async function stopScript(jobId) {
    addLogMessage(`⏹️ توقف اسکریپت: ${jobId}`, 'info');

    try {
        const response = await fetch(`/api/scripts/stop/${jobId}`, {
            method: 'POST'
        });

        if (!response.ok) {
            throw new Error(`HTTP Error: ${response.status}`);
        }

        const result = await response.json();

        if (result.success) {
            addLogMessage(`✅ اسکریپت ${jobId} متوقف شد`, 'success');

            // Refresh active scripts list
            await loadActiveScripts();
        } else {
            addLogMessage(`❌ خطا در توقف اسکریپت: ${result.message}`, 'error');
        }
    } catch (error) {
        console.error('Error stopping script:', error);
        addLogMessage(`❌ خطا در توقف اسکریپت ${jobId}: ${error.message}`, 'error');
    }
}

// Monitor script execution status
async function monitorScriptExecution(jobId) {
    const checkStatus = async () => {
        try {
            const response = await fetch(`/api/scripts/status/${jobId}`);
            const data = await response.json();

            if (data.status === 'running') {
                // Continue monitoring
                setTimeout(checkStatus, 2000);
            } else {
                // Script finished
                const statusClass = data.status === 'completed' ? 'success' : 'error';
                const statusMessage = data.status === 'completed' ? 'با موفقیت تکمیل شد' : 'با خطا متوقف شد';
                addLogMessage(`📋 اسکریپت ${jobId} ${statusMessage}`, statusClass);

                if (data.stderr) {
                    addLogMessage(`❌ خطای اسکریپت: ${data.stderr}`, 'error');
                }

                // Update active scripts list
                await loadActiveScripts();
            }
        } catch (error) {
            addLogMessage(`❌ خطا در مانیتورینگ اسکریپت: ${error.message}`, 'error');
        }
    };

    // Start monitoring
    setTimeout(checkStatus, 1000);
}

// Data download function
async function downloadData() {
    addLogMessage('📥 شروع دانلود داده‌های جدید...', 'info');

    try {
        const response = await fetch('/api/scripts/run', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                script_name: 'download_yahoo',
                config_file: 'config-test.json',
                timeout: 300
            })
        });

        if (!response.ok) {
            throw new Error(`HTTP Error: ${response.status}`);
        }

        const result = await response.json();

        addLogMessage(`✅ دانلود داده‌ها شروع شد - Job ID: ${result.job_id}`, 'success');
        monitorScriptExecution(result.job_id);
    } catch (error) {
        console.error('Error downloading data:', error);
        addLogMessage(`❌ خطا در دانلود داده‌ها: ${error.message}`, 'error');
    }
}

// Configuration management
async function loadConfigs() {
    try {
        const response = await fetch('/api/configs/list');
        const configs = await response.json();

        addLogMessage(`${configs.length} فایل کانفیگ یافت شد`, 'info');
    } catch (error) {
        console.error('Error loading configs:', error);
        addLogMessage(`خطا در بارگذاری کانفیگ‌ها: ${error.message}`, 'error');
    }
}

async function loadConfigsForScripts() {
    try {
        const response = await fetch('/api/configs/list');
        const configs = await response.json();

        // Update config dropdowns for each script
        document.querySelectorAll('select[id^="config-"]').forEach(select => {
            select.innerHTML = '<option value="">انتخاب کانفیگ</option>';
            configs.forEach(config => {
                const configName = config.name || config.filename || 'نامشخص';
                const configValue = config.name || config.filename || config.path;
                const option = document.createElement('option');
                option.value = configValue;
                option.textContent = configName;
                select.appendChild(option);
            });
        });
    } catch (error) {
        console.error('Error loading configs for scripts:', error);
    }
}

// Log management functions
function addLogMessage(message, level = 'info') {
    const logContainer = document.getElementById('live-logs');
    if (!logContainer) return;

    const timestamp = new Date().toLocaleTimeString('fa-IR');
    const logEntry = document.createElement('div');
    logEntry.className = `log-line log-${level} p-2 border-b border-gray-100`;

    const levelIcon = {
        'info': 'ℹ️',
        'success': '✅',
        'error': '❌',
        'warning': '⚠️',
        'stdout': '📤',
        'stderr': '📥'
    }[level] || 'ℹ️';

    logEntry.innerHTML = `
        <span class="timestamp text-gray-500 text-sm">[${timestamp}]</span>
        <span class="log-level">${levelIcon}</span>
        <span class="log-message">${message}</span>
    `;

    logContainer.appendChild(logEntry);

    // Auto scroll if enabled
    if (autoScroll) {
        logContainer.scrollTop = logContainer.scrollHeight;
    }

    // Limit log entries to prevent memory issues
    const maxLogs = 1000;
    while (logContainer.children.length > maxLogs) {
        logContainer.removeChild(logContainer.firstChild);
    }
}

function clearLogs() {
    const logContainer = document.getElementById('live-logs');
    if (logContainer) {
        logContainer.innerHTML = '';
        addLogMessage('لاگ‌ها پاک شدند', 'info');
    }
}

function toggleAutoScroll() {
    autoScroll = !autoScroll;
    const toggleBtn = document.getElementById('toggle-scroll-btn');
    if (toggleBtn) {
        toggleBtn.textContent = autoScroll ? 'توقف اسکرول' : 'شروع اسکرول';
    }
    addLogMessage(`اسکرول خودکار ${autoScroll ? 'فعال' : 'غیرفعال'} شد`, 'info');
}

// Cleanup on page unload
window.addEventListener('beforeunload', function () {
    if (resourceMonitorInterval) {
        clearInterval(resourceMonitorInterval);
    }
});

// New functions for signals and connections
async function loadLatestSignal() {
    try {
        const response = await fetch('/api/signals/latest');
        const data = await response.json();
        updateLatestSignalUI(data);
    } catch (error) {
        console.error('Error loading latest signal:', error);
        const signalElement = document.getElementById('latest-signal');
        if (signalElement) {
            signalElement.innerHTML = '<p class="text-red-500">خطا در بارگیری سیگنال</p>';
        }
    }
}

function updateLatestSignalUI(data) {
    const signalElement = document.getElementById('latest-signal');
    if (!signalElement) return;

    if (data.status === 'no_recent_signals' && data.signal) {
        const signal = data.signal;
        const html = `
            <div class="bg-gradient-to-r from-blue-50 to-indigo-50 p-4 rounded-lg border border-blue-200">
                <div class="flex justify-between items-start mb-2">
                    <div class="flex items-center space-x-2 space-x-reverse">
                        <span class="text-lg font-bold text-gray-800">${signal.symbol}</span>
                        <span class="px-2 py-1 rounded-full text-xs font-medium ${signal.action === 'BUY' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}">
                            ${signal.action === 'BUY' ? 'خرید' : 'فروش'}
                        </span>
                    </div>
                    <span class="text-sm text-gray-500">${new Date(signal.timestamp).toLocaleString('fa-IR')}</span>
                </div>
                <div class="grid grid-cols-2 gap-4 text-sm">
                    <div>
                        <span class="text-gray-600">قیمت:</span>
                        <span class="font-medium text-gray-800">${signal.price.toLocaleString()}</span>
                    </div>
                    <div>
                        <span class="text-gray-600">اعتماد:</span>
                        <span class="font-medium text-gray-800">${(signal.confidence * 100).toFixed(1)}%</span>
                    </div>
                </div>
                ${signal.indicators ? `
                    <div class="mt-3 pt-3 border-t border-blue-200">
                        <div class="flex flex-wrap gap-2 text-xs">
                            ${signal.indicators.rsi ? `<span class="bg-blue-100 text-blue-700 px-2 py-1 rounded">RSI: ${signal.indicators.rsi}</span>` : ''}
                            ${signal.indicators.macd ? `<span class="bg-blue-100 text-blue-700 px-2 py-1 rounded">MACD: ${signal.indicators.macd}</span>` : ''}
                            ${signal.indicators.volume ? `<span class="bg-blue-100 text-blue-700 px-2 py-1 rounded">Volume: ${signal.indicators.volume}</span>` : ''}
                        </div>
                    </div>
                ` : ''}
            </div>
        `;
        signalElement.innerHTML = html;
    } else {
        signalElement.innerHTML = '<p class="text-gray-500 text-center py-4">هیچ سیگنال جدیدی یافت نشد</p>';
    }
}

async function loadConnectionsStatus() {
    try {
        const response = await fetch('/api/system/connections');
        const data = await response.json();
        updateConnectionsStatusUI(data);
    } catch (error) {
        console.error('Error loading connections status:', error);
        const connectionsElement = document.getElementById('connection-status');
        if (connectionsElement) {
            connectionsElement.innerHTML = '<p class="text-red-500">خطا در بارگیری وضعیت اتصالات</p>';
        }
    }
}

function updateConnectionsStatusUI(data) {
    const connectionsElement = document.getElementById('connection-status');
    if (!connectionsElement) return;

    if (data.connections) {
        let html = '<div class="grid grid-cols-1 md:grid-cols-2 gap-3">';

        Object.values(data.connections).forEach(connection => {
            const statusColor = connection.status === 'connected' ? 'green' :
                connection.status === 'partial' ? 'yellow' : 'red';

            html += `
                <div class="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                    <div class="flex items-center space-x-3 space-x-reverse">
                        <div class="w-3 h-3 rounded-full bg-${statusColor}-500"></div>
                        <span class="font-medium text-gray-800">${connection.name}</span>
                    </div>
                    <span class="text-xs text-gray-500">
                        ${connection.status === 'connected' ? 'متصل' :
                    connection.status === 'partial' ? 'جزئی' : 'قطع'}
                    </span>
                </div>
            `;
        });

        html += '</div>';

        // Add overall status
        html += `
            <div class="mt-4 p-3 bg-gradient-to-r from-gray-50 to-gray-100 rounded-lg">
                <div class="flex justify-between items-center">
                    <span class="text-sm font-medium text-gray-700">وضعیت کلی:</span>
                    <span class="text-sm font-bold ${data.status === 'healthy' ? 'text-green-600' : 'text-yellow-600'}">
                        ${data.active_connections}/${data.total_connections} فعال
                    </span>
                </div>
            </div>
        `;

        connectionsElement.innerHTML = html;
    } else {
        connectionsElement.innerHTML = '<p class="text-gray-500 text-center py-4">اطلاعات اتصالات در دسترس نیست</p>';
    }
}

// Add to initialization
document.addEventListener('DOMContentLoaded', function () {
    // Load signals and connections on page load
    loadLatestSignal();
    loadConnectionsStatus();

    // Update signals and connections periodically
    setInterval(loadLatestSignal, 30000); // Every 30 seconds
    setInterval(loadConnectionsStatus, 60000); // Every 60 seconds
});

// Initial log messages
addLogMessage('✅ داشبورد آماده است', 'info');
addLogMessage('🚀 Script Runner فعال - سیستم آماده کار است', 'success');
