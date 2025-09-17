// Global variables
let autoScroll = true;
let resourceMonitorInterval;
// Jobs history pagination state
let jobsHistoryState = {
    limit: 10,
    offset: 0,
    total: 0,
    lastQuery: {}
};

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

    // Jobs history controls
    const refreshJobsBtn = document.getElementById('refresh-jobs');
    if (refreshJobsBtn) refreshJobsBtn.addEventListener('click', () => { jobsHistoryState.offset = 0; loadJobsHistory(); });

    const prevBtn = document.getElementById('jobs-prev');
    const nextBtn = document.getElementById('jobs-next');
    if (prevBtn) prevBtn.addEventListener('click', () => { if (jobsHistoryState.offset >= jobsHistoryState.limit) { jobsHistoryState.offset -= jobsHistoryState.limit; loadJobsHistory(); } });
    if (nextBtn) nextBtn.addEventListener('click', () => { if (jobsHistoryState.offset + jobsHistoryState.limit < jobsHistoryState.total) { jobsHistoryState.offset += jobsHistoryState.limit; loadJobsHistory(); } });

    const filterStatus = document.getElementById('filter-status');
    const filterScript = document.getElementById('filter-script');
    if (filterStatus) filterStatus.addEventListener('change', () => { jobsHistoryState.offset = 0; loadJobsHistory(); });
    if (filterScript) filterScript.addEventListener('input', () => { jobsHistoryState.offset = 0; loadJobsHistory(); });
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

    // If showing jobs history, load data
    if (sectionName === 'jobs-history') {
        // reset offset when opening
        jobsHistoryState.offset = 0;
        loadJobsHistory();
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
    // Also refresh recent jobs for monitoring area
    updateResourceMetrics();
    loadMonitoringJobs();
    resourceMonitorInterval = setInterval(() => {
        updateResourceMetrics();
        loadMonitoringJobs();
    }, 5000);
}


async function loadMonitoringJobs() {
    try {
        const resp = await fetch('/api/scripts/history?limit=5');
        if (!resp.ok) return;
        const data = await resp.json();
        const jobs = data.jobs || [];
        const container = document.getElementById('monitor-jobs');
        if (!container) return;

        if (jobs.length === 0) {
            container.innerHTML = '<div class="text-gray-500">هیچ job اخیر یافت نشد</div>';
            return;
        }

        let html = '<div class="space-y-2">';
        jobs.forEach(j => {
            const start = j.start_time ? new Date(j.start_time * 1000).toLocaleString('fa-IR') : '-';
            const statusBadge = getStatusBadge(j.status || 'stopped');
            html += `
                <div class="flex items-center justify-between p-2 border rounded bg-white">
                    <div class="flex-1">
                        <div class="text-sm font-medium">${j.script || j.job_id}</div>
                        <div class="text-xs text-gray-500">شروع: ${start}</div>
                    </div>
                    <div class="flex items-center space-x-2">
                        ${statusBadge}
                        <a href="/api/scripts/logs/${j.job_id}/download" target="_blank" class="px-2 py-1 bg-gray-100 rounded text-xs">دانلود</a>
                    </div>
                </div>
            `;
        });
        html += '</div>';
        container.innerHTML = html;
    } catch (e) {
        // ignore for monitoring
    }
}

// Jobs history loader
async function loadJobsHistory() {
    try {
        const status = document.getElementById('filter-status') ? document.getElementById('filter-status').value : '';
        const script = document.getElementById('filter-script') ? document.getElementById('filter-script').value : '';

        const params = new URLSearchParams();
        params.set('limit', jobsHistoryState.limit);
        params.set('offset', jobsHistoryState.offset);
        if (status) params.set('status', status);
        if (script) params.set('script', script);

        jobsHistoryState.lastQuery = { status, script };

        const resp = await fetch(`/api/scripts/history?${params.toString()}`);
        if (!resp.ok) {
            document.getElementById('jobs-history-list').innerHTML = '<div class="text-red-500">خطا در بارگزاری تاریخچه</div>';
            return;
        }

        const data = await resp.json();
        jobsHistoryState.total = data.total || 0;

        renderJobsHistory(data.jobs || []);
        renderJobsPagination();
    } catch (e) {
        console.error('Error loading jobs history', e);
    }
}

function renderJobsHistory(jobs) {
    const container = document.getElementById('jobs-history-list');
    if (!container) return;

    if (!jobs || jobs.length === 0) {
        container.innerHTML = '<div class="text-gray-500">نتیجه‌ای یافت نشد</div>';
        return;
    }

    let html = '';
    jobs.forEach(j => {
        const start = j.start_time ? new Date(j.start_time * 1000).toLocaleString('fa-IR') : '-';
        const end = j.end_time ? new Date(j.end_time * 1000).toLocaleString('fa-IR') : '-';
        const statusBadge = getStatusBadge(j.status || 'stopped');
        const hasEnv = !!j.env_snapshot_path;
        html += `
            <div class="p-3 bg-gray-50 rounded-lg border ${j.status === 'failed' ? 'border-red-300' : 'border-gray-200'}">
                <div class="flex items-center justify-between">
                    <div class="flex-1">
                        <div class="font-medium">${j.script || j.job_id}</div>
                        <div class="text-sm text-gray-500">Job ID: ${j.job_id} <button class="ml-2 px-2 py-0.5 text-xs bg-gray-200 rounded" onclick="copyText('${j.job_id}')">کپی</button></div>
                        <div class="text-xs text-gray-500">Config: ${j.config || '-'}</div>
                        <div class="text-xs text-gray-500">Return code: ${j.returncode === null || j.returncode === undefined ? '-' : j.returncode}</div>
                        <div class="text-xs text-gray-500">شروع: ${start} — پایان: ${end}</div>
                    </div>
                    <div class="flex items-center space-x-2">
                        ${statusBadge}
                        <a href="/api/scripts/logs/${j.job_id}/download" target="_blank" class="px-3 py-1 bg-gray-200 text-gray-800 text-sm rounded hover:bg-gray-300">دانلود لاگ</a>
                        ${hasEnv ? `<button class="px-3 py-1 bg-gray-100 text-gray-800 text-sm rounded hover:bg-gray-200" onclick="openEnvModal('${j.job_id}')">Env</button>` : ''}
                    </div>
                </div>
            </div>
        `;
    });

    container.innerHTML = html;
}

// Env modal helpers
async function openEnvModal(jobId) {
    try {
        const resp = await fetch(`/api/scripts/env/${encodeURIComponent(jobId)}`);
        if (!resp.ok) {
            showEnvModalText(`خطا در دریافت env: ${resp.status}`);
            return;
        }
        const data = await resp.json();
        showEnvModalText(JSON.stringify(data, null, 2));
    } catch (e) {
        showEnvModalText(`خطای شبکه: ${e?.message || e}`);
    }
}

function copyText(text) {
    try {
        navigator.clipboard.writeText(text);
    } catch (e) { }
}

function showEnvModalText(text) {
    const modal = document.getElementById('env-modal');
    const pre = document.getElementById('env-modal-content');
    const closer = document.getElementById('env-modal-close');
    if (!modal || !pre || !closer) return;
    pre.textContent = text;
    modal.classList.remove('hidden');
    modal.classList.add('flex');
    const closeFn = () => { modal.classList.add('hidden'); modal.classList.remove('flex'); closer.removeEventListener('click', closeFn); };
    closer.addEventListener('click', closeFn);
    modal.addEventListener('click', (ev) => { if (ev.target === modal) closeFn(); });
}

function renderJobsPagination() {
    const pagination = document.getElementById('jobs-pagination');
    if (!pagination) return;

    const start = jobsHistoryState.offset + 1;
    const end = Math.min(jobsHistoryState.offset + jobsHistoryState.limit, jobsHistoryState.total);
    pagination.textContent = `نمایش ${start}–${end} از ${jobsHistoryState.total}`;
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
        const startTime = new Date(script.start_time * 1000).toLocaleString('fa-IR');

        // Calculate duration (always real-time)
        const currentTime = Date.now() / 1000;
        const endTime = script.end_time || currentTime;
        const duration = Math.round(endTime - script.start_time);
        const durationStr = duration > 60 ? `${Math.floor(duration / 60)}دقیقه ${duration % 60}ثانیه` : `${duration}ثانیه`;

        // Error details
        let errorDetails = '';
        if (script.status === 'failed' && script.stderr) {
            errorDetails = `
                <div class="mt-2 p-2 bg-red-50 border border-red-200 rounded text-sm">
                    <div class="font-medium text-red-800">خطا:</div>
                    <div class="text-red-700 font-mono text-xs mt-1 whitespace-pre-wrap">${script.stderr.slice(0, 200)}${script.stderr.length > 200 ? '...' : ''}</div>
                </div>
            `;
        }

        html += `
            <div class="p-3 bg-gray-50 rounded-lg border ${script.status === 'failed' ? 'border-red-300' : 'border-gray-200'}">
                <div class="flex items-center justify-between">
                    <div class="flex-1">
                        <div class="font-medium">${script.script}</div>
                        <div class="text-sm text-gray-500">شروع: ${startTime}</div>
                        <div class="text-sm text-gray-500">مدت: ${durationStr}</div>
                        ${script.returncode !== undefined ? `<div class="text-sm text-gray-500">کد خروج: ${script.returncode}</div>` : ''}
                    </div>
                    <div class="flex items-center space-x-2">
                        ${statusBadge}
                        <a href="/api/scripts/logs/${script.job_id}/download" target="_blank" 
                           class="px-3 py-1 bg-gray-200 text-gray-800 text-sm rounded hover:bg-gray-300">
                           دانلود لاگ
                        </a>
                        ${script.status === 'running' ? `
                            <button onclick="stopScript('${script.job_id}')" 
                                    class="px-3 py-1 bg-red-500 text-white text-sm rounded hover:bg-red-600">
                                توقف
                            </button>
                        ` : ''}
                    </div>
                </div>
                ${errorDetails}
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
        'starting': '<span class="px-2 py-1 bg-yellow-100 text-yellow-800 text-xs rounded-full">🚀 شروع</span>',
        'running': '<span class="px-2 py-1 bg-green-100 text-green-800 text-xs rounded-full">🔄 در حال اجرا</span>',
        'completed': '<span class="px-2 py-1 bg-blue-100 text-blue-800 text-xs rounded-full">✅ تکمیل شده</span>',
        'failed': '<span class="px-2 py-1 bg-red-100 text-red-800 text-xs rounded-full">❌ ناموفق</span>',
        'error': '<span class="px-2 py-1 bg-red-100 text-red-800 text-xs rounded-full">💥 خطا</span>',
        'stopped': '<span class="px-2 py-1 bg-gray-100 text-gray-800 text-xs rounded-full">⏹️ متوقف شده</span>'
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
            method: 'DELETE'
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
    // Try to open a Server-Sent Events connection to stream live logs.
    // This endpoint tails the per-job stdout file written by the server.
    let es = null;
    try {
        es = new EventSource(`/api/scripts/logs/${jobId}/stream`);
        es.onmessage = (e) => {
            const text = e.data || '';
            // Server emits control messages like [FINISHED] or [ERROR]
            if (text.startsWith('[FINISHED]') || text.startsWith('[ERROR]')) {
                const level = text.startsWith('[ERROR]') ? 'error' : 'info';
                addLogMessage(text, level);
            } else {
                // Split possibly multi-line payload into separate log entries
                text.split('\n').forEach(line => {
                    if (line && line.trim()) addLogMessage(line, 'stdout');
                });
            }
        };
        es.onerror = (ev) => {
            // On error, close the connection and fall back to polling
            try { es.close(); } catch (e) { }
            es = null;
            console.warn('SSE connection error for job', jobId, ev);
        };
    } catch (e) {
        console.warn('Could not open EventSource for live logs:', e);
        es = null;
    }

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

                // Close SSE if open
                if (es) {
                    try { es.close(); } catch (e) { }
                    es = null;
                }
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

        // Try to fetch dashboard-selected default config (new endpoint)
        let defaultConfig = null;
        try {
            const sresp = await fetch('/api/setup/selected-config');
            if (sresp.ok) {
                const sdata = await sresp.json();
                defaultConfig = sdata.selected_config || null;
            }
        } catch (e) {
            // fail silently - fall back to client-side heuristics
            defaultConfig = null;
        }

        // Update config dropdowns for each script
        document.querySelectorAll('select[id^="config-"]').forEach(select => {
            select.innerHTML = '<option value="">Select config</option>';
            configs.forEach(config => {
                const configName = config.name || config.filename || 'Unnamed';
                // Use the full path (or relative path) returned by the API so the server can resolve it
                const configValue = config.path || config.name || config.filename || '';
                const option = document.createElement('option');
                option.value = configValue;
                option.textContent = configName;
                // Pre-select if it matches the dashboard-selected config
                if (defaultConfig) {
                    const candidate = defaultConfig.replace('configs/', '').split('/').pop();
                    if (option.value === defaultConfig || option.value.endsWith(candidate)) {
                        option.selected = true;
                    }
                }
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

                        // Try to fetch saved setup default config
                        let defaultConfig = null;
                        try {
                            const setupResp = await fetch('/api/setup');
                            // If the server serves the HTML page, try the JSON file instead
                        } catch (e) {
                            // ignore
                        }
                        try {
                            // Read setup file via API (we don't have a direct API; try known locations)
                            const resp1 = await fetch('/setup_config.json');
                            if (resp1.ok) {
                                const js1 = await resp1.json();
                                defaultConfig = js1.selected_config || js1.default_config || null;
                            }
                        } catch (e) {
                            // ignore
                        }

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

                            // If defaultConfig is available, attempt to pre-select it
                            if (defaultConfig) {
                                // defaultConfig may be relative like 'configs/config-test.json' or filename
                                const candidate = defaultConfig.replace('configs/', '').split('/').pop();
                                for (let i = 0; i < select.options.length; i++) {
                                    const opt = select.options[i];
                                    if (opt.value && (opt.value === defaultConfig || opt.value.endsWith(candidate))) {
                                        opt.selected = true;
                                        break;
                                    }
                                }
                            }
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
