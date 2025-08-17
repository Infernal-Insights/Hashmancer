// Hashmancer Portal JavaScript Application

// Global application state
window.HashmancerApp = {
    config: {
        wsReconnectInterval: 5000,
        refreshInterval: 30000,
        apiBaseUrl: '/api'
    },
    state: {
        isConnected: false,
        lastUpdate: null,
        activeWorkers: 0,
        totalCost: 0
    },
    ws: null,
    intervals: {}
};

// Initialize application when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
});

// Main initialization function
function initializeApp() {
    console.log('Initializing Hashmancer Portal...');
    
    // Initialize WebSocket connection
    initializeWebSocket();
    
    // Setup periodic refresh
    setupPeriodicRefresh();
    
    // Initialize tooltips
    initializeTooltips();
    
    // Setup global error handling
    setupErrorHandling();
    
    // Initialize navigation
    initializeNavigation();
    
    console.log('Hashmancer Portal initialized successfully');
}

// WebSocket Connection Management
function initializeWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/live-updates`;
    
    try {
        HashmancerApp.ws = new WebSocket(wsUrl);
        
        HashmancerApp.ws.onopen = function(event) {
            console.log('WebSocket connected');
            HashmancerApp.state.isConnected = true;
            updateConnectionStatus(true);
        };
        
        HashmancerApp.ws.onmessage = function(event) {
            try {
                const data = JSON.parse(event.data);
                handleLiveUpdate(data);
            } catch (error) {
                console.error('Error parsing WebSocket message:', error);
            }
        };
        
        HashmancerApp.ws.onclose = function(event) {
            console.log('WebSocket disconnected');
            HashmancerApp.state.isConnected = false;
            updateConnectionStatus(false);
            
            // Attempt to reconnect
            setTimeout(initializeWebSocket, HashmancerApp.config.wsReconnectInterval);
        };
        
        HashmancerApp.ws.onerror = function(error) {
            console.error('WebSocket error:', error);
        };
        
    } catch (error) {
        console.error('Failed to initialize WebSocket:', error);
        updateConnectionStatus(false);
    }
}

// Handle live updates from WebSocket
function handleLiveUpdate(data) {
    console.log('Received live update:', data);
    
    HashmancerApp.state.lastUpdate = new Date(data.timestamp);
    HashmancerApp.state.activeWorkers = data.running_workers || 0;
    HashmancerApp.state.totalCost = data.total_cost || 0;
    
    // Update UI elements
    updateDashboardStats(data);
    updateConnectionIndicator();
    
    // Trigger custom event for other components
    window.dispatchEvent(new CustomEvent('hashmancer:update', {
        detail: data
    }));
}

// Update dashboard statistics
function updateDashboardStats(data) {
    // Update worker count
    const workerCountElement = document.querySelector('[data-stat="active-workers"]');
    if (workerCountElement) {
        workerCountElement.textContent = data.running_workers || 0;
    }
    
    // Update cost
    const costElement = document.querySelector('[data-stat="total-cost"]');
    if (costElement) {
        costElement.textContent = '$' + (data.total_cost || 0).toFixed(2);
    }
    
    // Update hourly cost
    const hourlyCostElement = document.querySelector('[data-stat="hourly-cost"]');
    if (hourlyCostElement) {
        hourlyCostElement.textContent = '$' + (data.hourly_cost || 0).toFixed(2);
    }
}

// Update connection status indicators
function updateConnectionStatus(isConnected) {
    const indicators = document.querySelectorAll('.connection-indicator');
    indicators.forEach(indicator => {
        if (isConnected) {
            indicator.classList.remove('text-danger');
            indicator.classList.add('text-success');
            indicator.innerHTML = '<i class="bi bi-circle-fill"></i> Connected';
        } else {
            indicator.classList.remove('text-success');
            indicator.classList.add('text-danger');
            indicator.innerHTML = '<i class="bi bi-circle-fill"></i> Disconnected';
        }
    });
}

// Update connection indicator with last update time
function updateConnectionIndicator() {
    const indicator = document.querySelector('.last-update-indicator');
    if (indicator && HashmancerApp.state.lastUpdate) {
        const timeAgo = getTimeAgo(HashmancerApp.state.lastUpdate);
        indicator.textContent = `Last update: ${timeAgo}`;
    }
}

// Periodic refresh setup
function setupPeriodicRefresh() {
    // Refresh connection indicator every 5 seconds
    HashmancerApp.intervals.connectionUpdate = setInterval(updateConnectionIndicator, 5000);
    
    // Auto-refresh certain pages every 30 seconds
    const autoRefreshPages = ['workers', 'jobs', 'costs'];
    const currentPage = getCurrentPageName();
    
    if (autoRefreshPages.includes(currentPage)) {
        HashmancerApp.intervals.pageRefresh = setInterval(() => {
            if (document.visibilityState === 'visible') {
                refreshPageData();
            }
        }, HashmancerApp.config.refreshInterval);
    }
}

// Get current page name from URL
function getCurrentPageName() {
    const path = window.location.pathname;
    if (path === '/') return 'dashboard';
    return path.substring(1).split('/')[0];
}

// Refresh current page data
function refreshPageData() {
    // Only refresh if no modals are open
    const openModals = document.querySelectorAll('.modal.show');
    if (openModals.length === 0) {
        console.log('Auto-refreshing page data...');
        // This would typically call specific refresh functions
        // For now, we'll just update the timestamp
        updateConnectionIndicator();
    }
}

// Initialize Bootstrap tooltips
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// Global error handling
function setupErrorHandling() {
    window.addEventListener('error', function(event) {
        console.error('Global error:', event.error);
        showErrorNotification('An unexpected error occurred. Please refresh the page.');
    });
    
    window.addEventListener('unhandledrejection', function(event) {
        console.error('Unhandled promise rejection:', event.reason);
        showErrorNotification('A network error occurred. Please check your connection.');
    });
}

// Navigation enhancements
function initializeNavigation() {
    // Highlight active navigation item
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.navbar-nav .nav-link');
    
    navLinks.forEach(link => {
        const href = link.getAttribute('href');
        if (href === currentPath || (currentPath !== '/' && href !== '/' && currentPath.startsWith(href))) {
            link.classList.add('active');
        }
    });
}

// Utility Functions
function getTimeAgo(date) {
    const now = new Date();
    const diff = now - date;
    const seconds = Math.floor(diff / 1000);
    
    if (seconds < 60) return `${seconds}s ago`;
    
    const minutes = Math.floor(seconds / 60);
    if (minutes < 60) return `${minutes}m ago`;
    
    const hours = Math.floor(minutes / 60);
    if (hours < 24) return `${hours}h ago`;
    
    const days = Math.floor(hours / 24);
    return `${days}d ago`;
}

// API Helper Functions
function apiRequest(endpoint, options = {}) {
    const url = HashmancerApp.config.apiBaseUrl + endpoint;
    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
        }
    };
    
    return fetch(url, { ...defaultOptions, ...options })
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .catch(error => {
            console.error('API request failed:', error);
            showErrorNotification(`API request failed: ${error.message}`);
            throw error;
        });
}

function apiGet(endpoint) {
    return apiRequest(endpoint, { method: 'GET' });
}

function apiPost(endpoint, data) {
    return apiRequest(endpoint, {
        method: 'POST',
        body: JSON.stringify(data)
    });
}

function apiPut(endpoint, data) {
    return apiRequest(endpoint, {
        method: 'PUT',
        body: JSON.stringify(data)
    });
}

function apiDelete(endpoint) {
    return apiRequest(endpoint, { method: 'DELETE' });
}

// Notification System
function showNotification(message, type = 'info', duration = 5000) {
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(notification);
    
    // Auto-remove after duration
    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, duration);
}

function showSuccessNotification(message) {
    showNotification(message, 'success');
}

function showErrorNotification(message) {
    showNotification(message, 'danger');
}

function showWarningNotification(message) {
    showNotification(message, 'warning');
}

function showInfoNotification(message) {
    showNotification(message, 'info');
}

// Loading States
function showLoading(element) {
    if (typeof element === 'string') {
        element = document.querySelector(element);
    }
    if (element) {
        element.classList.add('loading');
        const spinner = document.createElement('div');
        spinner.className = 'spinner-border spinner-border-sm me-2';
        spinner.setAttribute('role', 'status');
        element.insertBefore(spinner, element.firstChild);
    }
}

function hideLoading(element) {
    if (typeof element === 'string') {
        element = document.querySelector(element);
    }
    if (element) {
        element.classList.remove('loading');
        const spinner = element.querySelector('.spinner-border');
        if (spinner) {
            spinner.remove();
        }
    }
}

// Form Utilities
function serializeForm(form) {
    const formData = new FormData(form);
    const data = {};
    for (let [key, value] of formData.entries()) {
        data[key] = value;
    }
    return data;
}

function resetForm(form) {
    if (typeof form === 'string') {
        form = document.querySelector(form);
    }
    if (form) {
        form.reset();
        // Clear any validation states
        form.querySelectorAll('.is-valid, .is-invalid').forEach(el => {
            el.classList.remove('is-valid', 'is-invalid');
        });
    }
}

// Local Storage Utilities
function saveToStorage(key, data) {
    try {
        localStorage.setItem(key, JSON.stringify(data));
    } catch (error) {
        console.error('Failed to save to localStorage:', error);
    }
}

function loadFromStorage(key, defaultValue = null) {
    try {
        const data = localStorage.getItem(key);
        return data ? JSON.parse(data) : defaultValue;
    } catch (error) {
        console.error('Failed to load from localStorage:', error);
        return defaultValue;
    }
}

// Copy to Clipboard
function copyToClipboard(text) {
    if (navigator.clipboard && window.isSecureContext) {
        return navigator.clipboard.writeText(text).then(() => {
            showSuccessNotification('Copied to clipboard');
        }).catch(error => {
            console.error('Failed to copy to clipboard:', error);
            showErrorNotification('Failed to copy to clipboard');
        });
    } else {
        // Fallback for older browsers
        const textArea = document.createElement('textarea');
        textArea.value = text;
        textArea.style.position = 'fixed';
        textArea.style.left = '-999999px';
        textArea.style.top = '-999999px';
        document.body.appendChild(textArea);
        textArea.focus();
        textArea.select();
        
        try {
            document.execCommand('copy');
            showSuccessNotification('Copied to clipboard');
        } catch (error) {
            console.error('Failed to copy to clipboard:', error);
            showErrorNotification('Failed to copy to clipboard');
        } finally {
            textArea.remove();
        }
    }
}

// Format Numbers
function formatNumber(num) {
    return new Intl.NumberFormat().format(num);
}

function formatCurrency(amount, currency = 'USD') {
    return new Intl.NumberFormat('en-US', {
        style: 'currency',
        currency: currency
    }).format(amount);
}

function formatBytes(bytes, decimals = 2) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB'];
    
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

// Cleanup function
function cleanup() {
    // Clear intervals
    Object.values(HashmancerApp.intervals).forEach(interval => {
        clearInterval(interval);
    });
    
    // Close WebSocket
    if (HashmancerApp.ws) {
        HashmancerApp.ws.close();
    }
}

// Cleanup on page unload
window.addEventListener('beforeunload', cleanup);

// Export functions for global use
window.HashmancerApp.utils = {
    apiGet,
    apiPost,
    apiPut,
    apiDelete,
    showNotification,
    showSuccessNotification,
    showErrorNotification,
    showWarningNotification,
    showInfoNotification,
    showLoading,
    hideLoading,
    serializeForm,
    resetForm,
    saveToStorage,
    loadFromStorage,
    copyToClipboard,
    formatNumber,
    formatCurrency,
    formatBytes,
    getTimeAgo
};

console.log('Hashmancer Portal JavaScript loaded successfully');