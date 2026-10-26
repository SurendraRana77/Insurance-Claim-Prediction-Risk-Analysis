/**
 * Application Core JavaScript
 * Handles Claymorphic interactions, mobile sidebar toggles, toast notifications,
 * Chart.js rendering fallbacks, and async form submission states.
 */

document.addEventListener('DOMContentLoaded', () => {
    initSidebarToggle();
    initRangeSliders();
    initFormSubmitLocks();
});

// 1. Mobile Sidebar Toggle
function initSidebarToggle() {
    const toggleBtn = document.getElementById('sidebar-toggle');
    const sidebar = document.querySelector('.app-sidebar');
    
    if (toggleBtn && sidebar) {
        toggleBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            sidebar.classList.toggle('open');
        });
        
        document.addEventListener('click', (e) => {
            if (sidebar.classList.contains('open') && !sidebar.contains(e.target) && e.target !== toggleBtn) {
                sidebar.classList.remove('open');
            }
        });
    }
}

// 2. Dynamic Range Slider Labels
function initRangeSliders() {
    const sliders = document.querySelectorAll('.clay-range');
    sliders.forEach(slider => {
        const outputId = slider.dataset.output;
        const outputEl = outputId ? document.getElementById(outputId) : null;
        if (outputEl) {
            slider.addEventListener('input', () => {
                const isPercent = slider.dataset.format === 'percent';
                if (isPercent) {
                    outputEl.textContent = `${Math.round(slider.value * 100)}%`;
                } else {
                    outputEl.textContent = slider.value;
                }
            });
        }
    });
}

// 3. Form Submit Locks (Prevent Duplicate Submissions)
function initFormSubmitLocks() {
    const forms = document.querySelectorAll('form[data-prevent-duplicate]');
    forms.forEach(form => {
        form.addEventListener('submit', () => {
            const submitBtn = form.querySelector('button[type="submit"]');
            if (submitBtn) {
                submitBtn.disabled = true;
                const originalText = submitBtn.innerHTML;
                submitBtn.innerHTML = `
                    <svg class="spinner" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="animation: spin 1s linear infinite; display: inline-block;">
                        <circle cx="12" cy="12" r="10" stroke-opacity="0.25"></circle>
                        <path d="M12 2a10 10 0 0 1 10 10" stroke-linecap="round"></path>
                    </svg> Processing...
                `;
                // Release lock after 15s in case of network issue
                setTimeout(() => {
                    submitBtn.disabled = false;
                    submitBtn.innerHTML = originalText;
                }, 15000);
            }
        });
    });
}

// 4. Toast Notification System
function showToast(message, type = 'info') {
    let container = document.getElementById('toast-container');
    if (!container) {
        container = document.createElement('div');
        container.id = 'toast-container';
        document.body.appendChild(container);
    }
    
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    
    let iconSvg = '';
    if (type === 'success') {
        iconSvg = '<svg width="20" height="20" fill="none" stroke="#16A34A" stroke-width="2" viewBox="0 0 24 24"><polyline points="20 6 9 17 4 12"></polyline></svg>';
    } else if (type === 'danger') {
        iconSvg = '<svg width="20" height="20" fill="none" stroke="#DC2626" stroke-width="2" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"></circle><line x1="15" y1="9" x2="9" y2="15"></line><line x1="9" y1="9" x2="15" y2="15"></line></svg>';
    } else {
        iconSvg = '<svg width="20" height="20" fill="none" stroke="#2563EB" stroke-width="2" viewBox="0 0 24 24"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="16" x2="12" y2="12"></line><line x1="12" y1="8" x2="12.01" y2="8"></line></svg>';
    }
    
    toast.innerHTML = `${iconSvg} <span>${message}</span>`;
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.style.transition = 'all 0.3s ease';
        toast.style.opacity = '0';
        toast.style.transform = 'translateY(20px)';
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// 5. Safe Chart.js Initialization Helper
function renderSafeChart(canvasId, chartConfig) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return null;
    
    if (typeof Chart === 'undefined') {
        const parent = canvas.parentElement;
        canvas.remove();
        const fallback = document.createElement('div');
        fallback.className = 'clay-alert clay-alert-warning';
        fallback.innerHTML = `
            <strong>Visualization Unavailable:</strong> The Chart.js library could not be loaded from CDN. 
            All tabular analytics remain accessible.
        `;
        parent.appendChild(fallback);
        return null;
    }
    
    try {
        return new Chart(canvas, chartConfig);
    } catch (err) {
        console.error(`Chart render error on #${canvasId}:`, err);
        return null;
    }
}
