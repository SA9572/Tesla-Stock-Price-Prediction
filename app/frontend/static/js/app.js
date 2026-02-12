(function () {
    // Use `window.API_BASE_URL` (set in Vercel env) when deploying frontend separately.
    // If not set, fall back to relative paths so local dev and monorepo deploys keep working.
    const API_BASE_URL = (typeof window !== 'undefined' && window.API_BASE_URL) ? window.API_BASE_URL.replace(/\/$/, '') : '';
    const form = document.getElementById('predictForm');
    const resultEl = document.getElementById('result');
    const noResultEl = document.getElementById('no-result');
    const predictedPriceEl = document.getElementById('predictedPrice');
    const resultModelEl = document.getElementById('resultModel');
    const resultHorizonEl = document.getElementById('resultHorizon');
    const resultMetaEl = document.getElementById('resultMeta');
    const errorEl = document.getElementById('error');
    const loadingEl = document.getElementById('loading');
    const predictBtn = document.getElementById('predictBtn');
    const refreshBtn = document.getElementById('refreshBtn');
    const latestPriceEl = document.getElementById('latestPrice');

    function showLoading() {
        loadingEl.classList.remove('hidden');
        resultEl.classList.add('hidden');
        noResultEl.classList.add('hidden');
        errorEl.classList.add('hidden');
        predictBtn.disabled = true;
    }

    function hideLoading() {
        loadingEl.classList.add('hidden');
        predictBtn.disabled = false;
    }

    function showResult(price, model, horizon) {
        const modelName = model === 'lstm' ? 'LSTM' : 'SimpleRNN';
        predictedPriceEl.textContent = '$' + price.toLocaleString('en-US', { minimumFractionDigits: 2 });
        resultModelEl.textContent = modelName;
        resultHorizonEl.textContent = horizon + '-day ahead';
        resultMetaEl.innerHTML = '<span style="display: inline-block; background: rgba(0, 208, 132, 0.2); padding: 0.5rem 1rem; border-radius: 8px; border: 1px solid rgba(0, 208, 132, 0.5);">✓ Prediction Successfully Generated</span>';
        resultEl.classList.remove('hidden');
        noResultEl.classList.add('hidden');
        errorEl.classList.add('hidden');
    }

    function showError(msg) {
        errorEl.textContent = msg;
        errorEl.classList.remove('hidden');
        resultEl.classList.add('hidden');
        noResultEl.classList.add('hidden');
    }

    function showNoResult() {
        resultEl.classList.add('hidden');
        noResultEl.classList.remove('hidden');
        errorEl.classList.add('hidden');
    }

    async function loadLatestPrice() {
        try {
            const res = await fetch(`${API_BASE_URL}/api/latest`);
            const data = await res.json();
            if (data.status === 'success') {
                latestPriceEl.textContent = '$' + data.latest_price.toLocaleString('en-US', { minimumFractionDigits: 2 });
            }
        } catch (err) {
            console.log('Could not load latest price');
        }
    }

    form.addEventListener('submit', async function (e) {
        e.preventDefault();
        const model = document.querySelector('input[name="model"]:checked').value;
        const horizon = parseInt(document.querySelector('input[name="horizon"]:checked').value, 10);

        showLoading();

        try {
            const res = await fetch(`${API_BASE_URL}/api/predict?model=${encodeURIComponent(model)}&horizon=${horizon}`);
            const data = await res.json();

            if (!res.ok) {
                showError(data.error || 'Prediction failed');
                return;
            }

            if (data.status === 'success') {
                showResult(data.predicted_price, data.model, data.horizon);
            } else {
                showError(data.error || 'Unknown error');
            }
        } catch (err) {
            showError('Network error. Please try again.');
        } finally {
            hideLoading();
        }
    });

    if (refreshBtn) {
        refreshBtn.addEventListener('click', async function (e) {
            e.preventDefault();
            loadLatestPrice();
        });
    }

    // Load latest price on page load
    loadLatestPrice();
    showNoResult();
})();
