const form = document.getElementById('query-form');
const textInput = document.getElementById('query-text');
const encoderSelect = document.getElementById('encoder-select');
const topkSelect = document.getElementById('topk-select');
const customWrap = document.getElementById('custom-topk-wrap');
const customInput = document.getElementById('topk-custom');
const submitBtn = document.getElementById('submit-btn');
const statusEl = document.getElementById('status');
const table = document.getElementById('results-table');
const tbody = document.getElementById('results-body');

// All frontend calls go through this Spring Boot app (same origin),
// which proxies to the Python encoder service. No CORS needed.
const API_BASE = '';

topkSelect.addEventListener('change', () => {
    customWrap.hidden = topkSelect.value !== 'custom';
});

// Grey out encoders the backend can't currently serve (e.g. torch not installed).
async function loadEncoderAvailability() {
    try {
        const res = await fetch(`${API_BASE}/api/encoders`);
        if (!res.ok) return;
        const data = await res.json();
        for (const opt of encoderSelect.options) {
            const info = data.encoders.find((e) => e.id === opt.value);
            if (!info) {
                continue;
            }
            opt.textContent = info.label;
            if (!info.available) {
                opt.disabled = true;
                opt.textContent = `${info.label} (unavailable)`;
            }
        }
    } catch (err) {
        // Non-fatal: backend may not be up yet. Selecting it will just error on submit.
        console.warn('Could not load encoder availability', err);
    }
}

function currentTopK() {
    if (topkSelect.value === 'custom') {
        const n = parseInt(customInput.value, 10);
        return Number.isFinite(n) && n > 0 ? n : 10;
    }
    return parseInt(topkSelect.value, 10);
}

function renderResults(results) {
    tbody.innerHTML = '';
    for (const row of results) {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${row.rank}</td>
            <td class="word">${escapeHtml(row.word)}</td>
            <td class="score">${row.score.toFixed(4)}</td>
            <td>${escapeHtml(row.definition)}</td>
        `;
        tbody.appendChild(tr);
    }
    table.hidden = results.length === 0;
}

function escapeHtml(str) {
    return String(str)
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;');
}

form.addEventListener('submit', async (event) => {
    event.preventDefault();
    const text = textInput.value.trim();
    if (!text) return;

    const payload = {
        text,
        encoder: encoderSelect.value,
        top_k: currentTopK(),
    };

    submitBtn.disabled = true;
    statusEl.textContent = 'Searching…';
    statusEl.classList.remove('error');
    table.hidden = true;

    try {
        const res = await fetch(`${API_BASE}/api/query`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload),
        });

        const data = await res.json();

        if (!res.ok) {
            throw new Error(data.detail || `Request failed (${res.status})`);
        }

        renderResults(data.results);
        statusEl.textContent = `${data.results.length} result(s) from "${data.encoder}" + "${data.predictor}" in ${data.took_ms} ms.`;
    } catch (err) {
        statusEl.textContent = err.message || 'Something went wrong.';
        statusEl.classList.add('error');
    } finally {
        submitBtn.disabled = false;
    }
});

loadEncoderAvailability();
