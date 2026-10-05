/* ============================================================
   ChurnIQ — Main JavaScript
   ============================================================ */

// ── Predict Page: Single Customer ──────────────────────────
function predictChurn() {
    const form = document.getElementById('predictForm');
    if (!form) return;

    const fields = [
        'tenure', 'Contract', 'MonthlyCharges', 'InternetService',
        'PaymentMethod', 'TechSupport', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'StreamingTV', 'StreamingMovies', 'gender',
        'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
        'MultipleLines', 'PaperlessBilling'
    ];

    const data = {};
    for (const f of fields) {
        const el = document.getElementById(f);
        if (el) data[f] = el.value;
    }

    // Validate required fields
    if (!data.tenure || !data.MonthlyCharges) {
        alert('Please enter Tenure and Monthly Charges.');
        return;
    }

    // Show loading
    const btn = document.getElementById('predictBtn');
    const origText = btn.innerHTML;
    btn.innerHTML = '<span class="spinner"></span> Predicting...';
    btn.disabled = true;

    fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data)
    })
    .then(r => r.json())
    .then(result => {
        btn.innerHTML = origText;
        btn.disabled = false;
        showPredictionResult(result);
    })
    .catch(err => {
        btn.innerHTML = origText;
        btn.disabled = false;
        alert('Prediction failed: ' + err.message);
    });
}

function showPredictionResult(result) {
    const panel = document.getElementById('resultPanel');
    const probVal = document.getElementById('probValue');
    const riskBadge = document.getElementById('riskBadge');
    const riskDesc = document.getElementById('riskDesc');
    const factorsGrid = document.getElementById('factorsGrid');
    const probCircle = document.querySelector('.probability-circle');
    const resultPanelInner = document.querySelector('.result-panel');

    // Set probability
    probVal.textContent = result.probability + '%';

    // Set risk badge
    const riskLower = result.risk.toLowerCase();
    riskBadge.className = 'risk-badge ' + riskLower;
    riskBadge.textContent = result.risk.toUpperCase() + ' RISK';

    // Set circle color
    probCircle.className = 'probability-circle risk-' + riskLower;

    // Set panel color
    resultPanelInner.className = 'result-panel risk-' + riskLower;

    // Risk description
    const descs = {
        high: 'This customer has a high probability of churning. Immediate retention action is recommended.',
        medium: 'This customer has a moderate churn risk. Proactive engagement is recommended.',
        low: 'This customer has a low churn risk. Continue standard service and monitor.'
    };
    riskDesc.textContent = descs[riskLower] || '';

    // Factors
    factorsGrid.innerHTML = '';
    if (result.factors && result.factors.length > 0) {
        result.factors.forEach(f => {
            const card = document.createElement('div');
            card.className = 'factor-card ' + f.direction;
            card.innerHTML = `
                <div class="factor-name">
                    <i class="fas fa-${f.direction === 'negative' ? 'arrow-up' : 'arrow-down'}" 
                       style="color:${f.direction === 'negative' ? '#dc3545' : '#28a745'}"></i>
                    ${f.name}
                </div>
                <div class="factor-impact">Impact: ${f.impact.toFixed(3)}</div>
            `;
            factorsGrid.appendChild(card);
        });
    }

    // Show and scroll
    panel.style.display = 'block';
    panel.scrollIntoView({ behavior: 'smooth', block: 'start' });
}


// ── Bulk Page: CSV Upload ──────────────────────────────────
function setupBulkUpload() {
    const area = document.getElementById('uploadArea');
    const input = document.getElementById('csvInput');
    if (!area || !input) return;

    // Click to browse
    area.addEventListener('click', () => input.click());

    // Drag & drop
    area.addEventListener('dragover', e => {
        e.preventDefault();
        area.classList.add('dragover');
    });
    area.addEventListener('dragleave', () => area.classList.remove('dragover'));
    area.addEventListener('drop', e => {
        e.preventDefault();
        area.classList.remove('dragover');
        if (e.dataTransfer.files.length) {
            input.files = e.dataTransfer.files;
            handleBulkUpload(e.dataTransfer.files[0]);
        }
    });

    // File selected
    input.addEventListener('change', () => {
        if (input.files.length) handleBulkUpload(input.files[0]);
    });
}

function handleBulkUpload(file) {
    if (!file.name.endsWith('.csv')) {
        alert('Please upload a CSV file.');
        return;
    }

    // Show loading
    const overlay = document.getElementById('loadingOverlay');
    overlay.style.display = 'flex';

    const formData = new FormData();
    formData.append('file', file);

    fetch('/api/bulk', { method: 'POST', body: formData })
    .then(r => r.json())
    .then(result => {
        overlay.style.display = 'none';
        showBulkResults(result);
    })
    .catch(err => {
        overlay.style.display = 'none';
        alert('Upload failed: ' + err.message);
    });
}

let bulkCSVData = null;

function showBulkResults(result) {
    const section = document.getElementById('bulkResults');
    section.style.display = 'block';

    // KPIs
    document.getElementById('totalProcessed').textContent = result.total;
    document.getElementById('highRiskCount').textContent = result.high_risk;
    document.getElementById('mediumRiskCount').textContent = result.medium_risk;
    document.getElementById('lowRiskCount').textContent = result.low_risk;

    // Store CSV for download
    bulkCSVData = result.csv_data;

    // Table
    const tbody = document.querySelector('#resultsTable tbody');
    tbody.innerHTML = '';

    const rows = result.rows || [];
    const display = rows.slice(0, 100); // Show max 100

    display.forEach(row => {
        const riskLower = (row.Risk_Category || '').replace(' Risk', '').toLowerCase();
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${row.customerID || '—'}</td>
            <td>${row.tenure || '—'}</td>
            <td>${row.Contract || '—'}</td>
            <td>$${parseFloat(row.MonthlyCharges || 0).toFixed(2)}</td>
            <td>${row.InternetService || '—'}</td>
            <td><strong>${row.Churn_Probability}%</strong></td>
            <td><span class="risk-badge ${riskLower}">${row.Risk_Category}</span></td>
        `;
        tbody.appendChild(tr);
    });

    section.scrollIntoView({ behavior: 'smooth' });
}

function downloadResults() {
    if (!bulkCSVData) return;
    const blob = new Blob([bulkCSVData], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'churn_predictions.csv';
    a.click();
    URL.revokeObjectURL(url);
}


// ── Init ────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    setupBulkUpload();
});
