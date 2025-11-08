// Model Comparison JavaScript
document.addEventListener('DOMContentLoaded', function() {
    // Initialize model configuration dropdowns
    fetchModels();

    // Event listeners for model type changes
    document.getElementById('model1-type').addEventListener('change', function() {
        updateModelConfigs('model1-type', 'model1-config');
    });

    document.getElementById('model2-type').addEventListener('change', function() {
        updateModelConfigs('model2-type', 'model2-config');
    });

    // Show metadata on config selection
    document.getElementById('model1-config').addEventListener('change', function() {
        showModelMetadata('model1-type', 'model1-config', 'model1-name');
    });
    document.getElementById('model2-config').addEventListener('change', function() {
        showModelMetadata('model2-type', 'model2-config', 'model2-name');
    });

    // Train button
    document.getElementById('train-button').addEventListener('click', trainAndCompareModels);

    // Cancel buttons
    const c1 = document.getElementById('cancel-model1');
    const c2 = document.getElementById('cancel-model2');
    if (c1) c1.addEventListener('click', () => cancelJob('model1'));
    if (c2) c2.addEventListener('click', () => cancelJob('model2'));

    // Test buttons
    document.getElementById('test-model1').addEventListener('click', () => testModel('model1'));
    document.getElementById('test-model2').addEventListener('click', () => testModel('model2'));

    // Export and history
    document.getElementById('export-results').addEventListener('click', exportResults);
    document.getElementById('clear-history').addEventListener('click', clearHistory);

    // Load history from localStorage
    loadHistory();
});

// Global variable to store all models
let allModels = {};
let state = {
    jobs: { model1: null, model2: null },
    results: { model1: null, model2: null },
    intervals: { model1: null, model2: null },
    history: []
};

// Fetch all models from the API
async function fetchModels() {
    try {
        const response = await fetch('/api/models');
        if (!response.ok) {
            throw new Error('Failed to fetch models');
        }
        
        allModels = await response.json();
        
        // Initialize model config dropdowns
        updateModelConfigs('model1-type', 'model1-config');
        updateModelConfigs('model2-type', 'model2-config');
    } catch (error) {
        console.error('Error fetching models:', error);
    }
}

// Update model configuration dropdown based on selected model type
function updateModelConfigs(typeSelectId, configSelectId) {
    const modelType = document.getElementById(typeSelectId).value;
    const configSelect = document.getElementById(configSelectId);
    
    // Clear existing options
    configSelect.innerHTML = '<option value="">Selecione uma configuração</option>';
    
    // Map model type values to keys in allModels
    const modelTypeMap = {
        'logistic': 'Regressão Logística',
        'knn': 'KNN',
        'svc': 'SVC',
        'tree': 'Decision Tree',
        'rf': 'Random Forest'
    };
    
    const selectedType = modelTypeMap[modelType];
    
    // If we have models of this type
    if (allModels[selectedType]) {
        // Add each model configuration as an option
        allModels[selectedType].forEach(model => {
            const option = document.createElement('option');
            option.value = model.filename;
            option.textContent = model.filename.replace('.json', '');
            configSelect.appendChild(option);
        });
    }
}

// Show selected model metadata under the results titles
function showModelMetadata(typeSelectId, configSelectId, titleElementId) {
    const typeVal = document.getElementById(typeSelectId).value;
    const configVal = document.getElementById(configSelectId).value;
    const map = {
        'logistic': 'Regressão Logística',
        'knn': 'KNN',
        'svc': 'SVC',
        'tree': 'Decision Tree',
        'rf': 'Random Forest'
    };
    const typeKey = map[typeVal];
    const titleEl = document.getElementById(titleElementId);
    if (typeKey && allModels[typeKey]) {
        const meta = allModels[typeKey].find(m => m.filename === configVal);
        if (meta) {
            const paramsPreview = Object.entries(meta.params || {})
                .slice(0, 3)
                .map(([k, v]) => `${k}: ${v}`)
                .join(' • ');
            titleEl.textContent = `${meta.nome_modelo} — ${meta.filename.replace('.json','')}`;
            if (paramsPreview) {
                titleEl.title = paramsPreview; // tooltip
            }
        }
    }
}

// Train and compare the selected models
async function trainAndCompareModels() {
    const model1Type = document.getElementById('model1-type').value;
    const model1Config = document.getElementById('model1-config').value;
    const model2Type = document.getElementById('model2-type').value;
    const model2Config = document.getElementById('model2-config').value;
    let epochs = document.getElementById('epochs').value;
    const learningRate = document.getElementById('learning-rate').value;
    const testSplit = document.getElementById('test-split').value;
    const mode = document.getElementById('train-mode') ? document.getElementById('train-mode').value : 'both';

    // Validate selections according to mode
    if (mode === 'both' && (!model1Config || !model2Config)) {
        alert('Por favor, selecione configurações para ambos os modelos.');
        return;
    }
    if (mode === 'only-model1' && !model1Config) {
        alert('Selecione uma configuração para o Modelo 1.');
        return;
    }
    if (mode === 'only-model2' && !model2Config) {
        alert('Selecione uma configuração para o Modelo 2.');
        return;
    }

    // Show loading state
    const trainButton = document.getElementById('train-button');
    const originalButtonText = trainButton.textContent;
    trainButton.textContent = 'Treinando...';
    trainButton.disabled = true;
    document.getElementById('progress-section').classList.remove('hidden');
    // Reset progress UI
    setProgressUI('model1', { epoch: 0, total_epochs: epochs, loss: '--', accuracy: '--', eta_seconds: null });
    setProgressUI('model2', { epoch: 0, total_epochs: epochs, loss: '--', accuracy: '--', eta_seconds: null });
    // Reset results and charts
    state.results.model1 = null;
    state.results.model2 = null;
    document.getElementById('results-container').classList.add('hidden');
    const chartContainer = document.getElementById('comparison-chart');
    if (chartContainer) chartContainer.innerHTML = '';
    const cm1El = document.getElementById('cm1');
    const cm2El = document.getElementById('cm2');
    if (cm1El) cm1El.innerHTML = `
        <tr><td>Real -1</td><td>--</td><td>--</td></tr>
        <tr><td>Real 1</td><td>--</td><td>--</td></tr>
    `;
    if (cm2El) cm2El.innerHTML = `
        <tr><td>Real -1</td><td>--</td><td>--</td></tr>
        <tr><td>Real 1</td><td>--</td><td>--</td></tr>
    `;

    try {
        // Extract dataset
        const dataset = extractDatasetFromTable();
        // Sanitize numeric inputs
        let ep = parseInt(epochs, 10);
        if (!Number.isFinite(ep) || ep <= 0) {
            ep = 100; // fallback default
            const epEl = document.getElementById('epochs');
            if (epEl) epEl.value = String(ep);
        }
        // Clamp to reasonable bounds from UI
        ep = Math.max(1, Math.min(1000, ep));
        const trainingParams = {
            epochs: ep,
            learning_rate: parseFloat(learningRate),
            test_split: parseInt(testSplit) / 100,
            batch_size: 32,
            validation_split: 0.2
        };

        const startJob = async (which, type, config) => {
            const res = await fetch('/api/train/start', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model: { type, config },
                    training_params: trainingParams,
                    dataset
                })
            });
            if (!res.ok) throw new Error('Falha ao iniciar o treinamento');
            const data = await res.json();
            state.jobs[which] = data.job_id;
            return data.job_id;
        };

        const combined = {};
        if (mode === 'only-model1') {
            await startJob('model1', model1Type, model1Config);
            const r1 = await pollUntilComplete('model1');
            if (r1) { state.results.model1 = r1; combined.model1 = r1; saveToHistory('model1', r1); }
        } else if (mode === 'only-model2') {
            await startJob('model2', model2Type, model2Config);
            const r2 = await pollUntilComplete('model2');
            if (r2) { state.results.model2 = r2; combined.model2 = r2; saveToHistory('model2', r2); }
        } else {
            // Iniciar ambos os jobs em paralelo para evitar qualquer condição que impeça o Modelo 2 de começar
            await Promise.all([
                startJob('model1', model1Type, model1Config),
                startJob('model2', model2Type, model2Config)
            ]);
            const [r1, r2] = await Promise.all([
                pollUntilComplete('model1'),
                pollUntilComplete('model2')
            ]);
            if (r1) { state.results.model1 = r1; combined.model1 = r1; saveToHistory('model1', r1); }
            if (r2) { state.results.model2 = r2; combined.model2 = r2; saveToHistory('model2', r2); }
        }

        if (combined.model1 || combined.model2) displayResults(combined);

    } catch (error) {
        console.error('Error training models:', error);
        alert('Ocorreu um erro ao treinar os modelos. Por favor, tente novamente.');
    } finally {
        trainButton.textContent = originalButtonText;
        trainButton.disabled = false;
    }
}

// Extract dataset from the hidden table
function extractDatasetFromTable() {
    const rows = Array.from(document.querySelectorAll('#dataset table tbody tr'));
    const features = [];
    const labels = [];
    rows.forEach(tr => {
        const tds = tr.querySelectorAll('td');
        const x1 = parseFloat(tds[1].textContent);
        const x2 = parseFloat(tds[2].textContent);
        const x3 = parseFloat(tds[3].textContent);
        const d = parseFloat(tds[4].textContent);
        features.push([x1, x2, x3]);
        labels.push(d === -1 ? -1 : 1);
    });
    return { features, labels };
}

// Display the comparison results
function displayResults(results) {
    // Show results container
    document.getElementById('results-container').classList.remove('hidden');

    // Update model names
    document.getElementById('model1-name').textContent = results.model1 ? (results.model1.name || 'Modelo 1') : 'Modelo 1';
    document.getElementById('model2-name').textContent = results.model2 ? (results.model2.name || 'Modelo 2') : 'Modelo 2';

    // Utilitário seguro para percentuais
    const toPercent = (v) => {
        if (v === null || v === undefined) return '-';
        const n = Number(v);
        if (Number.isNaN(n)) return '-';
        return (n * 100).toFixed(2) + '%';
    };

    // Update metrics for model 1
    if (results.model1) {
        document.getElementById('model1-accuracy').textContent = toPercent(results.model1.accuracy);
        document.getElementById('model1-precision').textContent = toPercent(results.model1.precision);
        document.getElementById('model1-recall').textContent = toPercent(results.model1.recall);
        document.getElementById('model1-f1').textContent = toPercent(results.model1.f1_score);
    } else {
        document.getElementById('model1-accuracy').textContent = '-';
        document.getElementById('model1-precision').textContent = '-';
        document.getElementById('model1-recall').textContent = '-';
        document.getElementById('model1-f1').textContent = '-';
    }

    // Update metrics for model 2
    if (results.model2) {
        document.getElementById('model2-accuracy').textContent = toPercent(results.model2.accuracy);
        document.getElementById('model2-precision').textContent = toPercent(results.model2.precision);
        document.getElementById('model2-recall').textContent = toPercent(results.model2.recall);
        document.getElementById('model2-f1').textContent = toPercent(results.model2.f1_score);
    } else {
        document.getElementById('model2-accuracy').textContent = '-';
        document.getElementById('model2-precision').textContent = '-';
        document.getElementById('model2-recall').textContent = '-';
        document.getElementById('model2-f1').textContent = '-';
    }

    // Create comparison chart
    createComparisonChart(results);

    // Populate confusion matrices if included
    const renderCM = (cm, elId) => {
        const el = document.getElementById(elId);
        if (!el) return;
        const safe = (Array.isArray(cm) && cm.length === 2 && Array.isArray(cm[0]) && Array.isArray(cm[1]))
            ? cm : [[0,0],[0,0]];
        el.innerHTML = `
            <tr><td>Real -1</td><td>${safe[0][0]}</td><td>${safe[0][1]}</td></tr>
            <tr><td>Real 1</td><td>${safe[1][0]}</td><td>${safe[1][1]}</td></tr>
        `;
    };
    if (results.model1) renderCM(results.model1.confusion_matrix, 'cm1');
    if (results.model2) renderCM(results.model2.confusion_matrix, 'cm2');

    try {
        const lossCtx = document.getElementById('loss-chart').getContext('2d');
        const accCtx = document.getElementById('acc-chart').getContext('2d');
        const m1h = (results.model1 && results.model1.history) ? results.model1.history : {};
        const m2h = (results.model2 && results.model2.history) ? results.model2.history : {};
        const maxEpochs = Math.max(
            (m1h.loss || []).length,
            (m2h.loss || []).length,
            (m1h.accuracy || []).length,
            (m2h.accuracy || []).length
        );
        const labels = Array.from({ length: maxEpochs }, (_, i) => i + 1);
        new Chart(lossCtx, {
            type: 'line',
            data: {
                labels,
                datasets: [
                    { label: 'Modelo 1 - Loss', data: m1h.loss || [], borderColor: '#3b82f6', fill: false },
                    { label: 'Modelo 1 - Val Loss', data: m1h.val_loss || [], borderColor: '#93c5fd', fill: false },
                    { label: 'Modelo 2 - Loss', data: m2h.loss || [], borderColor: '#10b981', fill: false },
                    { label: 'Modelo 2 - Val Loss', data: m2h.val_loss || [], borderColor: '#6ee7b7', fill: false }
                ]
            },
            options: { responsive: true, maintainAspectRatio: false }
        });
        new Chart(accCtx, {
            type: 'line',
            data: {
                labels,
                datasets: [
                    { label: 'Modelo 1 - Acc', data: m1h.accuracy || [], borderColor: '#3b82f6', fill: false },
                    { label: 'Modelo 1 - Val Acc', data: m1h.val_accuracy || [], borderColor: '#93c5fd', fill: false },
                    { label: 'Modelo 2 - Acc', data: m2h.accuracy || [], borderColor: '#10b981', fill: false },
                    { label: 'Modelo 2 - Val Acc', data: m2h.val_accuracy || [], borderColor: '#6ee7b7', fill: false }
                ]
            },
            options: { responsive: true, maintainAspectRatio: false }
        });
    } catch (e) { console.warn('Falha ao renderizar gráficos', e); }
}

// Create a bar chart to compare model metrics
function createComparisonChart(results) {
    const chartContainer = document.getElementById('comparison-chart');
    
    // Clear previous chart if exists
    chartContainer.innerHTML = '';
    
    // Create canvas for the chart
    const canvas = document.createElement('canvas');
    chartContainer.appendChild(canvas);
    
    // Prepare data for the chart
    const labels = ['Acurácia', 'Precisão', 'Recall', 'F1-Score'];
    const model1Data = results.model1 ? [
        results.model1.accuracy * 100,
        results.model1.precision * 100,
        results.model1.recall * 100,
        results.model1.f1_score * 100
    ] : [];
    const model2Data = results.model2 ? [
        results.model2.accuracy * 100,
        results.model2.precision * 100,
        results.model2.recall * 100,
        results.model2.f1_score * 100
    ] : [];
    
    // Create the chart
    new Chart(canvas, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                ...(results.model1 ? [{
                    label: results.model1.name || 'Modelo 1',
                    data: model1Data,
                    backgroundColor: 'rgba(54, 162, 235, 0.7)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }] : []),
                ...(results.model2 ? [{
                    label: results.model2.name || 'Modelo 2',
                    data: model2Data,
                    backgroundColor: 'rgba(255, 99, 132, 0.7)',
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 1
                }] : [])
            ]
        },
        options: {
            responsive: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    title: {
                        display: true,
                        text: 'Porcentagem (%)'
                    }
                }
            },
            plugins: {
                legend: {
                    position: 'top',
                },
                title: {
                    display: true,
                    text: 'Comparação de Métricas'
                }
            }
        }
    });
}

// Helpers: dataset extraction
// Duplicate removed. Single authoritative extractDatasetFromTable is defined above.

// Progress polling and UI updates
function startPolling(which) {
    const jobId = state.jobs[which];
    if (!jobId) return;

    // Clear previous interval
    if (state.intervals[which]) clearInterval(state.intervals[which]);
    state.intervals[which] = setInterval(async () => {
        try {
            const res = await fetch(`/api/train/status/${jobId}`);
            if (!res.ok) throw new Error('Status não disponível');
            const status = await res.json();
            setProgressUI(which, status);
            if (status.status === 'completed') {
                clearInterval(state.intervals[which]);
                const r = await fetch(`/api/train/result/${jobId}`);
                if (r.ok) {
                    const result = await r.json();
                    state.results[which] = result;
                    tryComposeComparison();
                    saveToHistory(which, result);
                }
            } else if (status.status === 'error') {
                clearInterval(state.intervals[which]);
                alert(`Erro ao treinar ${which}: ${status.error}`);
            } else if (status.status === 'canceled' || status.status === 'cancelling') {
                clearInterval(state.intervals[which]);
                const etaEl = document.getElementById(which === 'model1' ? 'm1-eta' : 'm2-eta');
                if (etaEl) etaEl.textContent = 'Cancelado';
                const prog = document.getElementById(which === 'model1' ? 'm1-progress' : 'm2-progress');
                if (prog) prog.style.width = '0%';
            }
        } catch (e) {
            console.error(e);
        }
    }, 1000);
}

function setProgressUI(which, status) {
    const prefix = which === 'model1' ? 'm1' : 'm2';
    document.getElementById(`${prefix}-epoch`).textContent = status.epoch ?? 0;
    // Ensure the total epochs displayed come from server status
    const totalEl = document.getElementById(`${prefix}-total`);
    if (totalEl) totalEl.textContent = (status.total_epochs ?? 0);
    document.getElementById(`${prefix}-loss`).textContent = formatMetric(status.loss);
    document.getElementById(`${prefix}-acc`).textContent = formatMetric(status.accuracy);
    const pct = status.total_epochs ? Math.min(100, Math.round((status.epoch / status.total_epochs) * 100)) : 0;
    document.getElementById(`${prefix}-progress`).style.width = `${pct}%`;
    document.getElementById(`${prefix}-eta`).textContent = formatEta(status.eta_seconds);
}

function formatMetric(v) {
    if (v === null || v === undefined || v === '--') return '--';
    const num = Number(v);
    if (Number.isNaN(num)) return '--';
    return num.toFixed(4);
}

function formatEta(sec) {
    if (!sec && sec !== 0) return '--';
    const m = Math.floor(sec / 60);
    const s = sec % 60;
    return `${m}m ${s}s`;
}

function tryComposeComparison() {
    const results = {};
    if (state.results.model1) results.model1 = state.results.model1;
    if (state.results.model2) results.model2 = state.results.model2;
    if (results.model1 || results.model2) displayResults(results);
}

// Testing trained models
async function testModel(which) {
    const jobId = state.jobs[which];
    if (!jobId) {
        alert('Treine o modelo antes de testar.');
        return;
    }
    const x1 = parseFloat(document.getElementById('test-x1').value);
    const x2 = parseFloat(document.getElementById('test-x2').value);
    const x3 = parseFloat(document.getElementById('test-x3').value);
    const dRaw = document.getElementById('test-label').value;
    if ([x1, x2, x3].some(v => Number.isNaN(v))) {
        alert('Insira valores válidos para x₁, x₂ e x₃.');
        return;
    }
    const payload = { features: [[x1, x2, x3]] };
    if (dRaw !== '') {
        const d = parseFloat(dRaw);
        if (!Number.isNaN(d)) payload.labels = [d];
    }

    try {
        // Loading indicator on buttons
        const btn = document.getElementById(which === 'model1' ? 'test-model1' : 'test-model2');
        const old = btn.textContent;
        btn.textContent = 'Testando...';
        btn.disabled = true;
        const res = await fetch(`/api/test/${jobId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });
        if (!res.ok) throw new Error('Falha ao testar');
        const data = await res.json();
        document.getElementById('prediction-output').textContent = Array.isArray(data.predictions) ? data.predictions[0] : data.predictions;
        const metricsEl = document.getElementById('test-metrics');
        if (data.accuracy !== undefined) {
            metricsEl.classList.remove('hidden');
            document.getElementById('test-accuracy').textContent = (data.accuracy * 100).toFixed(2) + '%';
            document.getElementById('test-precision').textContent = (data.precision * 100).toFixed(2) + '%';
            document.getElementById('test-recall').textContent = (data.recall * 100).toFixed(2) + '%';
            if (data.confusion_matrix) {
                document.getElementById('test-confusion').classList.remove('hidden');
                const tcm = data.confusion_matrix;
                document.getElementById('test-cm').innerHTML = `
                    <tr><td></td><td>Prev -1</td><td>Prev 1</td></tr>
                    <tr><td>Real -1</td><td>${tcm[0][0]}</td><td>${tcm[0][1]}</td></tr>
                    <tr><td>Real 1</td><td>${tcm[1][0]}</td><td>${tcm[1][1]}</td></tr>
                `;
            }
            if (Array.isArray(data.probabilities)) {
                const p = data.probabilities[0];
                const p1 = Array.isArray(p) ? p[1] : p;
                document.getElementById('test-proba').classList.remove('hidden');
                document.getElementById('test-proba-output').textContent = Number(p1).toFixed(4);
            }
        } else {
            metricsEl.classList.add('hidden');
        }
    } catch (e) {
        console.error(e);
        alert('Erro ao testar o modelo.');
    } finally {
        const btn = document.getElementById(which === 'model1' ? 'test-model1' : 'test-model2');
        btn.textContent = which === 'model1' ? 'Testar Modelo 1' : 'Testar Modelo 2';
        btn.disabled = false;
    }
}

// Cancel training for a given model
async function cancelJob(which) {
    const jobId = state.jobs[which];
    if (!jobId) return;
    try {
        const res = await fetch(`/api/train/cancel/${jobId}`, { method: 'POST' });
        if (!res.ok) throw new Error('Falha ao cancelar');
        const status = await res.json();
        const etaEl = document.getElementById(which === 'model1' ? 'm1-eta' : 'm2-eta');
        if (etaEl) etaEl.textContent = status.status === 'cancelling' ? 'Cancelando...' : 'Cancelado';
    } catch (e) {
        console.error(e);
        alert('Não foi possível cancelar o treinamento.');
    }
}

// Poll until job completes or cancels; updates progress UI as it goes
async function pollUntilComplete(which) {
    const jobId = state.jobs[which];
    if (!jobId) return null;
    while (true) {
        const res = await fetch(`/api/train/status/${jobId}`);
        if (!res.ok) throw new Error('Status não disponível');
        const status = await res.json();
        setProgressUI(which, status);
        if (status.status === 'completed') {
            const r = await fetch(`/api/train/result/${jobId}`);
            if (r.ok) return await r.json();
            return null;
        }
        if (status.status === 'error') {
            alert(`Erro ao treinar ${which}: ${status.error}`);
            return null;
        }
        if (status.status === 'canceled') {
            const etaEl = document.getElementById(which === 'model1' ? 'm1-eta' : 'm2-eta');
            if (etaEl) etaEl.textContent = 'Cancelado';
            return null;
        }
        await new Promise(r => setTimeout(r, 800));
    }
}

// Export and history
function exportResults() {
    if (!state.results.model1 && !state.results.model2) {
        alert('Sem resultados para exportar.');
        return;
    }
    const blob = new Blob([JSON.stringify({
        timestamp: new Date().toISOString(),
        results: state.results
    }, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'comparacao_resultados.json';
    a.click();
    URL.revokeObjectURL(url);
}

function saveToHistory(which, result) {
    const entry = {
        when: new Date().toLocaleString(),
        which,
        name: result.name,
        metrics: {
            accuracy: result.accuracy,
            precision: result.precision,
            recall: result.recall,
            f1_score: result.f1_score
        }
    };
    state.history.push(entry);
    try {
        localStorage.setItem('mc_history', JSON.stringify(state.history));
    } catch {}
    renderHistory();
}

function loadHistory() {
    try {
        const raw = localStorage.getItem('mc_history');
        if (raw) state.history = JSON.parse(raw);
    } catch {}
    renderHistory();
}

function renderHistory() {
    const list = document.getElementById('history-list');
    if (!list) return;
    list.innerHTML = '';
    state.history.slice().reverse().forEach(h => {
        const li = document.createElement('li');
        li.textContent = `${h.when} • ${h.which.toUpperCase()} • ${h.name} • Acc ${(h.metrics.accuracy*100).toFixed(1)}%`;
        list.appendChild(li);
    });
}

function clearHistory() {
    state.history = [];
    try { localStorage.removeItem('mc_history'); } catch {}
    renderHistory();
}