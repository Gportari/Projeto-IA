// Model Comparison JavaScript
document.addEventListener('DOMContentLoaded', function() {
    // Initialize model configuration dropdowns
    fetchModels();
    
    // Set up event listeners for model type changes
    document.getElementById('model1-type').addEventListener('change', function() {
        updateModelConfigs('model1-type', 'model1-config');
    });
    
    document.getElementById('model2-type').addEventListener('change', function() {
        updateModelConfigs('model2-type', 'model2-config');
    });
    
    // Set up train button event listener
    document.getElementById('train-button').addEventListener('click', trainAndCompareModels);
});

// Global variable to store all models
let allModels = {};

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

// Train and compare the selected models
async function trainAndCompareModels() {
    const model1Type = document.getElementById('model1-type').value;
    const model1Config = document.getElementById('model1-config').value;
    const model2Type = document.getElementById('model2-type').value;
    const model2Config = document.getElementById('model2-config').value;
    const epochs = document.getElementById('epochs').value;
    const learningRate = document.getElementById('learning-rate').value;
    const testSplit = document.getElementById('test-split').value;
    
    // Validate selections
    if (!model1Config || !model2Config) {
        alert('Por favor, selecione configurações para ambos os modelos.');
        return;
    }
    
    // Show loading state
    const trainButton = document.getElementById('train-button');
    const originalButtonText = trainButton.textContent;
    trainButton.textContent = 'Treinando...';
    trainButton.disabled = true;
    
    try {
        // Extract dataset from the hidden table
        const dataset = extractDatasetFromTable();
        
        // Send request to train and compare models
        const response = await fetch('/api/train-compare', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                model1: {
                    type: model1Type,
                    config: model1Config
                },
                model2: {
                    type: model2Type,
                    config: model2Config
                },
                training_params: {
                    epochs: parseInt(epochs),
                    learning_rate: parseFloat(learningRate),
                    test_split: parseInt(testSplit) / 100,
                    batch_size: 32,
                    validation_split: 0.2
                },
                dataset: dataset
            })
        });
        
        if (!response.ok) {
            throw new Error('Failed to train models');
        }
        
        const results = await response.json();
        displayResults(results);
        
    } catch (error) {
        console.error('Error training models:', error);
        alert('Ocorreu um erro ao treinar os modelos. Por favor, tente novamente.');
    } finally {
        // Restore button state
        trainButton.textContent = originalButtonText;
        trainButton.disabled = false;
    }
}

// Extract dataset from the hidden table
function extractDatasetFromTable() {
    const table = document.querySelector('#dataset table');
    const rows = table.querySelectorAll('tbody tr');
    
    const features = [];
    const labels = [];
    
    rows.forEach(row => {
        const cells = row.querySelectorAll('td');
        
        // Extract features (x₁, x₂, x₃)
        const rowFeatures = [
            parseFloat(cells[1].textContent),
            parseFloat(cells[2].textContent),
            parseFloat(cells[3].textContent)
        ];
        
        // Extract label (d)
        // Convert -1.0 to 0 for binary classification with Keras
        const rawLabel = parseFloat(cells[4].textContent);
        const label = rawLabel === -1.0 ? 0 : 1;
        
        features.push(rowFeatures);
        labels.push(label);
    });
    
    return {
        features: features,
        labels: labels
    };
}

// Display the comparison results
function displayResults(results) {
    // Show results container
    document.getElementById('results-container').classList.remove('hidden');
    
    // Update model names
    document.getElementById('model1-name').textContent = results.model1.name;
    document.getElementById('model2-name').textContent = results.model2.name;
    
    // Update metrics for model 1
    document.getElementById('model1-accuracy').textContent = (results.model1.accuracy * 100).toFixed(2) + '%';
    document.getElementById('model1-precision').textContent = (results.model1.precision * 100).toFixed(2) + '%';
    document.getElementById('model1-recall').textContent = (results.model1.recall * 100).toFixed(2) + '%';
    document.getElementById('model1-f1').textContent = (results.model1.f1_score * 100).toFixed(2) + '%';
    
    // Update metrics for model 2
    document.getElementById('model2-accuracy').textContent = (results.model2.accuracy * 100).toFixed(2) + '%';
    document.getElementById('model2-precision').textContent = (results.model2.precision * 100).toFixed(2) + '%';
    document.getElementById('model2-recall').textContent = (results.model2.recall * 100).toFixed(2) + '%';
    document.getElementById('model2-f1').textContent = (results.model2.f1_score * 100).toFixed(2) + '%';
    
    // Create comparison chart
    createComparisonChart(results);
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
    const model1Data = [
        results.model1.accuracy * 100,
        results.model1.precision * 100,
        results.model1.recall * 100,
        results.model1.f1_score * 100
    ];
    const model2Data = [
        results.model2.accuracy * 100,
        results.model2.precision * 100,
        results.model2.recall * 100,
        results.model2.f1_score * 100
    ];
    
    // Create the chart
    new Chart(canvas, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: results.model1.name,
                    data: model1Data,
                    backgroundColor: 'rgba(54, 162, 235, 0.7)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                },
                {
                    label: results.model2.name,
                    data: model2Data,
                    backgroundColor: 'rgba(255, 99, 132, 0.7)',
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 1
                }
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