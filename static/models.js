document.addEventListener('DOMContentLoaded', function() {
    const filter = document.getElementById('model-type-filter');
    const forms = {
        logistic: document.getElementById('form-logistic'),
        knn: document.getElementById('form-knn'),
        svm: document.getElementById('form-svm'),
        tree: document.getElementById('form-tree'),
        rf: document.getElementById('form-rf')
    };
    
    function showForm(type) {
        Object.keys(forms).forEach(key => {
            if (forms[key]) {
                forms[key].style.display = (key === type) ? '' : 'none';
            }
        });
    }
    
    // Carregar modelos do servidor
    function loadModelsFromServer() {
        fetch('/api/models')
            .then(response => response.json())
            .then(data => {
                // Converter objeto agrupado por tipo em lista plana
                window.saved_models = Object.values(data || {}).flat();
                // Atualizar a tabela com os modelos carregados
                updateModelsTable();
            })
            .catch(error => console.error('Erro ao carregar modelos:', error));
    }
    
    filter.addEventListener('change', function() {
        showForm(filter.value);
        updateModelsTable(filter.value);
    });
    
    showForm(filter.value); // Inicializa com o selecionado
    loadModelsFromServer(); // Carrega modelos do servidor
});
// Este é o conteúdo do que o backend deve exportar via: window.model_list = {{ model_list | tojson }};

window.model_list = {
    // Definindo a estrutura dos parâmetros para cada modelo
    "logistic": {
        "name": "Regressão Logística",
        "fixed_cols": [
            "ID_Teste",
            "Estrategia_Preprocessamento",
            "Estrategia_Validacao"
        ],
        "params": [
            "logreg_penalty",
            "logreg_C",
            "logreg_solver",
            "logreg_class_weight",
            "logreg_max_iter"
        ]
    },
    "knn": {
        "name": "KNN",
        "fixed_cols": [
            "ID_Teste",
            "Estrategia_Preprocessamento",
            "Estrategia_Validacao"
        ],
        "params": [
            "knn_n_neighbors",
            "knn_weights",
            "knn_metric",
            "knn_p"
        ]
    },
    "svm": {
        "name": "SVM",
        "fixed_cols": [
            "ID_Teste",
            "Estrategia_Preprocessamento",
            "Estrategia_Validacao"
        ],
        "params": [
            "svm_C",
            "svm_kernel",
            "svm_gamma",
            "svm_degree",
            "svm_class_weight",
            "svm_probability"
        ]
    },
    "tree": {
        "name": "DecisionTreeClassifier",
        "fixed_cols": [
            "ID_Teste",
            "Estrategia_Preprocessamento",
            "Estrategia_Validacao"
        ],
        "params": [
            "tree_criterion",
            "tree_max_depth",
            "tree_min_samples_split",
            "tree_min_samples_leaf",
            "tree_class_weight"
        ]
    },
    "rf": {
        "name": "RandomForestClassifier",
        "fixed_cols": [
            "ID_Teste",
            "Estrategia_Preprocessamento",
            "Estrategia_Validacao"
        ],
        "params": [
            "rf_n_estimators",
            "rf_max_depth",
            "rf_criterion",
            "rf_min_samples_split",
            "rf_min_samples_leaf",
            "rf_class_weight",
            "rf_oob_score"
        ]
    }
};
// Variáveis de controle globais (ou definidas no escopo do módulo)
const commonTestColumns = ['ID_Teste', 'Estrategia_Preprocessamento', 'Estrategia_Validacao'];
const fixedMetricColumns = ['Acuracia_Media', 'F1_Score_Medio', 'Desvio_Padrao_F1'];
window.saved_models = []; // Array para armazenar os modelos carregados do servidor

/**
 * Atualiza a tabela de modelos com base no tipo de modelo selecionado
 * @param {string} modelType - O tipo de modelo selecionado ('logistic', 'knn', etc.)
 */
function updateModelsTable() {
    const savedModels = window.saved_models || [];
    const tableHead = document.getElementById('models-table-head');
    const tableBody = document.getElementById('model-table-body');

    // Cabeçalho fixo conforme Models.html
    tableHead.innerHTML = '';
    const headerRow = document.createElement('tr');
    ['Model Name', 'Type', 'Parameters'].forEach(h => {
        const th = document.createElement('th');
        th.classList.add('px-6', 'py-3');
        th.setAttribute('scope', 'col');
        th.textContent = h;
        headerRow.appendChild(th);
    });
    tableHead.appendChild(headerRow);

    tableBody.innerHTML = '';

    // Mapa reverso para descobrir o conjunto de parâmetros pelo tipo do modelo
    const displayToKey = {
        'Regressão Logística': 'logistic',
        'KNN': 'knn',
        'SVM': 'svm',
        'Decision Tree': 'tree',
        'Random Forest': 'rf'
    };

    if (savedModels.length === 0) {
        const emptyRow = document.createElement('tr');
        const emptyCell = document.createElement('td');
        emptyCell.classList.add('px-6', 'py-4', 'text-center', 'text-gray-500', 'dark:text-[#92adc9]');
        emptyCell.setAttribute('colspan', '3');
        emptyCell.textContent = 'Nenhum modelo encontrado.';
        emptyRow.appendChild(emptyCell);
        tableBody.appendChild(emptyRow);
        return;
    }

    savedModels.forEach(model => {
        const row = document.createElement('tr');
        row.classList.add('bg-white', 'border-b', 'dark:bg-[#1f2d3a]', 'dark:border-[#2a3746]');

        // Nome e Tipo
        const nameCell = document.createElement('td');
        nameCell.classList.add('px-6', 'py-4', 'whitespace-nowrap');
        nameCell.textContent = model.nome_teste || model.filename || '-';

        const typeCell = document.createElement('td');
        typeCell.classList.add('px-6', 'py-4', 'whitespace-nowrap');
        typeCell.textContent = model.nome_modelo || '-';

        // Parâmetros
        const paramsCell = document.createElement('td');
        paramsCell.classList.add('px-6', 'py-4');

        const key = displayToKey[model.nome_modelo];
        const paramsObj = model.params || {};
        const paramKeys = (key && window.model_list[key]) ? window.model_list[key].params : Object.keys(paramsObj);

        const pairs = paramKeys
            .filter(k => paramsObj[k] !== undefined)
            .map(k => {
                const pretty = k.includes('_') ? k.split('_').slice(1).join('_') : k;
                return `${pretty}=${paramsObj[k]}`;
            });

        paramsCell.textContent = pairs.join(', ');

        row.appendChild(nameCell);
        row.appendChild(typeCell);
        row.appendChild(paramsCell);
        tableBody.appendChild(row);
    });
}


/**
 * Função de controle que é chamada ao mudar o filtro (SWITCH/CHANGE).
 */
function switchModelFormAndRenderTable() {
    const selector = document.getElementById('model-type-filter');
    const selectedModel = selector.value;
    
    // --- Lógica de Alternância de Formulário (já existe) ---
    const forms = {
        'logistic': document.getElementById('form-logistic'),
        'knn': document.getElementById('form-knn'),
        // ... inclua os outros 3 forms
        'svm': document.getElementById('form-svm'),
        'tree': document.getElementById('form-tree'),
        'rf': document.getElementById('form-rf'),
    };
    for (const key in forms) {
        if (forms[key]) {
            forms[key].style.display = (key === selectedModel) ? 'block' : 'none';
        }
    }
    
    // --- Lógica de Renderização Dinâmica da Tabela ---
    updateModelsTable(selectedModel);
}


// --- Execução no carregamento da página ---
document.addEventListener('DOMContentLoaded', () => {
    const selector = document.getElementById('model-type-filter');
    if (selector) {
        // Integra a função de renderização com o evento de mudança do filtro
        selector.addEventListener('change', switchModelFormAndRenderTable);
        
        // Renderiza o modelo selecionado por padrão na inicialização
        switchModelFormAndRenderTable(); 
    }
});
// ----------------------------------------------------------------------------------
// Execução no carregamento da página
// ----------------------------------------------------------------------------------

// Função para carregar os modelos da API
function fetchModels() {
    fetch('/api/models')
        .then(response => {
            if (!response.ok) {
                throw new Error('Erro ao carregar modelos');
            }
            return response.json();
        })
        .then(data => {
            // Flatten grouped response into array
            window.saved_models = Object.values(data || {}).flat();
            // Atualiza a tabela com todos os modelos
            updateModelsTable();
        })
        .catch(error => {
            console.error('Erro ao carregar modelos:', error);
        });
}

/**
 * Gera o cabeçalho dinâmico da tabela de modelos
 * Esta função é chamada na inicialização da página
 */
function generateDynamicTableHeader() {
    // Esta função é substituída pela updateModelsTable que já gera o cabeçalho
    // Mantemos esta função para compatibilidade com o código existente
    updateModelsTable();
}

document.addEventListener('DOMContentLoaded', () => {
    // 1. Gera o cabeçalho dinâmico assim que o DOM estiver pronto
    if (window.model_list && Object.keys(window.model_list).length > 0) {
        generateDynamicTableHeader();
    } else {
        // Fallback caso a lista esteja vazia
        document.getElementById('models-table-head').innerHTML = '<tr><th class="px-6 py-3" scope="col">Nenhum modelo configurado.</th></tr>';
    }

    // 2. Inicializa o seletor de formulários
    const selector = document.getElementById('model-type-filter');
    if (selector) {
        selector.addEventListener('change', function() {
            const selectedModel = this.value;
            switchModelFormAndRenderTable(selectedModel);
        });
        // Exibe o formulário inicial e carrega os modelos
        const initialModel = selector.value || 'logistic';
        switchModelFormAndRenderTable(initialModel);
        fetchModels();
    }
    
    // 3. Adiciona evento aos formulários para recarregar modelos após cadastro
    const formElements = document.querySelectorAll('form');
    formElements.forEach(form => {
        form.addEventListener('submit', function() {
            // Recarregar modelos após um pequeno delay para permitir o processamento do servidor
            setTimeout(fetchModels, 1000);
        });
    });
});
