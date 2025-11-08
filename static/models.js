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
                // Atualizar a variável global com os dados do servidor
                window.saved_models = data;
                // Atualizar a tabela com os modelos carregados
                updateModelsTable(filter.value);
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
function updateModelsTable(modelType) {
    const modelDefinition = window.model_list[modelType];
    const savedModels = window.saved_models || [];
    
    if (!modelDefinition) return;

    const tableHead = document.getElementById('models-table-head');
    const tableBody = document.getElementById('model-table-body');
    
    // Limpar tabela
    tableHead.innerHTML = '';
    tableBody.innerHTML = '';

    // Mapear tipos de modelo para nomes de exibição
    const modelTypeMap = {
        'logistic': 'Regressão Logística',
        'knn': 'KNN',
        'svm': 'SVM',
        'tree': 'DecisionTreeClassifier',
        'rf': 'RandomForestClassifier'
    };
    
    // Filtrar modelos pelo tipo selecionado
    const filteredModels = savedModels.filter(model => {
        return model.nome_modelo === modelTypeMap[modelType];
    });
    
    // Definir colunas para este tipo de modelo
    const columns = [
        'nome_teste',
        'estrategia_preprocessamento',
        'estrategia_validacao',
        ...modelDefinition.params
    ];
    
    // Criar cabeçalho da tabela
    const headerRow = document.createElement('tr');
    
    columns.forEach(column => {
        const th = document.createElement('th');
        th.classList.add('px-6', 'py-3');
        th.setAttribute('scope', 'col');
        
        // Formatar nome da coluna para exibição
        let displayName = column;
        if (column.includes('_')) {
            // Para parâmetros de modelo, remover o prefixo (ex: logreg_C -> C)
            if (modelDefinition.params.includes(column)) {
                displayName = column.split('_').slice(1).join('_');
            } else {
                // Para outras colunas, substituir underscore por espaço
                displayName = column.replace(/_/g, ' ');
            }
        }
        
        th.textContent = displayName.toUpperCase();
        headerRow.appendChild(th);
    });
    
    tableHead.appendChild(headerRow);
    
    // Preencher corpo da tabela
    if (filteredModels.length > 0) {
        filteredModels.forEach(model => {
            const row = document.createElement('tr');
            row.classList.add('bg-white', 'border-b', 'dark:bg-[#1f2d3a]', 'dark:border-[#2a3746]');
            
            columns.forEach(column => {
                const td = document.createElement('td');
                td.classList.add('px-6', 'py-4', 'whitespace-nowrap');
                
                // Obter valor da coluna do modelo
                const value = model[column] !== undefined ? model[column] : '—';
                
                td.textContent = value;
                row.appendChild(td);
            });
            
            tableBody.appendChild(row);
        });
    } else {
        // Mensagem quando não há modelos
        const emptyRow = document.createElement('tr');
        emptyRow.classList.add('bg-white', 'dark:bg-[#1f2d3a]');
        
        const emptyCell = document.createElement('td');
        emptyCell.classList.add('px-6', 'py-4', 'text-center', 'text-gray-500', 'dark:text-[#92adc9]');
        emptyCell.setAttribute('colspan', columns.length);
        emptyCell.textContent = `Nenhum modelo ${modelTypeMap[modelType]} encontrado.`;
        
        emptyRow.appendChild(emptyCell);
        tableBody.appendChild(emptyRow);
    }
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
            window.saved_models = data;
            // Atualiza a tabela com os modelos carregados
            const currentModelType = document.getElementById('model-type-filter').value || 'logistic';
            updateModelsTable(currentModelType);
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
    const currentModelType = document.getElementById('model-type-filter').value || 'logistic';
    updateModelsTable(currentModelType);
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
