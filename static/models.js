document.addEventListener('DOMContentLoaded', function() {
    const filter = document.getElementById('model-type-filter');
    const forms = {
        logistic: document.getElementById('form-logistic'),
        knn: document.getElementById('form-knn'),
        svc: document.getElementById('form-svc'),
        tree: document.getElementById('form-tree'),
        rf: document.getElementById('form-rf')
    };
    function showForm(type) {
        Object.keys(forms).forEach(key => {
            forms[key].style.display = (key === type) ? '' : 'none';
        });
    }
    filter.addEventListener('change', function() {
        showForm(filter.value);
        updateModelsTable(filter.value);
    });
    showForm(filter.value); // Inicializa com o selecionado
    updateModelsTable(filter.value); // Inicializa tabela
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
    "svc": {
        "name": "SVC",
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

/**
 * 1. Gera o cabeçalho e preenche o corpo da tabela de resultados
 * para o modelo atualmente SELECIONADO no filtro.
 * @param {string} modelKey - A chave do modelo ('logistic', 'knn', etc.).
 */
function renderResultsTable(modelKey) {
    const modelDefinition = window.model_list[modelKey];
    const registeredTests = window.registered_tests || []; // Usa os dados de testes registrados
    
    if (!modelDefinition) return;

    const head = document.getElementById('models-table-head');
    const body = document.getElementById('model-table-body');
    head.innerHTML = '';
    body.innerHTML = '';

    // --- 1. Definir as Colunas para este Modelo ---
    // Colunas Iniciais + Colunas Comuns + Parâmetros do Modelo + Métricas
    const modelColumns = [
        ...commonTestColumns,
        ...modelDefinition.params, // Apenas os parâmetros do modelo selecionado!
        ...fixedMetricColumns 
    ];

    // --- 2. Gerar o Cabeçalho (<thead>) ---
    const headerRow = document.createElement('tr');
    
    modelColumns.forEach(columnName => {
        const th = document.createElement('th');
        th.classList.add('px-6', 'py-3');
        th.setAttribute('scope', 'col');
        
        let display_name = columnName.replace('_', ' ');

        // Trata os nomes dos hiperparâmetros para remover o prefixo do modelo (logreg_, knn_, etc.)
        if (modelDefinition.params.includes(columnName)) {
            // Separa pelo primeiro '_' e pega o resto (Ex: logreg_C -> C)
            display_name = columnName.split('_').slice(1).join('_');
            if (display_name === '') display_name = columnName; // Para casos como knn_p
        }
        
        th.textContent = display_name.toUpperCase();
        headerRow.appendChild(th);
    });
    
    head.appendChild(headerRow);


    // --- 3. Preencher o Corpo (<tbody>) ---
    // Filtra os testes registrados para mostrar SÓ os do modelo selecionado
    const filteredTests = registeredTests.filter(test => test.Modelo_ML_Utilizado === modelKey);

    filteredTests.forEach(test => {
        const dataRow = document.createElement('tr');
        dataRow.classList.add('bg-white', 'border-b', 'dark:bg-[#1f2d3a]', 'dark:border-[#2a3746]');
        
        modelColumns.forEach(col => {
            const td = document.createElement('td');
            td.classList.add('px-6', 'py-4', 'whitespace-nowrap');
            
            // Pega o valor do teste para a coluna atual (se não existir, é NULO/vazio)
            const value = test[col] !== undefined ? test[col] : '—'; 
            
            td.textContent = value;
            dataRow.appendChild(td);
        });

        body.appendChild(dataRow);
    });
    
    // Se não houver testes registrados para o modelo selecionado
    if (filteredTests.length === 0) {
         body.innerHTML = `
            <tr class="bg-white dark:bg-[#1f2d3a]">
                <td colspan="${modelColumns.length}" class="px-6 py-4 text-center text-gray-500 dark:text-[#92adc9]">
                    Nenhum teste registrado para o modelo ${modelDefinition.name}.
                </td>
            </tr>
         `;
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
        'svc': document.getElementById('form-svc'),
        'tree': document.getElementById('form-tree'),
        'rf': document.getElementById('form-rf'),
    };
    for (const key in forms) {
        if (forms[key]) {
            forms[key].style.display = (key === selectedModel) ? 'block' : 'none';
        }
    }
    
    // --- Lógica de Renderização Dinâmica da Tabela ---
    renderResultsTable(selectedModel);
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
        selector.addEventListener('change', switchModelForm);
        // Exibe o formulário inicial (Regressão Logística, que é o primeiro no SELECT)
        switchModelForm(); 
    }
    
    // NOTA: Você precisará de outra função aqui que lê os dados dos testes SALVOS e preenche o <tbody> (model-table-body)
    // Usando o array finalColumns gerado em generateDynamicTableHeader() para mapear os dados.
});
