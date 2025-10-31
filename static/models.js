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

