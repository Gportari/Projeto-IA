from flask import Flask, render_template, request, redirect, jsonify
from model_registry import ModelRegistry
import os
import json

app = Flask(__name__)
registry = ModelRegistry()

# Garantir que o diretório models_json exista
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models_json')
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

@app.route('/')
def home():
    return render_template('Dashboard.html')

@app.route('/dashboard')
def dashboard():
    return render_template('Dashboard.html')

@app.route('/Dashboard.html')
def Dashboard():
    return render_template('Dashboard.html')

@app.route('/Models.html', methods=['GET'])
def models():
    # Carregar modelos dos arquivos JSON
    models_list = registry.listar_modelos()
    return render_template('Models.html', model_list=models_list)

@app.route('/api/models', methods=['GET'])
def get_models_api():
    # Endpoint API para obter a lista de modelos em formato JSON
    return jsonify(registry.listar_modelos())

@app.route('/cadastrar_modelo', methods=['POST'])
def cadastrar_modelo():
    nome_teste = request.form['id_teste']
    nome_modelo = request.form['modelo_ml_utilizado']

    if nome_modelo == 'Regressão Logística':
        registry.cadastrar_logistic(
            nome_teste,
            request.form.get('estrategia_preprocessamento'),
            request.form.get('estrategia_validacao'),
            request.form.get('logreg_penalty'),
            request.form.get('logreg_C'),
            request.form.get('logreg_solver'),
            request.form.get('logreg_class_weight'),
            request.form.get('logreg_max_iter')
        )
    elif nome_modelo == 'KNN':
        registry.cadastrar_knn(
            nome_teste,
            request.form.get('estrategia_preprocessamento'),
            request.form.get('estrategia_validacao'),
            request.form.get('knn_n_neighbors'),
            request.form.get('knn_weights'),
            request.form.get('knn_metric'),
            request.form.get('knn_p')
        )
    elif nome_modelo == 'SVC':
        registry.cadastrar_svc(
            nome_teste,
            request.form.get('estrategia_preprocessamento'),
            request.form.get('estrategia_validacao'),
            request.form.get('svm_C'),
            request.form.get('svm_kernel'),
            request.form.get('svm_gamma'),
            request.form.get('svm_degree'),
            request.form.get('svm_class_weight'),
            request.form.get('svm_probability')
        )
    elif nome_modelo == 'DecisionTreeClassifier':
        registry.cadastrar_tree(
            nome_teste,
            request.form.get('estrategia_preprocessamento'),
            request.form.get('estrategia_validacao'),
            request.form.get('tree_criterion'),
            request.form.get('tree_max_depth'),
            request.form.get('tree_min_samples_split'),
            request.form.get('tree_min_samples_leaf'),
            request.form.get('tree_class_weight')
        )
    elif nome_modelo == 'RandomForestClassifier':
        registry.cadastrar_rf(
            nome_teste,
            request.form.get('estrategia_preprocessamento'),
            request.form.get('estrategia_validacao'),
            request.form.get('rf_n_estimators'),
            request.form.get('rf_max_depth'),
            request.form.get('rf_criterion'),
            request.form.get('rf_min_samples_split'),
            request.form.get('rf_min_samples_leaf'),
            request.form.get('rf_class_weight'),
            request.form.get('rf_oob_score')
        )
    return redirect('/Models.html')

if __name__ == "__main__":
    app.run(debug=True)
