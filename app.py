from flask import Flask, render_template, request, redirect, jsonify
from model_registry import ModelRegistry
import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
# Import Keras-based model implementations
from trainingModels.LogisticRegression import train_and_evaluate as train_logistic
from trainingModels.KNN import train_and_evaluate as train_knn
from trainingModels.SVC import train_and_evaluate as train_svc
from trainingModels.DecisionTree import train_and_evaluate as train_tree
from trainingModels.RandomForest import train_and_evaluate as train_rf

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

@app.route('/ModelComparison.html')
def model_comparison():
    # Carregar modelos dos arquivos JSON para a página de comparação
    models_list = registry.listar_modelos()
    return render_template('ModelComparison.html', model_list=models_list)

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



@app.route('/api/train-compare', methods=['POST'])
def train_compare():
    data = request.json
    
    # Extract dataset
    features = np.array(data['dataset']['features'])
    labels = np.array(data['dataset']['labels'])
    
    # Split dataset
    test_size = data['training_params']['test_split']
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=42
    )
    
    # Train and evaluate models
    results = {
        'model1': train_and_evaluate_model(data['model1'], X_train, X_test, y_train, y_test, 
                                          data['training_params']),
        'model2': train_and_evaluate_model(data['model2'], X_train, X_test, y_train, y_test,
                                          data['training_params'])
    }
    
    return jsonify(results)

def train_and_evaluate_model(model_info, X_train, X_test, y_train, y_test, training_params):
    model_type = model_info['type']
    config_file = model_info['config']
    
    # Load model configuration
    model_config = load_model_config(model_type, config_file)
    
    # Use the appropriate training function based on model type
    if model_type == 'logistic':
        results, _ = train_logistic(model_config, X_train, X_test, y_train, y_test, training_params)
    elif model_type == 'knn':
        results, _ = train_knn(model_config, X_train, X_test, y_train, y_test, training_params)
    elif model_type == 'svc':
        results, _ = train_svc(model_config, X_train, X_test, y_train, y_test, training_params)
    elif model_type == 'tree':
        results, _ = train_tree(model_config, X_train, X_test, y_train, y_test, training_params)
    elif model_type == 'rf':
        results, _ = train_rf(model_config, X_train, X_test, y_train, y_test, training_params)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Add model name to results if not already present
    if 'name' not in results:
        results['name'] = config_file.replace('.json', '')
    
    return results

def load_model_config(model_type, config_file):
    model_type_map = {
        'logistic': 'Regressão Logística',
        'knn': 'KNN',
        'svc': 'SVC',
        'tree': 'Decision Tree',
        'rf': 'Random Forest'
    }
    
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models_json')
    config_path = os.path.join(model_dir, config_file)
    
    with open(config_path, 'r') as f:
        return json.load(f)
    
            

if __name__ == "__main__":
    app.run(debug=True)
