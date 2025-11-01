from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
from model_registry import ModelRegistry
import pandas as pd
import os
import datetime

app = Flask(__name__)
app.secret_key = "secret"  # Necessário para flash messages
registry = ModelRegistry()

DATASET_FOLDER = os.path.join(os.path.dirname(__file__), 'datasets')
os.makedirs(DATASET_FOLDER, exist_ok=True)

def list_datasets():
    files = []
    for fname in os.listdir(DATASET_FOLDER):
        if fname.endswith('.xlsx'):
            path = os.path.join(DATASET_FOLDER, fname)
            stat = os.stat(path)
            files.append({
                'name': fname,
                'size': int(stat.st_size / 1024),
                'mtime': datetime.datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M')
            })
    return files

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
    model_list = registry.model_definitions()  # Definição dos modelos (dict)
    registered_tests = registry.listar_modelos()  # Lista dos modelos cadastrados (list of dicts)
    print("DEBUG: model_list =", model_list)
    print("DEBUG: registered_tests =", registered_tests)
    return render_template('Models.html', model_list=model_list, registered_tests=registered_tests)

@app.route('/Datasets.html')
def datasets():
    files = list_datasets()
    return render_template('Datasets.html', files=files)

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

@app.route('/upload_dataset', methods=['POST'])
def upload_dataset():
    file = request.files.get('file')
    custom_name = request.form.get('custom_name', '').strip()
    if file and file.filename.endswith('.xlsx') and custom_name:
        filename = secure_filename(custom_name) + '.xlsx'
        save_path = os.path.join(DATASET_FOLDER, filename)
        file.save(save_path)
        flash("Arquivo carregado com sucesso!", "success")
        return redirect(url_for('datasets'))
    flash("Arquivo inválido. Envie um arquivo .xlsx e informe o nome.", "danger")
    return redirect(url_for('datasets'))

@app.route('/delete_dataset', methods=['POST'])
def delete_dataset():
    filename = request.form.get('filename')
    if filename:
        path = os.path.join(DATASET_FOLDER, filename)
        if os.path.exists(path):
            os.remove(path)
            flash("Arquivo removido.", "success")
        else:
            flash("Arquivo não encontrado.", "danger")
    return redirect(url_for('datasets'))

@app.route('/rename_dataset', methods=['POST'])
def rename_dataset():
    old_name = request.form.get('old_name')
    new_name = request.form.get('new_name')
    if old_name and new_name and new_name.endswith('.xlsx'):
        old_path = os.path.join(DATASET_FOLDER, old_name)
        new_path = os.path.join(DATASET_FOLDER, secure_filename(new_name))
        if os.path.exists(old_path):
            os.rename(old_path, new_path)
            flash("Arquivo renomeado.", "success")
        else:
            flash("Arquivo não encontrado.", "danger")
    else:
        flash("Nome inválido.", "danger")
    return redirect(url_for('datasets'))

if __name__ == "__main__":
    app.run(debug=True)
