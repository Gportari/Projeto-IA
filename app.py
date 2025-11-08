from flask import Flask, render_template, request, redirect, jsonify
import threading
import time
from tensorflow import keras
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

# Background training job registry
TRAIN_JOBS = {}
TRAIN_JOBS_LOCK = threading.Lock()

class EpochProgressCallback(keras.callbacks.Callback):
    def __init__(self, job_id, total_epochs):
        super().__init__()
        self.job_id = job_id
        self.total_epochs = total_epochs
        self.start_time = time.time()
        self.last_epoch_time = self.start_time

    def on_epoch_end(self, epoch, logs=None):
        now = time.time()
        elapsed = now - self.start_time
        avg_epoch_time = elapsed / (epoch + 1)
        remaining_epochs = self.total_epochs - (epoch + 1)
        eta_seconds = max(0, int(avg_epoch_time * remaining_epochs))
        with TRAIN_JOBS_LOCK:
            job = TRAIN_JOBS.get(self.job_id, {})
            job.update({
                'epoch': epoch + 1,
                'total_epochs': self.total_epochs,
                'loss': float(logs.get('loss', 0.0)) if logs else 0.0,
                'accuracy': float(logs.get('accuracy', 0.0)) if logs else 0.0,
                'val_loss': float(logs.get('val_loss', 0.0)) if logs and 'val_loss' in logs else None,
                'val_accuracy': float(logs.get('val_accuracy', 0.0)) if logs and 'val_accuracy' in logs else None,
                'eta_seconds': eta_seconds,
                'status': 'training'
            })
            # Check for cancel request
            if job.get('cancel_requested'):
                job['status'] = 'cancelling'
                self.model.stop_training = True
            TRAIN_JOBS[self.job_id] = job

def _create_model_instance(model_type, model_config, training_params):
    # Factory to get model instance for callback-driven training
    if model_type == 'logistic':
        from trainingModels.LogisticRegression import create_model_from_config as _create
    elif model_type == 'knn':
        from trainingModels.KNN import create_model_from_config as _create
    elif model_type == 'svc':
        from trainingModels.SVC import create_model_from_config as _create
    elif model_type == 'tree':
        from trainingModels.DecisionTree import create_model_from_config as _create
    elif model_type == 'rf':
        from trainingModels.RandomForest import create_model_from_config as _create
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return _create(model_config, training_params)

def _load_config_by_filename(filename):
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models_json')
    config_path = os.path.join(model_dir, filename)
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def _start_training_thread(job_id, model_type, config_file, dataset, training_params):
    def _runner():
        try:
            features = np.array(dataset['features'])
            labels = np.array(dataset['labels'])
            test_size = float(training_params.get('test_split', 0.2))
            X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=42)

            # Create model
            model_config = _load_config_by_filename(config_file)
            model = _create_model_instance(model_type, model_config, training_params)
            # Safely parse epochs for callback ETA
            raw_ep = training_params.get('epochs', 50)
            try:
                epochs = int(raw_ep)
                if epochs <= 0:
                    epochs = 50
            except Exception:
                epochs = 50
            callbacks = [EpochProgressCallback(job_id, epochs)]

            # Train with callbacks
            model.fit(X_train, y_train, validation_split=float(training_params.get('validation_split', 0.2)), callbacks=callbacks)

            # Evaluate unless canceled
            with TRAIN_JOBS_LOCK:
                cancelled = TRAIN_JOBS.get(job_id, {}).get('cancel_requested')
            if cancelled:
                with TRAIN_JOBS_LOCK:
                    TRAIN_JOBS[job_id].update({
                        'status': 'canceled',
                        'model_type': model_type,
                        'config_file': config_file,
                        'model_ref': model
                    })
            else:
                metrics = model.evaluate(X_test, y_test)
                results = {
                    'name': model_config.get('nome_modelo') or model_config.get('name', model_type),
                    **metrics
                }
                with TRAIN_JOBS_LOCK:
                    TRAIN_JOBS[job_id].update({
                        'status': 'completed',
                        'results': results,
                        'model_type': model_type,
                        'config_file': config_file,
                        'model_ref': model
                    })
        except Exception as e:
            with TRAIN_JOBS_LOCK:
                TRAIN_JOBS[job_id].update({
                    'status': 'error',
                    'error': str(e)
                })

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    return t

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
    # Endpoint API para obter a lista de modelos com metadados e filename
    models_raw = registry.listar_modelos()
    # Group by model type name for UI
    grouped = {}
    for m in models_raw:
        raw_tipo = m.get('nome_modelo') or m.get('tipo') or 'Desconhecido'
        # Normalize type names to match frontend expectations
        tipo_map = {
            'DecisionTreeClassifier': 'Decision Tree',
            'RandomForestClassifier': 'Random Forest'
        }
        tipo = tipo_map.get(raw_tipo, raw_tipo)
        grouped.setdefault(tipo, []).append({
            'filename': m.get('filename'),
            'nome_teste': m.get('nome_teste'),
            'nome_modelo': tipo,
            'params': {k: v for k, v in m.items() if k not in ['filename','nome_teste','nome_modelo','timestamp','id']}
        })
    return jsonify(grouped)

@app.route('/api/train/start', methods=['POST'])
def api_train_start():
    payload = request.json
    job_id = str(int(time.time() * 1000))
    # Safely parse epochs from payload; handle null/NaN/missing
    tp = payload.get('training_params', {})
    raw_epochs = tp.get('epochs')
    try:
        total_epochs = int(raw_epochs) if raw_epochs is not None else 50
        if total_epochs <= 0:
            total_epochs = 50
    except Exception:
        total_epochs = 50
    with TRAIN_JOBS_LOCK:
        TRAIN_JOBS[job_id] = {
            'status': 'queued',
            'epoch': 0,
            'total_epochs': total_epochs,
            'loss': None,
            'accuracy': None,
            'eta_seconds': None,
            'cancel_requested': False
        }
    _start_training_thread(job_id, payload['model']['type'], payload['model']['config'], payload['dataset'], payload.get('training_params', {}))
    return jsonify({'job_id': job_id})

@app.route('/api/train/status/<job_id>', methods=['GET'])
def api_train_status(job_id):
    with TRAIN_JOBS_LOCK:
        job = TRAIN_JOBS.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    # Remove non-serializable objects from the response
    sanitized = dict(job)
    if 'model_ref' in sanitized:
        sanitized.pop('model_ref')
    return jsonify(sanitized)

@app.route('/api/train/cancel/<job_id>', methods=['POST'])
def api_train_cancel(job_id):
    with TRAIN_JOBS_LOCK:
        job = TRAIN_JOBS.get(job_id)
        if not job:
            return jsonify({'error': 'Job not found'}), 404
        if job.get('status') in ['completed', 'error', 'canceled']:
            return jsonify({'status': job.get('status')}), 200
        job['cancel_requested'] = True
        TRAIN_JOBS[job_id] = job
    return jsonify({'status': 'cancelling'})

@app.route('/api/train/result/<job_id>', methods=['GET'])
def api_train_result(job_id):
    with TRAIN_JOBS_LOCK:
        job = TRAIN_JOBS.get(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    if job.get('status') != 'completed':
        return jsonify({'error': 'Job not completed yet'}), 400
    return jsonify(job.get('results'))

@app.route('/api/test/<job_id>', methods=['POST'])
def api_test_model(job_id):
    # Test a trained model with provided dataset, return predictions and metrics
    payload = request.json
    features = np.array(payload['features'])
    labels = np.array(payload.get('labels')) if payload.get('labels') is not None else None
    with TRAIN_JOBS_LOCK:
        job = TRAIN_JOBS.get(job_id)
    if not job or job.get('status') != 'completed':
        return jsonify({'error': 'Model not ready'}), 400
    model = job.get('model_ref')
    if model is None:
        return jsonify({'error': 'Model reference unavailable'}), 500
    # Predict
    y_pred = model.predict(features).tolist()
    response = {
        'predictions': y_pred
    }
    # If labels provided, compute metrics
    if labels is not None:
        # Convert to numpy and metrics consistent with trainingModels evaluate
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        yp = np.array(y_pred)
        acc = float(accuracy_score(labels, yp))
        prec = float(precision_score(labels, yp, zero_division=0))
        rec = float(recall_score(labels, yp, zero_division=0))
        f1 = float(f1_score(labels, yp, zero_division=0))
        cm = confusion_matrix(labels, yp, labels=[-1, 1]).tolist()

        # Try to provide raw probabilities if the model supports it
        proba = None
        try:
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(features).tolist()
        except Exception:
            proba = None

        response.update({
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'confusion_matrix': cm,
            'probabilities': proba
        })
    return jsonify(response)

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
