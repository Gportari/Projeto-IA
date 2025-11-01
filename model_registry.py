import json
import os
import uuid
from datetime import datetime

class ModelRegistry:
    def __init__(self):
        self.models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models_json')
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
        
        self.model_list = self._load_models_from_json()

    def _save_model_to_json(self, model_data):
        """Salva um modelo em um arquivo JSON individual"""
        if 'id' not in model_data:
            model_data['id'] = str(uuid.uuid4())
        
        model_data['timestamp'] = datetime.now().isoformat()
        
        filename = f"{model_data['nome_teste']}_{model_data['id']}.json"
        file_path = os.path.join(self.models_dir, filename)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, ensure_ascii=False, indent=4)
        
        return model_data

    def _load_models_from_json(self):
        """Carrega todos os modelos dos arquivos JSON"""
        models = []
        
        if not os.path.exists(self.models_dir):
            return models
        
        for filename in os.listdir(self.models_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(self.models_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        model_data = json.load(f)
                        models.append(model_data)
                except Exception as e:
                    print(f"Erro ao carregar o arquivo {filename}: {e}")
        
        models.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return models

    def cadastrar_logistic(self, nome_teste, estrategia_preprocessamento, estrategia_validacao,
                           penalty, C, solver, class_weight, max_iter):
        model_data = {
            'nome_teste': nome_teste,
            'nome_modelo': 'Regressão Logística',
            'estrategia_preprocessamento': estrategia_preprocessamento,
            'estrategia_validacao': estrategia_validacao,
            'logreg_penalty': penalty,
            'logreg_C': C,
            'logreg_solver': solver,
            'logreg_class_weight': class_weight,
            'logreg_max_iter': max_iter
        }
        
        saved_model = self._save_model_to_json(model_data)
        self.model_list.insert(0, saved_model)  
        return saved_model

    def cadastrar_knn(self, nome_teste, estrategia_preprocessamento, estrategia_validacao,
                      n_neighbors, weights, metric, p):
        model_data = {
            'nome_teste': nome_teste,
            'nome_modelo': 'KNN',
            'estrategia_preprocessamento': estrategia_preprocessamento,
            'estrategia_validacao': estrategia_validacao,
            'knn_n_neighbors': n_neighbors,
            'knn_weights': weights,
            'knn_metric': metric,
            'knn_p': p
        }
        
        saved_model = self._save_model_to_json(model_data)
        self.model_list.insert(0, saved_model)
        return saved_model

    def cadastrar_svc(self, nome_teste, estrategia_preprocessamento, estrategia_validacao,
                      C, kernel, gamma, degree, class_weight, probability):
        model_data = {
            'nome_teste': nome_teste,
            'nome_modelo': 'SVC',
            'estrategia_preprocessamento': estrategia_preprocessamento,
            'estrategia_validacao': estrategia_validacao,
            'svm_C': C,
            'svm_kernel': kernel,
            'svm_gamma': gamma,
            'svm_degree': degree,
            'svm_class_weight': class_weight,
            'svm_probability': probability
        }
        
        saved_model = self._save_model_to_json(model_data)
        self.model_list.insert(0, saved_model)
        return saved_model

    def cadastrar_tree(self, nome_teste, estrategia_preprocessamento, estrategia_validacao,
                      criterion, max_depth, min_samples_split, min_samples_leaf, class_weight):
        model_data = {
            'nome_teste': nome_teste,
            'nome_modelo': 'DecisionTreeClassifier',
            'estrategia_preprocessamento': estrategia_preprocessamento,
            'estrategia_validacao': estrategia_validacao,
            'tree_criterion': criterion,
            'tree_max_depth': max_depth,
            'tree_min_samples_split': min_samples_split,
            'tree_min_samples_leaf': min_samples_leaf,
            'tree_class_weight': class_weight
        }
        
        saved_model = self._save_model_to_json(model_data)
        self.model_list.insert(0, saved_model)
        return saved_model

    def cadastrar_rf(self, nome_teste, estrategia_preprocessamento, estrategia_validacao,
                    n_estimators, max_depth, criterion, min_samples_split, min_samples_leaf, class_weight, oob_score):
        model_data = {
            'nome_teste': nome_teste,
            'nome_modelo': 'RandomForestClassifier',
            'estrategia_preprocessamento': estrategia_preprocessamento,
            'estrategia_validacao': estrategia_validacao,
            'rf_n_estimators': n_estimators,
            'rf_max_depth': max_depth,
            'rf_criterion': criterion,
            'rf_min_samples_split': min_samples_split,
            'rf_min_samples_leaf': min_samples_leaf,
            'rf_class_weight': class_weight,
            'rf_oob_score': oob_score
        }
        
        saved_model = self._save_model_to_json(model_data)
        self.model_list.insert(0, saved_model)
        return saved_model

    def listar_modelos(self):
        self.model_list = self._load_models_from_json()
        return self.model_list


if __name__ == "__main__":
    registry = ModelRegistry()
    registry.cadastrar_logistic("Teste 1", "StandardScaler", "StratifiedKFold",
                                  1.0, 1.0, "liblinear", "balanced", 100)
    registry.cadastrar_knn("Teste 2", "StandardScaler", "StratifiedKFold",
                            5, "uniform", "minkowski", 2)
    registry.cadastrar_svc("Teste 3", "StandardScaler", "StratifiedKFold",
                            1.0, "linear", "auto", 3, "balanced", True)
    registry.cadastrar_tree("Teste 4", "StandardScaler", "StratifiedKFold",
                            "gini", 5, 2, 1, "balanced")
    registry.cadastrar_rf("Teste 5", "StandardScaler", "StratifiedKFold",
                            100, 5, "gini", 2, 1, "balanced", True)
    print(registry.listar_modelos())

