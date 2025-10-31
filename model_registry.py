class ModelRegistry:
    def __init__(self):
        # Lista de modelos, cada modelo é um dicionário com campos específicos por tipo
        self.model_list = []

    def cadastrar_logistic(self, nome_teste, estrategia_preprocessamento, estrategia_validacao,
                           penalty, C, solver, class_weight, max_iter):
        self.model_list.append({
            'nome_teste': nome_teste,
            'nome_modelo': 'Regressão Logística',
            'estrategia_preprocessamento': estrategia_preprocessamento,
            'estrategia_validacao': estrategia_validacao,
            'logreg_penalty': penalty,
            'logreg_C': C,
            'logreg_solver': solver,
            'logreg_class_weight': class_weight,
            'logreg_max_iter': max_iter
        })

    def cadastrar_knn(self, nome_teste, estrategia_preprocessamento, estrategia_validacao,
                      n_neighbors, weights, metric, p):
        self.model_list.append({
            'nome_teste': nome_teste,
            'nome_modelo': 'KNN',
            'estrategia_preprocessamento': estrategia_preprocessamento,
            'estrategia_validacao': estrategia_validacao,
            'knn_n_neighbors': n_neighbors,
            'knn_weights': weights,
            'knn_metric': metric,
            'knn_p': p
        })

    def cadastrar_svc(self, nome_teste, estrategia_preprocessamento, estrategia_validacao,
                      C, kernel, gamma, degree, class_weight, probability):
        self.model_list.append({
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
        })

    def cadastrar_tree(self, nome_teste, estrategia_preprocessamento, estrategia_validacao,
                      criterion, max_depth, min_samples_split, min_samples_leaf, class_weight):
        self.model_list.append({
            'nome_teste': nome_teste,
            'nome_modelo': 'DecisionTreeClassifier',
            'estrategia_preprocessamento': estrategia_preprocessamento,
            'estrategia_validacao': estrategia_validacao,
            'tree_criterion': criterion,
            'tree_max_depth': max_depth,
            'tree_min_samples_split': min_samples_split,
            'tree_min_samples_leaf': min_samples_leaf,
            'tree_class_weight': class_weight
        })

    def cadastrar_rf(self, nome_teste, estrategia_preprocessamento, estrategia_validacao,
                    n_estimators, max_depth, criterion, min_samples_split, min_samples_leaf, class_weight, oob_score):
        self.model_list.append({
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
        })

    def listar_modelos(self):
        return self.model_list

# Exemplo de uso:
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