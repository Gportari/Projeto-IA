class ModelRegistry:
    def __init__(self):
        # Lista de modelos, cada modelo é um dicionário com 'nome_teste', 'nome_modelo', 'epocas', 'batch_size', 'hiperparametros_utilizados'
        self.model_list = []

    def cadastrar_modelo(self, nome_teste, nome_modelo, epocas, batch_size, hiperparametros_utilizados):
        self.model_list.append({
            'nome_teste': nome_teste,
            'nome_modelo': nome_modelo,
            'epocas': epocas,
            'batch_size': batch_size,
            'hiperparametros_utilizados': hiperparametros_utilizados
        })

    def listar_modelos(self):
        return self.model_list

# Exemplo de uso:
if __name__ == "__main__":
    registry = ModelRegistry()
    registry.cadastrar_modelo("Teste 1", "KNeighborsClassifier", None, None, {'n_neighbors': 5, 'metric': 'minkowski'})
    registry.cadastrar_modelo("Teste 2", "SVC", None, None, {'C': 1.0, 'kernel': 'linear', 'class_weight': 'balanced'})
    registry.cadastrar_modelo("Teste 3", "RandomForestClassifier", None, None, {'n_estimators': 100, 'max_depth': 5})
    print(registry.listar_modelos())