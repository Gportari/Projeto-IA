class ModelRegistry:
    def __init__(self):
        # Lista de modelos, cada modelo é um dicionário com 'nome_teste' e 'nome_modelo'
        self.model_list = []

    def cadastrar_modelo(self, nome_teste, nome_modelo):
        self.model_list.append({
            'nome_teste': nome_teste,
            'nome_modelo': nome_modelo
        })

    def listar_modelos(self):
        return self.model_list

# Exemplo de uso:
if __name__ == "__main__":
    registry = ModelRegistry()
    registry.cadastrar_modelo("Teste 1", "ResNet50")
    registry.cadastrar_modelo("Teste 2", "BERT")
    print(registry.listar_modelos())
