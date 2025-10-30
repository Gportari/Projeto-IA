from flask import Flask, render_template, request, redirect
from model_registry import ModelRegistry

app = Flask(__name__)
registry = ModelRegistry()

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
    return render_template('Models.html', model_list=registry.listar_modelos())

@app.route('/cadastrar_modelo', methods=['POST'])
def cadastrar_modelo():
    nome_teste = request.form['nome_teste']
    nome_modelo = request.form['nome_modelo']
    registry.cadastrar_modelo(nome_teste, nome_modelo)
    return redirect('/Models.html')

if __name__ == "__main__":
    app.run(debug=True)
