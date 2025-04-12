from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
import joblib

# Criando uma instância do FastAPI
app = FastAPI()

# Criando uma classe que terá os dados do request body para a API
class request_body(BaseModel):
  horas_estudo : float

# Carregando modelo para realizar a predição

modelo_pontuacao = joblib.load(r'.\models\trained_model.pkl')

@app.post('/predict')
def predict(data : request_body):
  # Preparando os dados para predição
  input_feature = [[data.horas_estudo]]

  # Realizando a predição
  y_pred = modelo_pontuacao.predict(input_feature)[0].astype(int)

  # Retornando a predição
  return {'pontuacao_teste': y_pred.tolist()}
