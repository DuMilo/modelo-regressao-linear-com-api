from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
import joblib

# Criar uma instância do FastAPI
app = FastAPI()

# Criar uma classe que terá os dados do request body para a API
class request_body(BaseModel):
  horas_estudo : float

# Carregar modelo para realizar a predição

modelo_pontuacao = joblib.load('./modelo_regressao.pkl')

@app.post('/predict')
def predict(data : request_body):
  # Preparar os dados para predição
  input_feature = [[data.horas_estudo]]

  # Realizar a predição
  y_pred = modelo_pontuacao.predict(input_feature)[0].astype(int)

  # Retornar a predição
  return {'pontuacao_teste': y_pred.tolist()}
