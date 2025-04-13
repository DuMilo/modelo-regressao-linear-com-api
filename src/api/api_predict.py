from pydantic import BaseModel
from fastapi import FastAPI
import uvicorn
import joblib

# Criando uma instância do FastAPI
app = FastAPI(title="API de Predição de Pontuação", 
              description="Prevê pontuações baseadas em horas de estudo")

# Criando uma classe que terá os dados do request body para a API
class request_body(BaseModel):
  horas_estudo : float

# Carregando modelo para realizar a predição
modelo_pontuacao = joblib.load('models/trained_model.joblib')

@app.post('/predict')
def predict(data : request_body):
  try:
  # Preparando os dados para predição
    input_feature = [[data.horas_estudo]]

  # Realizando a predição
    y_pred = modelo_pontuacao.predict(input_feature)[0].astype(int)

  # Retornando a predição
    return {
            'horas_estudo': data.horas_estudo,
            'pontuacao_prevista': int(y_pred),
            'unidade': 'pontos'
          }
        
  except Exception as e:
    return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)