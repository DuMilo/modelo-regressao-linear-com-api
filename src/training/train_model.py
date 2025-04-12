from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing.preprocess import preprocess_data

def train_model():
    raw_data_file = fr'data\raw\pontuacao_teste.csv'
    processed_data_path = fr'data\processed'
    X, y = preprocess_data(fr'{raw_data_file}', fr'{processed_data_path}')
    
    model = LinearRegression()
    model.fit(X, y)
    
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    print(f"MSE do modelo: {mse:.2f}")
    
    joblib.dump(model, rf"models/modelo_regressao.joblib")
    
    return model