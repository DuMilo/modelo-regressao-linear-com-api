import joblib
import os

def train_model():
    model_file = "models/trained_model.pkl" 
    model_path = "models"
    model_joblib = rf"{model_path}/trained_model.joblib"
    
    try:
        model = joblib.load(model_file)
        print(f"✅ Modelo carregado de: {os.path.abspath(model_file)}")
        joblib.dump(model, model_joblib)
        print(f"✅ Modelo .joblib mandado para: {os.path.abspath(model_file)}")
        return model
    except Exception as e:
        print(f"🚨 Erro ao carregar o modelo: {str(e)}")
        return None

if __name__ == "__main__":
    train_model() 