import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

def preprocess_data(input_path, output_path):
    input_path = rf'data\raw\pontuacao_teste.csv'
    output_path = rf'data\processed'
    
    try:
        print(f"\n🔍 Input path: {input_path}")
        print(f"📂 Output path: {output_path}")
        
        df = pd.read_csv(input_path)
        print(f"📊 Dados carregados. Shape: {df.shape}\nColunas: {df.columns.tolist()}")
        
        required_columns = ['pontuacao_teste']  
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"❌ Coluna faltante: '{col}'")
        
        scaler = StandardScaler()
        pontuacao_scaled = scaler.fit_transform(df[['pontuacao_teste']]) 
        print("✅ Dados normalizados com sucesso")
        
        os.makedirs(output_path, exist_ok=True)
        
        output_data = rf"{output_path}/processed_data.joblib"
        output_scaler = rf"{output_path}/scaler.joblib"
        
        joblib.dump(pontuacao_scaled, output_data)  
        joblib.dump(scaler, output_scaler)
        
        print(f"💾 Arquivos salvos em:\n- {output_data}\n- {output_scaler}")
        return pontuacao_scaled
        
    except Exception as e:
        print(f"🚨 ERRO: {str(e)}")
        raise

if __name__ == "__main__":
    input_csv = r'data\raw\pontuacao_teste.csv'
    output_dir = r'data\processed'
    preprocess_data(input_csv, output_dir)