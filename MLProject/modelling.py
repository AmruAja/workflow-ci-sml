import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

def main():
    # Mengambil tracking URI dari environment variabel. Jika tidak ada, gunakan localhost port 5000
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(tracking_uri)
    
    mlflow.set_experiment("Car Price Prediction")
    
    # Autolog akan otomatis mencatat parameter dan metrik ke dalam Run yang dibuat oleh 'mlflow run'
    mlflow.sklearn.autolog()
    
    possible_paths = [
        os.path.join("Membangun_model", "namadataset_preprocessing"),
        "namadataset_preprocessing",
        "../namadataset_preprocessing",
        os.path.join("preprocessing", "namadataset_preprocessing"),
        os.path.join("MLProject", "namadataset_preprocessing")
    ]
    
    data_path = None
    for path in possible_paths:
        if os.path.exists(path):
            data_path = path
            break
            
    if data_path is None:
        print("[ERROR] File 'namadataset_preprocessing' tidak ditemukan!")
        return
        
    print(f"-> Membaca dataset dari: {data_path}")
    df = pd.read_csv(data_path)
    
    df = pd.get_dummies(df, drop_first=True)
    
    X = df.drop(columns=['price']) 
    y = df['price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # === PERBAIKAN: Menghilangkan start_run karena sudah di-handle otomatis oleh MLflow CLI ===
    print("Sedang melatih model Random Forest Regressor...")
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    # Lakukan prediksi pada data test
    predictions = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mape = mean_absolute_percentage_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print("\n=== Model Sukses Dilatih! ===")
    print(f"MAE       : {mae:.2f}")
    print(f"RMSE      : {rmse:.2f}")
    print(f"MAPE      : {mape:.2f}")
    print(f"R-Squared : {r2:.2f}")
    print("=============================")
    print("Semua parameter, metrik regresi, dan artifact otomatis tercatat via AUTOLOG.")

if __name__ == "__main__":
    main()
