import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_model():
    # 1. Konfigurasi MLflow Tracking Lokal
    mlflow.set_tracking_uri("http://127.0.0.1:5000/")
    mlflow.set_experiment("Latihan_FIFA_Player_Rating")
    
    # Aktifkan Autolog agar parameter & metrik Scikit-Learn otomatis tercatat
    mlflow.autolog()
    
    print("=== Membaca Data Bersih ===")
    train_path = os.path.join('namadataset_preprocessing', 'train_clean.csv')
    test_path = os.path.join('namadataset_preprocessing', 'test_clean.csv')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Data bersih tidak ditemukan di folder 'namadataset_preprocessing'!")
        
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)
    
    # Pisahkan fitur dan target (player_rating)
    X_train = train_data.drop(columns=['player_rating'])
    y_train = train_data['player_rating']
    X_test = test_data.drop(columns=['player_rating'])
    y_test = test_data['player_rating']
    
    print("=== Memulai Training Model dengan MLflow ===")
    with mlflow.start_run() as run:
        # Menggunakan RandomForestRegressor (bisa disesuaikan)
        # Batasi n_estimators agar training berjalan cepat saat lokal
        model = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=10)
        model.fit(X_train, y_train)
        
        # Prediksi dan Evaluasi tambahan
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print(f"Training Selesai. Metrics -> MSE: {mse:.4f} | R2 Score: {r2:.4f}")
        print(f"Run ID Berhasil Dicatat: {run.info.run_id}")

if __name__ == '__main__':
    train_model()