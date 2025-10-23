import os
import pickle
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    # --- Load data langsung ---
    df = pd.read_csv("data/personality_dataset.csv")  # Pastikan file ini ada di repo
    print(f"✅ Dataset loaded, shape: {df.shape}")

    # Misal kolom target bernama 'label' — ganti sesuai dataset kamu
    target_col = "label"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # --- Split data ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Set parameter model ---
    params = {"max_iter": 1000, "random_state": 42}

    # --- Mulai experiment MLflow ---
    mlflow.set_tracking_uri("file:./mlruns")  # Simpan log ke folder lokal mlruns/
    mlflow.set_experiment("personality-ml-experiment")

    with mlflow.start_run():
        # Log parameter
        mlflow.log_params(params)

        # --- Train model ---
        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        # --- Evaluasi ---
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"✅ Akurasi model: {acc:.4f}")

        # Log metric ke MLflow
        mlflow.log_metric("accuracy", acc)

        # --- Simpan model lokal ---
        os.makedirs("models", exist_ok=True)
        with open("models/model.pkl", "wb") as f:
            pickle.dump(model, f)
        print("✅ Model disimpan ke models/model.pkl")

        # Log model ke MLflow
        mlflow.sklearn.log_model(model, "model")

    print("🏁 Training selesai & tercatat di MLflow.")
