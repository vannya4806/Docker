import os
import pickle
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from src.preprocess import load_and_preprocess

if __name__ == "__main__":
    # --- Load data ---
    X_train, X_test, y_train, y_test = load_and_preprocess("personality_dataset.csv")

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

        # --- Evaluasi sederhana ---
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
