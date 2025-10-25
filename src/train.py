import os
import pickle
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

if __name__ == "__main__":
    # --- Load data langsung ---
    df = pd.read_csv("data/personality_dataset.csv")
    print(f"✅ Dataset loaded, shape: {df.shape}")

    target_col = "Personality"
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # --- Tangani NaN ---
    for col in X.columns:
        if X[col].isnull().sum() > 0:
            if X[col].dtype == 'object':
                X[col].fillna(X[col].mode()[0], inplace=True)
            else:
                X[col].fillna(X[col].mean(), inplace=True)

    # --- Encode fitur kategorikal ---
    cat_cols = X.select_dtypes(include=["object"]).columns
    if len(cat_cols) > 0:
        print("🔄 Encoding kolom kategorikal:", cat_cols.tolist())
        le = LabelEncoder()
        for col in cat_cols:
            X[col] = le.fit_transform(X[col].astype(str))

    # Encode target juga kalau berupa string
    if y.dtype == "object":
        y = LabelEncoder().fit_transform(y)

    # --- Split data ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Set parameter model ---
    params = {"max_iter": 1000, "random_state": 42}

    # --- MLflow setup ---
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("personality-ml-experiment")

    with mlflow.start_run():
        mlflow.log_params(params)

        model = LogisticRegression(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"✅ Akurasi model: {acc:.4f}")

        mlflow.log_metric("accuracy", acc)

        os.makedirs("models", exist_ok=True)
        with open("models/model.pkl", "wb") as f:
            pickle.dump(model, f)
        print("✅ Model disimpan ke models/model.pkl")

        mlflow.sklearn.log_model(model, "model")

    print("🏁 Training selesai & tercatat di MLflow.")
