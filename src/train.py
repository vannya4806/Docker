import os
import joblib
from src.preprocess import load_data
from src.model import build_model

def train_and_save():
    # 1. Ambil data
    X_train, X_test, y_train, y_test = load_data()

    # 2. Build model
    model = build_model()

    # 3. Train
    model.fit(X_train, y_train)

    # 4. Pastikan folder models ada
    os.makedirs("models", exist_ok=True)

    # 5. Simpan model
    joblib.dump(model, "src/model.py")

    print("✅ Model berhasil dilatih dan disimpan ke models/model.pkl")

if __name__ == "__main__":
    train_and_save()
