# src/train.py
from src.preprocess import load_and_preprocess
from sklearn.linear_model import LogisticRegression
import pickle
import os
if __name__ == "__main__":
  # Load & preprocess data
  X_train, X_test, y_train, y_test = load_and_preprocess("personality_dataset.csv")
  # Buat model langsung
  model = LogisticRegression(max_iter=1000, random_state=42)
  model.fit(X_train, y_train)
  # Simpan ke models/model.pkl
  os.makedirs("models", exist_ok=True)
  with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)
  print("✅ Model dilatih dan disimpan ke models/model.pkl")
