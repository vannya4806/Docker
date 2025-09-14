import joblib
from sklearn.ensemble import RandomForestClassifier
from preprocessing import load_and_preprocess

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_preprocess("data/personality_dataset.csv")

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    print(f"✅ Model trained with accuracy: {acc:.2f}")

    joblib.dump(model, "models/model.pkl")
