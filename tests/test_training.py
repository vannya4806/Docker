from sklearn.ensemble import RandomForestClassifier
from src.preprocessing import load_and_preprocess

def test_training():
    X_train, X_test, y_train, y_test = load_and_preprocess("data/personality_dataset.csv")
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    assert acc > 0.5
