from src.preprocessing import load_and_preprocess

def test_preprocessing():
    X_train, X_test, y_train, y_test = load_and_preprocess("data/personality_dataset.csv")
    assert X_train.isnull().sum().sum() == 0
    assert X_test.isnull().sum().sum() == 0
    assert all([col.dtype != "object" for col in X_train.dtypes])
    assert X_train.shape[0] > 0 and X_test.shape[0] > 0
