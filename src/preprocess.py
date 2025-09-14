from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

def load_data(test_size=0.2, random_state=42):
    """
    Contoh fungsi load dataset Iris dari sklearn.
    Return: X_train, X_test, y_train, y_test
    """
    data = load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test
