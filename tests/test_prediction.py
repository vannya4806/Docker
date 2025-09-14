import pickle
import numpy as np

def test_single_prediction():
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)

    sample = np.array([23, 1, 6.5, 5.1, 7.0, 6.2, 5.8]).reshape(1, -1)
    pred = model.predict(sample)

    assert pred is not None
