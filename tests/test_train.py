import os
import pickle

def test_model_exists_after_training():
    assert os.path.exists("src/model.py")

def test_model_can_be_loaded():
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)
    # Model harus punya method predict
    assert hasattr(model, "predict")
