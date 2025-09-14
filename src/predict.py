# src/predict.py
import pickle

def load_model():
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

def predict(input_data):
    model = load_model()
    return model.predict([input_data])
