import joblib
import numpy as np

def predict_personality(input_data):
    model = joblib.load("models/model.pkl")
    prediction = model.predict([np.array(input_data)])
    return "Introvert" if prediction[0] == 1 else "Extrovert"
