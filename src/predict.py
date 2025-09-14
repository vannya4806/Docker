# src/predict.py
import pickle

def load_model():
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

def predict(input_data):
    model = load_model()
    return model.predict([input_data])

if __name__ == "__main__":
    # Contoh input (ubah sesuai datasetmu)
    sample_input = [0, 1, 2, 3, 4]
    prediction = predict(sample_input)
    print("Prediksi:", prediction)
