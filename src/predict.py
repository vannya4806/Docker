import joblib

def predict(input_data):
    """
    Prediksi menggunakan model yang sudah disimpan.
    input_data: numpy array atau list (1 sample atau banyak sample)
    """
    model = joblib.load("models/model.pkl")
    return model.predict(input_data)
