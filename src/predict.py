import pickle
import pandas as pd

# Load model & encoders
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/encoders.pkl", "rb") as f:
    encoders = pickle.load(f)

def predict(sample):
    # Convert dict → DataFrame
    df = pd.DataFrame([sample])

    # Encode with same encoders
    for col, le in encoders.items():
        if col in df.columns:
            df[col] = le.transform(df[col])

    return model.predict(df)[0]

if __name__ == "__main__":
    sample_input = {
        "Gender": "Male",
        "Openness": "High",
        "Neuroticism": "Low",
        "Personality": "???"  # target ignored
    }
    prediction = predict(sample_input)
    print("✅ Prediction:", prediction)
