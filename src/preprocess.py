import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
import os

def preprocess_data(df, training=True):
    """Preprocess dataset: encode categorical features & target"""

    encoders = {}
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    if training:
        os.makedirs("models", exist_ok=True)
        with open("models/encoders.pkl", "wb") as f:
            pickle.dump(encoders, f)

    return df

if __name__ == "__main__":
    df = pd.read_csv("data/personality_dataset.csv")
    df = preprocess_data(df, training=True)
    df.to_csv("data/personality_preprocessed.csv", index=False)
    print("✅ Preprocessing done, saved to data/personality_preprocessed.csv")
