import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)

    # Handle missing values
    df = df.fillna(df.mode().iloc[0])

    # Encode categorical
    label_enc = LabelEncoder()
    df["Stage_fear"] = label_enc.fit_transform(df["Stage_fear"])
    df["Drained_after_socializing"] = label_enc.fit_transform(df["Drained_after_socializing"])
    df["Personality"] = label_enc.fit_transform(df["Personality"])

    X = df.drop("Personality", axis=1)
    y = df["Personality"]

    return train_test_split(X, y, test_size=0.2, random_state=42)
