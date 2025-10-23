import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import os

def load_and_preprocess(test_size=0.2, random_state=42):
    """Load dataset Iris dan bagi menjadi train/test"""
    data = load_iris()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Gabungkan jadi satu DataFrame untuk contoh sederhana
    df_train = pd.DataFrame(X_train, columns=data.feature_names)
    df_train["target"] = y_train

    df_test = pd.DataFrame(X_test, columns=data.feature_names)
    df_test["target"] = y_test

    return df_train, df_test


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df_train, df_test = load_and_preprocess()

    # Pastikan direktori tujuan ada
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    # Simpan hasil preprocess ke file CSV (contoh: hanya train)
    df_train.to_csv(args.output, index=False)

    print(f"✅ Data preprocessed disimpan ke: {args.output}")
