import argparse
import pandas as pd
import joblib
import json

from sklearn.metrics import classification_report

def evaluate_model(model_path: str, data_path: str, output_path: str):
    df = pd.read_csv(data_path)
    if 'target' not in df.columns:
        raise ValueError("Dataset harus punya kolom 'target'")
    X = df.drop('target', axis=1)
    y = df['target']

    model = joblib.load(model_path)
    preds = model.predict(X)
    report = classification_report(y, preds, output_dict=True)

    with open(output_path, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"Metrics saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/model.pkl")
    parser.add_argument("--data", type=str, default="data/processed.csv")
    parser.add_argument("--output", type=str, default="metrics.json")
    args = parser.parse_args()
    evaluate_model(args.model, args.data, args.output)
