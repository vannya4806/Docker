import argparse
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import yaml

def load_params(path):
    with open(path) as f:
        return yaml.safe_load(f)

def train_model(input_path: str, model_out: str, params_path: str):
    params = load_params(params_path)
    df = pd.read_csv(input_path)
    if 'target' not in df.columns:
        raise ValueError("Dataset harus punya kolom 'target'")
    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=params['train']['test_size'], 
        random_state=params['train']['random_state']
    )

    mlflow.set_experiment("personality-classification")
    with mlflow.start_run():
        if params['train']['model'] == "LogisticRegression":
            model = LogisticRegression(max_iter=params['train']['max_iter'])
        else:
            raise NotImplementedError(f"Model {params['train']['model']} belum diimplementasi")

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # log params & metrics
        mlflow.log_params({
            "model": params['train']['model'],
            "max_iter": params['train']['max_iter']
        })
        mlflow.log_metric("accuracy", float(acc))

        # log model artifact
        mlflow.sklearn.log_model(model, "model")

        # save model lokal
        joblib.dump(model, model_out)
        print(f"Model saved to {model_out} | Accuracy: {acc}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/personality_dataset.csv")
    parser.add_argument("--model", type=str, default="models/model.pkl")
    parser.add_argument("--params", type=str, default="params.yaml")
    args = parser.parse_args()
    train_model(args.input, args.model, args.params)
