import argparse
import pickle
import numpy as np

# Load model
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

# Argument parser for CLI
parser = argparse.ArgumentParser()
parser.add_argument("--age", type=int, required=True)
parser.add_argument("--gender", type=str, required=True)
parser.add_argument("--openness", type=float, required=True)
parser.add_argument("--neuroticism", type=float, required=True)
parser.add_argument("--conscientiousness", type=float, required=True)
parser.add_argument("--agreeableness", type=float, required=True)
parser.add_argument("--extraversion", type=float, required=True)

args = parser.parse_args()

# Encode gender (contoh sederhana)
gender = 1 if args.gender.lower() == "male" else 0

features = np.array([
    args.age,
    gender,
    args.openness,
    args.neuroticism,
    args.conscientiousness,
    args.agreeableness,
    args.extraversion
]).reshape(1, -1)

prediction = model.predict(features)[0]
print(f"🎯 Predicted Personality: {prediction}")
