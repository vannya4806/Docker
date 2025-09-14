import pandas as pd
import pickle

# Load model
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

# Input CSV
df = pd.read_csv("data/input.csv")

# Encode gender
df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

# Predict
predictions = model.predict(df)
df["Prediction"] = predictions

# Save output
df.to_csv("data/output_predictions.csv", index=False)
print("✅ Predictions saved to data/output_predictions.csv")
