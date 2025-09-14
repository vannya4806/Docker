import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle
import os
from preprocess import preprocess_data

# Load dataset
df = pd.read_csv("data/personality_dataset.csv")

# Preprocess
df = preprocess_data(df, training=True)

# Split features & target
X = df.drop("Personality", axis=1)
y = df["Personality"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
os.makedirs("models", exist_ok=True)
with open("models/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved to models/model.pkl")
