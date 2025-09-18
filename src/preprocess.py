import argparse
import pandas as pd
from pathlib import Path

def preprocess(input_path: str, output_path: str):
    df = pd.read_csv(input_path)
    # Drop missing
    df = df.dropna()

    # Example encoding: gender -> numeric
    if 'gender' in df.columns:
        # map tunggal atau sesuaikan dataset kamu
        df['gender'] = df['gender'].map({'male': 0, 'female': 1})
    
    # Jika ada kolom kategori lain, lakukan encoding atau ordinal
    # Misalnya personality trait features sudah numeric: openess, etc.

    # Save processed
    outp = Path(output_path)
    outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="data/personality.csv")
    parser.add_argument("--output", type=str, default="data/processed.csv")
    args = parser.parse_args()
    preprocess(args.input, args.output)
