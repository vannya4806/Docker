import pandas as pd
from src.preprocess import preprocess_data

def test_preprocess_encodes_strings():
    df = pd.DataFrame({
        "Gender": ["Male", "Female"],
        "Openness": ["High", "Low"],
        "Personality": ["Introvert", "Extrovert"]
    })
    df_encoded = preprocess_data(df.copy(), training=False)

    # Semua kolom harus numerik
    assert all(dtype != "object" for dtype in df_encoded.dtypes)
    assert "Personality" in df_encoded.columns
