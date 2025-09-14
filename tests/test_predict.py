from src.predict import predict

def test_prediction_output():
    sample = {
        "Gender": "Male",
        "Openness": "High",
        "Neuroticism": "Low",
        "Personality": "Introvert"  
    }
    result = predict(sample)
    # Output harus berupa angka (encoded label)
    assert isinstance(result, (int, float))
