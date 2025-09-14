from src.predict import predict_personality

def test_prediction():
    sample_input = [5, 0, 6, 4, 1, 10, 5]
    result = predict_personality(sample_input)
    assert result in ["Introvert", "Extrovert"]
