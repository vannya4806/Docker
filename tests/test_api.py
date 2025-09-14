from fastapi.testclient import TestClient
from app.app import app

client = TestClient(app)

def test_api_predict():
    response = client.post("/predict", json={
        "Time_spent_Alone": 5,
        "Stage_fear": 0,
        "Social_event_attendance": 6,
        "Going_outside": 4,
        "Drained_after_socializing": 1,
        "Friends_circle_size": 10,
        "Post_frequency": 5
    })
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in ["Introvert", "Extrovert"]
