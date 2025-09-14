from fastapi import FastAPI
from pydantic import BaseModel
from src.predict import predict_personality

app = FastAPI()

class InputData(BaseModel):
    Time_spent_Alone: float
    Stage_fear: int
    Social_event_attendance: float
    Going_outside: float
    Drained_after_socializing: int
    Friends_circle_size: float
    Post_frequency: float

@app.post("/predict")
def predict(data: InputData):
    input_list = [
        data.Time_spent_Alone,
        data.Stage_fear,
        data.Social_event_attendance,
        data.Going_outside,
        data.Drained_after_socializing,
        data.Friends_circle_size,
        data.Post_frequency
    ]
    result = predict_personality(input_list)
    return {"prediction": result}
