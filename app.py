from fastapi import FastAPI
from pydantic import BaseModel
import joblib

app = FastAPI()

model = joblib.load("chat_model.pkl")

class MessageInput(BaseModel):
    message: str

class CategoryOutput(BaseModel):
    category: str
    confidence: float

@app.post("/categorize", response_model=CategoryOutput)
def categorize(input: MessageInput):
    prediction = model.predict([input.message])[0]
    probabilities = model.predict_proba([input.message])[0]
    confidence = max(probabilities)
    return {
        "category": prediction,
        "confidence": round(float(confidence), 2)
    }
