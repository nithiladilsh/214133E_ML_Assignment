from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os

# Add project root to path so we can import 'src'
# Assuming structure: Project_Root/backend/app/api.py
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(script_dir, '..', '..') 
sys.path.append(root_dir)

from src.predict import predict_next_day

app = FastAPI(title="LKR Forex AI API")

class PredictionRequest(BaseModel):
    currency: str
    latest_data: dict

@app.post("/predict")
async def get_prediction(request: PredictionRequest):
    try:
        # Call the predict function from your src folder
        price, factor = predict_next_day(request.currency, request.latest_data)
        return {
            "currency": request.currency,
            "predicted_price": float(price),
            "damping_factor": float(factor)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))