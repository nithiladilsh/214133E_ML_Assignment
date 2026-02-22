from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
import os

# 1. Path Setup: Ensure the backend can find the 'src' folder
script_dir = os.path.dirname(os.path.abspath(__file__))

root_dir = os.path.normpath(os.path.join(script_dir, '..')) 
if root_dir not in sys.path:
    sys.path.append(root_dir)

# 2. Import the prediction engine
try:
    from src.predict import predict_future_sequence
except ImportError:
    raise ImportError(f"Could not find 'src.predict'. Ensure you are running from the project root. Path searched: {root_dir}")

app = FastAPI(
    title="LKR Forex AI API",
    description="Recursive Stochastic Forecasting for the Sri Lankan Rupee",
    version="2.0.0"
)

# 3. Request Schema
class PredictionRequest(BaseModel):
    currency: str
    latest_data: dict
    days: int = 7  # Default to 7-day forecast if not specified
    usd_shock: float = 0.0  # New field (defaults to 0%)

@app.get("/")
def read_root():
    return {"status": "online", "message": "LKR AI Forecasting API is active."}

@app.post("/predict")
async def get_prediction(request: PredictionRequest):
    """
    Accepts the latest market row and returns a multi-day stochastic forecast.
    """
    try:
        # Call the recursive stochastic engine
        forecast_results = predict_future_sequence(
            request.currency, 
            request.latest_data, 
            request.days,
            usd_shock=request.usd_shock
        )
        
        # Explicitly cast to float() to ensure JSON compatibility 
        # (LightGBM/NumPy types sometimes confuse standard JSON encoders)
        return {
            "status": "success",
            "currency": request.currency,
            "predicted_price_t1": float(forecast_results['trend'][0]),
            "forecast_trend": [float(p) for p in forecast_results['trend']],
            "upper_bound": [float(p) for p in forecast_results['upper']],
            "lower_bound": [float(p) for p in forecast_results['lower']],
            "days_forecasted": int(request.days)
        }
        
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=404, detail=str(fnf))
    except Exception as e:
        # This catches any unexpected errors and reports them to the frontend
        raise HTTPException(status_code=500, detail=f"AI Engine Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)