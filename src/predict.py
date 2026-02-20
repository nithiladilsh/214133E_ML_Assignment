import joblib
import os
import pandas as pd
import numpy as np
from datetime import timedelta

def predict_next_day(currency, latest_data_series):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, '..', 'models', f'lgbm_{currency}.joblib')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model for {currency} not found at {model_path}")
        
    payload = joblib.load(model_path)
    model = payload['model']
    optimal_factor = payload['optimal_factor']
    features = payload['features']
    
    input_df = pd.DataFrame([latest_data_series])
    X_input = input_df[features]
    current_rate = input_df['LKR_Rate'].iloc[0]
    
    predicted_delta = model.predict(X_input)[0]
    return current_rate + (optimal_factor * predicted_delta)

def predict_future_sequence(currency, latest_row, days_to_forecast, usd_shock=0.0):
    forecast_sequence = []
    upper_bound = []
    lower_bound = []
    
    current_data = latest_row.copy()
    current_date = pd.to_datetime(current_data['Date'])
    
    # Keep noise tight (0.5 multiplier for stability)
    base_vol = float(current_data.get('LKR_Volatility_7', 0.8)) * 0.5 

    # FIX: Capture the base USD Index so it doesn't compound exponentially
    base_usd_index = float(current_data.get('USD_Index', 100.0))

    for i in range(days_to_forecast):
        current_date = current_date + timedelta(days=1)
        
        # Apply shock to the base value consistently
        if 'USD_Index' in current_data:
            current_data['USD_Index'] = base_usd_index * (1 + usd_shock)

        yesterday_price = float(current_data['LKR_Rate'])
        next_price_ai = predict_next_day(currency, current_data)
        
        # --- SHOCK NUDGE ---
        # This ensures the 'What-If' scenario is actually visible. 
        # A 1% shock causes a 0.1% nudge in the predicted price.
        if usd_shock != 0:
            next_price_ai += (yesterday_price * (usd_shock * 0.1))
        
        # Add Stable Noise
        natural_noise = np.random.normal(0, base_vol)
        next_price = next_price_ai + natural_noise
        forecast_sequence.append(next_price)
        
        # Uncertainty bounds (tighter spread)
        spread = base_vol * np.sqrt(i + 1) * 1.96
        upper_bound.append(next_price + spread)
        lower_bound.append(next_price - spread)
        
        # Recursive Update
        current_data['LKR_Lag_1'] = yesterday_price
        current_data['LKR_Rate'] = next_price
        current_data['LKR_MA_7'] = (current_data['LKR_MA_7'] * 6 + next_price) / 7
        current_data['LKR_MA_30'] = (current_data['LKR_MA_30'] * 29 + next_price) / 30

    return {"trend": forecast_sequence, "upper": upper_bound, "lower": lower_bound}