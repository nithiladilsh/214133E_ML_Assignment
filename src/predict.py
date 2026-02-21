import joblib
import os
import pandas as pd
import numpy as np
from datetime import timedelta

def predict_next_day(currency, latest_data_series):
    """
    Loads the specific local model and predicts the next day's price delta.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Assumes models folder is at the Project Root
    model_path = os.path.normpath(os.path.join(script_dir, '..', 'models', f'lgbm_{currency}.joblib'))
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model for {currency} not found at {model_path}")
        
    payload = joblib.load(model_path)
    model = payload['model']
    optimal_factor = payload['optimal_factor']
    features = payload['features']
    
    # Ensure input is a DataFrame with correct feature names
    input_df = pd.DataFrame([latest_data_series])
    X_input = input_df[features]
    
    current_rate = float(input_df['LKR_Rate'].iloc[0])
    
    # Predict the CHANGE (Delta), then add to current rate
    predicted_delta = model.predict(X_input)[0]
    return current_rate + (optimal_factor * predicted_delta)

def predict_future_sequence(currency, latest_row, days_to_forecast, usd_shock=0.0):
    """
    Generates a multi-step recursive forecast with stochastic noise and macro shocks.
    """
    forecast_sequence = []
    upper_bound = []
    lower_bound = []
    
    current_data = latest_row.copy()
    current_date = pd.to_datetime(current_data['Date'])
    
    # Standard deviation for stochastic noise (reduced for stability)
    base_vol = float(current_data.get('LKR_Volatility_7', 0.8)) * 0.5 
    base_usd_index = float(current_data.get('USD_Index', 100.0))

    for i in range(days_to_forecast):
        # 1. Advance the date
        current_date = current_date + timedelta(days=1)
        
        # 2. Update Calendar Features (Crucial for AI accuracy over time)
        current_data['Day_of_Week'] = current_date.dayofweek
        current_data['Month_Sin'] = np.sin(2 * np.pi * current_date.month / 12)
        current_data['Month_Cos'] = np.cos(2 * np.pi * current_date.month / 12)
        current_data['Date'] = str(current_date.date())

        # 3. Apply Macro Shock to USD Index
        if 'USD_Index' in current_data:
            current_data['USD_Index'] = base_usd_index * (1 + usd_shock)

        # 4. Generate AI Prediction
        yesterday_price = float(current_data['LKR_Rate'])
        next_price_ai = predict_next_day(currency, current_data)
        
        # 5. Apply Shock Nudge (Visible impact for stress testing)
        if usd_shock != 0:
            next_price_ai += (yesterday_price * (usd_shock * 0.1))
        
        # 6. Add Stochastic Noise (Random Walk behavior)
        natural_noise = np.random.normal(0, base_vol)
        next_price = next_price_ai + natural_noise
        
        forecast_sequence.append(next_price)
        
        # 7. Calculate Confidence Interval (Uncertainty grows over time)
        spread = base_vol * np.sqrt(i + 1) * 1.96
        upper_bound.append(next_price + spread)
        lower_bound.append(next_price - spread)
        
        # 8. RECURSIVE UPDATE: Feed today's prediction back as tomorrow's input
        current_data['LKR_Lag_1'] = yesterday_price
        current_data['LKR_Rate'] = next_price
        
        # Update rolling averages approximately
        current_data['LKR_MA_7'] = (current_data['LKR_MA_7'] * 6 + next_price) / 7
        current_data['LKR_MA_30'] = (current_data['LKR_MA_30'] * 29 + next_price) / 30

    return {
        "trend": forecast_sequence, 
        "upper": upper_bound, 
        "lower": lower_bound
    }