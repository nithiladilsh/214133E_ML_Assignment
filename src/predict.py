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
    model_path = os.path.normpath(os.path.join(script_dir, '..', 'models', f'lgbm_{currency}.joblib'))
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model for {currency} not found at {model_path}")
        
    payload = joblib.load(model_path)
    model = payload['model']
    optimal_factor = payload['optimal_factor']
    features = payload['features']
    
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
    
    # Standard deviation for stochastic noise
    base_vol = float(current_data.get('LKR_Volatility_7', 0.8))
    
    for i in range(days_to_forecast):
        # 1. Advance the date
        current_date = current_date + timedelta(days=1)
        
        # 2. Update Calendar Features
        current_data['Day_of_Week'] = current_date.dayofweek
        current_data['Month_Sin'] = np.sin(2 * np.pi * current_date.month / 12)
        current_data['Month_Cos'] = np.cos(2 * np.pi * current_date.month / 12)
        current_data['Date'] = str(current_date.date())

        # 3. Apply Macro Shock directly to the feature the AI reads
        current_data['USD_Index_Change'] = usd_shock

        # 4. Generate AI Prediction
        yesterday_price = float(current_data['LKR_Rate'])
        next_price_ai = predict_next_day(currency, current_data)
        
        # 5. Apply Shock Nudge (Calibrated to 0.02 for realistic, gradual impact)
        if usd_shock != 0:
            next_price_ai += (yesterday_price * (usd_shock * 0.02))
        
        # 6. Add Stochastic Noise (Random Walk behavior)
        # SCALE ADJUSTMENT: Daily step scaled by sqrt(1/30) to prevent blowout
        price_scaled_vol = (yesterday_price / 300.0) * base_vol
        daily_step_vol = price_scaled_vol * (1 / np.sqrt(30))
        noise_level = max(daily_step_vol, 0.01)
        
        natural_noise = np.random.normal(0, noise_level)
        next_price = next_price_ai + natural_noise
        
        forecast_sequence.append(next_price)
        
        # 7. Calculate Confidence Interval
        spread = price_scaled_vol * np.sqrt((i + 1) / 30) * 1.96
        upper_bound.append(next_price + spread)
        lower_bound.append(next_price - spread)
        
        # 8. FULL RECURSIVE UPDATE: Shift all lags correctly
        current_data['LKR_Lag_7'] = current_data.get('LKR_Lag_6', current_data['LKR_Lag_1'])
        current_data['LKR_Lag_2'] = current_data['LKR_Lag_1']
        current_data['LKR_Lag_1'] = yesterday_price
        current_data['LKR_Rate'] = next_price
        
        # Update rolling averages
        current_data['LKR_MA_7'] = (current_data['LKR_MA_7'] * 6 + next_price) / 7
        current_data['LKR_MA_30'] = (current_data['LKR_MA_30'] * 29 + next_price) / 30

    return {
        "trend": forecast_sequence, 
        "upper": upper_bound, 
        "lower": lower_bound
    }