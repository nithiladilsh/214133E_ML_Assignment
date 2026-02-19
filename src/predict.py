import joblib
import os
import pandas as pd

def predict_next_day(currency, latest_data_series):
    """
    Loads the saved model and predicts the T+1 exchange rate.
    """
    # 1. Bulletproof Pathing (Fixes the CWD issue)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, '..', 'models', f'lgbm_{currency}.joblib')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model for {currency} not found at {model_path}")
        
    # 2. Load the payload
    payload = joblib.load(model_path)
    model = payload['model']
    optimal_factor = payload['optimal_factor']
    features = payload['features']
    
    # 3. Convert input to a single-row DataFrame
    input_df = pd.DataFrame([latest_data_series])
    
    # 4. Apply JPY scaling to the inputs BEFORE predicting
    if currency == 'JPY':
        for col in ['LKR_Rate', 'LKR_Lag_1', 'LKR_Lag_2', 'LKR_Lag_7', 'LKR_MA_7', 'LKR_MA_30']:
            if col in input_df.columns:
                input_df[col] = input_df[col] / 10000
                
    X_input = input_df[features]
    current_rate = input_df['LKR_Rate'].iloc[0]
    
    # 5. Predict the Delta (Change)
    predicted_change = model.predict(X_input)[0]
    
    # 6. Reconstruct the final forecast
    final_prediction = current_rate + (optimal_factor * predicted_change)
    
    # 7. Reverse JPY scaling for the final output
    if currency == 'JPY':
        final_prediction *= 10000
        
    # Return BOTH values so Streamlit can display them
    return final_prediction, optimal_factor

# Simple test block (won't run when imported into Streamlit)
if __name__ == "__main__":
    print("Predict module ready. Import predict_next_day() to use.")