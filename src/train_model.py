import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os

def train_and_save_models():
    print("Initializing Production Model Training (CBSL Local Data)...")
    
    # 1. Setup paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    processed_data_path = os.path.join(script_dir, '..', 'data', 'processed', 'LKR_Forex_Processed.csv')
    models_dir = os.path.join(script_dir, '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # 2. Load Data
    if not os.path.exists(processed_data_path):
        raise FileNotFoundError(f"Processed data not found at {processed_data_path}. Run preprocessing first!")

    df = pd.read_csv(processed_data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Define the exact feature set used in your models and XAI
    features = [
        'LKR_Lag_1', 'LKR_Lag_2', 'LKR_Lag_7', 
        'LKR_MA_7', 'LKR_MA_30', 'LKR_Volatility_7',
        'USD_Index_Change', 'Day_of_Week', 
        'Month_Sin', 'Month_Cos', 'Is_Crisis_Period'
    ]

    # Hyperparameters tuned specifically to optimize MAE and mimic Efficient Market (Random Walk)
    # THESE MUST MATCH WHAT YOU WROTE IN THE REPORT!
    lgb_params = {
        'objective': 'mae',          
        'metric': 'mae',
        'learning_rate': 0.01,       
        'max_depth': 4,              
        'num_leaves': 15,
        'feature_fraction': 0.8,
        'reg_alpha': 1.0,            # L1 Regularization (The "Zero-Guess" enforcer)
        'reg_lambda': 0.5,           # L2 Regularization (Spike prevention)
        'n_estimators': 800,         
        'random_state': 42,
        'verbose': -1
    }

    # 3. Training Loop
    for curr in df['Currency'].unique():
        print(f"Processing {curr}/LKR...")
        curr_df = df[df['Currency'] == curr].sort_values('Date').dropna().reset_index(drop=True)
        
        # Train on the FULL dataset for the production model 
        # to ensure the AI has the most recent 2024/2025 context.
        X = curr_df[features]
        y_change = curr_df['Target_Next_Day'] - curr_df['LKR_Rate']

        # Train with the newly optimized parameters
        model = lgb.LGBMRegressor(**lgb_params)
        model.fit(X, y_change)
        
        # 4. Save Payload
        # Set 'optimal_factor' to 1.0 to ensure the Recursive Loop 
        # uses the full AI predictive signal.
        model_payload = {
            'model': model,
            'optimal_factor': 1.0, 
            'features': features
        }
        
        save_path = os.path.join(models_dir, f'lgbm_{curr}.joblib')
        joblib.dump(model_payload, save_path)
        print(f"{curr} model saved (Pure AI Mode with L1 Regularization).")

if __name__ == "__main__":
    train_and_save_models()
    print("\nSUCCESS: All local models exported to the /models/ folder!")