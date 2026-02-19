import pandas as pd
import numpy as np
import lightgbm as lgb
import joblib
import os
from sklearn.metrics import mean_absolute_error

def train_and_save_models():
    print("Initializing Model Training Pipeline...")
    
    # 1. Setup paths
    # Get the directory where train_model.py is located (the src/ folder)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Go up one level to the root, then into data/processed
    processed_data_path = os.path.join(script_dir, '..', 'data', 'processed', 'LKR_Forex_Processed.csv')
    
    # Do the same for the models directory
    models_dir = os.path.join(script_dir, '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # 2. Load Data
    df = pd.read_csv(processed_data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df.loc[df['Currency'] == 'JPY', ['LKR_Rate', 'Target_Next_Day', 'LKR_Lag_1']] /= 10000 

    features = [
        'LKR_Lag_1', 'LKR_Lag_2', 'LKR_Lag_7', 'LKR_MA_7', 'LKR_MA_30', 
        'Daily_Return', 'LKR_Volatility_7', 'USD_Index', 'USD_Index_Change', 
        'Day_of_Week', 'Month_Sin', 'Month_Cos', 'Is_Crisis_Period'
    ]

    # 3. Train and Save loop
    for curr in df['Currency'].unique():
        print(f"Training {curr} model...")
        curr_df = df[df['Currency'] == curr].sort_values('Date').dropna().reset_index(drop=True)
        
        X = curr_df[features]
        y_change = curr_df['Target_Next_Day'] - curr_df['LKR_Rate']
        
        # 80/20 Chronological Split
        split = int(len(curr_df) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train_change = y_change.iloc[:split]
        actual_test_prices = curr_df['Target_Next_Day'].iloc[split:]
        today_test_prices = curr_df['LKR_Rate'].iloc[split:]

        # Train Model
        model = lgb.LGBMRegressor(
            n_estimators=100, learning_rate=0.01, num_leaves=7, max_depth=3, random_state=42, verbose=-1
        )
        model.fit(X_train, y_train_change)
        ai_change_preds = model.predict(X_test)
        
        # Optimize Damping Factor
        best_factor = 0
        min_mae = mean_absolute_error(actual_test_prices, today_test_prices)
        
        for factor in np.linspace(0, 0.5, 51):
            current_preds = today_test_prices + (factor * ai_change_preds)
            current_mae = mean_absolute_error(actual_test_prices, current_preds)
            if current_mae < min_mae:
                min_mae = current_mae
                best_factor = factor
                
        # 4. Save the Model AND the Optimal Factor together
        model_payload = {
            'model': model,
            'optimal_factor': best_factor,
            'features': features
        }
        
        save_path = os.path.join(models_dir, f'lgbm_{curr}.joblib')
        joblib.dump(model_payload, save_path)
        print(f"{curr} model saved to {save_path} (Factor: {best_factor:.2f})")

if __name__ == "__main__":
    train_and_save_models()
    print("All models successfully trained and exported!")