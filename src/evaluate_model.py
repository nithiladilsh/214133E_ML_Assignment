import os
import joblib

def check_models():
    # 1. Use the 'GPS' pathing to find the models folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(script_dir, '..', 'models')
    
    if not os.path.exists(models_dir):
        print(f"❌ No models directory found at: {models_dir}")
        return
        
    files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
    
    if not files:
        print("⚠️ No .joblib model files found. Did you run train_model.py?")
        return

    print(f"✅ Found {len(files)} trained models in {models_dir}:")
    print("-" * 50)
    
    for f in files:
        # Load the payload (which contains our model and the best damping factor)
        payload = joblib.load(os.path.join(models_dir, f))
        
        # Display the results
        currency = f.replace('lgbm_', '').replace('.joblib', '')
        factor = payload['optimal_factor']
        print(f"Currency: {currency:4} | Damping Factor: {factor:.2f}")

if __name__ == "__main__":
    check_models()