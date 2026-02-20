import os
import joblib

def check_models():
    # 1. Pathing logic to locate the models folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.normpath(os.path.join(script_dir, '..', 'models'))
    
    if not os.path.exists(models_dir):
        print(f"❌ No models directory found at: {models_dir}")
        return
        
    files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
    
    if not files:
        print("⚠️ No .joblib model files found. Did you run train_model.py?")
        return

    print(f"✅ Found {len(files)} trained models in {models_dir}:")
    print("-" * 60)
    print(f"{'CURRENCY':<10} | {'STATUS':<15} | {'SIGNAL FACTOR':<15} | {'FEATURES'}")
    print("-" * 60)
    
    for f in files:
        try:
            # Load the payload
            payload = joblib.load(os.path.join(models_dir, f))
            
            # Extract metadata
            currency = f.replace('lgbm_', '').replace('.joblib', '')
            factor = payload.get('optimal_factor', 'N/A')
            feature_count = len(payload.get('features', []))
            
            # Validation Check
            status = "READY ✅" if feature_count > 0 else "ERROR ❌"
            
            print(f"{currency:<10} | {status:<15} | {factor:<15} | {feature_count} features")
        except Exception as e:
            print(f"❌ Error loading {f}: {e}")

if __name__ == "__main__":
    check_models()