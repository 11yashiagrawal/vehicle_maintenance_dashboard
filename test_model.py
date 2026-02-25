import joblib
import pandas as pd

# Test loading
model = joblib.load('vehicle_model.joblib')
features = pd.read_csv('features.csv', header=None)[0].tolist()

print("✅ Model loaded!")
print("✅ Features:", features[:19], "...")
print(f"✅ Total features: {len(features)}")