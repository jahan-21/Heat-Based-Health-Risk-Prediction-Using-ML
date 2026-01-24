import pandas as pd
import joblib
import os
import glob
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# 1. Setup Folders
if not os.path.exists('models_xgb'):
    os.makedirs('models_xgb')

# 2. Get all City Files
files = glob.glob('data/*_weather.csv')
print(f"🚀 Found {len(files)} city datasets. Starting XGBoost Training...")

# 3. Training Loop
for file in files:
    # Extract city name (e.g., 'data/Madurai_weather.csv' -> 'Madurai')
    city = os.path.basename(file).split('_')[0]
    print(f"\n📍 Training XGBoost for {city}...")
    
    try:
        df = pd.read_csv(file)
        
        # --- PREPROCESSING (Same Logic as Random Forest) ---
        # Create Targets (Shift -1 to predict tomorrow)
        df['Target_Temp'] = df['Max_Temp'].shift(-1)
        df['Target_Hum'] = df['Humidity'].shift(-1)
        df = df.dropna()

        # Features (X) and Targets (y)
        features = ['Max_Temp', 'Humidity', 'Wind_Speed', 'Rainfall']
        X = df[features]
        y_temp = df['Target_Temp']
        y_hum = df['Target_Hum']
        
        # --- MODEL 1: TEMPERATURE (XGBoost) ---
        # n_estimators=100: Number of boosting rounds
        # learning_rate=0.1: How much each tree contributes
        model_temp = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        model_temp.fit(X, y_temp)
        
        # Evaluate Accuracy
        acc_temp = r2_score(y_temp, model_temp.predict(X)) * 100
        print(f"   🌡️  Temp Model Accuracy: {acc_temp:.2f}%")

        # --- MODEL 2: HUMIDITY (XGBoost) ---
        model_hum = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
        model_hum.fit(X, y_hum)
        
        # Save Models into a new folder
        joblib.dump(model_temp, f'models_xgb/{city}_temp_xgb.pkl')
        joblib.dump(model_hum, f'models_xgb/{city}_hum_xgb.pkl')
        
    except Exception as e:
        print(f"   ❌ Error training {city}: {e}")

print("\n🎉 ALL XGBOOST MODELS TRAINED SUCCESSFULLY!")
print("📁 Models saved in 'models_xgb/' folder.")