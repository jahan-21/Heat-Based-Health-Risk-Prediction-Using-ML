import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

# 1. Setup: Create a folder to save the trained 'brains' (models)
if not os.path.exists('models'):
    os.makedirs('models')
    print("📂 Created a new folder called 'models'!")

# 2. Find all the CSV files we downloaded
data_folder = 'data'
files = [f for f in os.listdir(data_folder) if f.endswith('_weather.csv')]

print(f"🚀 Found {len(files)} cities. Starting training...")

# 3. Loop through each city and train its own AI
for file in files:
    city_name = file.replace('_weather.csv', '') # Extract "Chennai" from filename
    print(f"\n🧠 Training AI for {city_name}...")
    
    # Load the data
    df = pd.read_csv(os.path.join(data_folder, file))
    
    # --- THE LOGIC: LAG FEATURES ---
    # We want to use TODAY'S weather to predict TOMORROW'S.
    # So we create a 'Target' column by shifting data up by 1 row.
    
    df['Target_Temp'] = df['Max_Temp'].shift(-1)      # Tomorrow's Temp
    df['Target_Humidity'] = df['Humidity'].shift(-1)  # Tomorrow's Humidity
    
    # Drop the last row (because it has no tomorrow to predict)
    df = df.dropna()
    
    # Define Inputs (X) and Outputs (y)
    # Inputs: Today's stats
    features = ['Max_Temp', 'Humidity', 'Wind_Speed', 'Rainfall']
    X = df[features]
    
    # Outputs: Tomorrow's stats
    y_temp = df['Target_Temp']
    y_hum = df['Target_Humidity']
    
    # --- TRAINING THE MODELS ---
    # Model 1: Temperature Predictor
    model_temp = RandomForestRegressor(n_estimators=100, random_state=42)
    model_temp.fit(X, y_temp)
    
    # Model 2: Humidity Predictor
    model_hum = RandomForestRegressor(n_estimators=100, random_state=42)
    model_hum.fit(X, y_hum)
    
    # --- SAVING THE BRAINS ---
    # We save them as .pkl files so we can use them later in the app
    joblib.dump(model_temp, f'models/{city_name}_temp_model.pkl')
    joblib.dump(model_hum, f'models/{city_name}_hum_model.pkl')
    
    print(f"✅ Trained & Saved models for {city_name}!")

print("\n🎉 ALL TRAINING COMPLETE! Your AI is ready.")