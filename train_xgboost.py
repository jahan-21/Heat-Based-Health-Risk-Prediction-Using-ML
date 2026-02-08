import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib

# 1. Setup: Create a folder to save the trained 'brains' (models)
if not os.path.exists('models_xgb'):
    os.makedirs('models_xgb')
    print("📂 Created a new folder called 'models_xgb'!")

# 2. Find all the CSV files we downloaded
data_folder = 'data'
files = [f for f in os.listdir(data_folder) if f.endswith('_weather.csv')]

print(f"🚀 Found {len(files)} cities. Starting XGBoost Training...\n")

total_accuracy = 0

# 3. Loop through each city and train its own AI
for file in files:
    city_name = file.replace('_weather.csv', '') 
    
    # Load the data
    file_path = os.path.join(data_folder, file)
    df = pd.read_csv(file_path)
    
    # --- FEATURE ENGINEERING (Lag Features) ---
    df['Target_Temp'] = df['Max_Temp'].shift(-1)      # Tomorrow's Temp
    df['Target_Humidity'] = df['Humidity'].shift(-1)  # Tomorrow's Humidity
    
    # Drop the last row (because it has no tomorrow to predict)
    df = df.dropna()
    
    # Define Inputs (X) and Outputs (y)
    features = ['Max_Temp', 'Humidity', 'Wind_Speed', 'Rainfall']
    X = df[features]
    y_temp = df['Target_Temp']
    y_hum = df['Target_Humidity']
    
    # --- SPLIT DATA (To Calculate Accuracy) ---
    # 80% for Training, 20% for Testing
    X_train, X_test, y_train, y_test = train_test_split(X, y_temp, test_size=0.2, random_state=42)
    
    # --- TRAIN TEMP MODEL & CHECK ACCURACY ---
    model_temp = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model_temp.fit(X_train, y_train)
    
    # Calculate Accuracy (R2 Score)
    preds = model_temp.predict(X_test)
    accuracy = r2_score(y_test, preds) * 100
    total_accuracy += accuracy
    
    print(f"📍 Training XGBoost for {city_name}...")
    print(f"   🌡️  Temp Model Accuracy: {accuracy:.2f}%")
    
    # --- TRAIN HUMIDITY MODEL & SAVE FINAL VERSIONS ---
    # (We retrain on FULL data before saving to make them as smart as possible)
    model_temp.fit(X, y_temp)
    
    model_hum = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    model_hum.fit(X, y_hum)
    
    # Save as .pkl files
    joblib.dump(model_temp, f'models_xgb/{city_name}_temp_xgb.pkl')
    joblib.dump(model_hum, f'models_xgb/{city_name}_hum_xgb.pkl')

# --- FINAL SUMMARY ---
avg_acc = total_accuracy / len(files)
print(f"\n🎉 ALL XGBOOST MODELS TRAINED! Average Accuracy: {avg_acc:.2f}%")
print("📁 Models saved in 'models_xgb/' folder.")