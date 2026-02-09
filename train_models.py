import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib

if not os.path.exists('models'):
    os.makedirs('models')

data_folder = 'data'
files = [f for f in os.listdir(data_folder) if f.endswith('_weather.csv')]

print(f"🚀 Found {len(files)} cities. Starting Random Forest Training...\n")

total_accuracy = 0

for file in files:
    city_name = file.replace('_weather.csv', '')
    
    
    file_path = os.path.join(data_folder, file)
    df = pd.read_csv(file_path)
    
    df['Target_Temp'] = df['Max_Temp'].shift(-1)
    df['Target_Humidity'] = df['Humidity'].shift(-1)
    df = df.dropna()
    
    features = ['Max_Temp', 'Humidity', 'Wind_Speed', 'Rainfall']
    X = df[features]
    y_temp = df['Target_Temp']
    y_hum = df['Target_Humidity']
    
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_temp, test_size=0.2, random_state=42)
    
    model_temp = RandomForestRegressor(n_estimators=100, random_state=42)
    model_temp.fit(X_train, y_train)
   
    preds = model_temp.predict(X_test)
    accuracy = r2_score(y_test, preds) * 100
    total_accuracy += accuracy
    
    print(f"📍 Training RF for {city_name}...")
    print(f"   🌡️  Temp Model Accuracy: {accuracy:.2f}%")
    
    model_hum = RandomForestRegressor(n_estimators=100, random_state=42)
    model_hum.fit(X, y_hum)
    
    model_temp.fit(X, y_temp)
    
    joblib.dump(model_temp, f'models/{city_name}_temp_model.pkl')
    joblib.dump(model_hum, f'models/{city_name}_hum_model.pkl')

avg_acc = total_accuracy / len(files)
print(f"\n🎉 ALL RF MODELS TRAINED! Average Accuracy: {avg_acc:.2f}%")
print("📁 Models saved in 'models/' folder.")