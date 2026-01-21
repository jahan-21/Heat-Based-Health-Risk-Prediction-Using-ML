import joblib
import pandas as pd

# --- HELPER: The Health Risk Formula ---
def calculate_heat_index(T, RH):
    # This is a simplified version of the NOAA formula for the project demo
    # T = Temp in Celsius, RH = Humidity %
    return T + (0.55 - 0.0055 * RH) * (T - 14.5)

def get_risk_label(hi_value):
    if hi_value < 27: return "🟢 Low Risk (Safe)"
    elif 27 <= hi_value < 32: return "🟡 Moderate Risk (Caution)"
    elif 32 <= hi_value < 41: return "🟠 High Risk (Extreme Caution)"
    else: return "🔴 Severe Risk (Danger)"

# --- MAIN TEST ---
print("🧪 TESTING THE MODEL FOR: Madurai")

# 1. Load the Madurai Brains
try:
    model_temp = joblib.load('models/Madurai_temp_model.pkl')
    model_hum = joblib.load('models/Madurai_hum_model.pkl')
    print("✅ Models loaded successfully!")
except:
    print("❌ Error: Could not find model files. Did you run train_models.py?")
    exit()

# 2. Simulate User Input (Imagine today is hot and humid)
# [Max_Temp, Humidity, Wind_Speed, Rainfall]
today_weather = [[38.0, 60.0, 5.0, 0.0]] 

print(f"\n🌤️  Today's Input: Temp={today_weather[0][0]}°C, Humidity={today_weather[0][1]}%")

# 3. Ask the Brains to Predict Tomorrow
pred_temp = model_temp.predict(today_weather)[0]
pred_hum = model_hum.predict(today_weather)[0]

print(f"🔮 AI Prediction for Tomorrow: Temp={pred_temp:.1f}°C, Humidity={pred_hum:.1f}%")

# 4. Calculate Risk
hi = calculate_heat_index(pred_temp, pred_hum)
risk = get_risk_label(hi)

print(f"🌡️  Calculated Heat Index: {hi:.1f}")
print(f"⚠️  HEALTH FORECAST: {risk}")