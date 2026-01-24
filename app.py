import streamlit as st
import pandas as pd
import joblib
import requests
import plotly.graph_objects as go # For the Trend Chart

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="TN Heat Risk AI", page_icon="☀️", layout="wide")

# --- 2. FULL LIST OF TAMIL NADU DISTRICTS ---
# You must have .pkl files for ALL of these in your 'models/' folder!
tn_districts = {
    "Ariyalur": {"lat": 11.1401, "lon": 79.0786},
    "Chengalpattu": {"lat": 12.6939, "lon": 79.9757},
    "Chennai": {"lat": 13.0827, "lon": 80.2707},
    "Coimbatore": {"lat": 11.0168, "lon": 76.9558},
    "Cuddalore": {"lat": 11.7480, "lon": 79.7714},
    "Dharmapuri": {"lat": 12.1211, "lon": 78.1582},
    "Dindigul": {"lat": 10.3673, "lon": 77.9803},
    "Erode": {"lat": 11.3410, "lon": 77.7172},
    "Kallakurichi": {"lat": 11.7384, "lon": 78.9639},
    "Kancheepuram": {"lat": 12.8342, "lon": 79.7036},
    "Kanyakumari": {"lat": 8.0883, "lon": 77.5385},
    "Karur": {"lat": 10.9601, "lon": 78.0766},
    "Krishnagiri": {"lat": 12.5186, "lon": 78.2137},
    "Madurai": {"lat": 9.9252, "lon": 78.1198},
    "Mayiladuthurai": {"lat": 11.1018, "lon": 79.6525},
    "Nagapattinam": {"lat": 10.7656, "lon": 79.8424},
    "Namakkal": {"lat": 11.2189, "lon": 78.1675},
    "Nilgiris": {"lat": 11.4102, "lon": 76.6950},
    "Perambalur": {"lat": 11.2358, "lon": 78.8810},
    "Pudukkottai": {"lat": 10.3797, "lon": 78.8208},
    "Ramanathapuram": {"lat": 9.3639, "lon": 78.8395},
    "Ranipet": {"lat": 12.9292, "lon": 79.3323},
    "Salem": {"lat": 11.6643, "lon": 78.1460},
    "Sivaganga": {"lat": 9.8433, "lon": 78.4809},
    "Tenkasi": {"lat": 8.9594, "lon": 77.3129},
    "Thanjavur": {"lat": 10.7870, "lon": 79.1378},
    "Theni": {"lat": 10.0104, "lon": 77.4768},
    "Thoothukudi": {"lat": 8.7642, "lon": 78.1348},
    "Tiruchirappalli": {"lat": 10.7905, "lon": 78.7047},
    "Tirunelveli": {"lat": 8.7139, "lon": 77.7567},
    "Tirupathur": {"lat": 12.4925, "lon": 78.5623},
    "Tiruppur": {"lat": 11.1085, "lon": 77.3411},
    "Tiruvallur": {"lat": 13.1430, "lon": 79.9128},
    "Tiruvannamalai": {"lat": 12.2253, "lon": 79.0747},
    "Tiruvarur": {"lat": 10.7725, "lon": 79.6365},
    "Vellore": {"lat": 12.9165, "lon": 79.1325},
    "Viluppuram": {"lat": 11.9401, "lon": 79.5055},
    "Virudhunagar": {"lat": 9.5680, "lon": 77.9624}
}

# --- 3. FUNCTIONS ---
def load_models(city):
    try:
        temp_model = joblib.load(f'models_xgb/{city}_temp_xgb.pkl')
        hum_model = joblib.load(f'models_xgb/{city}_hum_xgb.pkl')
        return temp_model, hum_model
    except:
        return None, None

def calculate_heat_index(T, RH):
    return T + (0.55 - 0.0055 * RH) * (T - 14.5)

# --- 4. UI LAYOUT ---
st.title("☀️ Tamil Nadu Heat Risk Forecasting")
st.markdown("### 🏥 AI-Powered Warning System for All Districts")

# Sidebar
st.sidebar.header("📍 Select District")
# Sort the list alphabetically for better UX
sorted_districts = sorted(list(tn_districts.keys()))
city = st.sidebar.selectbox("Choose District:", sorted_districts)

if st.sidebar.button("🔮 Predict Risk"):
    
    st.info(f"📡 Fetching live satellite data for {city}...")
    
    # --- 5. THE API FIX (Fetching 3 Days) ---
    lat = tn_districts[city]['lat']
    lon = tn_districts[city]['lon']
    
    # Changed past_days=1 to past_days=3 to get more data
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&past_days=7&daily=temperature_2m_max,relative_humidity_2m_mean,wind_speed_10m_max,precipitation_sum&timezone=auto"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        # --- THE FIX: Select the LAST available day (Index -1) ---
        # data['daily']['...'] returns a list like [Day1, Day2, Day3, Today]
        # We use [-1] to pick the most recent valid data
        
        temps = data['daily']['temperature_2m_max']
        hums = data['daily']['relative_humidity_2m_mean']
        
        live_temp = temps[-1] # Last item in list
        live_hum = hums[-1]
        live_wind = data['daily']['wind_speed_10m_max'][-1]
        live_rain = data['daily']['precipitation_sum'][-1]
        
        st.success(f"✅ Data Received. Recent Max Temp: {live_temp}°C")

        # --- 6. PREDICTION ---
        model_t, model_h = load_models(city)
        
        if model_t is None:
            st.error(f"❌ Error: Model for {city} not found! You need to train it first.")
            st.warning("💡 Tip: Add this district to your '1_download_data.py' and re-run training.")
        else:
            # Predict
            input_data = [[live_temp, live_hum, live_wind, live_rain]]
            pred_temp = model_t.predict(input_data)[0]
            pred_hum = model_h.predict(input_data)[0]
            
            # Risk Calc
            hi = calculate_heat_index(pred_temp, pred_hum)
            
            # Risk Logic
            if hi < 27:
                risk = "LOW RISK"; color = "green"; bg = "#d4edda"
            elif 27 <= hi < 32:
                risk = "MODERATE RISK"; color = "orange"; bg = "#fff3cd"
            elif 32 <= hi < 41:
                risk = "HIGH RISK"; color = "red"; bg = "#f8d7da"
            else:
                risk = "SEVERE RISK"; color = "darkred"; bg = "#721c24"

            # --- 7. VISUALIZATION ---
            col1, col2, col3 = st.columns(3)
            col1.metric("🌡️ Tomorrow's Temp", f"{pred_temp:.1f}°C", delta=f"{pred_temp-live_temp:.1f}")
            col2.metric("💧 Tomorrow's Humidity", f"{pred_hum:.1f}%")
            col3.markdown(f"### Feels Like: {hi:.1f}°C")
            
            st.markdown(f"""
            <div style="background-color: {bg}; padding: 15px; border-radius: 10px; text-align: center; border: 2px solid {color};">
                <h1 style="color: {color}; margin:0;">{risk}</h1>
            </div>
            """, unsafe_allow_html=True)

            # BONUS: Trend Chart (Since we fetched 3 days!)
            st.subheader("📉 Recent Temperature Trend (Last 3 Days)")
            fig = go.Figure(data=go.Scatter(y=temps, mode='lines+markers', line=dict(color='firebrick', width=4)))
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"API Error: {e}")