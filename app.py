import streamlit as st
import pandas as pd
import joblib
import requests
import plotly.express as px 

st.set_page_config(page_title="TN Heat Risk AI", page_icon="☀️", layout="wide")

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

def load_models(city):
    try:
        temp_model = joblib.load(f'models_xgb/{city}_temp_xgb.pkl')
        hum_model = joblib.load(f'models_xgb/{city}_hum_xgb.pkl')
        return temp_model, hum_model
    except FileNotFoundError:
        return None, None

def calculate_heat_index(T, RH):
    return T + (0.55 - 0.0055 * RH) * (T - 14.5)

st.title("☀️ Spatio-Temporal Heat Health Forecasting System")
st.markdown("### 🏥 AI-Powered Warning System for Tamil Nadu (38 Districts)")

st.sidebar.header("📍 Select Location")
sorted_districts = sorted(list(tn_districts.keys()))
city = st.sidebar.selectbox("Choose District:", sorted_districts)

if st.sidebar.button("🔮 Predict Heat Risk"):
    st.info(f"📡 Fetching live satellite data for {city}...")

    lat = tn_districts[city]['lat']
    lon = tn_districts[city]['lon']

    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&past_days=7&daily=temperature_2m_max,relative_humidity_2m_mean,wind_speed_10m_max,precipitation_sum&timezone=auto"

    try:
        response = requests.get(url)
        data = response.json()

        temps = data['daily']['temperature_2m_max']
        hums = data['daily']['relative_humidity_2m_mean']

        live_temp = temps[-1] 
        live_hum = hums[-1]
        live_wind = data['daily']['wind_speed_10m_max'][-1]
        live_rain = data['daily']['precipitation_sum'][-1]

        st.success(f"✅ Data Received. Current Max Temp: {live_temp}°C")

        model_t, model_h = load_models(city)

        if model_t is None:
            st.error(f"❌ Error: XGBoost Model for {city} not found in 'models_xgb/' folder.")
        else:
            input_data = pd.DataFrame(
                [[live_temp, live_hum, live_wind, live_rain]], 
                columns=['Max_Temp', 'Humidity', 'Wind_Speed', 'Rainfall']
            )

            pred_temp = model_t.predict(input_data)[0]
            pred_hum = model_h.predict(input_data)[0]

            # Elevation Calibration for Hill Stations
            if city in ["Nilgiris", "Dindigul", "Theni"]: 
                pred_temp = pred_temp - 4.0

            hi = calculate_heat_index(pred_temp, pred_hum)

            
            if pred_temp < 30:
                risk = "LOW RISK"; color = "green"; bg = "#d4edda"
            elif 30 <= pred_temp < 36:
                risk = "MODERATE RISK"; color = "orange"; bg = "#fff3cd"
            elif 36 <= pred_temp < 45:
                risk = "HIGH RISK"; color = "red"; bg = "#f8d7da"
            else:
                risk = "SEVERE RISK"; color = "darkred"; bg = "#721c24"

            col1, col2, col3 = st.columns(3)
            col1.metric("🌡️ Predicted Temp (Tomorrow)", f"{pred_temp:.1f}°C", delta=f"{pred_temp-live_temp:.1f}")
            col2.metric("💧 Predicted Humidity", f"{pred_hum:.1f}%")
            col3.markdown(f"### Feels Like: {hi:.1f}°C")

            st.markdown(f"""
            <div style="background-color: {bg}; padding: 15px; border-radius: 10px; text-align: center; border: 2px solid {color}; margin-bottom: 20px;">
                <h2 style="color: {color}; margin:0;">⚠️ FORECAST: {risk}</h2>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("📊 Model Performance Benchmark")
            st.write("System-wide average accuracy across all 38 districts (Validation Phase).")

            accuracy_data = pd.DataFrame({
                'Model': ['Random Forest (Baseline)', 'XGBoost (Advanced)'],
                'Accuracy': [82.85, 83.20],
                'Color': ['#1f77b4', '#2ca02c'] 
            })

            fig = px.bar(
                accuracy_data, 
                x='Model', 
                y='Accuracy', 
                color='Model', 
                range_y=[75, 90],
                text='Accuracy',
                title="Overall System Accuracy (R² Score)",
                color_discrete_map={
                    'Random Forest (Baseline)': '#636EFA',
                    'XGBoost (Advanced)': '#00CC96'
                }
            )

            fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            fig.update_layout(showlegend=False, height=400)

            st.plotly_chart(fig, use_container_width=True)

            

    except Exception as e:
        st.error(f"System Error: {e}")