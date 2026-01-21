import os
import time
import requests
import pandas as pd

if not os.path.exists('data'):
    os.makedirs('data')
    print("📂 Created a new folder called 'data'!")

locations = {
    "Chennai":      {"lat": 13.0827, "lon": 80.2707}, 
    "Madurai":      {"lat": 9.9252,  "lon": 78.1198}, 
    "Coimbatore":   {"lat": 11.0168, "lon": 76.9558}, 
    "Vellore":      {"lat": 12.9165, "lon": 79.1325}, 
    "Nagapattinam": {"lat": 10.7656, "lon": 79.8424}  
}

start_date = "20100101"
end_date = "20240101"
base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"

print("🚀 Starting connection to NASA servers...")

for city, coords in locations.items():
    print(f"⏳ Downloading weather data for {city}...")
    
    params = {
        "parameters": "T2M_MAX,RH2M,WS2M,PRECTOTCORR",
        "community": "AG",
        "longitude": coords["lon"],
        "latitude": coords["lat"],
        "start": start_date,
        "end": end_date,
        "format": "JSON"
    }
    
    response = requests.get(base_url, params=params)
    
    if response.status_code == 200:
        data = response.json()['properties']['parameter']
        
        df = pd.DataFrame({
            'Date': pd.to_datetime(list(data['T2M_MAX'].keys())),
            'Max_Temp': list(data['T2M_MAX'].values()),
            'Humidity': list(data['RH2M'].values()),
            'Wind_Speed': list(data['WS2M'].values()),
            'Rainfall': list(data['PRECTOTCORR'].values())
        })
        
        filename = f"data/{city}_weather.csv"
        df.to_csv(filename, index=False)
        print(f"✅ Success! Saved {filename}")
    else:
        print(f"❌ Problem downloading {city}. (NASA might be busy)")

    time.sleep(2) 

print("\n🎉 MISSION COMPLETE: You have the dataset!")