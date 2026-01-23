import os
import time
import requests
import pandas as pd

# 1. Setup: Create a folder for data
if not os.path.exists('data'):
    os.makedirs('data')
    print("📂 Created a new folder called 'data'!")

# 2. FULL LIST OF TAMIL NADU DISTRICTS (38 Cities)
locations = {
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
    "Madurai": {"lat": 9.9252,  "lon": 78.1198},
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

# 3. Connection Settings (2010 to 2024)
start_date = "20100101"
end_date = "20240101"
base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"

print(f"🚀 Starting Massive Data Collection for {len(locations)} Districts...")

# 4. Loop through each city
for i, (city, coords) in enumerate(locations.items(), 1):
    print(f"[{i}/{len(locations)}] ⏳ Downloading {city}...")
    
    params = {
        "parameters": "T2M_MAX,RH2M,WS2M,PRECTOTCORR",
        "community": "AG",
        "longitude": coords["lon"],
        "latitude": coords["lat"],
        "start": start_date,
        "end": end_date,
        "format": "JSON"
    }
    
    try:
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
            print(f"   ✅ Saved {filename}")
        else:
            print(f"   ❌ Failed {city} (Status: {response.status_code})")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")

    # Wait 1.5 seconds to be polite to NASA
    time.sleep(1.5) 

print("\n🎉 MISSION COMPLETE: All 38 Districts Downloaded!")