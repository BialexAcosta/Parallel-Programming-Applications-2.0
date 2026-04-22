import requests, io
import pandas as pd
import os

def download_firms_data():
    API_KEY = "1ff65824800f37f4bea41edc09581ae0"
    os.makedirs("exercise_3", exist_ok=True)

    # Descargar datos de Yucatán (abril 2024)
    url = (
        f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/"
        f"{API_KEY}/VIIRS_SNPP_SP/-91.5,17.5,-86.5,21.5/5/2024-04-15"
    )
    
    print("Descargando datos de NASA FIRMS (puntos de calor)...")
    try:
        r = requests.get(url)
        if r.status_code == 200:
            df = pd.read_csv(io.StringIO(r.text))
            df.to_csv("exercise_3/firms_data.csv", index=False)
            print(f"✅ {len(df)} detecciones guardadas en exercise_3/firms_data.csv")
        else:
            print(f"Error al descargar datos: {r.status_code} - {r.text}")
    except Exception as e:
        print(f"Error en la descarga: {e}")

if __name__ == "__main__":
    download_firms_data()
