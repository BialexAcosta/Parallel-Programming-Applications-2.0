import urllib.request
import zipfile
import os

def download_data():
    # Crear carpeta para el ejercicio 2 si no existe
    os.makedirs("exercise_2", exist_ok=True)

    url = "http://data.celltrackingchallenge.net/training-datasets/DIC-C2DH-HeLa.zip"
    zip_path = "exercise_2/DIC-C2DH-HeLa.zip"
    extract_path = "exercise_2/DIC-C2DH-HeLa"

    if os.path.exists(extract_path):
        print(f"El directorio {extract_path} ya existe. Saltando descarga.")
        return

    print("Descargando dataset HeLa (aprox. 80MB)...")
    try:
        urllib.request.urlretrieve(url, zip_path)
    except Exception as e:
        print(f"Error al descargar: {e}")
        return

    print("Extrayendo archivos...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extraer en exercise_2, ya que el zip normalmente crea la carpeta DIC-C2DH-HeLa
            zip_ref.extractall("exercise_2")
    except Exception as e:
        print(f"Error al extraer: {e}")
        return

    print(f"¡Listo! Datos preparados en {extract_path}")
    
    # Limpiar el zip
    if os.path.exists(zip_path):
        os.remove(zip_path)

if __name__ == "__main__":
    download_data()
