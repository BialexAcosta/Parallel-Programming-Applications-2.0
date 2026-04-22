import os
import time
import pandas as pd
import numpy as np
from skimage import io, measure
import matplotlib.pyplot as plt

def run_cellpose_pipeline(data_dir="exercise_2/DIC-C2DH-HeLa/01"):
    print("=== Ejercicio 2: Segmentación con Cellpose ===")
    
    if not os.path.exists(data_dir):
        print(f"Directorio de datos {data_dir} no encontrado.")
        print("Asegúrese de ejecutar o tener los datos extraídos en esa ruta.")
        return
        
    try:
        from cellpose import models
    except ImportError:
        print("El módulo cellpose no está instalado. Instalándolo puede resolver esto (pip install cellpose).")
        return
        
    all_paths = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".tif")])
    
    if not all_paths:
        print("No se encontraron imágenes .tif en el directorio.")
        return

    print("Cargando modelo Cellpose (cyto2)...")
    # Para la prueba no usaremos GPU por defecto a menos que esté disponible de forma transparente.
    model = models.Cellpose(gpu=False, model_type='cyto2')
    
    results = []
    t0 = time.perf_counter()
    
    # Procesamos solo las primeras 5 imágenes como prueba para la validación
    sample_paths = all_paths[:5]
    print(f"Procesando {len(sample_paths)} imágenes de prueba...")
    
    for img_path in sample_paths:
        img = io.imread(img_path)
        # Evaluamos usando cellpose
        masks, flows, styles, diams = model.eval(img, diameter=None, channels=[0,0])
        
        props = measure.regionprops(masks)
        cells = []
        for p in props:
            if p.area < 50: # Filtro de ruido
                continue
            cells.append({
                "area": p.area,
                "major_axis": round(p.major_axis_length, 2),
                "minor_axis": round(p.minor_axis_length, 2),
                "bbox": p.bbox,
                "centroid": p.centroid,
            })
            
        results.append({
            "imagen": os.path.basename(img_path),
            "n_cells": len(cells),
            "avg_major": round(np.mean([c["major_axis"] for c in cells]), 2) if cells else 0,
            "avg_minor": round(np.mean([c["minor_axis"] for c in cells]), 2) if cells else 0,
            "avg_area": round(np.mean([c["area"] for c in cells]), 2) if cells else 0
        })
        print(f"Procesada {os.path.basename(img_path)}: {len(cells)} células.")
        
    t_elapsed = time.perf_counter() - t0
    print(f"Tiempo total (Cellpose serial - 5 imgs): {t_elapsed:.2f} s")
    
    df = pd.DataFrame(results)
    print("\nResultados con Cellpose:")
    print(df.to_string(index=False))
    
    # Guardar resultados
    os.makedirs("exercise_2/results", exist_ok=True)
    df.to_csv("exercise_2/results/cellpose_results.csv", index=False)
    print("\nGuardado en exercise_2/results/cellpose_results.csv")

if __name__ == "__main__":
    run_cellpose_pipeline()