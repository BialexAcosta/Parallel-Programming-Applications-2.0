import os
import time
import pandas as pd
from multiprocessing import Pool
from serial_pipeline import process_image

def process_image_path(img_path):
    try:
        _, _, cells = process_image(img_path)
        return {
            "imagen": os.path.basename(img_path),
            "n_cells": len(cells),
            "avg_major": round(sum(c["major_axis"] for c in cells) / len(cells), 2) if cells else 0,
            "avg_minor": round(sum(c["minor_axis"] for c in cells) / len(cells), 2) if cells else 0,
            "avg_area":  round(sum(c["area"] for c in cells) / len(cells), 2) if cells else 0
        }
    except Exception as e:
        print(f"Error procesando {img_path}: {e}")
        return None

def run_benchmark(img_dir="exercise_2/DIC-C2DH-HeLa/01", workers_list=[2, 4]):
    if not os.path.exists(img_dir):
        print(f"Directorio {img_dir} no encontrado.")
        return

    all_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".tif")])
    print(f"Total imágenes: {len(all_paths)}\n")

    # Serial
    t0 = time.perf_counter()
    results_serial = [process_image_path(p) for p in all_paths]
    t_serial = time.perf_counter() - t0
    print(f"Serial: {t_serial:.2f}s")

    # Parallel
    for w in workers_list:
        t0 = time.perf_counter()
        with Pool(processes=w) as pool:
            results_parallel = pool.map(process_image_path, all_paths)
        t_par = time.perf_counter() - t0
        speedup = t_serial / t_par
        print(f"Workers={w}: {t_par:.2f}s Speedup={speedup:.2f}")

    # Guardar resultados
    df = pd.DataFrame([r for r in results_serial if r is not None])
    os.makedirs("exercise_2/results", exist_ok=True)
    df.to_csv("exercise_2/results/exercise_2_results.csv", index=False)
    print("\nResultados guardados en exercise_2/results/exercise_2_results.csv")

if __name__ == "__main__":
    run_benchmark()
