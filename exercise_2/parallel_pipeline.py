import os
import time
import pandas as pd
from multiprocessing import Pool
from serial_pipeline import process_image

def process_image_path(img_path):
    try:
        _, _, cells = process_image(img_path)
        majors = [c["major_axis"] for c in cells]
        minors = [c["minor_axis"] for c in cells]
        areas = [c["area"] for c in cells]
        return {
            "image": os.path.basename(img_path),
            "n_cells": len(cells),
            "avg_major": round(sum(majors) / len(majors), 2) if majors else 0,
            "std_major": round(pd.Series(majors).std(ddof=0), 2) if majors else 0,
            "avg_minor": round(sum(minors) / len(minors), 2) if minors else 0,
            "std_minor": round(pd.Series(minors).std(ddof=0), 2) if minors else 0,
            "avg_area": round(sum(areas) / len(areas), 2) if areas else 0,
            "std_area": round(pd.Series(areas).std(ddof=0), 2) if areas else 0,
        }
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return None

def run_benchmark(img_dir="exercise_2/DIC-C2DH-HeLa/01", workers_list=[2, 4]):
    if not os.path.exists(img_dir):
        print(f"Directory {img_dir} not found.")
        return

    all_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".tif")])
    print(f"Total images: {len(all_paths)}\n")

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
        efficiency = speedup / w
        print(f"Workers={w}: {t_par:.2f}s Speedup={speedup:.2f} Efficiency={efficiency:.2f}")

    # Save results
    df = pd.DataFrame([r for r in results_serial if r is not None])
    os.makedirs("exercise_2/results", exist_ok=True)
    df.to_csv("exercise_2/results/exercise_2_results.csv", index=False)
    print("\nResults saved to exercise_2/results/exercise_2_results.csv")

if __name__ == "__main__":
    run_benchmark()
