import os
import time
import pandas as pd
import numpy as np
from skimage import io, measure
import matplotlib.pyplot as plt

def run_cellpose_pipeline(data_dir="exercise_2/DIC-C2DH-HeLa/01"):
    print("=== Exercise 2: Segmentation with Cellpose ===")
    
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} not found.")
        print("Please download/extract the dataset into that path before running this script.")
        return
        
    try:
        from cellpose import models
    except ImportError:
        print("The cellpose module is not installed. Install it with: pip install cellpose")
        return
        
    all_paths = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".tif")])
    
    if not all_paths:
        print("No .tif images were found in the directory.")
        return

    print("Loading Cellpose model (cyto2)...")
    # Support older and newer Cellpose APIs.
    if hasattr(models, "Cellpose"):
        model = models.Cellpose(gpu=False, model_type='cyto2')
        eval_kwargs = {"diameter": None, "channels": [0, 0]}
    elif hasattr(models, "CellposeModel"):
        model = models.CellposeModel(gpu=False, model_type='cyto2')
        eval_kwargs = {"diameter": None, "channels": [0, 0]}
    else:
        print("Unsupported Cellpose API version: missing Cellpose and CellposeModel classes.")
        return
    
    results = []
    t0 = time.perf_counter()
    
    # Process only the first 5 images for quick validation.
    sample_paths = all_paths[:5]
    print(f"Processing {len(sample_paths)} validation images...")
    
    for img_path in sample_paths:
        img = io.imread(img_path)
        # Evaluate using Cellpose.
        eval_result = model.eval(img, **eval_kwargs)
        if isinstance(eval_result, tuple):
            masks = eval_result[0]
        else:
            masks = eval_result
        
        props = measure.regionprops(masks)
        cells = []
        for p in props:
            if p.area < 50:  # Noise filter.
                continue
            cells.append({
                "area": p.area,
                "major_axis": round(p.axis_major_length, 2),
                "minor_axis": round(p.axis_minor_length, 2),
                "bbox": p.bbox,
                "centroid": p.centroid,
            })
            
        results.append({
            "image": os.path.basename(img_path),
            "n_cells": len(cells),
            "avg_major": round(np.mean([c["major_axis"] for c in cells]), 2) if cells else 0,
            "avg_minor": round(np.mean([c["minor_axis"] for c in cells]), 2) if cells else 0,
            "avg_area": round(np.mean([c["area"] for c in cells]), 2) if cells else 0
        })
        print(f"Processed {os.path.basename(img_path)}: {len(cells)} cells.")
        
    t_elapsed = time.perf_counter() - t0
    print(f"Total time (Cellpose serial - 5 images): {t_elapsed:.2f} s")
    
    df = pd.DataFrame(results)
    print("\nCellpose results:")
    print(df.to_string(index=False))
    
    # Save results.
    os.makedirs("exercise_2/results", exist_ok=True)
    df.to_csv("exercise_2/results/cellpose_results.csv", index=False)
    print("\nSaved to exercise_2/results/cellpose_results.csv")

if __name__ == "__main__":
    run_cellpose_pipeline()