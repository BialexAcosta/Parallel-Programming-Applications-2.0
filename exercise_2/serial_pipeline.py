import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage import io, filters, measure, morphology, segmentation
from skimage.feature import peak_local_max
from scipy.ndimage import distance_transform_edt
import pandas as pd
import time
import os

def process_image(img_path):
    img = io.imread(img_path)

    smoothed = filters.gaussian(img, sigma=2)
    thresh = filters.threshold_otsu(smoothed)
    binary = smoothed > thresh

    binary = morphology.remove_small_objects(binary, max_size=199)
    binary = morphology.remove_small_holes(binary, max_size=499)

    distance = distance_transform_edt(binary)
    coords = peak_local_max(distance, min_distance=15, labels=binary)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers = measure.label(mask)
    labels = segmentation.watershed(-distance, markers, mask=binary)

    props = measure.regionprops(labels)
    cells = []
    for p in props:
        if p.area < 200:
            continue
        cells.append({
            "area":       p.area,
            "major_axis": round(p.axis_major_length, 2),
            "minor_axis": round(p.axis_minor_length, 2),
            "bbox":       p.bbox,
            "centroid":   p.centroid,
        })
    return img, labels, cells

if __name__ == "__main__":
    # Test with the first image in sequence 01.
    img_path = "exercise_2/DIC-C2DH-HeLa/01/t001.tif"
    if os.path.exists(img_path):
        img, labels, cells = process_image(img_path)
        print(f"Detected cells: {len(cells)}")
        
        # Save visualization.
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img, cmap="gray")
        axes[0].set_title("Original")
        axes[1].imshow(labels, cmap="nipy_spectral")
        axes[1].set_title(f"Segmentation ({len(cells)} cells)")
        axes[2].imshow(img, cmap="gray")
        for c in cells:
            r0, c0, r1, c1 = c["bbox"]
            rect = mpatches.Rectangle((c0, r0), c1-c0, r1-r0,
                                        linewidth=1.5, edgecolor="lime", facecolor="none")
            axes[2].add_patch(rect)
        axes[2].set_title("Bounding boxes")
        plt.tight_layout()
        plt.savefig("exercise_2/results/sample_segmentation.png")
        print("Result saved to exercise_2/results/sample_segmentation.png")
    else:
        print(f"Image not found: {img_path}")
