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

    binary = morphology.remove_small_objects(binary, min_size=200)
    binary = morphology.remove_small_holes(binary, area_threshold=500)

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
            "major_axis": round(p.major_axis_length, 2),
            "minor_axis": round(p.minor_axis_length, 2),
            "bbox":       p.bbox,
            "centroid":   p.centroid,
        })
    return img, labels, cells

if __name__ == "__main__":
    # Probar con la primera imagen de la secuencia 01
    img_path = "exercise_2/DIC-C2DH-HeLa/01/t001.tif"
    if os.path.exists(img_path):
        img, labels, cells = process_image(img_path)
        print(f"Células detectadas: {len(cells)}")
        
        # Guardar visualización
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(img, cmap="gray")
        axes[0].set_title("Original")
        axes[1].imshow(labels, cmap="nipy_spectral")
        axes[1].set_title(f"Segmentación ({len(cells)} células)")
        axes[2].imshow(img, cmap="gray")
        for c in cells:
            r0, c0, r1, c1 = c["bbox"]
            rect = mpatches.Rectangle((c0, r0), c1-c0, r1-r0,
                                        linewidth=1.5, edgecolor="lime", facecolor="none")
            axes[2].add_patch(rect)
        axes[2].set_title("Bounding boxes")
        plt.tight_layout()
        plt.savefig("exercise_2/results/sample_segmentation.png")
        print("Resultado guardado en exercise_2/results/sample_segmentation.png")
    else:
        print(f"Imagen no encontrada: {img_path}")
