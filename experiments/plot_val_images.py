#!/usr/bin/env python
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
import numpy as np
import matplotlib.pyplot as plt
from src.utils.data_loader import get_data

def main():
    # Get val filepaths
    _, val_files, _ = get_data()

    # Take first 100
    val_files = val_files[:100]
    print(f"[INFO] Plotting {len(val_files)} images from val set.")

    # Load images
    images = []
    for path in val_files:
        img = np.load(path)  # Expect shape (3,64,64)
        if img.shape[0] == 3:
            img = np.transpose(img, (1,2,0))  # To (64,64,3)
        images.append(img)

    # Plot 10x10 grid
    fig, axes = plt.subplots(10, 10, figsize=(20,20))
    for ax, img in zip(axes.flatten(), images):
        ax.imshow(img)
        ax.axis("off")
    plt.tight_layout()

    # Save
    os.makedirs("experiments/plots", exist_ok=True)
    plot_path = os.path.join("experiments/plots", "val_grid.png")
    plt.savefig(plot_path)
    print(f"[INFO] Saved plot to {plot_path}")

if __name__ == "__main__":
    main()