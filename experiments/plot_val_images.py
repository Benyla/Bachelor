#!/usr/bin/env python
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
import numpy as np
import torch
import matplotlib.pyplot as plt
from src.utils.data_loader import get_data

def main():
    # Get test data
    _, test_files, _ = get_data()

    # Take first 100
    test_files = test_files[:100]
    print(f"[INFO] Plotting {len(test_files)} images from val set.")


    # Load images
    images = []
    for path in test_files:
        data = torch.load(path)
        print(f"[DEBUG] Loaded file: {path}, type: {type(data)}") # problems with data format?
        if isinstance(data, torch.Tensor):
            img = data.numpy()
        elif isinstance(data, dict):
            img = data['image'].numpy()  # can be removed - data is tensor
        if img.shape[0] == 3:
            img = np.transpose(img, (1,2,0))  # Convert (3,64,64) to (64,64,3)
        images.append(img)

    # Plot 10x10 grid
    fig, axes = plt.subplots(10, 10, figsize=(20,20))
    for ax, img in zip(axes.flatten(), images):
        ax.imshow(img)
        ax.axis("off")
    plt.tight_layout()

    # Save
    os.makedirs("experiments/plots", exist_ok=True)
    plot_path = os.path.join("experiments/plots", "test_grid.png")
    plt.savefig(plot_path)
    print(f"[INFO] Saved plot to {plot_path}")

if __name__ == "__main__":
    main()