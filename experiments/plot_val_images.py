#!/usr/bin/env python
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.utils.config_loader import load_config
from src.utils.data_loader import get_data, SingleCellDataset

def main():
    p = argparse.ArgumentParser(description="Plot first 100 val images (10×10) with optional MoA filter")
    p.add_argument("--config", required=True, help="Path to your YAML config")
    p.add_argument("--moa", default=None, help="Only plot cells with this moa class")
    args = p.parse_args()

    # 1) load config & metadata
    config = load_config(args.config)
    meta = pd.read_csv(config["metadata_csv"])
    # strip “.npy” so it matches filenames
    meta["Single_Cell_Image_Name"] = (
        meta["Single_Cell_Image_Name"]
            .astype(str)
            .str.replace(".npy", "", regex=False)
    )

    # 2) get val filepaths
    _, val_files, _ = get_data()
    # extract IDs (filenames without .npy)
    file_ids = [os.path.splitext(os.path.basename(p))[0] for p in val_files]

    # 3) if moa filter, select only matching IDs
    if args.moa:
        keep = set(meta.loc[meta["moa"] == args.moa, "Single_Cell_Image_Name"])
        file_ids = [fid for fid in file_ids if fid in keep]
        val_files = [p for p in val_files if os.path.splitext(os.path.basename(p))[0] in keep]
        print(f"[INFO] Filtering for moa={args.moa}, {len(val_files)} images remain.")
    else:
        print(f"[INFO] No MoA filter, {len(val_files)} total val images.")

    # 4) take first 100
    val_files = val_files[:100]
    ids_100 = [os.path.splitext(os.path.basename(p))[0] for p in val_files]
    print("Plotting the following image IDs:")
    for i, fid in enumerate(ids_100, 1):
        print(f"{i}: {fid}")

    # 5) load images into array
    images = [np.load(path) for path in val_files]  # assumes each is a 2D array

    # 6) plot 10×10 grid
    fig, axes = plt.subplots(10, 10, figsize=(20,20))
    for ax, img in zip(axes.flatten(), images):
        ax.imshow(img, cmap="gray")
        ax.axis("off")
    plt.tight_layout()
    os.makedirs("experiments/plots", exist_ok=True)
    plot_path = os.path.join("experiments/plots", f"val_grid_{args.moa or 'all'}.png")
    plt.savefig(plot_path)
    print(f"[INFO] Saved plot to {plot_path}")


if __name__ == "__main__":
    main()