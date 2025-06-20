#!/usr/bin/env python
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
import argparse

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# make sure your src/ is on the path
sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "src")
    )
)
from src.utils.data_loader import get_data, SingleCellDataset

def parse_args():
    p = argparse.ArgumentParser(
        description="Plot a grid of validation images, optionally filtered by MOA class"
    )
    p.add_argument(
        "--moa",
        type=str,
        default=None,
        help="If set, only plot images whose metadata MOA equals this string"
    )
    p.add_argument(
        "--n",
        type=int,
        default=100,
        help="Total number of images to plot (default: 100)"
    )
    p.add_argument(
        "--metadata",
        type=str,
        default="/zhome/70/5/14854/nobackup/deeplearningf22/bbbc021/singlecell/metadata.csv",
        help="Path to metadata CSV (must contain Single_Cell_Image_Name and moa columns)"
    )
    return p.parse_args()

def main():
    args = parse_args()

    # load metadata and index by image name (without .npy)
    meta = pd.read_csv(args.metadata)
    meta["Single_Cell_Image_Name"] = (
        meta["Single_Cell_Image_Name"]
        .astype(str)
        .str.replace(".npy", "", regex=False)
    )
    meta = meta.set_index("Single_Cell_Image_Name", drop=False)

    # get test files
    train_files, test_files, _ = get_data()
    print(f"[INFO] {len(test_files)} validation files found.")

    # filter for MOA 
    if args.moa is not None:
        print(f"[INFO] Filtering for MOA = {args.moa!r}")
        test_dataset = SingleCellDataset(test_files)
        filtered = []
        for path, (img_tensor, file_id) in zip(test_files, test_dataset):
            # file_id should match the stripped filename in metadata
            str_id = str(file_id)
            if str_id not in meta.index:
                print(f"[WARNING] ID {str_id} not in metadata! skipping.")
                continue
            if str(meta.loc[str_id, "moa"]) == args.moa:
                filtered.append(path)
        test_files = filtered
        print(f"[INFO] {len(test_files)} images match MOA = {args.moa!r}")

    # take up to N images
    n = min(len(test_files), args.n)
    test_files = test_files[:n]
    print(f"[INFO] Plotting {n} images from val set.")

    # load and convert to H×W×3 numpy arrays
    images = []
    for path in test_files:
        data = torch.load(path)
        if isinstance(data, torch.Tensor):
            img = data.numpy()
        elif isinstance(data, dict):
            img = data["image"].numpy()
        else:
            raise ValueError(f"Unsupported data format in {path}")
        # if channels-first cant remember
        if img.ndim == 3 and img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        images.append(img)

    # build grid
    grid_size = int(np.ceil(np.sqrt(n)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size*2, grid_size*2))
    axes = axes.flatten()
    for ax, img in zip(axes, images):
        ax.imshow(img)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")

    plt.tight_layout()

    # save
    out_dir = "experiments/plots"
    os.makedirs(out_dir, exist_ok=True)
    plot_path = os.path.join(out_dir, f"test_grid{('_'+args.moa) if args.moa else ''}.png")
    plt.savefig(plot_path)
    print(f"[INFO] Saved plot to {plot_path}")

if __name__ == "__main__":
    main()