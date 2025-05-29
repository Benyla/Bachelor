#!/usr/bin/env python3
# File: experiments/tSNE_sampled.py

import argparse
import os
import yaml
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.manifold import TSNE
from src.models.VAE import VAE
from src.utils.data_loader import get_data, SingleCellDataset
from src.utils.config_loader import load_config
from src.utils.latent_codes_and_metadata import get_latent_and_metadata
from torch.utils.data import DataLoader
from torchvision.transforms import ToPILImage

mpl.rcParams['font.family'] = 'serif'

# sample_size determines how many total data points to include.
# If sample_size is None, the entire dataset will be used without downsampling.
def subsample_equal(df, sample_size):
    df = df.dropna(subset=["moa"]).copy()
    classes = df["moa"].unique()
    n_classes = len(classes)
    per_class = sample_size // n_classes
    extras = sample_size - per_class * n_classes

    subs = []
    for i, moa in enumerate(classes):
        n = per_class + (1 if i < extras else 0)
        sub = df[df["moa"] == moa].sample(
            n=min(n, len(df[df["moa"] == moa])),
            random_state=42
        )
        subs.append(sub)
    return pd.concat(subs).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(
        description="t-SNE on VAE latents with scatter and image-grid visualization"
    )
    parser.add_argument("--config",     type=str, required=True,
                        help="Path to YAML config")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to .pth checkpoint")
    parser.add_argument("--sample-size",type=int, default=None,
                        help="Total points to sample (evenly by MOA). If not set, uses full dataset.")
    parser.add_argument("--output",     type=str, default="experiments/plots",
                        help="Where to save the output plots")
    parser.add_argument("--grid-size",  type=int, default=50,
                        help="Number of cells per axis in the image grid (e.g. 50 for 50x50)")
    args = parser.parse_args()

    # load config & checkpoint
    config = load_config(args.config)
    config["model"]["checkpoint_path"] = args.model_path

    # prepare val_loader
    _, val_files, _ = get_data()
    val_loader = DataLoader(
        SingleCellDataset(val_files),
        batch_size=config["training"]["batch_size"],
        shuffle=False, drop_last=False
    )

    # get latent codes and metadata
    df = get_latent_and_metadata(config, 49)

    # subsample if requested
    if args.sample_size is None:
        df_sub = df
        print(f"[INFO] Using full dataset: {len(df_sub)} points across {df_sub['moa'].nunique()} MOAs")
    else:
        df_sub = subsample_equal(df, args.sample_size)
        print(f"[INFO] Subsampled to {len(df_sub)} points across {df_sub['moa'].nunique()} MOAs")

    # extract latent vectors
    z_cols = [c for c in df_sub.columns if c.startswith("z")]
    Z = df_sub[z_cols].values

    # run t-SNE
    tsne = TSNE(n_components=2, init='random', random_state=42)
    X_tsne = tsne.fit_transform(Z)
    df_sub['TSNE1'], df_sub['TSNE2'] = X_tsne[:,0], X_tsne[:,1]

    # create output dir
    os.makedirs(args.output, exist_ok=True)

    # 1) Scatter plot colored by MOA
    plt.figure(figsize=(15,12), dpi=300)
    moas = df_sub['moa'].astype('category')
    df_sub['moa_code'] = moas.cat.codes
    cmap = plt.get_cmap('tab20', len(moas.cat.categories))
    plt.scatter(df_sub['TSNE1'], df_sub['TSNE2'],
                c=df_sub['moa_code'].to_numpy(),
                cmap=cmap, s=15, alpha=0.7)
    handles = [
        plt.Line2D([0],[0],marker='o',color='w',
                   markerfacecolor=cmap(i),markersize=6)
        for i in range(len(moas.cat.categories))
    ]
    plt.legend(handles, moas.cat.categories,
               title='MOA', bbox_to_anchor=(1,1))
    plt.xlabel('t-SNE1'); plt.ylabel('t-SNE2')
    plt.title(f't-SNE of {len(df_sub)} latent codes (even by MOA)')
    scatter_out = os.path.join(args.output, 'tSNE_scatter.png')
    plt.tight_layout()
    plt.savefig(scatter_out)
    print(f"[INFO] Saved t-SNE scatter plot to {scatter_out}")
    plt.close()

    # 2) Compute t-SNE on full validation set for image grid visualization
    print(f"[INFO] Computing t-SNE on full dataset ({len(df)} points) for image grid...")
    tsne_full = TSNE(n_components=2, init='random', random_state=42)
    X_full = tsne_full.fit_transform(df[z_cols].values)
    df['TSNE1_full'], df['TSNE2_full'] = X_full[:,0], X_full[:,1]

    # 3) Image grid based on full validation embeddings
    grid_size = args.grid_size
    # define grid centers
    x_min, x_max = df['TSNE1_full'].min(), df['TSNE1_full'].max()
    y_min, y_max = df['TSNE2_full'].min(), df['TSNE2_full'].max()
    centers_x = np.linspace(x_min, x_max, grid_size)
    # invert y for top-down plotting
    centers_y = np.linspace(y_max, y_min, grid_size)

    # map id to image
    id_set = set(df['id'].astype(str).tolist())
    to_pil = ToPILImage()
    id2img = {}
    print(f"[INFO] Loading images for {len(id_set)} embedded points...")
    for imgs, ids in val_loader:
        for img, idx in zip(imgs, ids):
            i = str(idx) if isinstance(idx, str) else str(idx.item())
            if i in id_set and i not in id2img:
                id2img[i] = to_pil(img.cpu())
        if len(id2img) >= len(id_set):
            break

    # assign nearest image to each grid cell
    used = set()
    grid_assign = {}
    pts = df[['TSNE1_full','TSNE2_full']].values
    ids_array = df['id'].astype(str).values
    for row in range(grid_size):
        for col in range(grid_size):
            cx, cy = centers_x[col], centers_y[row]
            # compute distances
            dists = np.sqrt((pts[:,0]-cx)**2 + (pts[:,1]-cy)**2)
            idx_min = int(np.argmin(dists))
            img_id = ids_array[idx_min]
            if img_id not in used:
                grid_assign[(row, col)] = img_id
                used.add(img_id)
            else:
                grid_assign[(row, col)] = None

    # plot image grid
    fig = plt.figure(figsize=(20,20), dpi=300)
    for (row, col), img_id in grid_assign.items():
        if img_id is None:
            continue
        left = col / grid_size
        bottom = (grid_size - row - 1) / grid_size
        width = 1.0 / grid_size
        height = 1.0 / grid_size
        ax = fig.add_axes([left, bottom, width, height])
        ax.imshow(id2img[img_id])
        ax.axis('off')

    grid_out = os.path.join(args.output, f'tSNE_image_grid_{grid_size}x{grid_size}.png')
    plt.savefig(grid_out)
    print(f"[INFO] Saved t-SNE image grid to {grid_out}")
    plt.close()


if __name__ == '__main__':
    main()