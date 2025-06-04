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
from scipy.stats import gaussian_kde

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
    
    for moa in moas.cat.categories:
        class_data = df_sub[df_sub['moa'] == moa]
        if len(class_data) < 10:
            continue  # skip small classes
        x = class_data['TSNE1'].values
        y = class_data['TSNE2'].values
        try:
            kde = gaussian_kde(np.vstack([x, y]))
            xi, yi = np.mgrid[x.min():x.max():100j, y.min():y.max():100j]
            zi = kde(np.vstack([xi.flatten(), yi.flatten()]))
            plt.contour(xi, yi, zi.reshape(xi.shape), levels=3, alpha=0.5, linewidths=1)
        except np.linalg.LinAlgError:
            print(f"[WARN] Skipping contour for '{moa}' due to singular covariance matrix.")
    
    handles = [
        plt.Line2D([0],[0],marker='o',color='w',
                   markerfacecolor=cmap(i),markersize=6)
        for i in range(len(moas.cat.categories))
    ]
    plt.legend(handles, moas.cat.categories,
               title='MOA', bbox_to_anchor=(1,1))
    plt.xlabel('t-SNE1'); plt.ylabel('t-SNE2')
    plt.title(f't-SNE of {len(df_sub)} latent codes (even by MOA)')
    scatter_out = os.path.join(args.output, 'tSNE_scatter_new.png')
    plt.tight_layout()
    plt.savefig(scatter_out)
    print(f"[INFO] Saved t-SNE scatter plot to {scatter_out}")
    plt.close()


if __name__ == '__main__':
    main()