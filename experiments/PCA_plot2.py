#!/usr/bin/env python3
# File: experiments/PCA_sampled.py

import argparse
import os
import yaml
import torch
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from src.models.VAE import VAE
from src.utils.data_loader import get_data, SingleCellDataset
from src.utils.config_loader import load_config
from torch.utils.data import DataLoader

def get_latent_and_metadata(config, val_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load model
    model = VAE(
        in_channels=config["model"]["in_channels"],
        latent_dim=config["model"]["latent_dim"]
    ).to(device)
    ckpt = torch.load(config["model"]["checkpoint_path"], map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # collect latent means and IDs
    all_mu, all_ids = [], []
    with torch.no_grad():
        for batch, ids in val_loader:
            batch = batch.to(device)
            # get (z, mu, logvar), but we only need mu
            z, mu, logvar = model.encode(batch, return_stats=True)
            all_mu.append(mu.cpu())
            all_ids.extend(ids)
    latents = torch.cat(all_mu, dim=0).numpy()  # (N, latent_dim)

    # load metadata
    meta = pd.read_csv(config["data"]["metadata_path"])
    # strip .npy if present
    meta["Single_Cell_Image_Name"] = (
        meta["Single_Cell_Image_Name"].astype(str)
            .str.replace(".npy", "", regex=False)
    )

    # build dataframe
    z_cols = [f"z{i}" for i in range(latents.shape[1])]
    df = pd.DataFrame(latents, columns=z_cols)
    df["Single_Cell_Image_Name"] = all_ids
    df = df.merge(
        meta[["Single_Cell_Image_Name", "moa"]],
        on="Single_Cell_Image_Name", how="left"
    )
    return df

def subsample_equal(df, sample_size):
    # drop rows without moa
    df = df.dropna(subset=["moa"]).copy()
    # how many classes?
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
        description="PCA on a sampled subset of VAE latents, equal per MOA"
    )
    parser.add_argument("--config",       type=str, required=True,
                        help="Path to YAML config")
    parser.add_argument("--model-path",   type=str, required=True,
                        help="Path to .pth checkpoint")
    parser.add_argument("--sample-size",  type=int, default=None,
                        help="Total points to sample (evenly by MOA). If not set, uses the full dataset.")
    parser.add_argument("--output",       type=str, default="experiments/plots",
                        help="Where to save the PCA plot")
    args = parser.parse_args()

    # load config & inject
    config = load_config(args.config)
    config["model"]["checkpoint_path"] = args.model_path

    # prepare val_loader
    _, val_files, _ = get_data()
    val_loader = DataLoader(
        SingleCellDataset(val_files),
        batch_size=config["training"]["batch_size"],
        shuffle=False, drop_last=False
    )

    # get full dataframe
    df = get_latent_and_metadata(config, val_loader)
    print(f"[INFO] Total points with MOA: {df['moa'].notna().sum()}")

    # optionally subsample equally by class if sample_size is provided
    if args.sample_size is None:
        df_sub = df
        print(f"[INFO] Using full dataset: {len(df_sub)} points across {df_sub['moa'].nunique()} MOAs")
    else:
        df_sub = subsample_equal(df, args.sample_size)
        print(f"[INFO] Subsampled to {len(df_sub)} points across {df_sub['moa'].nunique()} MOAs")

    # run PCA
    z_cols = [c for c in df_sub.columns if c.startswith("z")]
    pca = PCA(n_components=2, svd_solver="randomized")
    pcs = pca.fit_transform(df_sub[z_cols])

    df_sub["PC1"], df_sub["PC2"] = pcs[:,0], pcs[:,1]

    # plot
    moas = df_sub["moa"].astype("category")
    df_sub["moa_code"] = moas.cat.codes
    cmap = plt.get_cmap("tab20", len(moas.cat.categories))

    plt.figure(figsize=(8,6))
    plt.scatter(df_sub["PC1"], df_sub["PC2"],
                c=df_sub["moa_code"].to_numpy(),
                cmap=cmap, s=15, alpha=0.7)
    handles = [
        plt.Line2D([0],[0],marker="o",color="w",
                   markerfacecolor=cmap(i),markersize=6)
        for i in range(len(moas.cat.categories))
    ]
    plt.legend(handles, moas.cat.categories,
               title="MOA", bbox_to_anchor=(1,1))
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title(f"PCA of {len(df_sub)} latent codes (even by MOA)")
    os.makedirs(args.output, exist_ok=True)
    outpath = os.path.join(args.output, "PCA_sampled.png")
    plt.tight_layout()
    plt.savefig(outpath)
    print(f"[INFO] Saved PCA plot to {outpath}")
    plt.show()

if __name__ == "__main__":
    main()