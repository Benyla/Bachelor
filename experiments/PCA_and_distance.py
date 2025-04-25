#!/usr/bin/env python3
# File: experiments/PCA_sampled.py

import argparse
import os
import yaml
import torch
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from src.models.VAE import VAE
from src.utils.data_loader import get_data, SingleCellDataset
from src.utils.config_loader import load_config
from torch.utils.data import DataLoader
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.gridspec as gridspec

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
            z, mu, logvar = model.encode(batch, return_stats=True)
            all_mu.append(mu.cpu())
            all_ids.extend(ids)
    latents = torch.cat(all_mu, dim=0).numpy()  # (N, latent_dim)

    # load metadata
    meta = pd.read_csv("/zhome/70/5/14854/nobackup/deeplearningf22/bbbc021/singlecell/metadata.csv")
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

def subsample_equal(df, sample_size): # gets subsample of val_loader with equally distributed moa
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
    parser.add_argument(
        "--mode", type=str, choices=["pca", "distance"], default="pca",
        help="Select analysis mode: 'pca' for PCA plot or 'distance' for MOA distance matrix"
    )
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

    # prepare df_sub according to mode
    if args.mode == "distance":
        # Compute MOA centroid distance matrix
        print(f"[INFO] Computing distance matrix for {len(df_sub)} points across {df_sub['moa'].nunique()} MOAs")
        latent_cols = [c for c in df_sub.columns if c.startswith("z")]
        # Compute centroids per MOA
        centroids = df_sub.groupby("moa")[latent_cols].mean()
        # Compute pairwise Euclidean distances
        dist_matrix = pd.DataFrame(
            pairwise_distances(centroids, metric="euclidean"),
            index=centroids.index, columns=centroids.index
        )
        # Hierarchical clustering to reorder matrix
        Z = linkage(dist_matrix.values, method="average")
        dendro = dendrogram(Z, no_plot=True)
        order = dendro["leaves"]
        dist_matrix = dist_matrix.iloc[order, order]
        # PCA on centroids for scatter
        cent_pca = PCA(n_components=2).fit_transform(centroids.values[order])

        # Plot cluster heatmap + dendrogram + centroid scatter
        fig = plt.figure(figsize=(16,8), dpi=300)
        gs = gridspec.GridSpec(2, 3, width_ratios=[0.2, 1, 1], height_ratios=[0.2, 1], wspace=0.05, hspace=0.05)
        ax_dendro_top = fig.add_subplot(gs[0,1])
        ax_dendro_left = fig.add_subplot(gs[1,0])
        ax_heat = fig.add_subplot(gs[1,1])
        ax_scatter = fig.add_subplot(gs[:,2])

        # Top dendrogram (no labels)
        dendrogram(Z, ax=ax_dendro_top, orientation="top", no_labels=True, color_threshold=None)
        ax_dendro_top.axis("off")

        # Left dendrogram (no labels)
        dendrogram(Z, ax=ax_dendro_left, orientation="right", no_labels=True, color_threshold=None)
        ax_dendro_left.axis("off")

        # Heatmap
        im = ax_heat.imshow(dist_matrix.values, aspect="auto", origin="lower", cmap="viridis")
        ax_heat.set_xticks(range(len(dist_matrix)))
        ax_heat.set_xticklabels(dist_matrix.columns, rotation=90, fontsize=6)
        ax_heat.set_yticks(range(len(dist_matrix)))
        ax_heat.set_yticklabels(dist_matrix.index, fontsize=6)
        ax_heat.set_title("MOA Distance Matrix (clustered)")

        # Colorbar
        cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
        cbar.set_label("Euclidean distance", rotation=270, labelpad=15)

        # Centroid PCA scatter
        sc = ax_scatter.scatter(cent_pca[:,0], cent_pca[:,1], c=cent_pca[:,0], cmap="plasma", s=60)
        for i, label in enumerate(dist_matrix.index):
            ax_scatter.text(cent_pca[i,0], cent_pca[i,1], label, fontsize=6, ha="center", va="center")
        ax_scatter.set_xlabel("PC1")
        ax_scatter.set_ylabel("PC2")
        ax_scatter.set_title("MOA Centroid PCA")

        # Save figure
        os.makedirs(args.output, exist_ok=True)
        outpath = os.path.join(args.output, "MOA_distance_matrix_clustered.png")
        plt.tight_layout()
        plt.savefig(outpath, dpi=300)
        print(f"[INFO] Saved clustered distance figure to {outpath}")
        plt.show()
        return

    else:
        if args.sample_size is None:
            df_sub = df
            print(f"[INFO] Using full dataset: {len(df_sub)} points across {df_sub['moa'].nunique()} MOAs")
        else:
            df_sub = subsample_equal(df, args.sample_size)
            print(f"[INFO] Subsampled to {len(df_sub)} points across {df_sub['moa'].nunique()} MOAs")

    if args.mode == "pca":
        # run PCA
        z_cols = [c for c in df_sub.columns if c.startswith("z")]
        pca = PCA(n_components=2, svd_solver="randomized")
        pcs = pca.fit_transform(df_sub[z_cols])
        df_sub["PC1"], df_sub["PC2"] = pcs[:,0], pcs[:,1]

        print("Explained variance ratios:", pca.explained_variance_ratio_[:2])

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
        suffix = f"{args.sample_size}" if args.sample_size is not None else "full_valset"
        outpath = os.path.join(args.output, f"PCA_plot_{suffix}.png")
        plt.tight_layout()
        plt.savefig(outpath)
        print(f"[INFO] Saved PCA plot to {outpath}")
        plt.show()

if __name__ == "__main__":
    main()