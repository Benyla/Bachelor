import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import os
import argparse
import json

# Import the provided function to get latent codes
from src.utils.latent_codes_and_metadata import get_latent_and_metadata

def plot_latent_covariance_heatmap(config, epoch):
    """
    Loads latent codes for a given epoch and plots a heatmap of their covariance matrix.

    Args:
        config (dict): Configuration dictionary containing:
            - model.use_adv (bool)
            - model.latent_dim (int)
            - metadata_csv (str)
        epoch (int): Epoch number to load (e.g., 20 → "VAE+_256_latent_epoch_20.pth")
    """
    # Retrieve the DataFrame with latent codes
    df_latents = get_latent_and_metadata(config, epoch)
    
    # Extract only the latent code columns (z0, z1, ..., z_{latent_dim-1})
    latent_columns = [col for col in df_latents.columns if col.startswith("z")]
    Z = df_latents[latent_columns].values  # shape: (N, latent_dim)
    
    # Compute the empirical covariance matrix (latent_dim x latent_dim)
    cov_matrix = np.cov(Z, rowvar=False)
    
    # Plot the heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cov_matrix,
        cmap="vlag",
        center=0,
        xticklabels=latent_columns,
        yticklabels=latent_columns
    )
    plt.title(f"Empirical Covariance of Latent Codes (Epoch {epoch})")
    plt.xlabel("Latent Dimensions")
    plt.ylabel("Latent Dimensions")
    plt.tight_layout()
    output_dir = "experiments/plots"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"latent_cov_epoch_{epoch}.png"))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot latent covariance heatmap.")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file.")
    parser.add_argument("--epoch", type=int, required=True, help="Epoch number.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    plot_latent_covariance_heatmap(config, args.epoch)
