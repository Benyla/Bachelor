import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import argparse
from src.utils.config_loader import load_config

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
    latent_columns = [col for col in df_latents.columns if col.startswith("z")]
    Z = df_latents[latent_columns].values  # shape: (N, latent_dim)

    # Compute variance per dimension
    var_vec = np.var(Z, axis=0)

    # Plot bar chart of variances
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(latent_columns)), var_vec, edgecolor='black')
    plt.axhline(1.0, color='red', linestyle='--', label='Prior Variance = 1')
    plt.title(f"Empirical Variance of Latent Codes per Dimension (Epoch {epoch})")
    plt.xlabel("Latent Dimension Index")
    plt.ylabel("Empirical Variance")
    plt.legend()
    plt.tight_layout()
    output_dir = "experiments/plots"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"latent_variance_epoch_{epoch}.png"))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot latent covariance heatmap.")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file.")
    parser.add_argument("--epoch", type=int, required=True, help="Epoch number.")
    args = parser.parse_args()

    config = load_config(args.config)

    plot_latent_covariance_heatmap(config, args.epoch)
