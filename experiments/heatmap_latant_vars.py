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

def plot_latent_means_barplot(config, epoch):
    """
    Loads latent codes for a given epoch and plots a barplot of their mean values.

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

    # Compute mean per dimension
    mean_vec = np.mean(Z, axis=0)

    # Plot bar chart of means
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(latent_columns)), mean_vec, edgecolor='black')
    plt.axhline(0.0, color='red', linestyle='--', label='Expected Mean = 0')
    plt.title(f"Empirical Mean of Latent Codes per Dimension (Epoch {epoch})")
    plt.xlabel("Latent Dimension Index")
    plt.ylabel("Empirical Mean")
    plt.legend()
    plt.tight_layout()
    output_dir = "experiments/plots"
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, f"latent_mean_epoch_{epoch}.png"))
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot latent covariance heatmap.")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file.")
    parser.add_argument("--epoch", type=int, required=True, help="Epoch number.")
    args = parser.parse_args()

    config = load_config(args.config)

    plot_latent_means_barplot(config, args.epoch)
