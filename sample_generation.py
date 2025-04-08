import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import torch
import matplotlib.pyplot as plt
import numpy as np

# Import your VAE model (ensure your PYTHONPATH is set correctly or adjust sys.path)
from src.models.VAE import VAE

def load_latest_model(model_dir, device):
    """
    Searches the specified directory for model checkpoint files (*.pt) and returns
    the full path of the file with the latest modification time.
    """
    model_files = [f for f in os.listdir(model_dir) if f.endswith('.pt')]
    if not model_files:
        raise FileNotFoundError(f"No model checkpoint files found in {model_dir}.")
    # Select the latest file based on modification time
    latest_file = max(model_files, key=lambda f: os.path.getmtime(os.path.join(model_dir, f)))
    latest_path = os.path.join(model_dir, latest_file)
    print(f"Loading model checkpoint: {latest_path}")
    return latest_path

def generate_samples(model, num_samples, latent_dim, device):
    """
    Samples num_samples latent codes from a standard normal distribution and
    decodes them using the provided model's decoder.
    """
    # Sample latent vectors: shape (num_samples, latent_dim)
    z = torch.randn(num_samples, latent_dim).to(device)
    with torch.no_grad():
        # Assumes the model has a decode() method
        generated_samples = model.decode(z)
    return generated_samples

def plot_generated_samples(samples, num_rows, num_cols, save_path="generated_samples.png"):
    """
    Plots the provided generated samples in a grid.
    Assumes samples is a tensor of shape (N, C, H, W). Adjust if necessary.
    """
    samples = samples.cpu().numpy()
    num_samples = samples.shape[0]
    
    # Create a matplotlib grid to display images
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
    
    # If a single row, make sure axes is always iterable
    if num_rows == 1:
        axes = [axes]
    
    # For simplicity, assuming we want a single-row grid (num_rows=1)
    for idx in range(num_samples):
        ax = axes[0][idx] if num_rows == 1 else axes[idx // num_cols][idx % num_cols]
        img = samples[idx]
        # If the image has 1 channel, squeeze it for grayscale plotting
        if img.shape[0] == 1:
            img = img.squeeze(0)
            ax.imshow(img, cmap="gray")
        else:
            # If image has 3 channels (RGB), re-order axes
            img = np.transpose(img, (1, 2, 0))
            ax.imshow(img)
        ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def main():
    # Configuration parameters (adjust these if necessary)
    model_dir = "trained_models"
    in_channels = 1       # e.g., number of channels in your cell images
    latent_dim = 64       # must match what you used in training
    num_samples = 8       # number of random samples to generate

    # Set device for computation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Instantiate your VAE model
    model = VAE(in_channels=in_channels, latent_dim=latent_dim).to(device)
    
    # Load the latest model checkpoint
    latest_model_path = load_latest_model(model_dir, device)
    checkpoint = torch.load(latest_model_path, map_location=device)
    # Adjust depending on how your checkpoint is saved
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # Generate samples from random latent codes
    samples = generate_samples(model, num_samples, latent_dim, device)
    
    # Plot and save a grid of the generated images (1 row x 8 columns)
    plot_generated_samples(samples, num_rows=1, num_cols=num_samples)

if __name__ == "__main__":
    main()