# src/train.py

import os
import yaml
import torch
import torch.optim as optim
import neptune.new as neptune
from data.mnist_dummy_data import load_mnist_data

from models.VAE import VAE  # Ensure your model is in src/models/VAE.py

# ---------------------------
# Load configuration
# ---------------------------
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

# ---------------------------
# Create Dummy Data
# ---------------------------

def dummy_data(config):
    """
    Returns an MNIST DataLoader resized to whatever the
    VAE expects (e.g., 64x64) or the original MNIST size (28x28),
    depending on how you implemented load_mnist_data.

    Args:
        config (dict): Configuration dictionary loaded from config.yaml.

    Returns:
        DataLoader: A DataLoader yielding MNIST batches.
    """
    batch_size = config["training"]["batch_size"]
    return load_mnist_data(batch_size)

# ---------------------------
# Main Training Loop
# ---------------------------
def train():
    # Load configuration
    config = load_config()

    # Initialize Neptune run using the new API
    run = neptune.init_run(
        project=config["experiment"]["neptune_project"],
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI0ZGE0NDljMi04NGIwLTRhNDEtOGU1ZC1kNmNhZWNlZTRhOTUifQ==",
        name=config["experiment"]["name"],
        tags=["dummy-data", "vae"]
    )

    # Log the configuration parameters as metadata
    run["parameters"] = config

    # Create dummy data loader
    dataloader = dummy_data(config)

    # Instantiate model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(
        in_channels=config["model"]["in_channels"],
        latent_dim=config["model"]["latent_dim"],
        image_size=config["model"]["image_size"]
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    # Training loop
    num_epochs = config["training"]["epochs"]
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in dataloader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            loss = model.loss(x, mu, logvar)  # Loss that includes reconstruction and KL divergence
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        
        # Log training loss to Neptune
        run["train/loss"].log(avg_loss, step=epoch)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Finish Neptune run
    run.stop()

if __name__ == "__main__":
    train()


