# src/train.py

import os
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import neptune.new as neptune

from models.vae import VAE  # Ensure your model is in src/models/vae.py

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
def create_dummy_data(config):
    # For instance, create random images of shape (batch_size, in_channels, image_size, image_size)
    num_samples = 1000
    channels = config["model"]["in_channels"]
    image_size = config["model"]["image_size"]
    dummy_images = torch.rand(num_samples, channels, image_size, image_size)
    dataset = TensorDataset(dummy_images)
    return DataLoader(dataset, batch_size=config["training"]["batch_size"], shuffle=True)

# ---------------------------
# Main Training Loop
# ---------------------------
def train():
    # Load configuration
    config = load_config()

    # Initialize Neptune
    run = neptune.init(
        project=config["experiment"]["neptune_project"],
        api_token=config["experiment"]["neptune_api_token"],
        name=config["experiment"]["name"],
        tags=["dummy-data", "vae"]
    )

    # Create dummy data loader
    dataloader = create_dummy_data(config)

    # Instantiate model, optimizer, etc.
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
            loss = model.loss(x, mu, logvar)  # using our loss_function that includes the reconstruction and KL terms
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        
        # Log metrics to Neptune
        run["train/loss"].log(avg_loss)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Finish Neptune run
    run.stop()

if __name__ == "__main__":
    train()
