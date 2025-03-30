import yaml
import torch
import os
import random
import torch.optim as optim
from torchvision.transforms.functional import to_pil_image
import neptune.new as neptune
import neptune.types
from data.mnist_dummy_data import load_mnist_data
from torch.utils.data import DataLoader
from data.Cell_data import CellDataset
from models.VAE import VAE

# ---------------------------
# Load configuration
# ---------------------------
def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

# ---------------------------
# Dummy Data function
# ---------------------------
def dummy_data(config):
    batch_size = config["training"]["batch_size"]
    return load_mnist_data(batch_size)

# ---------------------------
# Cell data function
# ---------------------------
def get_dataloader(folder_path, batch_size=32):
    dataset = CellDataset(folder_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader


# ---------------------------
# Training function
# ---------------------------
def train():
    config = load_config() # Load configuration

    run = neptune.init_run( # Initialize Neptune run 
        project=config["experiment"]["neptune_project"],
        api_token = os.getenv("NEPTUNE_API_TOKEN"),
        name=config["experiment"]["name"],
        tags=["dummy-data", "vae"]
    )

    run["parameters"] = config # Log the configuration parameters as metadata


    # < ---- Load data ---- >
    if config["data"]["dummy"] == True:
        dataloader = dummy_data(config)
    else:
        dataloader = get_dataloader(
            folder_path=config["data"]["folder_path"],
            batch_size=config["training"]["batch_size"]
        )


    # < ---- Init model and optimizer ---- >
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(
        in_channels=config["model"]["in_channels"],
        latent_dim=config["model"]["latent_dim"],
        image_size=config["model"]["image_size"]
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    # < ---- Training loop ---- >
    num_epochs = config["training"]["epochs"]
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for batch in dataloader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            loss = model.loss(x, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        
        # < ---- Log training loss to Neptune ---- >
        run["train/loss"].log(avg_loss, step=epoch)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # < ---- Log origional and reconstruction to Neptune ---- >
        # Random sample from each epoch
        idx = random.randint(0, x.size(0) - 1)
        original = x[idx].cpu()
        reconstructed = recon_x[idx].detach().cpu()
        run[f"visuals/original_epoch_{epoch}"] = neptune.types.File.as_image(to_pil_image(original))
        run[f"visuals/reconstruction_epoch_{epoch}"] = neptune.types.File.as_image(to_pil_image(reconstructed))

    run.stop()

if __name__ == "__main__":
    train()


