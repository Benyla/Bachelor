import yaml
import torch
import os
import random
import torch.optim as optim
import tqdm
from torchvision.transforms.functional import to_pil_image
import neptune
import neptune.types
from data.mnist_dummy_data import load_mnist_data
from torch.utils.data import DataLoader
from models.VAE import VAE
from data_works import get_data, transform_cell_image, SingleCellDataset

# ---------------------------
# Load configuration
# ---------------------------
def load_config(config_path="config3.yaml"):
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
# Training function
# ---------------------------
def train():
    config = load_config() # Load configuration

    run = neptune.init_run( # Initialize Neptune run 
        project=config["experiment"]["neptune_project"],
        api_token = os.getenv("NEPTUNE_API_TOKEN"),
        name=config["experiment"]["name"],
        tags=["vae"]
    )

    run["parameters"] = config # Log the configuration parameters as metadata


    # < ---- Load data ---- >
    if config["data"]["dummy"] == True:
        train_loader = dummy_data(config)
        print("selecting dummy data")
    else:
        train_files, val_files, test_files = get_data()
        train_dataset = SingleCellDataset(train_files, transform=transform_cell_image)
        val_dataset = SingleCellDataset(val_files, transform=transform_cell_image)
        test_dataset = SingleCellDataset(test_files, transform=transform_cell_image)
        train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"], shuffle=False)


    # < ---- Init model and optimizer ---- >
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(
        in_channels=config["model"]["in_channels"],
        latent_dim=config["model"]["latent_dim"]
        ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    # < ---- Training loop ---- >
    num_epochs = config["training"]["epochs"]
    for epoch in tqdm(range(num_epochs)):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            x = batch.to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            loss = model.loss(x, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        
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


