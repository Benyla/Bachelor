import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Ensure repo access

import yaml
import torch
import argparse
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.VAE import VAE
from data_works import get_data, SingleCellDataset
from neptuneLogger import NeptuneLogger
from torch.utils.data import Subset


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def train(config, logger, train_loader):
    # Initialize model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(
        in_channels=config["model"]["in_channels"],
        latent_dim=config["model"]["latent_dim"]
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    num_epochs = config["training"]["epochs"]
    logging_frequency = config["training"].get("logging_frequency", 10)
    
    for epoch in tqdm(range(num_epochs), desc="Training epochs"):
        model.train()
        
        for batch_idx, (batch, ids) in enumerate(train_loader):
            x = batch.to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            loss = model.loss(x, mu, logvar)
            loss.backward()
            optimizer.step()
            
            global_step = epoch * len(train_loader) + batch_idx
            logger.log_loss(loss.item(), step=global_step, kind="batch")

        # Log the original and reconstructed images as a combined figure
        logger.log_images(x, recon_x, step=epoch)
    
    logger.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    
    # Load configuration from file
    config = load_config(args.config)
    
    # Initialize Neptune logger
    logger = NeptuneLogger(config)
    
    # Load data
    train_files, val_files, test_files = get_data()
    train_dataset = SingleCellDataset(train_files)
    if config["data"]["test"] == True:
        train_dataset = Subset(train_dataset, list(range(10000)))
        print('Training on subset of images (testing)')
        print(f"Subset size: {len(train_dataset)}")


    # drop_last=True ensures that the last incomplete batch is dropped (caused problems with loss visualization)
    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True, drop_last=True) 
    
    # Start training
    train(config, logger, train_loader)


if __name__ == "__main__":
    main()

