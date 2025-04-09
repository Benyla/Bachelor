import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Ensure repo access

import yaml
import torch
import argparse
import time
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.VAE import VAE
from utils.data_loader import get_data, SingleCellDataset
from utils.neptuneLogger import NeptuneLogger
from torch.utils.data import Subset
from evaluate import validate
from utils.save_model import save_model


def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def train(config, logger, train_loader, val_loader):
    # Initialize model and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(
        in_channels=config["model"]["in_channels"],
        latent_dim=config["model"]["latent_dim"]
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config["training"].get("lr_scheduler_factor", 0.1),
        patience=config["training"].get("lr_scheduler_patience", 3),
        verbose=True
    )

    num_epochs = config["training"]["epochs"]
    
    for epoch in tqdm(range(num_epochs), desc="Training epochs"):
        start_time = time.time()
        model.train()
        train_loss_total = 0.0
        recon_loss_total = 0.0
        kl_loss_total = 0.0
        
        for batch_idx, (batch, ids) in enumerate(train_loader):
            x = batch.to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar = model(x)
            recon_loss, kl_loss = model.loss(x, mu, logvar)
            loss = recon_loss + kl_loss
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()
            recon_loss_total += recon_loss.item()
            kl_loss_total += kl_loss.item()
        
        avg_train_loss = train_loss_total / (len(train_loader)*config["training"]["batch_size"])
        avg_recon_loss = recon_loss_total / (len(train_loader)*config["training"]["batch_size"])
        avg_kl_loss = kl_loss_total / (len(train_loader)*config["training"]["batch_size"])

        avg_val_loss, average_recon_loss, average_kl_loss = validate(model, val_loader, device, config=config)

        logger.log_metrics({"train": avg_train_loss, "recon_loss": avg_recon_loss, "kl_loss": avg_kl_loss}, step=epoch, prefix="loss")
        logger.log_metrics({"val": avg_val_loss, "recon_loss": average_recon_loss, "kl_loss": average_kl_loss}, step=epoch, prefix="val_loss")
        logger.log_images(x, recon_x, step=epoch)

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        logger.log_metrics({"learning_rate": current_lr}, step=epoch)

        logger.log_time({"epoch_time": time.time() - start_time}, step=epoch)

        save_model(logger, model, epoch, optimizer=optimizer, config=config)
    
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
    val_dataset = SingleCellDataset(val_files)

    if config["data"]["test"] == True:
        train_dataset = Subset(train_dataset, list(range(2560)))
        val_dataset = Subset(val_dataset, list(range(256)))
        print('Training on subset of images (testing)')
        print(f"Subset size: {len(train_dataset)}")


    # drop_last=True ensures that the last incomplete batch is dropped (caused problems with loss visualization)
    train_loader = DataLoader(train_dataset, batch_size=config["training"]["batch_size"], shuffle=True, drop_last=True) 
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], shuffle=False, drop_last=True)
    
    # Start training
    train(config, logger, train_loader, val_loader)


if __name__ == "__main__":
    main()

