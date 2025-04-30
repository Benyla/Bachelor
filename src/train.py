import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Ensure repo access

import yaml
import torch
import argparse, time
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from models.VAE import VAE
from utils.data_loader import get_data, SingleCellDataset
from utils.neptuneLogger import NeptuneLogger
from evaluate import validate
from utils.save_model import save_model
from utils.config_loader import load_config

def train(config, logger, train_loader, val_loader):
    # Initialize device and model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(
        in_channels=config["model"]["in_channels"],
        latent_dim=config["model"]["latent_dim"],
        use_adv=config["model"].get("use_adv", False),
        beta=config["model"].get("beta", 1.0),
        T=config["model"].get("T", 2500),
        overfit=config["model"].get("overfit", False)
    ).to(device)

    # Separate parameters for VAE and Discriminator
    vae_params = (
        list(model.enc1.parameters()) +
        list(model.enc2.parameters()) +
        list(model.enc3.parameters()) +
        list(model.enc4.parameters()) +
        list(model.fc_mu.parameters()) +
        list(model.fc_logvar.parameters()) +
        list(model.decoder_input.parameters()) +
        list(model.up1.parameters()) +
        list(model.up2.parameters()) +
        list(model.up3.parameters()) +
        list(model.up4.parameters())
    )
    optimizer_VAE = optim.Adam(
        vae_params,
        lr=config["training"]["lr_VAE"],
    )

    if config["model"].get("use_adv", False):
        optimizer_D = optim.SGD(
            model.discriminator.parameters(),
            lr=config["training"]["lr_D"],
            momentum=0.9
        )

    batch_size = config["training"]["batch_size"]

    # Training loop
    for epoch in range(config["training"]["epochs"]):
        start_time = time.time()
        model.train()
        epoch_losses = {"total":0, "recon":0, "kl":0, "adv":0}

        for x, _ in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
            x = x.to(device)

            # Forward & single sample
            x_rec, mu, logvar, _ = model(x)

            # ----- Discriminator update -----
            if config["model"].get("use_adv", False):
                optimizer_D.zero_grad()
                # Only classification loss
                d_loss = model.loss_discriminator(x, x_rec)
                d_loss.backward()
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer_D.step()

            # ----- VAE update -----
            recon, kl, adv_fm_loss = model.loss_generator(x, x_rec, mu, logvar)
            loss = recon + model.get_beta() * kl + adv_fm_loss

            optimizer_VAE.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer_VAE.step()

            # Track losses
            epoch_losses["total"] += loss.item()
            epoch_losses["recon"] += recon.item()
            epoch_losses["kl"]    += kl.item()
            epoch_losses["adv"]   += (adv_fm_loss.item() if isinstance(adv_fm_loss, torch.Tensor) else adv_fm_loss)

        # Compute averages
        avg = lambda k: epoch_losses[k] / (len(train_loader) * batch_size)
        val_loss, val_recon, val_kl, val_adv, val_x, val_x_rec = validate(model, val_loader, device, config=config, epoch=epoch)

        # Log metrics
        log_dict = {
            "train/total": avg("total"),
            "train/recon": avg("recon"),
            "train/kl":    avg("kl"),
            "val/total":   val_loss,
            "val/recon":   val_recon,
            "val/kl":      val_kl,
        }
        if config["model"].get("use_adv", False):
            log_dict.update({"train/adv": avg("adv"), "val/adv": val_adv})

        logger.log_metrics(log_dict, step=epoch)
        logger.log_images(x, x_rec, step=epoch, prefix="train")
        logger.log_images(val_x, val_x_rec, step=epoch, prefix="val")
        save_model(logger, model, epoch, optimizer=optimizer_VAE, config=config)

    logger.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    config = load_config(args.config)
    logger = NeptuneLogger(config)

    # Load data
    train_files, val_files, test_files = get_data()
    train_dataset = SingleCellDataset(train_files)
    val_dataset   = SingleCellDataset(val_files)

    if config["data"]["test"]:
        train_dataset = Subset(train_dataset, list(range(64)))
        val_dataset   = Subset(val_dataset,   list(range(8)))
        print('Training on subset of images (testing)')
        print(f"Subset size: {len(train_dataset)}")

    train_loader = DataLoader(train_dataset,
                              batch_size=config["training"]["batch_size"],
                              shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_dataset,
                              batch_size=config["training"]["batch_size"],
                              shuffle=False, drop_last=True)

    train(config, logger, train_loader, val_loader)

if __name__ == "__main__":
    main()
