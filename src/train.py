import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))  # Ensure repo access

import yaml
import torch
import argparse
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

    # < -- Initialize model -- >
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(
        in_channels=config["model"]["in_channels"],
        latent_dim=config["model"]["latent_dim"],
        use_adv=config["model"].get("use_adv", False),
        beta=config["model"].get("beta", 1.0),
        T=config["model"].get("T", 2500),
        overfit=config["model"].get("overfit", False)
    ).to(device)

    # < -- log all model hyperparameters -- >
    hyperparams_str = (
        f"use_adv={model.use_adv}, overfit={model.overfit}\n"
        f"beta={model.beta}, T={model.T}\n"
        f"latent_dim={model.latent_dim}, in_channels={model.in_channels}\n"
        f"device={device}\n"
    )
    logger.run["model/parameters"] = hyperparams_str


    # < -- Separate parameters for VAE and Discriminator and create optimizers -- >
    vae_params = (
        list(model.encoder.parameters()) +
        list(model.fc_mu.parameters()) + list(model.fc_logvar.parameters()) +
        list(model.decoder_input.parameters()) + list(model.decoder.parameters())
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
        model.train()
        epoch_losses = {"total":0, "recon":0, "kl":0, "adv":0}
        clipping_counter = {"vae": 0, "d": 0}

        for x, _ in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):

            x = x.to(device)
            x_rec, mu, logvar, _ = model(x)

            # < -- Discriminator update -- >
            if config["model"].get("use_adv", False):
                optimizer_D.zero_grad()
                d_loss = model.loss_discriminator(x, x_rec)
                d_loss.backward()
                
                # To log the clip rate for discriminator
                norm_d_grad = clip_grad_norm_(model.discriminator.parameters(), max_norm=1.0)
                if norm_d_grad > 1.0:
                    clipping_counter["d"] += 1
                
                clip_grad_norm_(model.discriminator.parameters(), max_norm=1.0) # clip
                optimizer_D.step()

            # < -- VAE update -- >
            recon, kl, adv_fm_loss = model.loss(x, x_rec, mu, logvar)
            loss = recon + model.beta * kl + adv_fm_loss # adv_fm_loss is scaled by _gamma method if use_adv=True otherwise 0

            optimizer_VAE.zero_grad()
            loss.backward()

            # To log the clip rate for vae
            norm_vae_grad = clip_grad_norm_(vae_params, max_norm=1.0)
            if norm_vae_grad > 1.0:
                clipping_counter["vae"] += 1

            clip_grad_norm_(vae_params, max_norm=1.0) # clip
            optimizer_VAE.step()

            # Track losses
            epoch_losses["total"] += loss.item()
            epoch_losses["recon"] += recon.item()
            epoch_losses["kl"]    += kl.item()
            epoch_losses["adv"]   += (adv_fm_loss.item() if isinstance(adv_fm_loss, torch.Tensor) else adv_fm_loss)

        # Compute averages
        avg = lambda k: epoch_losses[k] / (len(train_loader) * batch_size)
        val_loss, val_recon, val_kl, val_adv, val_x, val_x_rec = validate(model, val_loader, device, config=config, epoch=epoch)

        # < -- Log metrics and save model -- >
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
        logger.log_metrics({"train/clipping_vae": clipping_counter["vae"], "train/clipping_d": clipping_counter["d"]}, step=epoch)
        logger.log_images(x, x_rec, step=epoch, prefix="train")
        logger.log_images(val_x, val_x_rec, step=epoch, prefix="val")
        # Log progressive feature-matching weights for each discriminator layer
        if config["model"].get("use_adv", False):
            num_layers = len(model.discriminator.layers)
            gamma_dict = {}
            for i in range(num_layers):
                gamma_value = model._gamma(i)
                gamma_dict[f"train/gamma_layer_{i}"] = gamma_value
            logger.log_metrics(gamma_dict, step=epoch)
        save_model(logger, model, epoch, optimizer=optimizer_VAE, d_optimizer=optimizer_D if config["model"].get("use_adv", False) else None, config=config)

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
        train_dataset = Subset(train_dataset, list(range(1280)))
        val_dataset   = Subset(val_dataset,   list(range(64)))
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
