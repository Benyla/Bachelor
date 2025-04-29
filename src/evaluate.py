import os
import torch

def validate(model, val_loader, device, config=None, epoch=None):
    """
    Runs validation, computes loss metrics, and optionally saves latent codes with IDs.

    Returns:
        Tuple: average_val_loss, average_recon_loss, average_kl_loss, average_adv_loss
    """
    model.eval()
    loss_acc = {"total": 0.0, "recon": 0.0, "kl": 0.0, "adv": 0.0}
    latents, ids_all = [], []

    batch_size = config["training"]["batch_size"]

    with torch.no_grad():
        for x, ids in val_loader:
            x = x.to(device)
            x_rec, mu, logvar = model(x)
            recon, kl, adv, _ = model.loss(x, mu, logvar)
            total = recon + model.beta * kl + adv

            # Accumulate losses
            loss_acc["total"] += total.item()
            loss_acc["recon"] += recon.item()
            loss_acc["kl"]    += kl.item()
            loss_acc["adv"]   += adv.item()

            # Collect latent codes and IDs
            latents.append(mu.cpu())
            ids_all.extend(ids)

    # Normalize losses by total number of samples
    scale = len(val_loader) * batch_size
    avg_loss = {k: v / scale for k, v in loss_acc.items()}

    # Save latent codes and IDs
    if epoch is not None:
        os.makedirs("latent_codes", exist_ok=True)
        output_filename = (
            f"latent_codes/VAE+_latent_epoch_{epoch}.pth"
            if config["model"].get("use_adv", False)
            else f"latent_codes/VAE_latent_epoch_{epoch}.pth"
        )
        torch.save({
            "latent_codes": torch.cat(latents, dim=0),
            "ids": ids_all
        }, output_filename)
        print(f"[Validation] Saved latent codes for epoch {epoch}")

    return avg_loss["total"], avg_loss["recon"], avg_loss["kl"], avg_loss["adv"]