import os
import torch
import re

def validate(model, val_loader, device, config=None, epoch=None):

    model.eval()
    loss_acc = {"total": 0.0, "recon": 0.0, "kl": 0.0, "adv": 0.0}
    latents, ids_all = [], []

    batch_size = config["training"]["batch_size"]

    with torch.no_grad():
        for x, ids in val_loader:
            x = x.to(device)
            x_rec, mu, logvar, z = model(x)
            recon, kl, adv = model.loss(x, x_rec, mu, logvar)
            total = recon + model.beta * kl + adv

            # Accumulate losses
            loss_acc["total"] += total.item()
            loss_acc["recon"] += recon.item()
            loss_acc["kl"] += kl.item()
            loss_acc["adv"] += adv.item() if isinstance(adv, torch.Tensor) else adv

            # Collect latent codes and IDs
            latents.append(mu.cpu())
            ids_all.extend(ids)

    # Normalize losses by total number of samples
    scale = len(val_loader) * batch_size
    avg_loss = {k: v / scale for k, v in loss_acc.items()}

    latent_dim = config["model"]["latent_dim"]
    use_adv = config["model"].get("use_adv", False)
    prefix = "VAE+" if use_adv else "VAE"
    filename = f"{prefix}_{latent_dim}_latent_epoch_{epoch}.pth"
    output_path = os.path.join("latent_codes", filename)

    # Delete older latent codes, keeping only current and milestones
    for fname in os.listdir("latent_codes"):
        if fname.startswith(prefix) and fname.endswith(".pth"):
            match = re.search(r"epoch_(\d+)", fname)
            if match:
                ep = int(match.group(1))
                if ep not in {epoch, 10, 20, 30, 40, 49}:
                    path_to_delete = os.path.join("latent_codes", fname)
                    try:
                        os.remove(path_to_delete)
                        print(f"Deleted old latent code file: {path_to_delete}")
                    except Exception as e:
                        print(f"Could not delete {path_to_delete}: {e}")

    # Save latent codes and IDs
    if epoch is not None:
        os.makedirs("latent_codes", exist_ok=True)

        torch.save({
            "latent_codes": torch.cat(latents, dim=0),
            "ids": ids_all
        }, output_path)
        print(f"[Validation] Saved latent codes for epoch {epoch}")

    return avg_loss["total"], avg_loss["recon"], avg_loss["kl"], avg_loss["adv"], x, x_rec