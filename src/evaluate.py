import os
import torch

def validate(model, val_loader, device, config=None, epoch=None):
    """
    Runs validation, computes loss metrics, and saves latent codes with their IDs
    for later analysis.

    Args:
        model: The VAE model.
        val_loader: DataLoader yielding tuples (images, ids).
        device: The device (cpu or cuda).
        config: Configuration dict (used here for batch size).
        epoch: The current epoch number. If provided, latent codes and IDs are saved.

    Returns:
        average_val_loss, average_recon_loss, average_kl_loss
    """
    model.eval()
    total_val_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0

    all_latents = []   # list to accumulate latent codes (here we use mu)
    all_ids = []       # list to accumulate ids

    with torch.no_grad():
        for batch, ids in val_loader:
            x = batch.to(device)
            recon_x, mu, logvar = model(x)
            recon_loss, kl_loss  = model.loss(x, mu, logvar)
            loss = recon_loss + kl_loss

            total_val_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()

            # Store latent representations and the corresponding ids.
            all_latents.append(mu.cpu())
            all_ids.extend(ids)

    average_val_loss = total_val_loss / (len(val_loader) * config["training"]["batch_size"])
    average_recon_loss = total_recon_loss / (len(val_loader) * config["training"]["batch_size"])
    average_kl_loss = total_kl_loss / (len(val_loader) * config["training"]["batch_size"])

    # Save latent codes and IDs for later analysis if epoch is provided.
    if epoch is not None:
        # Concatenate latent codes from all batches into a single tensor.
        all_latents = torch.cat(all_latents, dim=0)
        output_folder = "latent_codes"
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, f"latent_epoch_{epoch}.pth")
        torch.save({"latent_codes": all_latents, "ids": all_ids}, output_path)
        print(f"Saved latent codes and IDs for epoch {epoch} to {output_path}")

    return average_val_loss, average_recon_loss, average_kl_loss
