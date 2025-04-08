import torch

def validate(model, val_loader, device, config=None):
    model.eval()
    total_val_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    with torch.no_grad():
        for batch, _ in val_loader:
            x = batch.to(device)
            recon_x, mu, logvar = model(x)
            recon_loss, kl_loss  = model.loss(x, mu, logvar)
            loss = recon_loss + kl_loss
            total_val_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
    average_val_loss = total_val_loss / (len(val_loader)*config["training"]["batch_size"])
    average_recon_loss = total_recon_loss / (len(val_loader)*config["training"]["batch_size"])
    average_kl_loss = total_kl_loss / (len(val_loader)*config["training"]["batch_size"])
    return average_val_loss, average_recon_loss, average_kl_loss

