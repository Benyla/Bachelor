import torch

def validate(model, logger, val_loader, device, global_step):
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for batch, _ in val_loader:
            x = batch.to(device)
            recon_x, mu, logvar = model(x)
            loss = model.loss(x, mu, logvar)
            total_val_loss += loss.item()
    average_val_loss = total_val_loss / len(val_loader)
    return average_val_loss

