import torch
import os
import copy

def save_model(logger, model, epoch, optimizer=None, d_optimizer=None, config=None):

    save_dir = "/zhome/e9/c/186947/Bachelor/trained_models"
    
    model_copy = copy.deepcopy(model)
    model_copy.to("cpu")
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_copy.state_dict(),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if d_optimizer is not None and config["model"].get("use_adv", False):
        checkpoint["d_optimizer_state_dict"] = d_optimizer.state_dict()

    latent_dim = config["model"].get("latent_dim", 256)

    output_filename = (
        f"VAE+_{latent_dim}_epoch_{epoch}.pth"
        if config["model"].get("use_adv", False)
        else f"VAE_{latent_dim}_epoch_{epoch}.pth"
    )

    save_path = os.path.join(save_dir, output_filename)
    torch.save(checkpoint, save_path)
        
    print(f"Checkpoint saved to {save_path}")