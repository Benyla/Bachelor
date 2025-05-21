import torch
import os
import copy
import re

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
    use_adv = config["model"].get("use_adv", False)

    # Define filename pattern
    prefix = "VAE+" if use_adv else "VAE"
    output_filename = f"{prefix}_{latent_dim}_epoch_{epoch}.pth"
    save_path = os.path.join(save_dir, output_filename)

    # Clean up older checkpoints not on a multiple of 10
    for fname in os.listdir(save_dir):
        if fname.startswith(prefix) and fname.endswith(".pth"):
            match = re.search(r"epoch_(\d+)", fname)
            if match:
                ep = int(match.group(1))
                if ep not in {epoch, 10, 20, 30, 40, 49}:
                    path_to_delete = os.path.join(save_dir, fname)
                    try:
                        os.remove(path_to_delete)
                        print(f"Deleted old checkpoint: {path_to_delete}")
                    except Exception as e:
                        print(f"Could not delete {path_to_delete}: {e}")

    torch.save(checkpoint, save_path) 
    print(f"Checkpoint saved to {save_path}")
