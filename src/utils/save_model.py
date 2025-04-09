import torch
import os
import copy

def save_model(logger, model, epoch, optimizer=None, config=None):

    save_dir = "/zhome/e9/c/186947/Bachelor/trained_models"
    
    model_copy = copy.deepcopy(model)
    model_copy.to("cpu")
    
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_copy.state_dict(),
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    save_path = os.path.join(save_dir, f"model_epoch_{epoch}.pth")
    torch.save(checkpoint, save_path)
        
    print(f"Checkpoint saved to {save_path}")