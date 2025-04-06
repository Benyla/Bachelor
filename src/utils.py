import torch
import os
import copy

def save_model(logger, model, epoch, optimizer=None, config=None):
    """
    Save a checkpoint of the VAE model after an epoch.
    
    This function creates a deepcopy of the model, moves it to CPU,
    and saves a checkpoint that includes the model state, epoch, 
    and optionally the optimizer state (for resuming training).
    It then logs the saved checkpoint to Neptune as an artifact.
    
    Args:
        logger: The NeptuneLogger instance.
        model (torch.nn.Module): The model to save.
        epoch (int): The current epoch number.
        optimizer (torch.optim.Optimizer, optional): Optimizer to save.
        config (dict, optional): Additional configuration parameters.
    """
    # Create temporary directory if it doesn't exist
    save_dir = "/zhome/e9/c/186947/Bachelor/trained_models"
    
    # Create a deepcopy of the model and move it to CPU for safe saving
    model_copy = copy.deepcopy(model)
    model_copy.to("cpu")
    
    # Prepare a checkpoint dictionary
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model_copy.state_dict(),
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    
    # Save the checkpoint to a file
    save_path = os.path.join(save_dir, f"model_epoch_{epoch}.pth")
    torch.save(checkpoint, save_path)
    
    # Log the saved model as an artifact in Neptune
    logger.run[f"artifacts/model_epoch_{epoch}.pth"].upload(save_path)
    
    print(f"Checkpoint saved to {save_path} and logged to Neptune.")