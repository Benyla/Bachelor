import os
import torch
import pandas as pd

def get_latent_and_metadata(config, epoch):
    # Build filename based on whether we used adversarial
    latent_dir = "latent_codes"
    use_adv    = config["model"].get("use_adv", False)
    latent_dim = config["model"].get("latent_dim", 256)
    prefix     = "VAE+" if use_adv else "VAE"
    fname      = f"{prefix}_{latent_dim}_latent_epoch_{epoch}.pth"
    path       = os.path.join(latent_dir, fname)
    
    # Load the saved dictionary - comes from evaluate.py
    data = torch.load(path, map_location="cpu")
    latents = data["latent_codes"]    
    ids     = data["ids"]            

    # Load and clean metadata
    meta = pd.read_csv(config["metadata_csv"])
    meta["Single_Cell_Image_Name"] = (
        meta["Single_Cell_Image_Name"]
            .astype(str)
            .str.replace(".npy", "", regex=False)
    )

    # Build the DataFrame
    z_cols = [f"z{i}" for i in range(latents.shape[1])]
    df = pd.DataFrame(latents.numpy(), columns=z_cols)
    df["id"] = ids
    df = df.merge(
        meta[["Single_Cell_Image_Name", "moa"]],
        left_on="id", right_on="Single_Cell_Image_Name",
        how="left"
    ).drop(columns="Single_Cell_Image_Name")

    return df