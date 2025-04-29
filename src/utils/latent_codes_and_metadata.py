import os
import torch
import pandas as pd

def get_latent_and_metadata(config, epoch):
    """
    Loads precomputed latent codes and metadata for a given epoch.
    
    Args:
        config (dict): Must contain:
            - model.use_adv (bool)
            - latent_codes_dir (str, optional; default "latent_codes")
            - metadata_csv (str)
        epoch (int): Which epoch’s file to load (e.g. 20 → "VAE+_latent_epoch_20.pth")
    
    Returns:
        pd.DataFrame with columns ["id", "moa", "z0","z1",...]
    """
    # 1) Build filename based on whether you used adversarial
    latent_dir = config.get("latent_codes_dir", "latent_codes")
    use_adv    = config["model"].get("use_adv", False)
    prefix     = "VAE+" if use_adv else "VAE"
    fname      = f"{prefix}_latent_epoch_{epoch}.pth"
    path       = os.path.join(latent_dir, fname)
    
    # 2) Load the saved dictionary
    data = torch.load(path, map_location="cpu")
    latents = data["latent_codes"]    # shape (N, latent_dim)
    ids     = data["ids"]             # list of length N

    # 3) Load and clean metadata
    meta = pd.read_csv(config["metadata_csv"])
    # assume your IDs match the filename without “.npy”
    meta["Single_Cell_Image_Name"] = (
        meta["Single_Cell_Image_Name"]
            .astype(str)
            .str.replace(".npy", "", regex=False)
    )

    # 4) Build the DataFrame
    z_cols = [f"z{i}" for i in range(latents.shape[1])]
    df = pd.DataFrame(latents.numpy(), columns=z_cols)
    df["id"] = ids
    df = df.merge(
        meta[["Single_Cell_Image_Name", "moa"]],
        left_on="id", right_on="Single_Cell_Image_Name",
        how="left"
    ).drop(columns="Single_Cell_Image_Name")

    return df