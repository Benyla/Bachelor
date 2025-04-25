import torch
from src.models.VAE import VAE
import pandas as pd

def get_latent_and_metadata(config, val_loader, device=None):
    """
    Encodes all validation samples to latents and returns a DataFrame:
    ['id', 'moa', 'z0', 'z1', ...]
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = VAE(
        in_channels=config["model"]["in_channels"],
        latent_dim=config["model"]["latent_dim"]
    ).to(device)
    ckpt = torch.load(config["model"]["checkpoint_path"], map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Encode all data
    all_mu, all_ids = [], []
    with torch.no_grad():
        for batch, ids in val_loader:
            batch = batch.to(device)
            z, mu, logvar = model.encode(batch, return_stats=True)
            all_mu.append(mu.cpu())
            all_ids.extend(ids)
    latents = torch.cat(all_mu, dim=0).numpy()

    # Load metadata
    metadata_path = config.get("metadata_csv", "/path/to/metadata.csv")
    meta = pd.read_csv(metadata_path)
    meta["Single_Cell_Image_Name"] = meta["Single_Cell_Image_Name"].astype(str).str.replace(".npy", "", regex=False)

    # Build dataframe
    z_cols = [f"z{i}" for i in range(latents.shape[1])]
    df = pd.DataFrame(latents, columns=z_cols)
    df["id"] = all_ids
    df = df.merge(meta[["Single_Cell_Image_Name", "moa"]], left_on="id", right_on="Single_Cell_Image_Name", how="left")
    return df