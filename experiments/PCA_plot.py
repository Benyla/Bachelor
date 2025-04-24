import argparse
import yaml
import torch
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from src.models.VAE import VAE
from src.utils.data_loader import get_data, SingleCellDataset
from torch.utils.data import DataLoader
from src.utils.config_loader import load_config

def get_latent_codes_and_run_PCA(config: dict, val_loader: DataLoader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model architecture and weights
    model = VAE(
        in_channels=config["model"]["in_channels"],
        latent_dim=config["model"]["latent_dim"]
    ).to(device)
    ckpt = torch.load(config["model"]["checkpoint_path"], map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Collect latent codes and IDs
    all_latents = []
    all_ids = []
    with torch.no_grad():
        for batch, ids in val_loader:
            batch = batch.to(device)
            # Use the encoder to get the latent mean vector
            z, mu, logvar = model.encode(batch, return_stats=True)
            all_latents.append(mu.cpu())
            all_ids.extend(ids)

    latents = torch.cat(all_latents, dim=0).numpy()  # shape: (N_samples, latent_dim)
    print(f"[DEBUG] Latent matrix shape for PCA: {latents.shape}")

    # Load metadata and merge with latent codes
    meta = pd.read_csv("/zhome/70/5/14854/nobackup/deeplearningf22/bbbc021/singlecell/metadata.csv")
    z_columns = [f"z{i}" for i in range(latents.shape[1])]
    df_latent = pd.DataFrame(latents, columns=z_columns)
    df_latent["Multi_Cell_Image_Name"] = all_ids

    # Force both columns to be strings so merge works
    df_latent["Multi_Cell_Image_Name"] = df_latent["Multi_Cell_Image_Name"].astype(str)
    meta["Multi_Cell_Image_Name"] = meta["Multi_Cell_Image_Name"].astype(str)

    df = df_latent.merge(
        meta[["Multi_Cell_Image_Name", "moa"]],
        on="Multi_Cell_Image_Name",
        how="left"
    )
    if df["moa"].isnull().any():
        print("Warning: some IDs had no MOA in metadata.csv")
    missing_moa_count = df["moa"].isnull().sum()
    print(f"[DEBUG] Number of datapoints with missing MOA: {missing_moa_count}")

    # Run PCA down to 2 components
    pca = PCA(n_components=2)
    pcs = pca.fit_transform(df[z_columns])
    df["PC1"] = pcs[:, 0]
    df["PC2"] = pcs[:, 1]

    # Plot scatter colored by MOA
    moas = df["moa"].astype("category")
    df["moa_code"] = moas.cat.codes
    cmap = plt.get_cmap("tab20", len(moas.cat.categories))

    plt.figure(figsize=(8, 6))
    plt.scatter(
        df["PC1"], df["PC2"],
        c=df["moa_code"].to_numpy(),
        cmap=cmap,
        s=10, alpha=0.8
    )
    # Legend entries per MOA
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=cmap(i), markersize=6)
        for i in range(len(moas.cat.categories))
    ]
    plt.legend(handles, moas.cat.categories, title="MOA", bbox_to_anchor=(1, 1))
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("2D PCA of latent space\ncolored by MOA")
    plt.tight_layout()
    plt.savefig("experiments/plots/PCA_plot.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Run PCA on VAE latent space and visualize by MOA.")
    parser.add_argument("--config",       type=str, required=True, help="Path to YAML config file")
    parser.add_argument("--model-path",   type=str, required=True, help="Path to trained model checkpoint (pth)")
    args = parser.parse_args()

    # Load config and inject paths
    config = load_config(args.config)
    config["model"]["checkpoint_path"] = args.model_path

    # Prepare validation DataLoader
    _, val_files, _ = get_data()
    val_dataset = SingleCellDataset(val_files)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        drop_last=False
    )

    # Execute PCA pipeline
    get_latent_codes_and_run_PCA(config, val_loader)


if __name__ == "__main__":
    main()
