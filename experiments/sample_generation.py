#!/usr/bin/env python3
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import re
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from src.models.VAE import VAE
from src.utils.config_loader import load_config
from src.utils.data_loader import get_data, SingleCellDataset

def load_latest_model(model_dir, prefix=''):
    def extract_epoch(filename):
        match = re.search(r'epoch_(\d+)', filename)
        return int(match.group(1)) if match else -1
    ckpts = [
        f for f in os.listdir(model_dir)
        if f.endswith(".pth") and f.startswith(prefix)
    ]
    if not ckpts:
        raise FileNotFoundError(f"No .pth files in {model_dir} matching prefix '{prefix}'")
    latest = max(ckpts, key=extract_epoch)
    print(f"[Model] Loading checkpoint: {latest}")
    return os.path.join(model_dir, latest)


def generate_grid_variations(model, ref_img, sigma, grid_size=9, dims=None):
    """
    Given a single reference image (1xCxHxW), traverse two latent dimensions
    over a grid centered at the image's latent mean.
    Returns a Tensor of shape (grid_size, grid_size, C, H, W).
    """
    device = next(model.parameters()).device
    ref_img = ref_img.to(device)

    if dims is None:
        raise ValueError("Must provide dims to generate_grid_variations")

    with torch.no_grad():
        _, mu, _ = model.encode(ref_img)  # (1, D)
        mu = mu.squeeze(0)                # (D,)
        offsets = torch.linspace(-4*sigma, 4*sigma, steps=grid_size, device=device)
        zs = []
        for dx in offsets:
            for dy in offsets:
                z = mu.clone()
                z[dims[0]] += dx
                z[dims[1]] += dy
                zs.append(z)
        zs = torch.stack(zs, dim=0)      # (grid_size*grid_size, D)
        samples = model.decode(zs)       # (grid_size*grid_size, C, H, W)
        # reshape to grid
        return samples.view(grid_size, grid_size, *samples.shape[1:]).cpu()

def plot_grid_reference_and_samples(grid_samples, ref_img, save_path, dims=None):
    """
    Display and save a grid of images.
    grid_samples: Tensor of shape (G, G, C, H, W).
    """
    # Place original reference image in the center
    G = grid_samples.shape[0]
    center = G // 2
    orig = ref_img.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    grid_samples[center, center] = torch.from_numpy(orig).permute(2, 0, 1)

    C, H, W = grid_samples.shape[2:]
    fig, axes = plt.subplots(G, G, figsize=(2*G, 2*G))
    for i in range(G):
        for j in range(G):
            img = grid_samples[i, j].numpy().transpose(1, 2, 0)
            axes[i, j].imshow(img)
            axes[i, j].axis("off")

    # Add axis labels and title
    if dims is not None:
        fig.supxlabel(f'Offset in latent dim {dims[1]} (σ)', fontsize=12)
        fig.supylabel(f'Offset in latent dim {dims[0]} (σ)', fontsize=12)
    fig.suptitle('Latent Space Traversal Grid', fontsize=16)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path)
    print(f"[Plot] saved to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      required=True, help="YAML config path")
    parser.add_argument("--ref_idx",     type=int,   default=None, help="Global index of reference image")
    parser.add_argument("--sigma",       type=float, default=None, help="Std-dev around reference latent")
    args = parser.parse_args()

    # --- Load config --------------------------------------------------------
    cfg = load_config(args.config)
    sample_cfg = cfg.get("sample", {})

    # --- Prepare dataset ----------------------------------------------------
    _, val_files, _ = get_data()
    val_ds = SingleCellDataset(val_files)
    total = len(val_ds)

    # choose ref index
    ref_idx = args.ref_idx \
        if args.ref_idx is not None else sample_cfg.get("ref_idx", 0)
    if not (0 <= ref_idx < total):
        raise IndexError(f"ref_idx={ref_idx} out of range [0,{total})")

    ref_img, ref_id = val_ds[ref_idx]
    ref_img = ref_img.unsqueeze(0)  # add batch dim

    sigma = args.sigma \
        if args.sigma is not None else sample_cfg.get("sigma", 0.1)

    # --- Instantiate model --------------------------------------------------
    use_adv    = cfg["model"].get("use_adv", False)
    in_ch      = cfg["model"]["in_channels"]
    latent_dim = cfg["model"]["latent_dim"]
    beta       = cfg["model"].get("beta", 1.0)
    T          = cfg["model"].get("T", 2500)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare checkpoint path for loading weights into temp model
    model_dir  = cfg.get("paths", {}).get("model_dir", "trained_models")
    prefix     = f"VAE+_{latent_dim}_" if use_adv else f"VAE_{latent_dim}_"
    ckpt_path  = load_latest_model(model_dir, prefix)

    if use_adv:
        model_name = "VAE+_" + str(latent_dim)
    else:
        model_name = "VAE_" + str(latent_dim)

    # Determine top-2 latent dims by variance, with caching
    stats_path = os.path.join("experiments", f"{model_name}_latent_stats.pth")
    if os.path.exists(stats_path):
        stats = torch.load(stats_path)
        dims = stats["dims"]
    else:
        model_temp = VAE(in_ch, latent_dim, use_adv=use_adv).to(device)
        # Load trained weights into the temp model before encoding
        ckpt   = torch.load(ckpt_path, map_location=device)
        state  = ckpt.get("model_state_dict", ckpt)
        model_temp.load_state_dict(state)
        model_temp.eval()
        latents = []
        with torch.no_grad():
            for img, _ in val_ds:
                img = img.unsqueeze(0).to(device)
                _, mu, _ = model_temp.encode(img)
                latents.append(mu.squeeze(0).cpu())
        latents = torch.stack(latents, dim=0)
        variances = torch.var(latents, dim=0)
        dims = torch.topk(variances, 2).indices.tolist()
        os.makedirs("experiments", exist_ok=True)
        torch.save({"dims": dims}, stats_path)
        del model_temp
    print(f"Using latent dims for traversal: {dims}")

    model = VAE(in_ch, latent_dim, use_adv=use_adv).to(device)

    # --- Load latest checkpoint ---------------------------------------------
    model_dir   = cfg.get("paths", {}).get("model_dir", "trained_models")
    latent_dim  = cfg["model"]["latent_dim"]
    prefix      = f"VAE+_{latent_dim}_" if use_adv else f"VAE_{latent_dim}_"
    ckpt_path   = load_latest_model(model_dir, prefix)
    ckpt = torch.load(ckpt_path, map_location=device)
    # support both full-state and bare-state dicts
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()

    # --- Generate grid of latent traversals around reference image ----
    grid_size = 5
    grid_samples = generate_grid_variations(model, ref_img, sigma, grid_size=grid_size, dims=dims)

    # --- Plot + save grid ----
    prefix_plot = f"VAE+_{latent_dim}_near_traversal" if use_adv else f"VAE_{latent_dim}_near_traversal"
    out_dir = os.path.join("experiments", "plots")
    os.makedirs(out_dir, exist_ok=True)
    fname = f"{prefix_plot}_dims{dims[0]}-{dims[1]}_ref{ref_idx}_grid{grid_size}.png"
    save_path = os.path.join(out_dir, fname)
    plot_grid_reference_and_samples(grid_samples, ref_img, save_path, dims=dims)

if __name__ == "__main__":
    main()