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

def load_latest_model(model_dir):
    def extract_epoch(filename):
        match = re.search(r'epoch_(\d+)', filename)
        return int(match.group(1)) if match else -1

    ckpts = [f for f in os.listdir(model_dir) if f.endswith(".pth")]
    if not ckpts:
        raise FileNotFoundError(f"No .pth files in {model_dir}")
    latest = max(ckpts, key=extract_epoch)
    print(f"[Model] Loading checkpoint: {latest}")
    return os.path.join(model_dir, latest)

def generate_variations(model, ref_img, num_samples, sigma):
    """
    Given a single reference image (1xCxHxW), sample around its latent mean.
    Returns a Tensor of shape (num_samples, C, H, W).
    """
    device = next(model.parameters()).device
    ref_img = ref_img.to(device)
    with torch.no_grad():
        _, mu, _ = model.encode(ref_img)  # (1, D)
        eps = torch.randn(num_samples, mu.size(1), device=device) * sigma
        zs = mu + eps  # (num_samples, D)
        samples = model.decode(zs)  # (num_samples, C, H, W)
    return samples.cpu()

def plot_reference_and_samples(ref_img, samples, save_path):
    """
    Display one row: [ reference | sample1 | sample2 | ... ]
    """
    # combine into (1+N, C, H, W)
    grid = torch.cat([ref_img.cpu(), samples], dim=0).numpy()
    n = grid.shape[0]
    fig, axes = plt.subplots(1, n, figsize=(2*n, 2))
    for i, ax in enumerate(axes):
        img = np.transpose(grid[i], (1,2,0))
        ax.imshow(img)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[Plot] saved to {save_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      required=True, help="YAML config path")
    parser.add_argument("--ref_idx",     type=int,   default=None, help="Global index of reference image")
    parser.add_argument("--num_samples", type=int,   default=None, help="How many variations to sample")
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

    num_samples = args.num_samples \
        if args.num_samples is not None else sample_cfg.get("num_samples", 8)
    sigma = args.sigma \
        if args.sigma is not None else sample_cfg.get("sigma", 0.1)

    # --- Instantiate model --------------------------------------------------
    use_adv    = cfg["model"].get("use_adv", False)
    in_ch      = cfg["model"]["in_channels"]
    latent_dim = cfg["model"]["latent_dim"]
    beta       = cfg["model"].get("beta", 1.0)
    T          = cfg["model"].get("T", 2500)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VAE(in_ch, latent_dim, use_adv=use_adv).to(device)

    # --- Load latest checkpoint ---------------------------------------------
    model_dir = cfg.get("paths", {}).get("model_dir", "trained_models")
    ckpt_path = load_latest_model(model_dir)
    ckpt = torch.load(ckpt_path, map_location=device)
    # support both full-state and bare-state dicts
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.eval()

    # --- Generate variations ------------------------------------------------
    samples = generate_variations(model, ref_img, num_samples, sigma)

    # --- Plot + save --------------------------------------------------------
    prefix = "VAE+" if use_adv else "VAE"
    base   = os.path.splitext(os.path.basename(ckpt_path))[0]
    out_dir = sample_cfg.get("output_dir", "generated_samples")
    os.makedirs(out_dir, exist_ok=True)

    fname = f"{prefix}_{base}_ref{ref_id}_σ{sigma}_n{num_samples}.png"
    save_path = os.path.join(out_dir, fname)
    plot_reference_and_samples(ref_img, samples, save_path)

if __name__ == "__main__":
    main()