#!/usr/bin/env python3
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
import re
import argparse
import torch
import matplotlib.pyplot as plt
from src.utils.data_loader import get_data
from src.models.VAE import VAE

def load_image_tensor(path, device):
    data = torch.load(path, map_location=device)
    if isinstance(data, torch.Tensor):
        img = data
    elif isinstance(data, dict):
        img = data.get('image')
        if img is None:
            raise ValueError(f"No 'image' key in data dict from {path}")
    else:
        raise ValueError(f"Unsupported data format in {path}")
    return img

def parse_model_name(model_str):
    # Decide adversarial flag by presence of '+' in the prefix
    use_adv = model_str.startswith("VAE+")
    # Extract latent dimension (first number after '_')
    m = re.search(r'_(\d+)', model_str)
    if not m:
        raise ValueError(f"Cannot parse latent dimension from model name: {model_str}")
    latent_dim = int(m.group(1))
    return use_adv, latent_dim

def main():
    p = argparse.ArgumentParser(
        description="Plot originals vs. reconstructions for selected val-set images"
    )
    p.add_argument(
        "--model", required=True,
        help="Checkpoint filename in trained_models/, e.g. VAE+_128_epoch_49.pth"
    )
    p.add_argument(
        "--indices", type=int, nargs="+", required=True,
        help="List of val-set indices, e.g. --indices 0 5 10 15 20"
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load validation file paths
    _, val_files, _ = get_data()

    # 2) Validate indices
    for idx in args.indices:
        if idx < 0 or idx >= len(val_files):
            raise IndexError(f"Index {idx} is out of bounds (0–{len(val_files)-1})")

    # 3) Instantiate & load model
    use_adv, latent_dim = parse_model_name(args.model)
    ckpt = os.path.join("trained_models", args.model)
    state = torch.load(ckpt, map_location=device)
    model = VAE(in_channels=3, latent_dim=latent_dim, use_adv=use_adv)
    sd = state.get("model_state_dict", state)
    model.load_state_dict(sd)
    model.to(device).eval()

    # 4) Load images & compute reconstructions
    originals, recons = [], []
    for idx in args.indices:
        path = val_files[idx]
        img = load_image_tensor(path, device)
        originals.append(img.cpu().squeeze())
        with torch.no_grad():
            rec, mu, logvar, _ = model(img.unsqueeze(0).to(device))
        recons.append(rec.cpu().squeeze())

    # 5) Plot grid
    n = len(args.indices)
    fig, axes = plt.subplots(2, n, figsize=(n * 3, 2 * 3))
    # Add overall title
    fig.suptitle(f"Reconstruction of {n} images", fontsize=16)
    axes[0, 0].set_ylabel('Original', rotation=90, fontsize=12, labelpad=10)
    axes[1, 0].set_ylabel('Reconstructed', rotation=90, fontsize=12, labelpad=10)
    for col, idx in enumerate(args.indices):
        # Original
        orig = originals[col]
        disp_o = orig.permute(1,2,0) if orig.ndim == 3 else orig
        axo = axes[0, col]
        axo.imshow(disp_o, cmap="gray" if disp_o.ndim == 2 else None)
        axo.set_xticks([]); axo.set_yticks([])
        for spine in axo.spines.values(): spine.set_visible(False)

        # Reconstruction
        rec = recons[col]
        disp_r = rec.permute(1,2,0) if rec.ndim == 3 else rec
        axr = axes[1, col]
        axr.imshow(disp_r, cmap="gray" if disp_r.ndim == 2 else None)
        axr.set_xticks([]); axr.set_yticks([])
        for spine in axr.spines.values(): spine.set_visible(False)

    # 6) Tweak spacing
    os.makedirs("experiments/plots", exist_ok=True)
    out_path = os.path.join("experiments/plots", "reconstructions.png")
    # Apply spacing like test.py
    plt.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.05, wspace=0.1, hspace=0.1)
    plt.savefig(out_path, dpi=300)
    print(f"Saved figure to {out_path}")

if __name__ == "__main__":
    main()