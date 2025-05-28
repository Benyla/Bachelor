#!/usr/bin/env python3
"""
latent_traversal.py

Interpolate in the VAE latent space from a control cell to a target class centroid,
generate intermediate images, and plot the sequence.
"""
import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset
from src.models.VAE import VAE
from src.utils.data_loader import get_data, SingleCellDataset
from src.utils.config_loader import load_config
from src.utils.latent_codes_and_metadata import get_latent_and_metadata

def decode_batch(model, zs, device):
    """
    Decode a numpy array of latent vectors (N, latent_dim) to torch images (N, C, H, W).
    """
    model.eval()
    zs = torch.from_numpy(zs).to(device)
    with torch.no_grad():
        recon = model.decode(zs)
    return recon.cpu().numpy()

def plot_interpolation(images, output, prefix):
    """
    Plot a sequence of images in a row.
    images: numpy array (N, C, H, W)
    """
    n = len(images)
    images = np.transpose(images, (0, 2, 3, 1))  # (N, H, W, C)
    rows, cols = 4, 5
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

    idx = 0
    for r in range(rows):
        for c in range(cols):
            ax = axes[r, c]
            if idx < n:
                img = images[idx]
                ax.imshow(img)
                ax.set_title(f'{idx+1}/{n}')
            else:
                ax.axis('off')
            ax.axis('off')
            idx += 1

    os.makedirs(output, exist_ok=True)
    outpath = os.path.join(output, f'{prefix}_traversal_VAE.png')
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    print(f'[INFO] Saved interpolation figure to {outpath}')


def main():
    parser = argparse.ArgumentParser(description='Latent traversal from control to target class')
    parser.add_argument('--config', type=str, required=True, help='YAML config path')
    parser.add_argument('--model-path', type=str, required=True, help='VAE checkpoint .pth')
    parser.add_argument('--control-class', type=str, required=True, help='MOA label for control cells')
    parser.add_argument('--target-class', type=str, required=True, help='MOA label for target cells')
    parser.add_argument('--steps', type=int, default=20, help='Number of interpolation steps')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for encoding')
    parser.add_argument('--output', type=str, default='experiments/traversal', help='Output directory')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load config & set checkpoint
    config = load_config(args.config)
    config['model']['checkpoint_path'] = args.model_path
    use_adv = config['model'].get('use_adv', False)
    # add metadata path
    # assume config contains metadata_csv key or override here
    if 'metadata_csv' not in config:
        config['metadata_csv'] = os.path.expanduser('~/data/metadata.csv')  # adjust as needed

    # prepare validation loader
    _, val_files, _ = get_data()
    val_dataset = SingleCellDataset(val_files)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # get all latents and metadata
    df = get_latent_and_metadata(config, 49)

    # select control latent (first of control class)
    ctrl_df = df[df['moa'] == args.control_class]
    if ctrl_df.empty:
        raise ValueError(f"No samples found for control class '{args.control_class}'")
    z_ctrl = ctrl_df.iloc[5][[c for c in df.columns if c.startswith('z')]].values

    # Select target latent (first of target class)
    tgt_df = df[df['moa'] == args.target_class]
    if tgt_df.empty:
        raise ValueError(f"No samples found for target class '{args.target_class}'")
    z_tgt = tgt_df.iloc[0][[c for c in df.columns if c.startswith('z')]].values

    # interpolate
    alphas = np.linspace(0, 1, args.steps)
    z_interp = np.array([(1 - a) * z_ctrl + a * z_tgt for a in alphas], dtype=np.float32)

    # decode
    model = VAE(in_channels=config['model']['in_channels'], latent_dim=config['model']['latent_dim'], use_adv=use_adv).to(device)
    ckpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    imgs = decode_batch(model, z_interp, device)

    # plot
    model_base = os.path.splitext(os.path.basename(args.model_path))[0]
    plot_interpolation(imgs, args.output, f"{model_base}_{args.control_class}_to_{args.target_class}")

if __name__ == '__main__':
    main()
