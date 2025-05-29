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

def plot_interpolation_and_latent_changes(images, zs, z_ctrl, z_tgt, output, prefix):
    """
    Plot interpolation images (2x5) and latent speed/distance metrics in one figure.
    """
    n = len(images)
    images = np.transpose(images, (0, 2, 3, 1))  # (N, H, W, C)
    
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 2, 1])
    
    # Create a grid for interpolation images
    interp_gs = gridspec.GridSpecFromSubplotSpec(2, 5, subplot_spec=gs[:2])
    for idx in range(n):
        ax = fig.add_subplot(interp_gs[idx // 5, idx % 5])
        img = images[idx]
        ax.imshow(img)
        ax.set_title(f'Step {idx+1}/{n}', fontsize=12)
        ax.axis('off')

    # Plot latent speed and relative distances
    ax_metrics = fig.add_subplot(gs[2])
    steps = np.arange(1, n+1)
    # compute speed between consecutive z's
    speeds = np.linalg.norm(zs[1:] - zs[:-1], axis=1)
    speeds = np.concatenate([[0.0], speeds])
    # compute distances to control and target
    dist_ctrl = np.linalg.norm(zs - np.expand_dims(z_ctrl, 0), axis=1)
    dist_tgt = np.linalg.norm(zs - np.expand_dims(z_tgt, 0), axis=1)
    ax_metrics.plot(steps, speeds, '-o', label='Latent Speed')
    ax_metrics.plot(steps, dist_ctrl, '-s', label='Distance to Control')
    ax_metrics.plot(steps, dist_tgt, '-^', label='Distance to Target')
    ax_metrics.set_xlabel('Step')
    ax_metrics.set_ylabel('Value')
    ax_metrics.set_title('Latent Speed & Distances')
    ax_metrics.legend()

    fig.suptitle('Latent Space Traversal', fontsize=16)
    
    os.makedirs(output, exist_ok=True)
    outpath = os.path.join(output, f'{prefix}_traversal_and_latent_changes.png')
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    print(f'[INFO] Saved combined figure to {outpath}')


def main():
    parser = argparse.ArgumentParser(description='Latent traversal from control to target class')
    parser.add_argument('--config', type=str, required=True, help='YAML config path')
    parser.add_argument('--model-path', type=str, required=True, help='VAE checkpoint .pth')
    parser.add_argument('--control-class', type=str, required=True, help='MOA label for control cells')
    parser.add_argument('--target-class', type=str, required=True, help='MOA label for target cells')
    parser.add_argument('--steps', type=int, default=10, help='Number of interpolation steps')
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
    z_ctrl = ctrl_df.iloc[70][[c for c in df.columns if c.startswith('z')]].values

    # Select target latent (first of target class)
    tgt_df = df[df['moa'] == args.target_class]
    if tgt_df.empty:
        raise ValueError(f"No samples found for target class '{args.target_class}'")
    z_tgt = tgt_df.iloc[23][[c for c in df.columns if c.startswith('z')]].values

    # interpolate
    alphas = np.linspace(0, 1, args.steps)
    z_interp = np.array([(1 - a) * z_ctrl + a * z_tgt for a in alphas], dtype=np.float32)

    # decode
    model = VAE(in_channels=config['model']['in_channels'], latent_dim=config['model']['latent_dim'], use_adv=use_adv).to(device)
    ckpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    imgs = decode_batch(model, z_interp, device)

    # replace endpoints with original control/target images using df 'id'
    ctrl_id = ctrl_df.iloc[70]['id']
    tgt_id = tgt_df.iloc[23]['id']
    orig_ctrl, orig_tgt = None, None
    for img_tensor, file_id in val_dataset:
        if file_id == ctrl_id:
            orig_ctrl = img_tensor.cpu().numpy()
        if file_id == tgt_id:
            orig_tgt = img_tensor.cpu().numpy()
        if orig_ctrl is not None and orig_tgt is not None:
            break
    if orig_ctrl is None or orig_tgt is None:
        raise ValueError("Original control or target image not found in val_dataset")
    # overwrite first and last images
    imgs[0] = orig_ctrl
    imgs[-1] = orig_tgt

    # plot
    model_base = os.path.splitext(os.path.basename(args.model_path))[0]
    plot_interpolation_and_latent_changes(
        imgs,
        z_interp,
        z_ctrl,
        z_tgt,
        args.output,
        f"{model_base}_{args.control_class}_to_{args.target_class}"
    )

if __name__ == '__main__':
    main()
