#!/usr/bin/env python3
import argparse, os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import transforms
from tqdm import tqdm
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance

from src.utils.config_loader import load_config
from src.utils.data_loader import get_data, SingleCellDataset
from src.models.VAE import VAE

def get_activations(loader, model, device):
    model.eval()
    acts = []
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    with torch.no_grad():
        for batch in tqdm(loader, desc="Activations"):
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(device)
            if batch.shape[1] == 1:
                batch = batch.repeat(1, 3, 1, 1)
            batch = F.interpolate(batch, size=(299, 299),
                                  mode='bilinear', align_corners=False)
            batch = normalize(batch)
            pred = model(batch)[0]       # pool3 features
            pred = pred.squeeze(-1).squeeze(-1)
            acts.append(pred.cpu().numpy())
    return np.concatenate(acts, axis=0)

def calculate_stats(acts):
    return acts.mean(0), np.cov(acts, rowvar=False)

def generate_images(vae, num, ld, device, bs):
    vae.eval()
    imgs = []
    with torch.no_grad():
        for i in range(0, num, bs):
            curr = min(bs, num - i)
            z = torch.randn(curr, ld, device=device)
            recon = vae.decode(z).cpu()
            imgs.append(recon)
    return torch.cat(imgs, 0)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config',     required=True)
    p.add_argument('--model-path', required=True)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--num-samples',type=int, default=None)
    p.add_argument('--output',     default='experiments/fid')
    args = p.parse_args()

    cfg = load_config(args.config)
    ld  = cfg['model']['latent_dim']
    adv = cfg['model'].get('use_adv', False)
    name = f"VAE+_{ld}" if adv else f"VAE_{ld}"

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vae = VAE(in_channels=cfg['model']['in_channels'],
              latent_dim=ld, use_adv=adv).to(device)
    ckpt = torch.load(args.model_path, map_location=device)
    vae.load_state_dict(ckpt['model_state_dict'])

    # Inception for FID
    block = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
    inc   = InceptionV3([block]).to(device)

    # Real data
    _, val_files, _ = get_data()
    full_ds = SingleCellDataset(val_files)
    total   = len(full_ds)
    nsamp   = args.num_samples or total

    if args.num_samples:
        np.random.seed(0)
        idxs = np.random.choice(total, nsamp, replace=False)
        real_ds = Subset(full_ds, idxs)
    else:
        real_ds = full_ds

    real_loader = DataLoader(real_ds, batch_size=args.batch_size)

    # Cache folder
    cache = os.path.expanduser("~/.cache/bachelor_fid/real_acts.npy")
    os.makedirs(os.path.dirname(cache), exist_ok=True)

    if args.num_samples is None and os.path.exists(cache):
        real_acts = np.load(cache)
    else:
        real_acts = get_activations(real_loader, inc, device)
        if args.num_samples is None:
            np.save(cache, real_acts)

    # Generated data
    gen_imgs = generate_images(vae, nsamp, ld, device, args.batch_size)
    gen_loader = DataLoader(TensorDataset(gen_imgs), batch_size=args.batch_size)
    gen_acts   = get_activations(gen_loader, inc, device)

    # Stats & FID
    mu_r, sig_r = calculate_stats(real_acts)
    mu_g, sig_g = calculate_stats(gen_acts)
    fid = calculate_frechet_distance(mu_r, sig_r, mu_g, sig_g)

    os.makedirs(args.output, exist_ok=True)
    out_file = os.path.join(args.output, f"{name}_fid.txt")
    with open(out_file, 'w') as f:
        f.write(f"{fid:.6f}\n")

    print(f"{name} FID: {fid:.6f}")
    print(f"Saved to {out_file}")

if __name__ == '__main__':
    main()