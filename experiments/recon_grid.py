#!/usr/bin/env python3
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
import torch
import matplotlib.pyplot as plt
from src.models.VAE import VAE
from src.utils.data_loader import get_data

# Configuration
CHECKPOINT_DIR = 'trained_models' # dir with terained models
IMAGE_IDX      = 1  # index of the image in the test set
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define models
models = {
    'VAE_128':  (False, 128),
    'VAE_256':  (False, 256),
    'VAE+_128': (True, 128),
    'VAE+_256': (True, 256),
}
# Epoch checkpoints 
epochs = [10, 20, 30, 40, 49]

# Load test data
_, test_files, _ = get_data()

# Load and preprocess the selected image 
path = test_files[IMAGE_IDX]
data = torch.load(path)
if isinstance(data, torch.Tensor):
    img = data
elif isinstance(data, dict):
    img = data['image'] # can be removed - data is tensor
else:
    raise ValueError(f"Unsupported data format in {path}")
img = img.unsqueeze(0).to(DEVICE)

# Prepare matplotlib grid (4 rows × 6 columns: 5 epochs + 1 original)
n_rows = len(models)
n_cols = len(epochs) + 1
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3 - 3, n_rows * 3 - 2))

# Overall title 
fig.suptitle('Model Reconstructions Across Epochs', fontsize=16)

# Explicit y-axis labels for each model row
axes[0, 0].set_ylabel('VAE_128', rotation=90, fontsize=12, labelpad=10)
axes[1, 0].set_ylabel('VAE_256', rotation=90, fontsize=12, labelpad=10)
axes[2, 0].set_ylabel('VAE+_128', rotation=90, fontsize=12, labelpad=10)
axes[3, 0].set_ylabel('VAE+_256', rotation=90, fontsize=12, labelpad=10)

# Plot reconstructions for each epoch, per model
for i, (model_name, (use_adv, latent_dim)) in enumerate(models.items()):
    # Build & load the model once per row
    model = VAE(in_channels=3, latent_dim=latent_dim, use_adv=use_adv)
    model.to(DEVICE).eval()

    for j, epoch in enumerate(epochs):
        ckpt = os.path.join(CHECKPOINT_DIR, f"{model_name}_epoch_{epoch}.pth")
        state = torch.load(ckpt, map_location=DEVICE)
        sd = state.get('model_state_dict', state)
        model.load_state_dict(sd)

        with torch.no_grad():
            recon, mu, logvar, _ = model(img)
        recon_img = recon.cpu().squeeze()
        display_img = recon_img.permute(1, 2, 0) if recon_img.ndim == 3 else recon_img

        ax = axes[i, j]
        ax.imshow(display_img, cmap='gray' if display_img.ndim == 2 else None)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        if i == 0:
            ax.set_title(f"Epoch {epoch}", fontsize=12)

    # Plot the original image in the last column
    ax_orig = axes[i, n_cols - 1]
    orig = img.cpu().squeeze()
    display_orig = orig.permute(1, 2, 0) if orig.ndim == 3 else orig
    ax_orig.imshow(display_orig, cmap='gray' if display_orig.ndim == 2 else None)
    ax_orig.set_xticks([]); ax_orig.set_yticks([])
    for spine in ax_orig.spines.values():
        spine.set_visible(False)
    if i == 0:
        ax_orig.set_title('Original', fontsize=12)

# Tweak spacing 
os.makedirs('experiments/plots', exist_ok=True)
out_path = os.path.join('experiments/plots', 'reconstructions_grid.png')
plt.subplots_adjust(left=0.04, right=0.96, top=0.92, bottom=0.05, wspace=0.1, hspace=0.1)
plt.savefig(out_path, dpi=300)
print(f"Saved reconstruction grid to {out_path}")
plt.show()