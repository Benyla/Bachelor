import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))
import torch
import matplotlib.pyplot as plt
from src.models.VAE import VAE
from src.utils.data_loader import get_data

# Configuration
CHECKPOINT_DIR = 'trained_models'          # directory with your .pth files
IMAGE_IDX      = 60                        # index of the image in the val set
DEVICE         = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the models you trained
models = {
    'VAE+_256': (True, 256),
    'VAE+_128': (True, 128),
    'VAE_256':  (False, 256),
    'VAE_128':  (False, 128),
}
# Epoch checkpoints you saved
epochs = [10, 20, 30, 40, 49]


# Load validation file paths
_, val_files, _ = get_data()

# Load and preprocess the selected image from file
path = val_files[IMAGE_IDX]
data = torch.load(path)
if isinstance(data, torch.Tensor):
    img = data
elif isinstance(data, dict):
    img = data['image']  # Change 'image' if necessary
else:
    raise ValueError(f"Unsupported data format in {path}")
img = img.unsqueeze(0).to(DEVICE)

# Prepare matplotlib grid
n_rows = len(models)
n_cols = len(epochs)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3))

for i, (model_name, (use_adv, latent_dim)) in enumerate(models.items()):
    for j, epoch in enumerate(epochs):
        # Instantiate VAE with appropriate adversarial flag
        model = VAE(in_channels=3, latent_dim=latent_dim, use_adv=use_adv)
        # Load checkpoint
        ckpt_path = os.path.join(
            CHECKPOINT_DIR,
            f"{model_name}_epoch_{epoch}.pth"
        )
        state = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(state)
        model.to(DEVICE)
        model.eval()

        # Forward pass
        with torch.no_grad():
            recon, mu, logvar = model(img)
        # Move to CPU and remove batch dim
        recon_img = recon.cpu().squeeze()
        # If multichannel (e.g., 3xHxW), permute dims
        if recon_img.ndim == 3:
            display_img = recon_img.permute(1, 2, 0)
        else:
            display_img = recon_img

        ax = axes[i, j]
        ax.imshow(display_img, cmap='gray' if display_img.ndim == 2 else None)
        ax.axis('off')

        # Titles and labels
        if i == 0:
            ax.set_title(f"Epoch {epoch}", fontsize=12)
        if j == 0:
            ax.set_ylabel(model_name, fontsize=12)

# Adjust layout and save
plt.tight_layout()
out_path = 'plots/reconstructions_grid.png'
plt.savefig(out_path, dpi=300)
print(f"Saved reconstruction grid to {out_path}")
plt.show()