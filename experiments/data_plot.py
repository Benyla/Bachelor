import pandas as pd
import matplotlib.pyplot as plt
import torch
import os
import random

# === 1. Load metadata ===
metadata_path = "/zhome/70/5/14854/nobackup/deeplearningf22/bbbc021/singlecell/metadata.csv"
df = pd.read_csv(metadata_path)

# === 2. Count instances per class in 'moa' column ===
moa_counts = df["moa"].value_counts()

# === 3. Plot the class distribution ===
plt.figure(figsize=(14, 6))

# Subplot 1: Class distribution bar plot
plt.subplot(1, 2, 1)
moa_counts.plot(kind='bar')
plt.title("Class Distribution in 'moa'")
plt.xlabel("MOA Class")
plt.ylabel("Number of Instances")
plt.xticks(rotation=45)
plt.grid(True)

# === 4. Load a random .pt file with 3 channels ===
image_dir = "/work3/s224194/cell_images_processed"
pt_files = [f for f in os.listdir(image_dir) if f.endswith(".pt")]

selected_file = random.choice(pt_files)
image_tensor = torch.load(os.path.join(image_dir, selected_file))

# Check and adjust shape if necessary
if image_tensor.shape[0] != 3:
    raise ValueError(f"Expected 3 channels, but got shape: {image_tensor.shape}")

# === 5. Plot the image channels ===
plt.subplot(1, 2, 2)
# Assume shape [3, H, W], normalize to [0, 1]
image_np = image_tensor.numpy()
image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())

# Display as RGB
plt.imshow(image_np.transpose(1, 2, 0))
plt.title(f"Random Cell Image: {selected_file}")
plt.axis("off")

plt.tight_layout()
plt.savefig("experiments/plots/class_distribution_with_image.png", dpi=300, bbox_inches='tight')
