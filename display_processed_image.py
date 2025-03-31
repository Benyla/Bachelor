import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt

# Specify the full path to your processed images folder
processed_folder = "/zhome/e9/c/186947/cell_images_processed"

# Get a list of all .pt files in that folder
pt_files = glob.glob(os.path.join(processed_folder, "*.pt"))

if not pt_files:
    print("No processed image files found in", processed_folder)
    exit(1)

# Sort files (optional) and pick the first file
first_file = sorted(pt_files)[0]

# Load the tensor using torch.load
image_tensor = torch.load(first_file)

# Print information about the file and the tensor
print("File Name:", os.path.basename(first_file))
print("Tensor Shape:", image_tensor.shape)
print("Pixel Values - min:", image_tensor.min().item(),
      "max:", image_tensor.max().item(),
      "mean:", image_tensor.mean().item())

# Convert tensor to a numpy array for plotting.
# Assuming the tensor shape is (3, 64, 64), we need to transpose it to (64, 64, 3)
if image_tensor.ndim == 3 and image_tensor.shape[0] == 3:
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
else:
    image_np = image_tensor.cpu().numpy()

# Plot the image
plt.imshow(image_np)
plt.title(f"File: {os.path.basename(first_file)}\nShape: {image_tensor.shape}")
plt.axis("off")
plt.show()