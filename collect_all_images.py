import os
import glob
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

# Set the path to your raw cell images
data_root = '/zhome/70/5/14854/nobackup/deeplearningf22/bbbc021/singlecell/singh_cp_pipeline_singlecell_images'
pattern = os.path.join(data_root, '**', '*.npy')
all_files = glob.glob(pattern, recursive=True)
print(f"Found {len(all_files)} raw files.")

def transform_cell_image(np_img):
    """
    Expects np_img of shape (68, 68, 3).
    Transposes to (3, 68, 68), normalizes each channel, and resizes to (3, 64, 64).
    """
    # Transpose to (3, 68, 68)
    img = np.transpose(np_img, (2, 0, 1))
    img = img.astype(np.float32)
    
    # Per-channel min-max normalization to [0,1]
    for c in range(img.shape[0]):
        cmin = img[c].min()
        cmax = img[c].max()
        if cmax > cmin:
            img[c] = (img[c] - cmin) / (cmax - cmin)
        else:
            img[c] = 0.0
            
    # Convert to a tensor and resize using bilinear interpolation
    # Note: F.interpolate expects a 4D tensor so we add a batch dimension
    img_tensor = torch.from_numpy(img).unsqueeze(0)  # Shape: (1, 3, 68, 68)
    img_resized = F.interpolate(img_tensor, size=(64, 64), mode="bilinear", align_corners=False)
    img_resized = img_resized.squeeze(0)  # Shape: (3, 64, 64)
    
    return img_resized

# Define where to store the processed images
output_folder = '/work3/s224194/cell_images_processed'

# Process each file and save the tensor to disk
for file_path in tqdm(all_files, desc="Processing images"):
    file_id = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(output_folder, file_id + '.pt')
    try:
        np_img = np.load(file_path)
        processed_tensor = transform_cell_image(np_img)
        torch.save(processed_tensor, output_path)
    except Exception as e:
        print(f"Error processing {file_id}: {e}")

