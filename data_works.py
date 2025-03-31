import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

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

class SingleCellDataset(Dataset):
    def __init__(self, file_list, transform=None):
        """
        file_list: list of paths to npy files.
        transform: a function that takes a np.array and returns a tensor.
        """
        self.file_list = file_list
        self.transform = transform
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        # Load the image npy file (assumed shape: (68, 68, 3))
        np_img = np.load(file_path)
        
        # Apply transform if provided (default normalization & resizing)
        if self.transform:
            img_tensor = self.transform(np_img)
        else:
            img_tensor = torch.from_numpy(np.transpose(np_img, (2, 0, 1))).float()
            
        # Use the file name (or part of it) as the unique ID
        file_id = os.path.basename(file_path)
        
        return img_tensor, file_id

def get_data():
    """
    Recursively collects all .npy files under the given root directory.
    Returns a list of file paths.
    """
    data_root = '/zhome/70/5/14854/nobackup/deeplearningf22/bbbc021/singlecell/singh_cp_pipeline_singlecell_images'
    pattern = os.path.join(data_root, '**', '*.npy')
    all_files = glob.glob(pattern, recursive=True)
    print(f"Found {len(all_files)} files.")
    train_files, temp_files = train_test_split(all_files, test_size=0.3, random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)
    print(f"Train files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")
    print(f"Test files: {len(test_files)}")
    return train_files, val_files, test_files
    
