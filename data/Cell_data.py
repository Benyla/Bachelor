import os
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F

class CellDataset(Dataset):
    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path
        self.npy_files = [f for f in os.listdir(folder_path) if f.endswith('.npy')]

    def __len__(self):
        return len(self.npy_files)

    def __getitem__(self, idx):
        npy_path = os.path.join(self.folder_path, self.npy_files[idx])
        img = np.load(npy_path) 
        img = np.transpose(img, (2, 0, 1))
        img = img.astype(np.float32)
        for c in range(img.shape[0]):
            cmin = img[c].min()
            cmax = img[c].max()
            if cmax > cmin: 
                img[c] = (img[c] - cmin) / (cmax - cmin)
            else:
                img[c] = 0.0
        # < -- Resize the image to 64x64 -- >
        img_tensor = torch.from_numpy(img).unsqueeze(0)
        img_resized = F.interpolate(img_tensor, size=(64, 64), mode="bilinear", align_corners=False)
        img_resized = img_resized.squeeze(0) 
        # < ------------------------------- >
        return img_resized