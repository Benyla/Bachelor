import os
import glob
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class SingleCellDataset(Dataset):
    def __init__(self, file_list):
        """
        file_list: list of paths to preprocessed .pt files.
        """
        self.file_list = file_list
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        # Load the preprocessed tensor (assumed shape: (3, 64, 64))
        img_tensor = torch.load(file_path)
        
        # Use the file name (without extension) as a unique ID
        file_id = os.path.splitext(os.path.basename(file_path))[0]
        
        return img_tensor, file_id

def get_data():
    """
    Recursively collects all processed .pt files under the given root directory.
    Splits the data into train, validation, and test sets.
    """
    # Updated data root: processed files are now stored here
    #data_root = '/zhome/e9/c/186947/cell_images_processed'
    data_root = '/work3/s224194/cell_images_processed'
    pattern = os.path.join(data_root, '**', '*.pt')
    all_files = glob.glob(pattern, recursive=True)
    print(f"Found {len(all_files)} files.")
    
    # Split into train (70%), validation (15%), and test (15%)
    train_files, temp_files = train_test_split(all_files, test_size=0.3, random_state=7)
    val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=7)
    
    print(f"Train files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")
    print(f"Test files: {len(test_files)}")
    
    return train_files, val_files, test_files
