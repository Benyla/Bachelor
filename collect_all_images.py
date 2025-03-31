import os
import glob
import torch
import numpy as np
import torch.nn.functional as F
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

data_root = '/zhome/70/5/14854/nobackup/deeplearningf22/bbbc021/singlecell/singh_cp_pipeline_singlecell_images'
pattern = os.path.join(data_root, '**', '*.npy')
all_files = glob.glob(pattern, recursive=True)
print(f"Found {len(all_files)} files.")

output_folder = '/zhome/e9/c/186947/cell_images_processed'
os.makedirs(output_folder, exist_ok=True)

def transform_cell_image(np_img):
    img = np.transpose(np_img, (2, 0, 1))
    img = img.astype(np.float32)
    for c in range(img.shape[0]):
        cmin = img[c].min()
        cmax = img[c].max()
        if cmax > cmin:
            img[c] = (img[c] - cmin) / (cmax - cmin)
        else:
            img[c] = 0.0
    img_tensor = torch.from_numpy(img).unsqueeze(0)
    img_resized = F.interpolate(img_tensor, size=(64, 64), mode="bilinear", align_corners=False)
    return img_resized.squeeze(0)

def process_file(file_path):
    file_id = os.path.splitext(os.path.basename(file_path))[0]
    output_path = os.path.join(output_folder, file_id + '.pt')
    try:
        np_img = np.load(file_path)
        processed_tensor = transform_cell_image(np_img)
        torch.save(processed_tensor, output_path)
        return f"Processed {file_id}"
    except Exception as e:
        return f"Error processing {file_id}: {e}"

# Use a process pool to parallelize the work
with ProcessPoolExecutor() as executor:
    results = list(tqdm(executor.map(process_file, all_files), total=len(all_files), desc="Processing images"))

