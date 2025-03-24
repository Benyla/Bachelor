import os
import numpy as np
import matplotlib.pyplot as plt

FOLDER_PATH = "B02_s1_w16F89C55C-7808-4136-82E4-E066F8E3CB10"

def inspect_first_npy(folder_path):
    files = sorted([f for f in os.listdir(folder_path) if f.endswith(".npy")])
    if not files:
        print("No .npy files found.")
        return

    file_path = os.path.join(folder_path, files[1])
    try:
        data = np.load(file_path)
        print(f"📄 {file_path}")
        print(f"  Shape     : {data.shape}")
        print(f"  Dtype     : {data.dtype}")
        print(f"  Min value : {np.min(data)}")
        print(f"  Max value : {np.max(data)}")
        print(f"  Mean      : {np.mean(data)}")

        # Visualize properly based on shape
        if data.ndim == 3 and data.shape[2] == 3:
            # Normalize to [0, 1] for display
            img = data.astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min())
            plt.imshow(img)
            plt.title(os.path.basename(file_path))
            plt.axis("off")
            plt.show()

        else:
            plt.imshow(data, cmap='gray')
            plt.title(os.path.basename(file_path))
            plt.colorbar()
            plt.axis("off")
            plt.show()

    except Exception as e:
        print(f"❌ Error reading {file_path}: {e}")

if __name__ == "__main__":
    inspect_first_npy(FOLDER_PATH)
