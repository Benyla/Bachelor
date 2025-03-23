import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import tensorflow as tf
import numpy as np

# Define global constants for device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_mnist_data(BATCH_SIZE, seed=420) -> DataLoader:
    """
    Load, preprocess, and resize the MNIST training dataset to 64x64.

    Args:
        BATCH_SIZE (int): The batch size for the DataLoader.

    Returns:
        train_loader (DataLoader): An iterable over batches of resized training images.
    """
    # Load MNIST data using TensorFlow's Keras datasets, ignoring labels
    (x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

        # Set random seed and sample 1000 random indices
    np.random.seed(seed)
    indices = np.random.choice(len(x_train), size=1000, replace=False)

    # Select 1000 random images
    x_train = x_train[indices]

    # Normalize pixel values to [0, 1]
    x_train = x_train / 255.0

    # Convert to torch.Tensor with shape (N, 1, 28, 28)
    x_train_torch = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)

    # Resize to (N, 1, 64, 64) using bilinear interpolation
    x_train_torch = F.interpolate(x_train_torch, size=(64, 64), mode='bilinear', align_corners=False)

    # Move to the specified device (GPU/CPU)
    x_train_torch = x_train_torch.to(DEVICE)

    # Create a TensorDataset and DataLoader
    train_dataset = TensorDataset(x_train_torch)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader
