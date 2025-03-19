import torch
from torch.utils.data import DataLoader, TensorDataset
import tensorflow as tf

# Define global constants for device and batch size
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_mnist_data(BATCH_SIZE) -> tuple[DataLoader, torch.Tensor]:
    """Load and preprocess the MNIST dataset.

    Returns:
        A tuple (train_loader, x_test):
            - train_loader: DataLoader for training data.
            - x_test: A torch.Tensor containing the normalized test images.
                      These images will be used for visualization during training.
    """
    # Load MNIST data using TensorFlow's Keras datasets
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

    # Normalize pixel values to [0, 1] and reshape the images to vectors (28*28)
    x_train = x_train.reshape(-1, 28 * 28) / 255.0
    x_test = x_test.reshape(-1, 28 * 28) / 255.0

    # Convert the numpy arrays to torch.Tensor and move to the specified device
    x_train_torch = torch.tensor(x_train, dtype=torch.float32).to(DEVICE)
    x_test_torch = torch.tensor(x_test, dtype=torch.float32).to(DEVICE)

    # Create a TensorDataset and DataLoader for the training data
    train_dataset = TensorDataset(x_train_torch)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    return train_loader, x_test_torch