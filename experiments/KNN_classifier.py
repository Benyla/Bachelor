import os
import argparse
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt

# Visualization functions
def visualize_confusion_matrix(cm, classes, title=None, save_path=None):
    fig, ax = plt.subplots()
    im = ax.imshow(cm)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=90)
    ax.set_yticklabels(classes)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    if title:
        ax.set_title(title)
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close(fig)

# Import the helper function for loading latent codes, metadata, and configuration
from src.utils.latent_codes_and_metadata import get_latent_and_metadata
from src.utils.config_loader import load_config

def train_evaluate_knn(config, epoch, n_neighbors=5, test_size=0.2, random_state=42):
    """
    Loads latent representations and trains a KNN classifier on the MOA labels.

    Args:
        config (dict): configuration with keys:
            - model.use_adv (bool)
            - model.latent_dim (int)
            - latent_codes_dir (str)
            - metadata_csv (str)
        epoch (int): epoch number to load latent codes from
        n_neighbors (int): number of neighbors for KNN
        test_size (float): fraction of data to reserve for testing
        random_state (int): seed for reproducibility

    Returns:
        dict: containing trained model, evaluation metrics, and splits
    """
    # 1. Load dataframe with latent codes and MOA labels
    df = get_latent_and_metadata(config, epoch)

    # 2. Prepare features and targets
    z_cols = [c for c in df.columns if c.startswith('z')]
    X = df[z_cols].values
    y = df['moa'].values

    # 3. Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 4. Initialize and train KNN
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)

    # 5. Evaluate on test set
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # Print results
    print(f"KNN Classification Report (k={n_neighbors}):")
    print(report)
    print(f"Accuracy: {acc:.4f}\n")

    # Visualize classification performance
    plot_path = os.path.join("experiments", "plots", f"knn_confusion_matrix_k{n_neighbors}_epoch{epoch}.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    visualize_confusion_matrix(cm, knn.classes_, title=f"KNN Confusion Matrix (k={n_neighbors})", save_path=plot_path)

    return {
        'model': knn,
        'accuracy': acc,
        'report': report,
        'confusion_matrix': cm,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train and evaluate a KNN classifier in the latent space.'
    )
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file (JSON or similar)')
    parser.add_argument('--latent-dim', type=int, default=256,
                        help='Dimensionality of the latent codes')
    parser.add_argument('--epoch', type=int, default=49,
                        help='Epoch number to load latent codes from (default: 49)')
    parser.add_argument('-k', '--n-neighbors', type=int, default=5,
                        help='Number of neighbors for the KNN classifier')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Fraction of the dataset to use as test set')
    parser.add_argument('--random-state', type=int, default=42,
                        help='Random seed for reproducibility')
    args = parser.parse_args()

    config = load_config(args.config)

    # Train and evaluate
    train_evaluate_knn(
        config,
        epoch=args.epoch,
        n_neighbors=args.n_neighbors,
        test_size=args.test_size,
        random_state=args.random_state
    )