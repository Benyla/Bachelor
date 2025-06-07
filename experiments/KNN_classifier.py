import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import argparse
import torch
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from collections import Counter
import matplotlib.pyplot as plt

# Visualization functions
def visualize_confusion_matrix(cm, classes, title=None, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, aspect='auto')
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticklabels(classes)
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    if title:
        ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('Count')
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
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

    # Subsample to balance classes based on the smallest class count
    min_count = df['moa'].value_counts().min()
    df = df.groupby('moa', group_keys=False).apply(lambda x: x.sample(min_count, random_state=random_state))

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
    report_dict = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    # Print results
    print(f"KNN Classification Report (k={n_neighbors}):")
    print(report)
    print(f"Accuracy: {acc:.4f}\n")
    print("Number of predictions per predicted MOA class:")
    pred_counts = Counter(y_pred)
    for moa, count in sorted(pred_counts.items(), key=lambda x: x[0]):
        print(f"{moa}: {count}")

    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=random_state)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    report_rf = classification_report(y_test, y_pred_rf, zero_division=0)
    print("Random Forest Classification Report:")
    print(report_rf)
    print(f"Accuracy: {acc_rf:.4f}\n")
    print("Number of predictions per predicted MOA class (RF):")
    pred_counts_rf = Counter(y_pred_rf)
    for moa, count in sorted(pred_counts_rf.items(), key=lambda x: x[0]):
        print(f"{moa}: {count}")

    # Encode labels for MLPClassifier
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # Train Neural Network
    nn_clf = MLPClassifier(hidden_layer_sizes=(128,64), alpha=1e-4, max_iter=200, early_stopping=True, random_state=random_state)
    nn_clf.fit(X_train, y_train_enc)
    y_pred_nn = nn_clf.predict(X_test)
    y_pred_nn_labels = le.inverse_transform(y_pred_nn)

    acc_nn = accuracy_score(y_test, y_pred_nn_labels)
    report_nn = classification_report(y_test, y_pred_nn_labels, zero_division=0)
    print("Neural Network Classification Report:")
    print(report_nn)
    print(f"Accuracy: {acc_nn:.4f}\n")
    print("Number of predictions per predicted MOA class (NN):")
    pred_counts_nn = Counter(y_pred_nn_labels)
    for moa, count in sorted(pred_counts_nn.items(), key=lambda x: x[0]):
        print(f"{moa}: {count}")

    # Visualize classification performance
    plot_path = os.path.join("experiments", "plots", f"knn_confusion_matrix_k{n_neighbors}_epoch{epoch}.png")
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    visualize_confusion_matrix(
        cm,
        knn.classes_,
        title=f"KNN Confusion Matrix (k={n_neighbors})",
        save_path=plot_path
    )

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