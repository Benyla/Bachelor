import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

import argparse
import torch
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.gridspec import GridSpec

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

    # Encode labels for MLPClassifier
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # Train Neural Network
    nn_clf = MLPClassifier(hidden_layer_sizes=(256,128,64), alpha=1e-4, max_iter=200, early_stopping=True, random_state=random_state)
    nn_clf.fit(X_train, y_train_enc)
    y_pred_nn = nn_clf.predict(X_test)
    y_pred_nn_labels = le.inverse_transform(y_pred_nn)

    # Prepare NN classification report dict
    report_nn_dict = classification_report(y_test, y_pred_nn_labels, zero_division=0, output_dict=True)
    report_nn = classification_report(y_test, y_pred_nn_labels, zero_division=0)

    acc_nn = accuracy_score(y_test, y_pred_nn_labels)
    print("Neural Network Classification Report:")
    print(report_nn)
    print(f"Accuracy: {acc_nn:.4f}\n")
    print("Number of predictions per predicted MOA class (NN):")
    pred_counts_nn = Counter(y_pred_nn_labels)
    for moa, count in sorted(pred_counts_nn.items(), key=lambda x: x[0]):
        print(f"{moa}: {count}")

    # --- Figure 1: Combined Confusion Matrices ---
    cm_nn = confusion_matrix(y_test, y_pred_nn_labels)
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(1, 2, width_ratios=[1, 1], figure=fig)

    # KNN CM
    ax0 = fig.add_subplot(gs[0])
    im0 = ax0.imshow(cm, aspect='auto')
    ax0.set_title(f"KNN Confusion Matrix (k={n_neighbors})")
    ax0.set_xticks(range(len(knn.classes_))); ax0.set_xticklabels(knn.classes_, rotation=45, ha='right')
    ax0.set_yticks(range(len(knn.classes_))); ax0.set_yticklabels(knn.classes_)
    ax0.set_ylabel('True label'); ax0.set_xlabel('Predicted label')
    cbar0 = fig.colorbar(im0, ax=ax0); cbar0.set_label('Count')

    # NN CM
    ax1 = fig.add_subplot(gs[1])
    im1 = ax1.imshow(cm_nn, aspect='auto')
    ax1.set_title("NN Confusion Matrix")
    ax1.set_xticks(range(len(le.classes_))); ax1.set_xticklabels(le.classes_, rotation=45, ha='right')
    ax1.set_yticks(range(len(le.classes_))); ax1.set_yticklabels(le.classes_)
    ax1.set_ylabel('True label'); ax1.set_xlabel('Predicted label')
    cbar1 = fig.colorbar(im1, ax=ax1); cbar1.set_label('Count')

    fig.tight_layout()
    fig_path1 = os.path.join("experiments", "plots", f"combined_confusion_k{n_neighbors}_epoch{epoch}.png")
    os.makedirs(os.path.dirname(fig_path1), exist_ok=True)
    fig.savefig(fig_path1, bbox_inches='tight')
    plt.close(fig)

    # --- Figure 2: Metrics Tables ---
    df_knn = pd.DataFrame(report_dict).T
    df_knn = df_knn[['precision','recall','f1-score','support']].round({'precision':2,'recall':2,'f1-score':2})
    df_knn['support'] = df_knn['support'].astype(int)
    df_nn = pd.DataFrame(report_nn_dict).T
    df_nn = df_nn[['precision','recall','f1-score','support']].round({'precision':2,'recall':2,'f1-score':2})
    df_nn['support'] = df_nn['support'].astype(int)

    # Remove macro/weighted averages and accuracy from metrics tables
    df_knn = df_knn.drop(index=['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
    df_nn = df_nn.drop(index=['accuracy', 'macro avg', 'weighted avg'], errors='ignore')

    # Prepare accuracy tables
    df_knn_acc = pd.DataFrame([["Accuracy", f"{acc:.4f}"]], columns=["Metric", "Value"])
    df_nn_acc = pd.DataFrame([["Accuracy", f"{acc_nn:.4f}"]], columns=["Metric", "Value"])

    fig2 = plt.figure(figsize=(15, 10))
    gs2 = GridSpec(2, 2, height_ratios=[4, 1], figure=fig2)

    # Metrics tables
    ax2 = fig2.add_subplot(gs2[0, 0]); ax2.axis('off')
    table_knn = ax2.table(cellText=df_knn.values,
                          rowLabels=df_knn.index,
                          colLabels=df_knn.columns,
                          loc='center')
    table_knn.auto_set_font_size(False); table_knn.set_fontsize(10); table_knn.scale(1, 1.5)
    ax2.set_title(f"KNN Metrics (k={n_neighbors})")

    ax3 = fig2.add_subplot(gs2[0, 1]); ax3.axis('off')
    table_nn = ax3.table(cellText=df_nn.values,
                         rowLabels=df_nn.index,
                         colLabels=df_nn.columns,
                         loc='center')
    table_nn.auto_set_font_size(False); table_nn.set_fontsize(10); table_nn.scale(1, 1.5)
    ax3.set_title("NN Metrics")

    # Accuracy tables
    ax4 = fig2.add_subplot(gs2[1, 0]); ax4.axis('off')
    acc_table_knn = ax4.table(cellText=df_knn_acc.values,
                              colLabels=df_knn_acc.columns,
                              loc='center')
    acc_table_knn.auto_set_font_size(False); acc_table_knn.set_fontsize(12); acc_table_knn.scale(1.5, 1.5)

    ax5 = fig2.add_subplot(gs2[1, 1]); ax5.axis('off')
    acc_table_nn = ax5.table(cellText=df_nn_acc.values,
                             colLabels=df_nn_acc.columns,
                             loc='center')
    acc_table_nn.auto_set_font_size(False); acc_table_nn.set_fontsize(12); acc_table_nn.scale(1.5, 1.5)

    fig2.tight_layout()
    fig_path2 = os.path.join("experiments", "plots", f"metrics_tables_k{n_neighbors}_epoch{epoch}.png")
    fig2.savefig(fig_path2, bbox_inches='tight')
    plt.close(fig2)

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