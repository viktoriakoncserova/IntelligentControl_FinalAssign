import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import pair_confusion_matrix


def evaluate_clusters_with_pair_confusion(best_params=None):
    """
    Evaluate the clustering results using pair confusion matrix.
    If best_params is not provided, it will be loaded from previously saved results.
    """
    # Load PCA results with true class labels
    try:
        pca_df = pd.read_csv("pca_results.csv")
        print("Loaded PCA results successfully.")
    except FileNotFoundError:
        print("PCA results file not found. Please run pca_analysis.py first.")
        return

    # Check if 'Class' exists in the data
    if "Class" not in pca_df.columns:
        print(
            "Warning: 'Class' column not found in the data. Cannot evaluate against true labels."
        )
        return

    # Extract PCA components (exclude metadata columns)
    pc_columns = [col for col in pca_df.columns if col.startswith("PC")]
    X_pca = pca_df[pc_columns].values

    # Get true labels - convert to numeric if needed
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y_true = le.fit_transform(pca_df["Class"])

    # Get the best parameters from optimization if available, otherwise use defaults
    if best_params is None:
        try:
            opt_results = pd.read_csv("pca_with_optimal_clusters.csv")
            if "OptimalCluster" in opt_results.columns:
                print("Using previously optimized clustering results.")
                y_pred = opt_results["OptimalCluster"].values
                # Count number of clusters to get n_clusters
                n_clusters = len(np.unique(y_pred))
                max_iter = 300  # default
            else:
                raise ValueError(
                    "OptimalCluster column not found in results file."
                )
        except (FileNotFoundError, ValueError) as e:
            print(f"Could not load optimal clusters: {e}")
            print("Using default KMeans parameters (6 clusters).")
            n_clusters = 6
            max_iter = 300
            # Train KMeans with default parameters
            kmeans = KMeans(
                n_clusters=n_clusters,
                max_iter=max_iter,
                random_state=42,
                n_init=10,
            )
            y_pred = kmeans.fit_predict(X_pca)
    else:
        # Extract best parameters
        n_clusters = best_params.get("n_clusters", 6)
        max_iter = best_params.get("max_iter", 300)

        # Train KMeans with best parameters
        print(
            f"Training KMeans with optimal parameters: n_clusters={n_clusters}, max_iter={max_iter}"
        )
        kmeans = KMeans(
            n_clusters=n_clusters,
            max_iter=max_iter,
            random_state=42,
            n_init=10,
        )
        y_pred = kmeans.fit_predict(X_pca)

    # Calculate pair confusion matrix
    pair_conf_matrix = pair_confusion_matrix(y_true, y_pred)

    # Print pair confusion matrix statistics
    print("\nPair Confusion Matrix Statistics:")
    print(f"Same pairs in both clusterings (TP): {pair_conf_matrix[1, 1]}")
    print(
        f"Different pairs in both clusterings (TN): {pair_conf_matrix[0, 0]}"
    )
    print(
        f"Same pairs in true but different in pred (FN): {pair_conf_matrix[1, 0]}"
    )
    print(
        f"Different pairs in true but same in pred (FP): {pair_conf_matrix[0, 1]}"
    )

    # Calculate metrics
    total_pairs = np.sum(pair_conf_matrix)
    rand_index = (
        pair_conf_matrix[0, 0] + pair_conf_matrix[1, 1]
    ) / total_pairs
    precision = pair_conf_matrix[1, 1] / (
        pair_conf_matrix[1, 1] + pair_conf_matrix[0, 1]
    )
    recall = pair_conf_matrix[1, 1] / (
        pair_conf_matrix[1, 1] + pair_conf_matrix[1, 0]
    )
    f1_score = 2 * precision * recall / (precision + recall)

    print("\nDerived Metrics:")
    print(f"Rand Index: {rand_index:.4f} (higher is better, range: [0, 1])")
    print(f"Pair Precision: {precision:.4f} (higher is better, range: [0, 1])")
    print(f"Pair Recall: {recall:.4f} (higher is better, range: [0, 1])")
    print(f"Pair F1-Score: {f1_score:.4f} (higher is better, range: [0, 1])")

    # Visualize pair confusion matrix with normalized values
    plt.figure(figsize=(10, 8))
    pair_conf_matrix_norm = pair_conf_matrix / total_pairs
    sns.heatmap(
        pair_conf_matrix_norm,
        annot=True,
        fmt=".4f",
        cmap="Blues",
        xticklabels=["Different in Pred", "Same in Pred"],
        yticklabels=["Different in True", "Same in True"],
    )
    plt.title("Normalized Pair Confusion Matrix")
    plt.tight_layout()
    plt.savefig("pair_confusion_matrix.png")
    plt.show()


    # Visualize clusters and true classes in 2D space
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Plot predicted clusters
    scatter1 = axes[0].scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=y_pred,
        cmap="viridis",
        alpha=0.7,
        edgecolor="k",
        s=50,
    )
    axes[0].set_title(f"Predicted Clusters (k={n_clusters})")
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=axes[0], label="Cluster")

    # Plot true classes
    scatter2 = axes[1].scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=y_true,
        cmap="plasma",
        alpha=0.7,
        edgecolor="k",
        s=50,
    )
    axes[1].set_title("True Classes")
    axes[1].set_xlabel("PC1")
    axes[1].set_ylabel("PC2")
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[1], label="Class")

    plt.tight_layout()
    plt.savefig("clusters_vs_true_classes_visualization.png")
    plt.show()

    return {
        "rand_index": rand_index,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }


if __name__ == "__main__":
    evaluate_clusters_with_pair_confusion()
