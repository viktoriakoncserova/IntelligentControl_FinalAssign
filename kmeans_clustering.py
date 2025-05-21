import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


def run_kmeans_clustering():
    # Load PCA results
    try:
        pca_df = pd.read_csv("pca_results.csv")
    except FileNotFoundError:
        print("PCA results file not found. Please run pca_analysis.py first.")
        return

    # Extract PCA components (exclude metadata columns)
    pc_columns = [col for col in pca_df.columns if col.startswith("PC")]
    X_pca = pca_df[pc_columns].values

    print(f"Loaded PCA data with {len(pc_columns)} principal components")

    # Determine optimal number of clusters using Elbow method
    inertias = []
    silhouette_scores = []
    k_range = range(2, 16)  # Testing from 2 to 15 clusters

    for k in k_range:
        # Train KMeans model
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_pca)

        # Calculate inertia (within-cluster sum of squares)
        inertias.append(kmeans.inertia_)

        # Calculate silhouette score
        if k > 1:  # Silhouette score requires at least 2 clusters
            silhouette_scores.append(silhouette_score(X_pca, kmeans.labels_))

    # Plot Elbow method results
    plt.figure(figsize=(12, 5))

    # Inertia plot
    plt.subplot(1, 2, 1)
    plt.plot(k_range, inertias, marker="o")
    plt.title("Elbow Method for Optimal k")
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia")
    plt.grid(True)

    # Silhouette score plot
    plt.subplot(1, 2, 2)
    # Convert range to list for plotting
    k_values = list(k_range)
    plt.plot(k_values, silhouette_scores, marker="o")
    plt.title("Silhouette Score for Optimal k")
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette Score")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("kmeans_optimal_k.png")
    plt.show()

    # Select optimal k based on silhouette score (higher is better)
    # if we would select optimal k based on elbow method, we would select the k, where
    # the inertia starts to decrease slower, however it is not always easy to determine
    # the elbow point, therefore i decided to use silhouette score, which is more reliable according to the internet.
    optimal_k = k_range[np.argmax(silhouette_scores) + 1]
    print(f"Optimal number of clusters based on silhouette score: {optimal_k}")

    # Train final KMeans model with optimal k
    final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    cluster_labels = final_kmeans.fit_predict(X_pca)

    # Add cluster labels to the dataframe
    pca_df["Cluster"] = cluster_labels

    # Evaluate clustering quality using various metrics
    silhouette = silhouette_score(X_pca, cluster_labels)
    db_score = davies_bouldin_score(X_pca, cluster_labels)
    ch_score = calinski_harabasz_score(X_pca, cluster_labels)

    print("\nClustering quality metrics:")
    print(
        f"  Silhouette Score: {silhouette:.4f} (higher is better, range: [-1, 1])"
    )
    print(f"  Davies-Bouldin Index: {db_score:.4f} (lower is better)")
    print(f"  Calinski-Harabasz Index: {ch_score:.4f} (higher is better)")

    # Visualize clusters in 2D space (first two principal components)
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=cluster_labels,
        cmap="viridis",
        alpha=0.7,
        edgecolor="k",
        s=50,
    )

    # Mark cluster centers
    centers = final_kmeans.cluster_centers_
    plt.scatter(
        centers[:, 0],
        centers[:, 1],
        c="red",
        marker="X",
        s=200,
        alpha=0.8,
        label="Cluster Centers",
    )

    plt.colorbar(scatter, label="Cluster")
    plt.title(f"KMeans Clustering Results (k={optimal_k})")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("kmeans_clusters.png")
    plt.show()

    # Compare clusters with original classes (if available)
    if "Class" in pca_df.columns:
        # Create a cross-tabulation of cluster assignments vs original classes
        cluster_class_comparison = pd.crosstab(
            pca_df["Cluster"],
            pca_df["Class"],
            rownames=["Cluster"],
            colnames=["Class"],
        )

        print("\nComparison of clusters with original classes:")
        print(cluster_class_comparison)

    # Save the complete results with cluster assignments
    pca_df.to_csv("pca_with_clusters.csv", index=False)

    print(
        "\nComplete results with cluster assignments saved to 'pca_with_clusters.csv'"
    )


if __name__ == "__main__":
    run_kmeans_clustering()
