import warnings

import matplotlib.pyplot as plt
import pandas as pd
from bayes_opt import BayesianOptimization
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
import seaborn as sns


def run_bayesian_optimization():
    # Suppress warnings
    warnings.filterwarnings("ignore")

    # Load PCA results
    try:
        pca_df = pd.read_csv("pca_results.csv")
        print("Loaded PCA results successfully.")
    except FileNotFoundError:
        print("PCA results file not found. Please run pca_analysis.py first.")
        return {}

    # Extract PCA components (exclude metadata columns)
    pc_columns = [col for col in pca_df.columns if col.startswith("PC")]
    X_pca = pca_df[pc_columns].values

    print(
        f"Running Bayesian Optimization on {len(pc_columns)} principal components"
    )

    # Define the objective function to maximize
    def kmeans_objective(n_clusters, max_iter):
        """
        Objective function for Bayesian Optimization:
        Balances silhouette score with Davies-Bouldin index
        """
        # Convert parameters to integers (Bayesian optimization works with floats)
        n_clusters_int = max(8, int(round(n_clusters)))
        max_iter_int = max(50, int(round(max_iter)))

        # Run KMeans with these parameters
        kmeans = KMeans(
            n_clusters=n_clusters_int,
            max_iter=max_iter_int,
            random_state=42,
            n_init=10,
        )

        # Fit and predict
        cluster_labels = kmeans.fit_predict(X_pca)

        # Calculate silhouette score - higher is better
        sil_score = silhouette_score(X_pca, cluster_labels)

        # Calculate Davies-Bouldin index - lower is better
        db_score = davies_bouldin_score(X_pca, cluster_labels)

        # Normalize DB score to similar range as silhouette
        norm_db_score = min(1.0, db_score / 4.0)

        # Create a combined score with a small bonus for higher cluster counts
        cluster_bonus = 0.02 * (n_clusters_int / 20)
        combined_score = (
            (0.7 * sil_score) - (0.3 * norm_db_score) + cluster_bonus
        )

        return combined_score

    # Define parameter bounds for Bayesian Optimization
    pbounds = {
        "n_clusters": (7, 20),  # Start from at least 7 clusters
        "max_iter": (50, 1000),
    }

    # Initialize Bayesian Optimization
    optimizer = BayesianOptimization(
        f=kmeans_objective, pbounds=pbounds, random_state=42, verbose=2
    )

    # Run optimization
    print(
        "\nStarting Bayesian Optimization to find optimal KMeans parameters..."
    )
    optimizer.maximize(init_points=10, n_iter=30)

    # Get best parameters
    best_params = optimizer.max["params"]
    best_score = optimizer.max["target"]

    # Convert to integers
    best_n_clusters = max(8, int(round(best_params["n_clusters"])))
    best_max_iter = max(50, int(round(best_params["max_iter"])))

    print("\n=== Optimization Results ===")
    print(f"Best Combined Score: {best_score:.4f}")
    print(f"Optimal number of clusters: {best_n_clusters}")
    print(f"Optimal maximum iterations: {best_max_iter}")

    # Run KMeans with optimal parameters
    print("\nRunning final KMeans with optimal parameters...")
    final_kmeans = KMeans(
        n_clusters=best_n_clusters,
        max_iter=best_max_iter,
        random_state=42,
        n_init=10,
    )

    # Fit and predict
    cluster_labels = final_kmeans.fit_predict(X_pca)

    # Add optimal cluster labels to the dataframe
    pca_df["OptimalCluster"] = cluster_labels

    # Save results
    pca_df.to_csv("pca_with_optimal_clusters.csv", index=False)

    # Visualize the optimization process
    plt.figure(figsize=(14, 6))

    # Extract optimization history safely
    n_clusters_history = []
    max_iter_history = []
    scores = []

    for res in optimizer.res:
        try:
            params = res.get("params", {})
            if params:
                n_clusters_history.append(
                    int(round(params.get("n_clusters", 8)))
                )
                max_iter_history.append(
                    int(round(params.get("max_iter", 100)))
                )
                scores.append(res.get("target", 0))
        except Exception as e:
            print(f"Warning: Could not process a result point: {e}")

    iterations = range(1, len(scores) + 1)

    # Plot scores vs iterations
    plt.subplot(1, 2, 1)
    plt.plot(iterations, scores, "o-")
    plt.axhline(
        y=best_score,
        color="r",
        linestyle="--",
        label=f"Best score: {best_score:.4f}",
    )
    plt.title("Combined Score Improvement")
    plt.xlabel("Iteration")
    plt.ylabel("Combined Score")
    plt.grid(True)
    plt.legend()

    # Plot parameter changes
    plt.subplot(1, 2, 2)
    plt.scatter(
        n_clusters_history, max_iter_history, c=scores, cmap="viridis", s=100
    )
    plt.colorbar(label="Score")
    plt.scatter(
        [best_n_clusters],
        [best_max_iter],
        color="red",
        s=200,
        marker="*",
        label=f"Best: {best_n_clusters} clusters, {best_max_iter} iterations",
    )
    plt.title("Parameter Space Exploration")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Maximum Iterations")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("bayesian_optimization.png")
    plt.show()

    # Visualize final clusters in 2D space
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

    plt.colorbar(scatter, label="Optimal Cluster")
    plt.title(
        f"Optimized KMeans Clustering (k={best_n_clusters}, max_iter={best_max_iter})"
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig("optimized_kmeans_clusters.png")
    plt.show()

    print("\nBayesian optimization completed successfully.")
    print("Results saved to 'pca_with_optimal_clusters.csv'")

    return {"n_clusters": best_n_clusters, "max_iter": best_max_iter}


if __name__ == "__main__":
    run_bayesian_optimization()
