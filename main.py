from bayesian_kmeans_opt import run_bayesian_optimization
from cluster_evaluation import evaluate_clusters_with_pair_confusion
from kmeans_clustering import run_kmeans_clustering
from pca_analysis import run_pca_analysis

# Task 1
print("=== Task 1: Running PCA Analysis ===")
run_pca_analysis()

# Task 2
print("\n\n=== Task 2: Running KMeans Clustering ===")
run_kmeans_clustering()

# Task 3
print("\n\n=== Task 3: Running Bayesian Optimization for KMeans ===")
best_params = run_bayesian_optimization()

# Task 4
print("\n\n=== Task 4: Evaluating Clusters with Pair Confusion Matrix ===")
evaluate_clusters_with_pair_confusion(best_params)

# CONCLUSION
# After analyzing my clustering results, I can see that my algorithm shows interesting
# but mixed performance:
# I achieved a strong recall of 0.9293, which means I'm successfully keeping most similar
# data points together that should be grouped. However, my precision is quite low at 0.1231,
# indicating my model is putting too many dissimilar points in the same clusters.

# My Rand Index of 0.8668 shows reasonably good overall agreement between predictions and
# true labels, but my F1-Score of 0.2174 reflects the imbalance between precision and recall.

# To improve the results, I should consider:
# 1. Increasing the number of clusters
# 2. Adjusting my algorithm parameters
# 3. Trying different clustering methods that might better separate my data
