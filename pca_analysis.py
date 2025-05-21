import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, RobustScaler


def run_pca_analysis():
    # Load dataset
    df = pd.read_csv("indicators_ID4.csv")

    # Convert Time column to datetime
    df["Time"] = pd.to_datetime(df["Time"], format="%d-%b-%Y %H:%M:%S")
    print("Time column converted to datetime format")

    # Display basic dataset info
    print("\nAll columns in dataset:")
    print(df.columns.tolist())
    print("\nData types:")
    print(df.dtypes)

    # Define metadata columns not used for PCA
    metadata_columns = ["Engine_ID", "Power", "Mounted", "Class", "Time"]

    print("\nMetadata columns:")
    print(df[metadata_columns].head())
    print("\nUnique values in 'Class' column:")
    print(df["Class"].unique())

    # Encode categorical Class values to numeric
    le = LabelEncoder()
    df["Class_encoded"] = le.fit_transform(df["Class"])

    print("\nEncoded 'Class' values:")
    print(df[["Class", "Class_encoded"]].drop_duplicates())

    # Select feature columns for PCA
    feature_columns = [
        col
        for col in df.columns
        if col not in metadata_columns and col != "Class_encoded"
    ]
    print(f"\nNumber of feature columns: {len(feature_columns)}")

    # Prepare data for PCA
    X = df[feature_columns]

    # Standardize data
    scaler = RobustScaler()  # StandardScaler()
    # first i was using StandardScaler, however i decided to use RobustScaler
    # to reduce the influence of outliers, i achieved better results with it
    X_scaled = scaler.fit_transform(X)

    # Apply PCA
    pca = PCA()
    pca.fit(X_scaled)

    # Calculate cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)

    # Find number of components needed for 95% variance
    n_components = np.argmax(cumulative_variance >= 0.95) + 1
    print(f"\nNumber of components needed for 95% variance: {n_components}")

    # Visualize
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(pca.explained_variance_ratio_) + 1),
        cumulative_variance,
        marker="o",
    )
    plt.axhline(y=0.95, color="r", linestyle="-", label="95% Threshold")
    plt.axvline(
        x=n_components,
        color="g",
        linestyle="--",
        label=f"Components needed: {n_components}",
    )
    plt.title("Cumulative Explained Variance vs. Number of Components")
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.grid(True)
    plt.legend()
    plt.savefig("pca_explained_variance.png")
    plt.show()

    # Apply PCA with optimal number of components
    final_pca = PCA(n_components=n_components)
    X_pca = final_pca.fit_transform(X_scaled)

    # Create dataframe with PCA results
    pca_df = pd.DataFrame(
        data=X_pca, columns=[f"PC{i + 1}" for i in range(n_components)]
    )

    # Add metadata columns back to results
    for col in metadata_columns:
        if col in df.columns:
            pca_df[col] = df[col].values

    # Save results to CSV
    pca_df.to_csv("pca_results.csv", index=False)

    # Output dimensionality reduction success
    print("\nPCA completed successfully!")
    print(
        f"Reduced from {len(feature_columns)} features to {n_components} principal components"
    )

    # Display explained variance for each component
    print("Explained variance ratio of each component:")
    for i, ratio in enumerate(final_pca.explained_variance_ratio_):
        print(f"  PC{i + 1}: {ratio:.4f} ({ratio * 100:.2f}%)")
    print(
        f"Total explained variance: {sum(final_pca.explained_variance_ratio_) * 100:.2f}%"
    )

    # Visualize first two principal components by class
    if "Class" in df.columns:
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=df["Class_encoded"],
            alpha=0.7,
            cmap="viridis",
            edgecolor="k",
            s=50,
        )
        plt.colorbar(scatter, label="Class")
        plt.title("First Two Principal Components by Class")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.grid(True, alpha=0.3)
        plt.savefig("pca_visualization.png")
        plt.show()

    # Analyze correlations between original features and principal components
    loadings = final_pca.components_
    feature_importance = pd.DataFrame(
        abs(loadings),
        columns=feature_columns,
        index=[f"PC{i + 1}" for i in range(n_components)],
    )

    # Identify most important features for each principal component
    print("\nTop features for each principal component:")
    for i in range(min(5, n_components)):
        pc = f"PC{i + 1}"
        top_features = feature_importance.loc[pc].nlargest(5)
        print(f"\n{pc}:")
        for feature, importance in top_features.items():
            print(f"  {feature}: {importance:.4f}")

    return X_pca, df["Class_encoded"].values, n_components


if __name__ == "__main__":
    run_pca_analysis()
