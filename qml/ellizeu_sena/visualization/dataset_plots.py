import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
    

def ensure_dir(path):
    if path:
        os.makedirs(path, exist_ok=True)

def dataset_summary(X, y):
    print("Número de amostras :", X.shape[0])
    print("Número de atributos:", X.shape[1])

    classes, counts = np.unique(y, return_counts=True)

    print("\nDistribuição das classes")

    for cls, count in zip(classes, counts):
        print(f"Classe {cls}: {count}")

def plot_correlation_heatmap(
    X,
    feature_names,
    output_path=None,
):
    import pandas as pd

    df = pd.DataFrame(
        X,
        columns=feature_names,
    )

    corr = df.corr()

    plt.figure(figsize=(12, 10))

    plt.imshow(
        corr,
        aspect="auto",
    )

    plt.colorbar()

    plt.title(
        "Feature Correlation Matrix"
    )

    if output_path:
        ensure_dir(
            os.path.dirname(output_path)
        )

        plt.savefig(
            output_path,
            bbox_inches="tight",
        )

    plt.show()

def plot_feature_histograms(
    X,
    feature_names,
    max_features=12,
):
    n = min(
        max_features,
        len(feature_names),
    )

    fig, axes = plt.subplots(
        nrows=3,
        ncols=4,
        figsize=(14, 10),
    )

    axes = axes.flatten()

    for i in range(n):
        axes[i].hist(
            X[:, i],
            bins=20,
        )

        axes[i].set_title(
            feature_names[i]
        )

    plt.tight_layout()
    plt.show()

def plot_feature_boxplots(
    X,
    feature_names,
    max_features=10,
):
    plt.figure(
        figsize=(14, 6)
    )

    plt.boxplot(
        X[:, :max_features]
    )

    plt.xticks(
        range(
            1,
            max_features + 1
        ),
        feature_names[:max_features],
        rotation=90,
    )

    plt.tight_layout()
    plt.show()

def descriptive_statistics(
    X,
    feature_names,
):
    df = pd.DataFrame(
        X,
        columns=feature_names,
    )

    return df.describe()


def plot_class_distribution(
    y,
    output_path=None,
):
    """
    Plot class distribution.
    """

    classes, counts = np.unique(y, return_counts=True)

    plt.figure(figsize=(6, 4))
    plt.bar(classes.astype(str), counts)

    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")

    if output_path:
        ensure_dir(os.path.dirname(output_path))
        plt.savefig(output_path, bbox_inches="tight")

    plt.show()

def plot_pca_pairplot(
    X,
    y,
):
    import pandas as pd
    from pandas.plotting import scatter_matrix

    pca = PCA(
        n_components=4
    )

    X_pca = pca.fit_transform(X)

    df = pd.DataFrame(
        X_pca,
        columns=[
            "PC1",
            "PC2",
            "PC3",
            "PC4",
        ],
    )

    scatter_matrix(
        df,
        figsize=(10, 10),
    )

    plt.show()


def plot_pca_projection(
    X,
    y,
    output_path=None,
):
    """
    PCA projection to 2 dimensions.
    """

    pca = PCA(n_components=2)

    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))

    for label in np.unique(y):
        mask = y == label

        plt.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            label=f"Class {label}",
            alpha=0.7,
        )

    plt.title("PCA Projection")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend()

    if output_path:
        ensure_dir(os.path.dirname(output_path))
        plt.savefig(output_path, bbox_inches="tight")

    plt.show()

    return pca


def plot_explained_variance(
    X,
    max_components=None,
    output_path=None,
):
    """
    Plot cumulative explained variance.
    """

    if max_components is None:
        max_components = min(X.shape)

    pca = PCA(n_components=max_components)

    pca.fit(X)

    cumulative_variance = np.cumsum(
        pca.explained_variance_ratio_
    )

    plt.figure(figsize=(8, 5))

    plt.plot(
        range(
            1,
            len(cumulative_variance) + 1,
        ),
        cumulative_variance,
        marker="o",
    )

    plt.title("Cumulative Explained Variance")
    plt.xlabel("Number of Components")
    plt.ylabel("Explained Variance")

    plt.grid(True)

    if output_path:
        ensure_dir(os.path.dirname(output_path))
        plt.savefig(output_path, bbox_inches="tight")

    plt.show()

    return cumulative_variance