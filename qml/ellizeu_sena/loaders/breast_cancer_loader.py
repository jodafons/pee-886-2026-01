import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def download_breast_cancer_dataset():
    """
    Load the Breast Cancer Wisconsin dataset.

    Returns
    -------
    X : np.ndarray
        Features.

    y : np.ndarray
        Labels.
    """

    dataset = load_breast_cancer()

    X = dataset.data
    y = dataset.target

    return X, y


def apply_pca(
    X_train,
    X_test,
    n_components=4,
):
    """
    Apply PCA dimensionality reduction.

    Parameters
    ----------
    X_train : np.ndarray
    X_test : np.ndarray
    n_components : int

    Returns
    -------
    X_train_pca : np.ndarray
    X_test_pca : np.ndarray
    """

    pca = PCA(n_components=n_components)

    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    return X_train_pca, X_test_pca


def load_breast_cancer_dataset(
    test_size=0.2,
    random_state=42,
    use_pca=True,
    n_components=4,
):
    """
    Load and preprocess the Breast Cancer Wisconsin dataset.

    Pipeline:
    1. Load dataset
    2. Train/test split
    3. Standardization
    4. Optional PCA

    Parameters
    ----------
    test_size : float
        Fraction used for test split.

    random_state : int
        Random seed for reproducibility.

    use_pca : bool
        Whether to apply PCA.

    n_components : int
        Number of PCA components.

    Returns
    -------
    X_train : np.ndarray
    X_test : np.ndarray
    y_train : np.ndarray
    y_test : np.ndarray
    """

    # Load dataset
    X, y = download_breast_cancer_dataset()

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # Standardization
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Optional PCA
    if use_pca:
        X_train, X_test = apply_pca(
            X_train,
            X_test,
            n_components=n_components,
        )

    return X_train, X_test, y_train, y_test