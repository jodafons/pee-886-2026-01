import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def download_breast_cancer_dataset(
    return_metadata=False,
):
    """
    Load the Breast Cancer Wisconsin dataset.

    Parameters
    ----------
    return_metadata : bool
        Whether to return feature names and target names.

    Returns
    -------
    X : np.ndarray
    y : np.ndarray

    Optional
    --------
    feature_names : np.ndarray
    target_names : np.ndarray
    """

    dataset = load_breast_cancer()

    X = dataset.data
    y = dataset.target

    if return_metadata:
        return (
            X,
            y,
            dataset.feature_names,
            dataset.target_names,
        )

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

def apply_standardization(
    X_train,
    X_test,
):
    """
    Apply feature standardization.

    Parameters
    ----------
    X_train : np.ndarray
    X_test : np.ndarray

    Returns
    -------
    X_train_scaled : np.ndarray
    X_test_scaled : np.ndarray
    """

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled

def preprocessing_pipeline(
    X_train,
    X_test,
    use_pca=True,
    n_components=4,
):
    """
    Complete preprocessing pipeline.

    Steps:
    1. Standardization
    2. Optional PCA

    Parameters
    ----------
    X_train : np.ndarray
    X_test : np.ndarray
    use_pca : bool
    n_components : int

    Returns
    -------
    X_train : np.ndarray
    X_test : np.ndarray
    """

    X_train, X_test = apply_standardization(
        X_train,
        X_test,
    )

    if use_pca:
        X_train, X_test = apply_pca(
            X_train,
            X_test,
            n_components=n_components,
        )

    return X_train, X_test

def process_dataset(
    X,
    y,
    test_size=0.2,
    random_state=42,
    use_pca=True,
    n_components=4,
):
    """
    Split and preprocess a dataset.

    Parameters
    ----------
    X : np.ndarray
        Features.

    y : np.ndarray
        Labels.

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

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    X_train, X_test = preprocessing_pipeline(
        X_train,
        X_test,
        use_pca=use_pca,
        n_components=n_components,
    )

    return X_train, X_test, y_train, y_test


def load_breast_cancer_dataset(
    test_size=0.2,
    random_state=42,
    use_pca=True,
    n_components=4,
):
    """
    Load and preprocess the Breast Cancer Wisconsin dataset.
    """

    X, y = download_breast_cancer_dataset()

    return process_dataset(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        use_pca=use_pca,
        n_components=n_components,
    )