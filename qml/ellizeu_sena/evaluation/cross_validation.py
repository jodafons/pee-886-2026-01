import numpy as np

from sklearn.model_selection import StratifiedKFold

from .metrics import compute_classification_metrics


def run_cross_validation(
    model_class,
    model_params,
    X,
    y,
    n_splits=5,
    random_state=42,
):
    """
    Run stratified cross-validation.

    Parameters
    ----------
    model_class : class
        Model class to instantiate.

    model_params : dict
        Parameters used to instantiate the model.

    X : np.ndarray
        Features.

    y : np.ndarray
        Labels.

    n_splits : int
        Number of folds.

    random_state : int
        Random seed.

    Returns
    -------
    results : dict
        Mean and standard deviation of metrics.
    """

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )

    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    f1_scores = []

    for train_index, test_index in skf.split(X, y):

        # Split fold
        X_train = X[train_index]
        X_test = X[test_index]

        y_train = y[train_index]
        y_test = y[test_index]

        # Create model
        model = model_class(**model_params)

        # Train
        model.fit(X_train, y_train)

        # Predict
        predictions = model.predict(X_test)

        # Metrics
        metrics = compute_classification_metrics(
            y_test,
            predictions,
        )

        accuracy_scores.append(metrics["accuracy"])
        precision_scores.append(metrics["precision"])
        recall_scores.append(metrics["recall"])
        f1_scores.append(metrics["f1_score"])

    results = {
        "accuracy_mean": np.mean(accuracy_scores),
        "accuracy_std": np.std(accuracy_scores),

        "precision_mean": np.mean(precision_scores),
        "precision_std": np.std(precision_scores),

        "recall_mean": np.mean(recall_scores),
        "recall_std": np.std(recall_scores),

        "f1_mean": np.mean(f1_scores),
        "f1_std": np.std(f1_scores),
    }

    return results