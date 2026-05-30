import numpy as np

from sklearn.model_selection import StratifiedKFold

from qml.ellizeu_sena.loaders import (
    preprocessing_pipeline,
)

from qml.ellizeu_sena.trainer import (
    ModelTrainer,
)

from .metrics import (
    compute_classification_metrics,
)


def run_cross_validation(
    model_class,
    model_params,
    X,
    y,
    n_splits=5,
    random_state=42,
    use_pca=True,
    n_components=4,
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

    use_pca : bool
        Whether to apply PCA inside each fold.

    n_components : int
        Number of PCA components.

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

        # -------------------------
        # Fold split
        # -------------------------

        X_train = X[train_index]
        X_test = X[test_index]

        y_train = y[train_index]
        y_test = y[test_index]

        # -------------------------
        # Preprocessing
        # -------------------------

        X_train, X_test = preprocessing_pipeline(
            X_train,
            X_test,
            use_pca=use_pca,
            n_components=n_components,
        )

        # -------------------------
        # Model
        # -------------------------

        model = model_class(
            **model_params
        )

        trainer = ModelTrainer(
            model
        )

        # -------------------------
        # Training + Prediction
        # -------------------------

        predictions = trainer.fit_predict(
            X_train,
            y_train,
            X_test,
        )

        # -------------------------
        # Evaluation
        # -------------------------

        metrics = compute_classification_metrics(
            y_test,
            predictions,
        )

        accuracy_scores.append(
            metrics["accuracy"]
        )

        precision_scores.append(
            metrics["precision"]
        )

        recall_scores.append(
            metrics["recall"]
        )

        f1_scores.append(
            metrics["f1_score"]
        )

    # -------------------------
    # Aggregate results
    # -------------------------

    results = {
        "accuracy_mean": float(
            np.mean(accuracy_scores)
        ),
        "accuracy_std": float(
            np.std(accuracy_scores)
        ),

        "precision_mean": float(
            np.mean(precision_scores)
        ),
        "precision_std": float(
            np.std(precision_scores)
        ),

        "recall_mean": float(
            np.mean(recall_scores)
        ),
        "recall_std": float(
            np.std(recall_scores)
        ),

        "f1_mean": float(
            np.mean(f1_scores)
        ),
        "f1_std": float(
            np.std(f1_scores)
        ),
    }

    return results