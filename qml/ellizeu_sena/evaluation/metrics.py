from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true,
    y_pred,
    y_score=None,
):
    """
    Compute classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels.

    y_pred : np.ndarray
        Predicted labels.

    y_score : np.ndarray, optional
        Prediction probabilities or decision scores.

    Returns
    -------
    metrics : dict
        Dictionary containing classification metrics.
    """

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }

    # Optional ROC-AUC
    if y_score is not None:
        metrics["roc_auc"] = roc_auc_score(
            y_true,
            y_score,
        )

    return metrics