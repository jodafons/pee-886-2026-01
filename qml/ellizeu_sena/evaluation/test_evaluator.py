import os

from .metrics import (
    compute_classification_metrics,
)


def evaluate_model(
    model,
    X_test,
    y_test,
):
    """
    Evaluate a trained model.
    """

    predictions = model.predict(
        X_test
    )

    return compute_classification_metrics(
        y_test,
        predictions,
    )


def evaluate_saved_model(
    model_class,
    model_path,
    X_test,
    y_test,
):
    """
    Load and evaluate a saved model.
    """

    model = model_class.load(
        model_path
    )

    return evaluate_model(
        model,
        X_test,
        y_test,
    )