import os
import json
import matplotlib.pyplot as plt

from sklearn.metrics import (
    ConfusionMatrixDisplay,
)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def plot_metric_comparison(
    classical_results,
    quantum_results,
    metric="accuracy",
    output_path=None,
):
    """
    Plot comparison between Classical and Quantum models.
    """

    classical_mean = classical_results[f"{metric}_mean"]
    classical_std = classical_results[f"{metric}_std"]

    quantum_mean = quantum_results[f"{metric}_mean"]
    quantum_std = quantum_results[f"{metric}_std"]

    labels = ["Classical SVM", "Quantum SVM"]

    means = [classical_mean, quantum_mean]
    stds = [classical_std, quantum_std]

    plt.figure(figsize=(6, 4))
    plt.bar(labels, means, yerr=stds, capsize=10)

    plt.title(f"{metric.upper()} Comparison")
    plt.ylabel(metric)

    if output_path:
        ensure_dir(os.path.dirname(output_path))
        plt.savefig(output_path, bbox_inches="tight")

    plt.show()


def plot_from_saved(
    save_path="data/ellizeu_sena/results",
    output_dir="data/ellizeu_sena/plots",
):
    """
    Load saved benchmark results and generate plots.
    """

    classical = load_json(f"{save_path}/cv_classical.json")
    quantum = load_json(f"{save_path}/cv_quantum.json")

    metrics = ["accuracy", "precision", "recall", "f1"]

    for metric in metrics:
        plot_metric_comparison(
            classical,
            quantum,
            metric=metric,
            output_path=f"{output_dir}/{metric}.pdf",
        )




def plot_confusion_matrix(
    confusion_matrix,
    class_names,
    title="Confusion Matrix",
    output_path=None,
):
    """
    Plot colored confusion matrix.

    Parameters
    ----------
    confusion_matrix : np.ndarray
        Matrix returned by sklearn.

    class_names : list[str]
        Labels for classes.

    title : str
        Figure title.

    output_path : str, optional
        Path to save PDF.
    """

    fig, ax = plt.subplots(
        figsize=(6, 5)
    )

    display = ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix,
        display_labels=class_names,
    )

    display.plot(
        ax=ax,
        colorbar=True,
    )

    ax.set_title(title)

    plt.tight_layout()

    if output_path is not None:

        plt.savefig(
            output_path,
            format="pdf",
            bbox_inches="tight",
        )

    plt.show()

    return fig