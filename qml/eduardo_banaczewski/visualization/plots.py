from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

__all__ = ["plot_confusion_matrix", "plot_training_curves"]
__all__.append("plot_fold_accuracy_comparison")
__all__.extend(
    [
        "plot_tsne_projection",
        "plot_pca_projection",
        "plot_quantum_circuit",
        "plot_experiment_error_bars",
        "plot_experiment_param_counts",
        "plot_experiment_loss_curves",
    ]
)


def _style_axes(ax: plt.Axes) -> None:
    """Apply a cleaner visual style to axes."""
    ax.grid(True, alpha=0.25, linestyle="--", linewidth=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save_figure(
    fig: plt.Figure,
    output_path: Path,
    dpi: int = 220,
    tight_layout_rect: Optional[Tuple[float, float, float, float]] = None,
) -> None:
    """Persist figure with consistent export settings."""
    if tight_layout_rect is None:
        fig.tight_layout()
    else:
        fig.tight_layout(rect=tight_layout_rect)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history: Dict[str, Sequence[float]], output_path: Path) -> None:
    """Plot training/validation loss and accuracy curves for one fold."""
    plt.style.use("seaborn-v0_8-whitegrid")
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="train_loss", linewidth=2.1)
    axes[0].plot(epochs, history["val_loss"], label="val_loss", linewidth=2.1)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Curves")
    axes[0].legend()
    _style_axes(axes[0])

    axes[1].plot(epochs, history["train_acc"], label="train_acc", linewidth=2.1)
    axes[1].plot(epochs, history["val_acc"], label="val_acc", linewidth=2.1)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy Curves")
    axes[1].legend()
    _style_axes(axes[1])
    _save_figure(fig, output_path, tight_layout_rect=(0.0, 0.14, 1.0, 1.0))


def plot_confusion_matrix(
    confusion_matrix: np.ndarray,
    class_names: Tuple[str, ...],
    output_path: Path,
) -> None:
    """Plot and export confusion matrix for one fold."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(confusion_matrix, cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    threshold = confusion_matrix.max() * 0.5 if confusion_matrix.size else 0.0
    for row in range(confusion_matrix.shape[0]):
        for col in range(confusion_matrix.shape[1]):
            color = "white" if confusion_matrix[row, col] > threshold else "black"
            ax.text(col, row, str(confusion_matrix[row, col]), ha="center", va="center", color=color)

    _style_axes(ax)
    _save_figure(fig, output_path)


def plot_fold_accuracy_comparison(
    val_acc: np.ndarray,
    test_acc: np.ndarray,
    output_path: Path,
) -> None:
    """Plot grouped bars comparing validation and test accuracy by fold."""
    plt.style.use("seaborn-v0_8-whitegrid")
    folds = np.arange(1, val_acc.size + 1)
    width = 0.28
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        folds - width / 2,
        val_acc,
        width=width,
        label="Validation Accuracy",
        color="#4C78A8",
        alpha=0.9,
    )
    ax.bar(
        folds + width / 2,
        test_acc,
        width=width,
        label="Test Accuracy",
        color="#F58518",
        alpha=0.9,
    )
    ax.set_xlabel("Fold")
    ax.set_ylabel("Accuracy")
    ax.set_title("Validation vs Test Accuracy by Fold")
    ax.set_xticks(folds)
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    _style_axes(ax)
    _save_figure(fig, output_path)


def _prepare_projection_data(
    embeddings: np.ndarray,
    labels: np.ndarray,
    seed: int,
    max_points: int = 3000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Subsample embeddings when needed for readable/fast projection plots."""
    if embeddings.shape[0] <= max_points:
        return embeddings, labels
    rng = np.random.default_rng(seed)
    idx = rng.choice(embeddings.shape[0], size=max_points, replace=False)
    return embeddings[idx], labels[idx]


def _plot_scatter_projection(
    projected: np.ndarray,
    labels: np.ndarray,
    title: str,
    output_path: Path,
) -> None:
    """Save a 2D scatter projection colored by class label."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        projected[:, 0],
        projected[:, 1],
        c=labels,
        cmap="tab10",
        s=14,
        alpha=0.8,
        linewidths=0.0,
    )
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    legend = ax.legend(
        *scatter.legend_elements(num=10),
        title="Class",
        loc="best",
        fontsize=8,
    )
    ax.add_artist(legend)
    _style_axes(ax)
    _save_figure(fig, output_path)


def plot_pca_projection(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    seed: int,
) -> None:
    """Run PCA over model logits/features and export a 2D scatter plot."""
    sampled_embeddings, sampled_labels = _prepare_projection_data(embeddings, labels, seed=seed)
    pca = PCA(n_components=2, random_state=seed)
    projected = pca.fit_transform(sampled_embeddings)
    plot_title = f"PCA Projection ({sampled_embeddings.shape[0]} samples)"
    _plot_scatter_projection(projected, sampled_labels, plot_title, output_path)


def plot_tsne_projection(
    embeddings: np.ndarray,
    labels: np.ndarray,
    output_path: Path,
    seed: int,
) -> None:
    """Run t-SNE over model logits/features and export a 2D scatter plot."""
    sampled_embeddings, sampled_labels = _prepare_projection_data(embeddings, labels, seed=seed)
    tsne = TSNE(n_components=2, random_state=seed, init="pca", learning_rate="auto")
    projected = tsne.fit_transform(sampled_embeddings)
    plot_title = f"t-SNE Projection ({sampled_embeddings.shape[0]} samples)"
    _plot_scatter_projection(projected, sampled_labels, plot_title, output_path)


def plot_quantum_circuit(
    model_name: str,
    n_qubits: int,
    n_q_layers: int,
    output_path: Path,
) -> None:
    """Export circuit diagram for quantum models; write placeholder for CNN baseline."""
    plt.style.use("seaborn-v0_8-whitegrid")
    if model_name == "cnn_benchmark":
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.text(
            0.5,
            0.5,
            "No quantum circuit for cnn_benchmark.",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.axis("off")
        _save_figure(fig, output_path)
        return

    variant = model_name.replace("qcnn_pure_", "qml_", 1)
    try:
        import pennylane as qml
    except ImportError as exc:
        raise ImportError("Pennylane is required to draw quantum circuits.") from exc

    device = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(device)
    def quantum_circuit(inputs, weights):
        if variant == "qml_baseline":
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
            for layer in range(n_q_layers):
                for wire in range(n_qubits):
                    qml.RY(weights[layer, wire], wires=wire)
                for wire in range(n_qubits):
                    qml.CNOT(wires=(wire, (wire + 1) % n_qubits))
        elif variant == "qml_strong":
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="X")
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        elif variant == "qml_data_reupload":
            for layer in range(n_q_layers):
                qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation="Y")
                for wire in range(n_qubits):
                    qml.RZ(weights[layer, wire, 0], wires=wire)
                    qml.RY(weights[layer, wire, 1], wires=wire)
                for wire in range(n_qubits):
                    qml.CZ(wires=(wire, (wire + 1) % n_qubits))
        else:
            raise ValueError(f"Unsupported QML variant: {variant}")
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    if variant == "qml_baseline":
        weights = np.zeros((n_q_layers, n_qubits))
    elif variant == "qml_strong":
        weights = np.zeros((n_q_layers, n_qubits, 3))
    else:
        weights = np.zeros((n_q_layers, n_qubits, 2))
    inputs = np.zeros((n_qubits,))
    fig, _ = qml.draw_mpl(quantum_circuit)(inputs, weights)
    _save_figure(fig, output_path)


def plot_experiment_error_bars(
    experiment_names: Sequence[str],
    means: np.ndarray,
    stds: np.ndarray,
    output_path: Path,
) -> None:
    """Plot mean test accuracy with standard deviation across experiments."""
    plt.style.use("seaborn-v0_8-whitegrid")
    x = np.arange(len(experiment_names))
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(
        x,
        means,
        yerr=stds,
        width=0.68,
        capsize=2,
        color="#4C78A8",
        alpha=0.92,
        edgecolor="#2F4B7C",
        linewidth=0.8,
        error_kw={"elinewidth": 0.7, "ecolor": "#2F4B7C"},
    )
    ax.set_xticks(x)
    ax.set_xticklabels(experiment_names, rotation=30, ha="right")
    ax.set_ylabel("Test Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Test Accuracy (mean ± std) by Experiment")
    _style_axes(ax)
    _save_figure(fig, output_path)


def plot_experiment_param_counts(
    experiment_names: Sequence[str],
    param_counts: np.ndarray,
    output_path: Path,
) -> None:
    """Plot trainable parameter counts for each experiment."""
    plt.style.use("seaborn-v0_8-whitegrid")
    x = np.arange(len(experiment_names))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        x,
        param_counts,
        width=0.62,
        color="#7E57C2",
        alpha=0.9,
        edgecolor="#4E2A8E",
        linewidth=0.8,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(experiment_names, rotation=30, ha="right")
    ax.set_ylabel("Trainable Parameters")
    ax.set_title("Model Complexity (Trainable Parameters)")
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    _style_axes(ax)
    _save_figure(fig, output_path)


def plot_experiment_loss_curves(
    experiment_names: Sequence[str],
    train_loss_means: Sequence[np.ndarray],
    train_loss_stds: Sequence[np.ndarray],
    val_loss_means: Sequence[np.ndarray],
    val_loss_stds: Sequence[np.ndarray],
    output_path: Path,
) -> None:
    """Plot mean loss curves with std bands for all experiments."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)
    colors = plt.cm.tab10(np.linspace(0, 1, len(experiment_names)))

    for index, experiment_name in enumerate(experiment_names):
        train_mean = train_loss_means[index]
        train_std = train_loss_stds[index]
        val_mean = val_loss_means[index]
        val_std = val_loss_stds[index]
        train_epochs = np.arange(1, train_mean.size + 1)
        val_epochs = np.arange(1, val_mean.size + 1)

        axes[0].plot(
            train_epochs,
            train_mean,
            label=experiment_name,
            color=colors[index],
            linewidth=2.0,
        )
        axes[0].fill_between(
            train_epochs,
            train_mean - train_std,
            train_mean + train_std,
            color=colors[index],
            alpha=0.15,
        )

        axes[1].plot(
            val_epochs,
            val_mean,
            label=experiment_name,
            color=colors[index],
            linewidth=2.0,
        )
        axes[1].fill_between(
            val_epochs,
            val_mean - val_std,
            val_mean + val_std,
            color=colors[index],
            alpha=0.15,
        )

    axes[0].set_title("Train Loss by Experiment (mean ± std)")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    _style_axes(axes[0])

    axes[1].set_title("Validation Loss by Experiment (mean ± std)")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    _style_axes(axes[1])

    for ax in axes:
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.7)

    axes[1].legend(
        loc="upper right",
        fontsize="small",
        frameon=False,
        handlelength=1.4,
        labelspacing=0.3,
    )
    _save_figure(fig, output_path)
