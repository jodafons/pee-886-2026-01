import numpy as np
import matplotlib.pyplot as plt


def compute_kernel_matrix(quantum_kernel, X):
    """
    Compute full kernel matrix K(x_i, x_j).
    """

    n_samples = len(X)
    kernel_matrix = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(n_samples):
            kernel_matrix[i, j] = quantum_kernel.evaluate(X[i], X[j])

    return kernel_matrix


def plot_kernel_heatmap(
    kernel_matrix,
    title="Quantum Kernel Heatmap",
    output_path=None,
):
    """
    Plot heatmap of kernel matrix.
    """

    plt.figure(figsize=(6, 5))
    plt.imshow(kernel_matrix, interpolation="nearest")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel("Samples")

    if output_path is not None:
        plt.savefig(output_path, bbox_inches="tight")

    plt.show()


def run_and_plot_kernel_heatmap(
    quantum_kernel,
    X,
    output_path=None,
):
    """
    Compute + plot kernel heatmap.
    """

    K = compute_kernel_matrix(quantum_kernel, X)
    plot_kernel_heatmap(K, output_path=output_path)

    return K