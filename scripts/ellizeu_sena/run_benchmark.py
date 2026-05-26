import os
import json
import numpy as np

from qml.ellizeu_sena.loaders import load_breast_cancer_dataset
from qml.ellizeu_sena.models import ClassicalSVM, QuantumSVM
from qml.ellizeu_sena.evaluation import run_cross_validation


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def run_model_cv(model_class, params, X, y, n_splits=5):
    """
    Wrapper for cross-validation execution.
    """
    return run_cross_validation(
        model_class=model_class,
        model_params=params,
        X=X,
        y=y,
        n_splits=n_splits,
    )


def save_json(path, data):
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def main():
    # ----------------------------
    # 1. LOAD DATA
    # ----------------------------
    print("Loading dataset...")
    X_train, X_test, y_train, y_test = load_breast_cancer_dataset(
        n_components=4
    )

    # ----------------------------
    # 2. CONFIG MODELS
    # ----------------------------
    classical_params = {
        "kernel": "rbf",
    }

    quantum_params = {
        "num_features": 4,
        "reps": 2,
    }

    # ----------------------------
    # 3. CREATE OUTPUT DIR
    # ----------------------------
    base_path = "data/ellizeu_sena"
    results_path = f"{base_path}/results"
    ensure_dir(results_path)

    # ----------------------------
    # 4. RUN CROSS VALIDATION
    # ----------------------------
    print("Running Classical SVM CV...")
    classical_results = run_model_cv(
        ClassicalSVM,
        classical_params,
        X_train,
        y_train,
        n_splits=5,
    )

    print("Running Quantum SVM CV...")
    quantum_results = run_model_cv(
        QuantumSVM,
        quantum_params,
        X_train,
        y_train,
        n_splits=5,
    )

    # ----------------------------
    # 5. SAVE RESULTS
    # ----------------------------
    print("Saving results...")

    save_json(f"{results_path}/cv_classical.json", classical_results)
    save_json(f"{results_path}/cv_quantum.json", quantum_results)

    benchmark = {
        "classical": classical_results,
        "quantum": quantum_results,
    }

    save_json(f"{results_path}/benchmark.json", benchmark)

    print("Benchmark completed successfully!")


if __name__ == "__main__":
    main()