import os

from qml.ellizeu_sena.loaders import (
    download_breast_cancer_dataset,
)

from qml.ellizeu_sena.models import (
    QuantumSVM,
)

from qml.ellizeu_sena.trainer import (
    run_grid_search,
    build_best_parameters_json,
)


DATA_PATH = os.path.join(
    "data",
    "ellizeu_sena",
)

GRID_SEARCH_PATH = os.path.join(
    DATA_PATH,
    "grid_search",
)


def main():

    X, y = download_breast_cancer_dataset()

    param_grid = {
        "num_features": [4],
        "C": [
            0.1,
            1,
            10,
        ],
        "reps": [
            1,
            2,
            3,
        ],
        "entanglement": [
            "linear",
            "full",
        ],
    }

    run_grid_search(
        model_class=QuantumSVM,
        model_name="quantum_svm",
        param_grid=param_grid,
        X=X,
        y=y,
        n_splits=5,
        use_pca=True,
        n_components=4,
        save_dir=GRID_SEARCH_PATH,
    )

    build_best_parameters_json(
        model_name="quantum_svm",
        save_dir=GRID_SEARCH_PATH,
    )


if __name__ == "__main__":
    main()