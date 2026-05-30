import os

from qml.ellizeu_sena.loaders import (
    load_breast_cancer_dataset,
)

from qml.ellizeu_sena.models import (
    ClassicalSVM,
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
    
    X, X_test, y, y_test = (
        load_breast_cancer_dataset(
            use_pca=False
        )
    )

    param_grid = {
        "C": [
            0.1,
            1,
            10,
            100,
        ],
        "gamma": [
            "scale",
            0.1,
            0.01,
            0.001,
        ],
    }

    run_grid_search(
        model_class=ClassicalSVM,
        model_name="classical_svm",
        param_grid=param_grid,
        X=X,
        y=y,
        n_splits=5,
        use_pca=True,
        n_components=4,
        save_dir=GRID_SEARCH_PATH,
    )

    build_best_parameters_json(
        model_name="classical_svm",
        save_dir=GRID_SEARCH_PATH,
    )


if __name__ == "__main__":
    main()