import os

from qml.ellizeu_sena.loaders import (
    download_breast_cancer_dataset,
    split_dataset,
    preprocessing_pipeline,
)

from qml.ellizeu_sena.models import (
    QuantumSVM,
)

from qml.ellizeu_sena.trainer import (
    run_grid_search,
    build_best_parameters_json,
)


PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.getcwd(),
        "..",
        "..",
    )
)
DATA_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "ellizeu_sena",
)
GRID_SEARCH_PATH = os.path.join(
    DATA_PATH,
    "grid_search",
)


def main():

    # 1. load raw data
    X, y = download_breast_cancer_dataset()
    
    # 2. split only
    X_train, X_test, y_train, y_test = split_dataset(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    param_grid = {
        "num_features": [30],
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
        X=X_train,
        y=y_train,
        n_splits=5,
        use_pca=False,
        n_components=4,
        save_dir=GRID_SEARCH_PATH,
    )

    best_summary = build_best_parameters_json(
        model_name="quantum_svm",
        save_dir=GRID_SEARCH_PATH,
    )
    
    best_params = best_summary[
        "best_params"
    ]
    
    best_params["num_features"] = int(
        best_params["num_features"]
    )
    
    best_params["reps"] = int(
        best_params["reps"]
    )
    
    best_params["C"] = float(
        best_params["C"]
    )

    X_train_processed, X_test_processed = (
        preprocessing_pipeline(
            X_train,
            X_test,
            use_pca=True,
            n_components=4,
        )
    )
    
    model = QuantumSVM(
        **best_params
    )
    
    model.fit(
        X_train_processed,
        y_train,
    )

    MODELS_PATH = os.path.join(
        DATA_PATH,
        "models",
    )
    
    os.makedirs(
        MODELS_PATH,
        exist_ok=True,
    )
    
    model.save(
        os.path.join(
            MODELS_PATH,
            "quantum_svm.joblib",
        )
    )
    
    print(
        "Final model trained and saved."
    )


if __name__ == "__main__":
    main()