import os

from qml.ellizeu_sena.loaders import (
    download_breast_cancer_dataset,
    split_dataset,
    preprocessing_pipeline,
)

from qml.ellizeu_sena.models import (
    ClassicalSVM,
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
        X=X_train,
        y=y_train,
        n_splits=5,
        use_pca=False,
        save_dir=GRID_SEARCH_PATH,
    )

    best_summary = build_best_parameters_json(
        model_name="classical_svm",
        save_dir=GRID_SEARCH_PATH,
    )

    best_params = best_summary[
        "best_params"
    ]

    best_params["C"] = float(
        best_params["C"]
    )
    
    if best_params["gamma"] != "scale":
        best_params["gamma"] = float(
            best_params["gamma"]
        )


    # -------------------------
    # Train final model
    # -------------------------
    
    X_train_processed, _ = preprocessing_pipeline(
        X_train,
        X_test,
        use_pca=False,
        n_components=4,
    )
    
    model = ClassicalSVM(
        **best_params
    )
    
    model.fit(
        X_train_processed,
        y_train,
    )
    
    # -------------------------
    # Save model
    # -------------------------
    
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
            "classical_svm.joblib",
        )
    )
    
    print(
        "Final model trained and saved."
    )


if __name__ == "__main__":
    main()