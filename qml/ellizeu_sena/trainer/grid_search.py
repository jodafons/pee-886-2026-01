import csv
import os
import time

from itertools import product

from qml.ellizeu_sena.evaluation import (
    run_cross_validation,
)


def ensure_dir(path):
    os.makedirs(
        path,
        exist_ok=True,
    )


def generate_parameter_combinations(
    param_grid,
):
    keys = list(
        param_grid.keys()
    )

    values = [
        param_grid[key]
        for key in keys
    ]

    for combination in product(*values):

        yield dict(
            zip(
                keys,
                combination,
            )
        )


def parameter_signature(
    params,
):
    return tuple(
        sorted(
            params.items()
        )
    )


def load_completed_runs(
    csv_path,
    param_grid,
):
    completed = set()

    if not os.path.exists(
        csv_path
    ):
        return completed

    parameter_names = set(
        param_grid.keys()
    )

    with open(
        csv_path,
        "r",
        newline="",
    ) as csvfile:

        reader = csv.DictReader(
            csvfile
        )

        for row in reader:

            params = {
                key: row[key]
                for key in parameter_names
            }

            completed.add(
                parameter_signature(
                    params
                )
            )

    return completed


def run_grid_search(
    model_class,
    model_name,
    param_grid,
    X,
    y,
    n_splits=5,
    random_state=42,
    use_pca=True,
    n_components=4,
    save_dir="data/ellizeu_sena/grid_search",
):
    """
    Execute grid search and save
    results incrementally to CSV.
    """

    ensure_dir(
        save_dir
    )

    csv_path = os.path.join(
        save_dir,
        f"{model_name}_results.csv",
    )

    csv_exists = os.path.exists(
        csv_path
    )

    completed_runs = (
        load_completed_runs(
            csv_path,
            param_grid,
        )
    )

    for params in generate_parameter_combinations(
        param_grid
    ):

        signature = (
            parameter_signature(
                {
                    k: str(v)
                    for k, v in params.items()
                }
            )
        )

        if signature in completed_runs:

            print(
                f"Skipping existing: {params}"
            )

            continue

        print(
            f"Testing: {params}"
        )

        start_time = (
            time.perf_counter()
        )

        results = run_cross_validation(
            model_class=model_class,
            model_params=params,
            X=X,
            y=y,
            n_splits=n_splits,
            random_state=random_state,
            use_pca=use_pca,
            n_components=n_components,
        )

        elapsed_time = (
            time.perf_counter()
            - start_time
        )

        row = {
            **params,
            **results,
            "execution_time_seconds": elapsed_time,
        }

        with open(
            csv_path,
            "a",
            newline="",
        ) as csvfile:

            writer = csv.DictWriter(
                csvfile,
                fieldnames=row.keys(),
            )

            if not csv_exists:

                writer.writeheader()
                csv_exists = True

            writer.writerow(
                row
            )

        completed_runs.add(
            signature
        )

    return csv_path