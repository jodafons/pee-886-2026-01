import csv
import json
import os


def get_best_result_from_csv(
    csv_path,
    metric="accuracy_mean",
):
    best_row = None
    best_score = float(
        "-inf"
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

            score = float(
                row[metric]
            )

            if score > best_score:

                best_score = score
                best_row = row

    return best_row

def build_best_parameters_json(
    model_name,
    save_dir="data/ellizeu_sena/grid_search",
    metric="accuracy_mean",
):
    """
    Generate best parameter JSON
    from CSV results.
    """

    csv_path = os.path.join(
        save_dir,
        f"{model_name}_results.csv",
    )

    json_path = os.path.join(
        save_dir,
        f"{model_name}_best_parameters.json",
    )

    best_row = (
        get_best_result_from_csv(
            csv_path,
            metric,
        )
    )

    metric_fields = {
        "accuracy_mean",
        "accuracy_std",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "f1_mean",
        "f1_std",
        "execution_time_seconds",
    }

    best_params = {
        k: v
        for k, v in best_row.items()
        if k not in metric_fields
    }

    best_results = {
        k: float(v)
        for k, v in best_row.items()
        if k in metric_fields
    }

    summary = {
        "best_params": best_params,
        "best_score": float(
            best_row[metric]
        ),
        "results": best_results,
    }

    with open(
        json_path,
        "w",
    ) as jsonfile:

        json.dump(
            summary,
            jsonfile,
            indent=4,
        )

    return summary