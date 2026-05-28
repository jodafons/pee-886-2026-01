import csv
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from qml.eduardo_banaczewski.experiment import CifarExperimentConfig
from qml.eduardo_banaczewski.models.factory import create_model
from qml.eduardo_banaczewski.visualization.plots import (
    plot_experiment_error_bars,
    plot_experiment_loss_curves,
    plot_experiment_param_counts,
)

EXPERIMENTS: List[str] = [
    "exp_cnn_bench",
    "exp_qml_baseline",
    "exp_qml_data_reupload",
    "exp_qml_strong",
    "exp_qcnn_pure_baseline",
    "exp_qcnn_pure_strong",
]


def _load_metrics(experiment_name: str) -> Dict[str, object]:
    metrics_path = PROJECT_ROOT / "outputs" / experiment_name / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"Missing metrics file: {metrics_path}. Run the experiment first, then re-run this script."
        )
    with metrics_path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _aggregate_loss_curves(folds: Dict[str, Dict[str, object]]) -> Dict[str, np.ndarray]:
    histories = [fold_metrics["history"] for _, fold_metrics in sorted(folds.items())]
    if not histories:
        raise ValueError("Cannot aggregate loss curves: no fold histories found.")

    aggregated: Dict[str, np.ndarray] = {}
    for metric_name in ("train_loss", "val_loss"):
        max_epochs = max(len(history[metric_name]) for history in histories)
        curve_values = np.full((len(histories), max_epochs), np.nan, dtype=float)
        for fold_index, history in enumerate(histories):
            values = np.asarray(history[metric_name], dtype=float)
            curve_values[fold_index, : values.size] = values
        aggregated[f"{metric_name}_mean"] = np.nanmean(curve_values, axis=0)
        aggregated[f"{metric_name}_std"] = np.nanstd(curve_values, axis=0)

    return {
        "train_loss_mean": aggregated["train_loss_mean"],
        "train_loss_std": aggregated["train_loss_std"],
        "val_loss_mean": aggregated["val_loss_mean"],
        "val_loss_std": aggregated["val_loss_std"],
    }


def _select_best_fold(folds: Dict[str, Dict[str, object]]) -> str:
    if not folds:
        raise ValueError("Cannot select representation fold: no fold metrics found.")
    for fold_key, metrics in folds.items():
        if "best_val_acc" not in metrics:
            raise ValueError(f"Fold metrics missing best_val_acc for {fold_key}.")
    best_fold, _ = max(
        folds.items(),
        key=lambda item: float(item[1]["best_val_acc"]),
    )
    return best_fold


def _copy_representation_artifacts(
    experiment_name: str,
    folds: Dict[str, Dict[str, object]],
    output_dir: Path,
) -> Dict[str, str]:
    best_fold = _select_best_fold(folds)
    fold_dir = PROJECT_ROOT / "outputs" / experiment_name / best_fold
    pca_src = fold_dir / "pca_projection.pdf"
    tsne_src = fold_dir / "tsne_projection.pdf"
    if not pca_src.exists():
        raise FileNotFoundError(f"Missing PCA projection: {pca_src}")
    if not tsne_src.exists():
        raise FileNotFoundError(f"Missing t-SNE projection: {tsne_src}")
    pca_name = f"{experiment_name}_pca_projection.pdf"
    tsne_name = f"{experiment_name}_tsne_projection.pdf"
    shutil.copy2(pca_src, output_dir / pca_name)
    shutil.copy2(tsne_src, output_dir / tsne_name)
    return {"fold": best_fold, "pca_plot": pca_name, "tsne_plot": tsne_name}


def _count_model_parameters(config: Dict[str, object]) -> Dict[str, int]:
    experiment_config = CifarExperimentConfig(**config)
    model = create_model(experiment_config)
    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {"total_params": int(total_params), "trainable_params": int(trainable_params)}


def main() -> None:
    experiment_names: List[str] = []
    means: List[float] = []
    stds: List[float] = []
    train_loss_means: List[np.ndarray] = []
    train_loss_stds: List[np.ndarray] = []
    val_loss_means: List[np.ndarray] = []
    val_loss_stds: List[np.ndarray] = []
    trainable_params: List[int] = []
    parameter_counts: Dict[str, Dict[str, int]] = {}
    representation_analysis: Dict[str, Dict[str, str]] = {}
    output_dir = PROJECT_ROOT / "outputs" / "exp_all_experiments_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    for experiment_name in EXPERIMENTS:
        metrics = _load_metrics(experiment_name=experiment_name)
        config = metrics.get("config")
        if not isinstance(config, dict):
            raise ValueError(f"Missing config for experiment '{experiment_name}'.")
        summary = metrics["summary"]
        folds = metrics["folds"]
        loss_curves = _aggregate_loss_curves(folds=folds)
        parameter_counts[experiment_name] = _count_model_parameters(config)
        trainable_params.append(parameter_counts[experiment_name]["trainable_params"])
        representation_analysis[experiment_name] = _copy_representation_artifacts(
            experiment_name=experiment_name,
            folds=folds,
            output_dir=output_dir,
        )
        experiment_names.append(experiment_name)
        means.append(float(summary["test_acc_mean"]))
        stds.append(float(summary["test_acc_std"]))
        train_loss_means.append(loss_curves["train_loss_mean"])
        train_loss_stds.append(loss_curves["train_loss_std"])
        val_loss_means.append(loss_curves["val_loss_mean"])
        val_loss_stds.append(loss_curves["val_loss_std"])

    plot_experiment_error_bars(
        experiment_names=experiment_names,
        means=np.asarray(means, dtype=float),
        stds=np.asarray(stds, dtype=float),
        output_path=output_dir / "test_accuracy_error_bars.pdf",
    )
    plot_experiment_param_counts(
        experiment_names=experiment_names,
        param_counts=np.asarray(trainable_params, dtype=float),
        output_path=output_dir / "model_parameter_counts.pdf",
    )
    plot_experiment_loss_curves(
        experiment_names=experiment_names,
        train_loss_means=train_loss_means,
        train_loss_stds=train_loss_stds,
        val_loss_means=val_loss_means,
        val_loss_stds=val_loss_stds,
        output_path=output_dir / "loss_curves_with_error.pdf",
    )
    with (output_dir / "model_parameter_counts.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["experiment", "trainable_params", "total_params"],
        )
        writer.writeheader()
        for experiment_name in experiment_names:
            counts = parameter_counts[experiment_name]
            writer.writerow(
                {
                    "experiment": experiment_name,
                    "trainable_params": counts["trainable_params"],
                    "total_params": counts["total_params"],
                }
            )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                "experiments": experiment_names,
                "test_acc_mean": means,
                "test_acc_std": stds,
                "loss_curve_plot": "loss_curves_with_error.pdf",
                "param_count_plot": "model_parameter_counts.pdf",
                "param_count_csv": "model_parameter_counts.csv",
                "representation_analysis": representation_analysis,
                "parameter_counts": parameter_counts,
            },
            stream,
            indent=2,
        )
    print(f"Saved aggregate analysis artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
