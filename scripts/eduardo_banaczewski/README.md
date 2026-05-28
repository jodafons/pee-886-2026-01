# Scripts - Eduardo Banaczewski

## CIFAR experiment runner

Run any individual experiment:

```bash
./scripts/eduardo_banaczewski/01_cnn_bench.sh
./scripts/eduardo_banaczewski/02_qml_baseline.sh
./scripts/eduardo_banaczewski/03_qml_data_reupload.sh
./scripts/eduardo_banaczewski/04_qml_strong.sh
./scripts/eduardo_banaczewski/05_qcnn_pure_baseline.sh
./scripts/eduardo_banaczewski/06_qcnn_pure_strong.sh
```

Run all experiments in sequence:

```bash
./scripts/eduardo_banaczewski/07_run_all_experiments.sh
```

After all experiments finish, generate aggregate analysis (accuracy error bars + loss curves with error bands):

```bash
./scripts/eduardo_banaczewski/08_run_all_experiments_with_analysis.sh
```

Each run creates `outputs/<experiment_name>/` with per-fold checkpoints, metrics, CSV predictions, and PDF plots (including circuit, PCA, and t-SNE). The aggregate script also writes:

- `outputs/exp_all_experiments_analysis/test_accuracy_error_bars.pdf`
- `outputs/exp_all_experiments_analysis/loss_curves_with_error.pdf`
- `outputs/exp_all_experiments_analysis/model_parameter_counts.pdf`
- `outputs/exp_all_experiments_analysis/model_parameter_counts.csv`
- `outputs/exp_all_experiments_analysis/<experiment>_pca_projection.pdf`
- `outputs/exp_all_experiments_analysis/<experiment>_tsne_projection.pdf`
- `outputs/exp_all_experiments_analysis/summary.json`

Runtime defaults:

- CUDA is required by default (`--allow-cpu` disables this check).
- QML models use PennyLane `default.qubit` (set by `--qml-device`, default: `default.qubit`).
- Shell scripts prefer `./.venv/bin/python` automatically (fallback: `python3`).
