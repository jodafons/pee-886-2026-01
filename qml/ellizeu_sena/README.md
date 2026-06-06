# 👤 Student Space

This directory is dedicated to the student's individual implementation. Here you should organize your modules, experiments, and technical documentation.

## 📜 Contribution Rules (Reminder)

1. **Naming Convention**: All files must use `snake_case` (lowercase letters and underscores). Ex: `my_model.py`.
2. **Location**: Work only within the folders identified with your letter (📁 `qml/<letter>/`, 📁 `notebooks/<letter>/`, 📁 `scripts/<letter>/`, 📁 `data/<letter>/`).
3. **Dependencies**: Add new packages only to the `requirements.txt` file at the root of the repository.

---

## 🛠️ About this Implementation

This project investigates the application of Quantum Support Vector Machines (QSVMs) to the Breast Cancer Wisconsin Diagnostic Dataset.

The main objective is to compare a classical Support Vector Machine (SVM) against a quantum kernel-based Support Vector Machine (QSVM), evaluating classification performance through cross-validation and statistical metrics.

The implementation follows the modular architecture proposed by the course repository and includes data loading, preprocessing, model implementation, evaluation, benchmarking, and visualization utilities.

### 🚀 Technologies and Architecture

#### Technologies

* Python
* NumPy
* Scikit-Learn
* Matplotlib
* Qiskit
* Qiskit Aer
* Qiskit Machine Learning
* Jupyter Notebook

#### Project Structure

```text
qml/ellizeu_sena/
├── loaders/
├── models/
├── trainer/
├── evaluation/
├── visualization/
└── README.md
```

##### loaders

Responsible for dataset loading and preprocessing.

Features:

* Breast Cancer Wisconsin dataset loader
* Feature standardization using StandardScaler
* Optional dimensionality reduction using PCA

##### models

Contains machine learning models.

Implemented models:

* ClassicalSVM
* QuantumSVM

The QuantumSVM uses:

* ZZFeatureMap
* FidelityQuantumKernel
* QSVC

##### trainer

Training utilities and model execution helpers.

##### evaluation

Evaluation and benchmarking tools.

Implemented features:

* Accuracy
* Precision
* Recall
* F1-score
* Stratified Cross Validation

##### visualization

Visualization utilities for analysis and reporting.

Implemented features:

* Quantum circuit visualization
* Quantum kernel heatmap
* Benchmark comparison plots
* Dataset exploration plots

### 📖 Usage Instructions

#### Loading the Dataset

```python
from qml.ellizeu_sena.loaders import load_breast_cancer_dataset

X_train, X_test, y_train, y_test = load_breast_cancer_dataset(
    use_pca=True,
    n_components=4,
)
```

#### Classical SVM

```python
from qml.ellizeu_sena.models import ClassicalSVM

model = ClassicalSVM()

model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

#### Quantum SVM

```python
from qml.ellizeu_sena.models import QuantumSVM

model = QuantumSVM(
    num_features=4,
    reps=2,
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
```

#### Cross Validation

```python
from qml.ellizeu_sena.evaluation import run_cross_validation

results = run_cross_validation(
    model_class=ClassicalSVM,
    model_params={},
    X=X,
    y=y,
    n_splits=5,
)
```

#### Circuit Visualization

```python
from qml.ellizeu_sena.visualization import draw_quantum_feature_map

draw_quantum_feature_map(
    num_features=4,
    reps=2,
)
```

### 📚 References

1. Havlíček, V. et al. *Supervised Learning with Quantum-Enhanced Feature Spaces*. Nature, 2019.

2. Schuld, M.; Petruccione, F. *Machine Learning with Quantum Computers*. Springer, 2021.

3. Qiskit Documentation.

4. Qiskit Machine Learning Documentation.

5. Scikit-Learn Documentation.

6. Breast Cancer Wisconsin Diagnostic Dataset Documentation.
