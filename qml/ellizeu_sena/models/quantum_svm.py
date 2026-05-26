from sklearn.metrics import accuracy_score

from qiskit.circuit.library import ZZFeatureMap

from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC


class QuantumSVM:
    """
    Quantum Support Vector Machine (QSVM).

    Uses:
    - ZZFeatureMap
    - FidelityQuantumKernel
    - QSVC
    """

    def __init__(
        self,
        num_features,
        reps=2,
        C=1.0,
    ):
        """
        Parameters
        ----------
        num_features : int
            Number of input features / qubits.

        reps : int
            Number of repetitions in the feature map.

        C : float
            SVM regularization parameter.
        """

        self.num_features = num_features
        self.reps = reps

        # Quantum feature map
        self.feature_map = ZZFeatureMap(
            feature_dimension=num_features,
            reps=reps,
        )

        # Quantum kernel
        self.quantum_kernel = FidelityQuantumKernel(
            feature_map=self.feature_map
        )

        # QSVM model
        self.model = QSVC(
            quantum_kernel=self.quantum_kernel,
            C=C,
        )

    def fit(
        self,
        X_train,
        y_train,
    ):
        """
        Train the QSVM model.
        """

        self.model.fit(X_train, y_train)

    def predict(
        self,
        X_test,
    ):
        """
        Perform predictions.
        """

        return self.model.predict(X_test)

    def score(
        self,
        X_test,
        y_test,
    ):
        """
        Compute accuracy score.
        """

        predictions = self.predict(X_test)

        return accuracy_score(y_test, predictions)

    def get_feature_map(self):
        """
        Return the quantum feature map.
        """

        return self.feature_map