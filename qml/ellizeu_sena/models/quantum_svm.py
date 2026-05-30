from sklearn.metrics import accuracy_score

from qiskit.circuit.library import ZZFeatureMap

from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC


class QuantumSVM:
    """
    Quantum Support Vector Machine (QSVM).

    Components
    ----------
    - ZZFeatureMap
    - FidelityQuantumKernel
    - QSVC
    """

    def __init__(
        self,
        num_features,
        reps=2,
        C=1.0,
        entanglement="full",
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

        entanglement : str
            Entanglement strategy used by ZZFeatureMap.
            Examples:
            - "linear"
            - "circular"
            - "full"
        """

        self.num_features = num_features
        self.reps = reps
        self.C = C
        self.entanglement = entanglement

        # Quantum feature map
        self.feature_map = ZZFeatureMap(
            feature_dimension=num_features,
            reps=reps,
            entanglement=entanglement,
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

        self.model.fit(
            X_train,
            y_train,
        )

        return self

    def predict(
        self,
        X_test,
    ):
        """
        Predict labels.
        """

        return self.model.predict(X_test)

    def score(
        self,
        X_test,
        y_test,
    ):
        """
        Compute classification accuracy.
        """

        predictions = self.predict(X_test)

        return accuracy_score(
            y_test,
            predictions,
        )

    def get_feature_map(self):
        """
        Return the quantum feature map.
        """

        return self.feature_map

    def get_quantum_kernel(self):
        """
        Return the quantum kernel.
        """

        return self.quantum_kernel

    def get_model(self):
        """
        Return the underlying QSVC model.
        """

        return self.model

    def get_params(self):
        """
        Return model hyperparameters.
        """

        return {
            "num_features": self.num_features,
            "reps": self.reps,
            "C": self.C,
            "entanglement": self.entanglement,
        }

    def __repr__(self):
        return (
            "QuantumSVM("
            f"num_features={self.num_features}, "
            f"reps={self.reps}, "
            f"C={self.C}, "
            f"entanglement='{self.entanglement}'"
            ")"
        )