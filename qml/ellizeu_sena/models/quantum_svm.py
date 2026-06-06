from sklearn.metrics import accuracy_score

from qiskit.circuit.library import ZZFeatureMap

from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC

import joblib


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
        feature_map_type="zz",
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
        self.feature_map_type = feature_map_type

        from qiskit.circuit.library import (
            ZZFeatureMap,
            ZFeatureMap,
            PauliFeatureMap,
        )
        
        if feature_map_type == "zz":
        
            self.feature_map = ZZFeatureMap(
                feature_dimension=num_features,
                reps=reps,
                entanglement=entanglement,
            )
        
        elif feature_map_type == "z":
        
            self.feature_map = ZFeatureMap(
                feature_dimension=num_features,
                reps=reps,
            )
        
        elif feature_map_type == "pauli":
        
            self.feature_map = PauliFeatureMap(
                feature_dimension=num_features,
                reps=reps,
                paulis=["Z", "ZZ"],
                entanglement=entanglement,
            )
        
        else:
        
            raise ValueError(
                f"Unknown feature map: {feature_map_type}"
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
            "feature_map_type": self.feature_map_type,
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
    
    def save(
        self,
        filepath,
    ):
        joblib.dump(
            self,
            filepath,
        )


    @classmethod
    def load(
        cls,
        filepath,
    ):
        return joblib.load(
            filepath
        )