from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import joblib


class ClassicalSVM:
    """
    Classical Support Vector Machine wrapper.

    Supports:
    - Linear kernel
    - RBF kernel
    """

    def __init__(
        self,
        kernel="rbf",
        C=1.0,
        gamma="scale",
    ):
        """
        Parameters
        ----------
        kernel : str
            SVM kernel type.

        C : float
            Regularization parameter.

        gamma : str or float
            Kernel coefficient for RBF.
        """

        self.kernel = kernel

        self.model = SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
        )

    def fit(
        self,
        X_train,
        y_train,
    ):
        """
        Train the SVM model.
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

    def save(
        self,
        filepath,
    ):
        """
        Save trained model.
        """

        joblib.dump(
            self.model,
            filepath,
        )


    @classmethod
    def load(
        cls,
        filepath,
    ):
        """
        Load trained model.
        """

        model = joblib.load(
            filepath,
        )

        instance = cls()

        instance.model = model

        return instance