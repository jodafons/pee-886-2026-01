class ModelTrainer:

    def __init__(self, model):
        self.model = model

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def fit_predict(
        self,
        X_train,
        y_train,
        X_test,
    ):
        self.fit(X_train, y_train)
        return self.predict(X_test)