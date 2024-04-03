from base_model import BaseModel
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier


class LibraryClassifier(BaseModel):
    model: object

    def fit(self, X, y, interactions=False):
        if interactions:
            X = self.create_interactions(X)
        else:
            self.interactions = False

        return self.model.fit(X, y)

    def predict(self, X):
        if self.interactions:
            X = self.create_interactions(X)
        return self.model.predict(X)


class LogisticRegressionLIB(LibraryClassifier):
    model: object

    def __init__(self, max_iter=500):
        self.model = SGDClassifier(max_iter=max_iter, penalty=None, fit_intercept=False)


class LinearDiscriminantAnalysisLIB(LibraryClassifier):
    def __init__(self):
        self.model = LinearDiscriminantAnalysis()


class QuadraticDiscriminantAnalysisLIB(LibraryClassifier):
    def __init__(self):
        self.model = QuadraticDiscriminantAnalysis()


class DecisionTreeLIB(LibraryClassifier):
    def __init__(self):
        self.model = DecisionTreeClassifier()


class RandomForestLIB(LibraryClassifier):
    def __init__(self):
        self.model = RandomForestClassifier()
