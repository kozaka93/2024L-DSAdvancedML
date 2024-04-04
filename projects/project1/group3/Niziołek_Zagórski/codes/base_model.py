import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures


class BaseModel:
    interactions: bool
    log_likelihood: list[float]

    def test_efficiency(
        self,
        X,
        y,
        test_size=0.2,
        random_state=2,
        interactions=False,
        repetition_count=5,
    ) -> float:
        scores = []
        for _ in range(repetition_count):
            train_X, test_X, train_y, test_y = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

            self.fit(train_X, train_y, interactions=interactions)
            pred = self.predict(test_X)
            scores.append(balanced_accuracy_score(pred, test_y))
            random_state += 1
        return np.mean(scores)

    def fit(self, X, y, interactions=False):
        raise NotImplementedError()

    def predict_proba(self, X):
        z = np.dot(X, self.beta)
        return self.sigmoid(z)

    def predict(self, X):
        raise NotImplementedError()

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def create_interactions(self, X):
        poly = PolynomialFeatures(include_bias=False, interaction_only=True)
        self.interactions = True
        return poly.fit_transform(X)

    def calculate_log_likelihood(self, X, y):
        value = sum(
            [
                np.log(max(p, 0.05)) if y == 1 else np.log(max(1 - p, 0.001))
                for p, y in zip(self.predict_proba(X), y)
            ]
        )
        return value
