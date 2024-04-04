#binary logisitc regression classifier
from optimizer import Optimizer
import numpy as np

def sigmoid(x):
    """Sigmoid activation function."""

    x = np.clip(x, a_min=-700, a_max=None)
    
    return 1 / (1 + np.exp(-x))

class LogisticRegressionClf:

    def __init__(self, optimizer: Optimizer, consider_interactions=False):
        '''
        Initialize the logistic regression classifier.
        '''
        self.optimizer = optimizer
        self.consider_interactions = consider_interactions
        self.weights = None
        self.log_loss_arr = []

    def _add_intercept(self, X):
        """Add intercept to the matrix of features."""
        intercept = np.ones((X.shape[0], 1))
        return np.hstack((intercept, X))

    def _expand_features(self, X):
        """Expand features to include interactions if required."""
        if self.consider_interactions:
            n = X.shape[1]
            expanded_features = []
            for i in range(n):
                for j in range(i, n):
                    expanded_features.append(X[:, i] * X[:, j])
            return np.column_stack(expanded_features)
        else:
            return X
        
    def fit(self, X, y, epochs=500, patience=20):
        '''
        Fit the model to the data using provided data and optimizer.
        '''

        if self.consider_interactions:
            X = self._expand_features(X)

        if isinstance(X, np.ndarray):
            X = self._add_intercept(X)
        else:
            raise ValueError("X needs to be a numpy ndarray.")

        self.optimizer.initialize(X.shape[1])
        self.weights = np.zeros(X.shape[1])

        best_log_loss = self.log_loss(X, y)
        counter = 0

        best_weights = self.weights

        for _ in range(epochs):
            self.weights = self.optimizer.update(self.weights, X, y)
            current_log_loss = self.log_loss(X, y)
            self.log_loss_arr.append(current_log_loss)

            if current_log_loss < best_log_loss:
                best_log_loss = current_log_loss
                best_weights = self.weights
                counter = 0
            else:
                counter += 1

            if counter >= patience:
                self.log_loss_arr = self.log_loss_arr[:len(self.log_loss_arr) - patience]
                break
        
        self.weights = best_weights

    def predict_proba(self, X):
        '''
        Return probability of class 1 for each observation in X.
        '''
        
        if self.consider_interactions:
            X = self._expand_features(X)

        if isinstance(X, np.ndarray):
            X = self._add_intercept(X)
        else:
            raise ValueError("X needs to be a numpy ndarray.")

        return sigmoid(np.dot(X, self.weights))

    def predict(self, X):
        '''
        Return predicted class for each observation in X.
        '''
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)
    
 
    def log_loss(self, X, y, epsilon=1e-5):
        '''
        Return minus log likelihood of the model given the data.
        '''

        z = np.dot(X, self.weights)
        sigmoid_z = sigmoid(z)

        log_loss = -np.sum(y * np.log(sigmoid_z + epsilon) + (1 - y) * np.log(1 - sigmoid_z + epsilon))

        return log_loss

    def params(self):
        '''
        Returns the parameters of the model.
        '''
        return self.weights