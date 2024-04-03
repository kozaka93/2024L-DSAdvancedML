import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from logistic_regression.utils.activations import sigmoid
from logistic_regression.optimizers.sgd import SGD  

class LogisticRegression:
    """Custom implementation of the logistic regression algorithm."""
    
    def __init__(self, optimizer=SGD(), interactions=False):
        """
        Initializes the Logistic Regression model with an optimizer.

        Parameters:
        - optimizer: An instance of an Optimizer to use for parameter updates. Defaults to SGD if none is provided.
        - interactions (boolean): Flag indicating whether to include interaction terms between input features in the model.
        If True, the model will not only base its predictions on the original features but also on the pairwise product of the features.
        """
        self.epochs_completed = None
        self.params = None
        self.optimizer = optimizer
        self.interactions = interactions
        self.cost_in_iterations = np.array([])


    def cost_function(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Computes the logistic regression cost function for all the training samples.

        Parameters:
        - X (np.ndarray): Design matrix with shape (m, n), where m is the number of samples and n is the number of features.
        - y (np.ndarray): Target vector with boolean values (0 or 1) with shape (m,).

        Returns:
        - cost (float): The computed cost.
        """
        m = len(y)
        fx = sigmoid(np.dot(X, self.params))
        epsilon = 1e-5  # To prevent log(0)
        cost = -np.sum(y * np.log(fx + epsilon) + (1 - y) * np.log(1 - fx + epsilon)) / m
        return cost

    def add_interactions(self, X):
        """ Adds interaction terms between features in the dataset X.

        Parameters:
        - X: A NumPy array of shape (n_samples, n_features), where each row represents a sample
             and each column represents a feature.
        
        Returns:
        - A NumPy array with original features followed by interaction terms.
        """
        n_samples, n_features = X.shape
        all_features = [X]

        for i in range(n_features):
            for j in range(i+1, n_features):
                interaction_term = X[:, i] * X[:, j]
                all_features.append(interaction_term[:, np.newaxis])

        return np.hstack(all_features)

    def fit(self, X: np.ndarray, y: np.ndarray, iterations: float=500, min_delta: float=0.000000001, patience:float=1000):
        """
        Fits the logistic regression model to the training data.

        Parameters:
        - X (np.ndarray): The input features for training.
        - y (np.ndarray): The target values for training.
        - iterations (int): The maximum number of iterations for training.
        - min_delta (float): The minimum change in the cost function to define convergence.
        - patience (int): The number of iterations with no improvement to wait before early stopping.
        """
        if self.interactions:
            X = self.add_interactions(X.copy())

        if self.params is None:
            self.params = np.random.randn(X.shape[1], 1)
        prev_cost = float('inf')  # Initialize previous cost as infinity
        no_improvement_count = 0

        # Split data into training and validation sets for early stopping rule
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

        for _ in tqdm(range(iterations), desc="Training Progress"):
            self.params = self.optimizer.optimize_parms(X_train, y_train, self.params)
            
            cost = self.cost_function(X_test, y_test)
            self.cost_in_iterations = np.append(self.cost_in_iterations, cost)
            cost_change = prev_cost - cost
            prev_cost = cost

            if cost_change < min_delta:
                no_improvement_count += 1
            else:
                no_improvement_count = 0
            
            if no_improvement_count >= patience:
                print("Early stopping triggered.")
                break

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts probability estimates for samples in X.

        Parameters:
        - X (np.ndarray): The input features.

        Returns:
        - np.ndarray: The probability estimates.
        """
        if self.interactions:
            X = self.add_interactions(X.copy())
        return sigmoid(np.dot(X, self.params))
    
    def predict(self, X: np.ndarray, threshold: float=0.5) -> np.ndarray:
        """
        Predicts labels for samples in X based on a probability threshold.

        Parameters:
        - X (np.ndarray): The input features.
        - threshold (float): The threshold for classifying samples as positive.

        Returns:
        - np.ndarray: The predicted labels.
        """
        if self.interactions:
            X = self.add_interactions(X.copy())
        prob_pred = sigmoid(np.dot(X, self.params))
        return (prob_pred > threshold).astype(int)