import numpy as np


class LogisticRegression:
    def __init__(
        self,
        optimizer,
        eta: float = 0.05,
        epochs: int = 100,
        random_seed: int = 1,
        early_stopping_rounds=None,
        tol=1e-4,
        include_interactions=False
    ) -> None:
        """
        Initialize the logistic regression model.
        Args:
            optimizer: The optimizer to use for updating the weights (class from optimizers folder)
            eta: Learning rate
            epochs: Number of passes over the training set
            random_seed: Seed for initializing random weights and shuffling the dataset
            early_stopping_rounds: Number of epochs to wait for the training loss to decrease before stopping training
            tol: Tolerance for the training loss to consider convergence
            include_interactions: Whether to include interaction terms in the model
        """
        self.eta = eta
        self.epochs = epochs
        self.early_stopping_rounds = early_stopping_rounds
        self.tol = tol
        self.random_seed = random_seed
        self.optimizer = optimizer
        self.losses = []
        self.include_interactions = include_interactions

    def initialize_weights(self, n_features: int) -> None:
        """
        Initialize weights randomly.
        Args:
            n_features: Number of features in the training set
        """
        # Set random seed for reproducibility
        np.random.seed(self.random_seed)
        if self.optimizer.__class__.__name__ == "IRLS":
            self.w = np.zeros(n_features)
        else:
            self.w = 2 * np.random.random(n_features) - 1

    def sigmoid(self, z: np.array) -> np.array:
        """
        Sigmoid activation function.
        Args:
            z: The input to the activation function
        """
        return 1 / (1 + np.exp(-z))
    
    def compute_loss(self, X, y):
        """
        Compute the negative log-likelihood loss.
        Args:
            X: The input samples
            y: The target values
        """
        z = np.dot(X, self.w)
        y_pred = self.sigmoid(z)
        epsilon = 1e-15  # Small value to avoid numerical instability
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        return loss

    def add_interactions(self, X: np.array) -> np.array:
        """
        Add interaction terms to the feature matrix.
        Args:
            X: The input feature matrix
        Returns:
            X_interactions: The feature matrix with interaction terms added
        """
        n_samples, n_features = X.shape
        interaction_terms = []

        for i in range(n_features):
            for j in range(i + 1, n_features):
                interaction_terms.append(X[:, i] * X[:, j])

        X_interactions = np.column_stack((X, *interaction_terms))
        return X_interactions

    def fit(self, X: np.array, y: np.array) -> None:
        """
        Fit the logistic regression model.
        Args:
            X: The training input samples
            y: The target values
        """
        prev_loss = float("inf")
        increasing_loss_count = 0
        n_samples, n_features = X.shape

        if self.include_interactions:
            X = self.add_interactions(X)
            n_features = X.shape[1]

        self.initialize_weights(n_features)

        for epoch in range(self.epochs):
            if self.optimizer.__class__.__name__ == "IRLS":
                self.w = self.optimizer.update(self.w, X, y)
                loss = self.compute_loss(X, y)
            else:
                for i in range(n_samples):
                    # Compute the negative gradient of the log-likelihood
                    # for the ith training sample
                    z = np.dot(X[i], self.w)
                    y_pred = self.sigmoid(z)
                    error = y_pred - y[i]
                    grad_wrt_w = X[i] * error
                    # Update the weights
                    self.w = self.optimizer.update(self.w, grad_wrt_w, X, y)
                loss = self.compute_loss(X, y)

            if loss - prev_loss > self.tol:
                increasing_loss_count += 1
                if (
                    self.early_stopping_rounds is not None
                    and increasing_loss_count >= self.early_stopping_rounds
                ):
                    print(f"Stopping early at epoch {epoch + 1} with increasing loss")
                    return
            else:
                increasing_loss_count = 0
            prev_loss = loss
            # append log-likelihood
            self.losses.append(loss)

    def predict(self, X: np.array) -> np.array:
        """
        Predict binary labels.
        Args:
            X: The input samples
        """
        # If interactions were included during training, add them to the input features for prediction
        if self.include_interactions:
            X = self.add_interactions(X)
        # Compute the output of the neuron
        z = np.dot(X, self.w)
        y_pred = np.round(self.sigmoid(z)).astype(int)
        return y_pred


