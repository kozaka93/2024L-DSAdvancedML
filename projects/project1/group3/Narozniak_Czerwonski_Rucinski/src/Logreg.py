import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from Adam import AdamOptim
from irls_optimizer import IRLS


class LogisticRegression:
    """
    Logistic Regression classifier.

    This class implements a logistic regression model for binary classification.
    It uses the sigmoid function as the activation function and applies an optimizer
    to update the weights and bias during training.

    Attributes:
        input_dim (int): The number of input features.
        weights (ndarray): The weights of the logistic regression model.
        bias (float): The bias term of the logistic regression model.
        weights_updates (list): List of weight updates during training.
        bias_updates (list): List of bias updates during training.
        losses (list): List of loss values during training.

    Methods:
        sigmoid(z): Applies the sigmoid function to the input.
        predict(X): Predicts the probability of the class being 1 for the input data.
        train(X, y, optimizer, epochs, batch_size): Trains the logistic regression model using the given data and optimizer.
        get_params(): Returns the current parameters of the logistic regression model.
        plot_params(): Plots the updates of weights, bias, and loss over time.
    """

    def __init__(self, input_dim : int) -> None:
        self.input_dim = input_dim
        self.weights = np.zeros(input_dim)
        self.bias = 0
        self.weights_updates = []
        self.bias_updates = []
        self.losses = []

    def sigmoid(self, z : float) -> float:
        z = z.astype(float)
        return 1 / (1 + np.exp(-z))
    
    def predict(self, X : np.ndarray) -> float:

        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def train(self, X : np.ndarray, y : np.ndarray, optimizer : AdamOptim, epochs : int, batch_size : int, patience = 10) -> None:

        m = X.shape[0]
        num_batches = m // batch_size

        best_loss = np.inf
        patience_counter = 0

        for epoch in range(epochs):
            epoch_loss = 0
            for batch in range(num_batches):
                
                start = batch * batch_size
                end = start + batch_size
                X_batch = X[start:end]
                y_batch = y[start:end]
                
                z = np.dot(X_batch, self.weights) + self.bias
                a = self.sigmoid(z)

                if isinstance(optimizer, AdamOptim):
                    dw = np.dot(X_batch.T, (a - y_batch)) / batch_size
                    db = np.mean(a - y_batch)

                    self.weights, self.bias = optimizer.update(epoch+1, self.weights, self.bias, dw, db)
                elif isinstance(optimizer, IRLS):
                    B = np.concatenate([np.array([self.bias]), self.weights])
                    self.weights, self.bias = optimizer.update(B, X, y)
                
                batch_loss = -np.mean(y_batch * np.log(a) + (1 - y_batch) * np.log(1 - a))
                epoch_loss += batch_loss

            epoch_loss /= num_batches
            self.losses.append(epoch_loss)
            self.weights_updates.append(self.weights)
            self.bias_updates.append(self.bias)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print('Early stopping after epoch', epoch)
                break


    def get_params(self):
        return self.weights, self.bias, self.weights_updates, self.bias_updates, self.losses

    def plot_params(self):
        plt.figure(figsize=(18, 6))
        plt.subplot(1, 3, 1)
        plt.plot(self.weights_updates)
        plt.title('Weights updates')
        plt.xlabel('Iteration')
        plt.ylabel('Weights')

        plt.subplot(1, 3, 2)
        plt.plot(self.bias_updates)
        plt.title('Bias updates')
        plt.xlabel('Iteration')
        plt.ylabel('Bias')

        plt.subplot(1, 3, 3)
        plt.plot(self.losses)
        plt.title('Loss over time')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')

        plt.tight_layout()
        plt.show()
