from optimizer import Optimizer
import numpy as np
from logRegClf import sigmoid

class AdamOptimizer(Optimizer):
    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-5, batch_size=64):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.m = None
        self.v = None
        self.t = 0

    def initialize(self, dims):
        #initialize moment vectors m and v
        self.m = np.zeros(dims)
        self.v = np.zeros(dims)
    
    def update(self, weights, X, y):
        #calculate no of batches and reamining samples
        num_samples = X.shape[0]
        num_batches = num_samples // self.batch_size
        rest = num_samples % self.batch_size

        for batch in range(num_batches):
            #select barch subset
            start_idx = batch * self.batch_size
            end_idx = (batch + 1) * self.batch_size
            X_batch = X[start_idx:end_idx]
            y_batch = y[start_idx:end_idx]

            #predict and calculate gradient
            predictions = sigmoid(np.dot(X_batch, weights))
            gradient = np.dot(X_batch.T, (predictions - y_batch))

            #update weights
            self.t += 1
            self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
            self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)
            m_hat = self.m / (1 - self.beta1 ** self.t)
            v_hat = self.v / (1 - self.beta2 ** self.t)
            weights -= self.alpha * m_hat / (np.sqrt(v_hat) + self.epsilon)

        #execute proc for remaining samples
        if rest != 0:
            X_batch = X[-rest:]
            y_batch = y[-rest:]

            #predict and calculate gradient
            predictions = sigmoid(np.dot(X_batch, weights))
            gradient = np.dot(X_batch.T, (predictions - y_batch))
            
            #update weights
            self.t += 1
            self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
            self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)
            m_hat = self.m / (1 - self.beta1 ** self.t)
            v_hat = self.v / (1 - self.beta2 ** self.t)
            weights -= self.alpha * m_hat / (np.sqrt(v_hat) + self.epsilon)

        return weights
