import numpy as np
from sklearn.metrics import mean_absolute_error, balanced_accuracy_score
from sklearn.utils import shuffle
from copy import deepcopy

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def batch_data(X, y, batch_size): 
    if batch_size == -1:
        return [X], [y]
    result_X = []
    result_y = []
    X_data = X[:]
    y_data = y[:]
    X_data, y_data = shuffle(X_data, y_data)
    for i in range(0, len(y_data), batch_size):
        result_X.append(X_data[i:i + batch_size])
        result_y.append(y_data[i:i + batch_size])
    return result_X, result_y

def add_data_interaction(data):
    result = []
    for row in np.array(data):
        new_row = row
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                new_row = np.append(new_row, row[i] * row[j])
        result.append(new_row)
    return np.array(result)


class LR:
    def __init__(self, learning_rate=0.01, n_iterations=10000, interaction_model=False, tol = 1e-4):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.interaction_model = interaction_model
        self.tol = tol

    def fit(self, X, y, optimization_algorithm='SGD', batch_size=-1, loglikelihood=False):
        if loglikelihood:
            loglikelihood_result = []
        
        if self.interaction_model:
            X = add_data_interaction(X)
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0

        X_batched, y_batched = batch_data(X, y, batch_size)
        # used for stop condition
        previous_loss = self.calculate_loss(X, y)
        n = 0

        if optimization_algorithm == 'SGD':

            for i in range(self.n_iterations):
                for X_batch, y_batch in zip(X_batched, y_batched):
                    batch_n_samples = X_batch.shape[0]
                    y_pred = sigmoid(np.dot(X_batch, self.weights) + self.bias)

                    dw = np.dot(X_batch.T, (y_pred - y_batch)) * self.learning_rate # *(1 / batch_n_samples)
                    db = np.sum(y_pred - y_batch) * self.learning_rate # *(1 / batch_n_samples)

                    self.weights -= dw
                    self.bias -= db
                # this stop condition utilizes the MAE loss function
                current_loss = self.calculate_loss(X, y)

                if self.stop_condition(previous_loss, current_loss):
                    if n == 10:
                        # print(f"SGD stopping at iteration {i}")
                        if loglikelihood:
                            return loglikelihood_result
                        return
                    else:
                        n+=1
                else:
                    n = 0
                previous_loss = current_loss

                    # doesn't work badly, just a very primitive approach
                    # if np.linalg.norm(dw) < self.tol:
                    #     print(f"SGD stopping at iteration {i}")
                    #     return
                if loglikelihood:
                   loglikelihood_result.append(np.sum([np.log(max(proba, 0.05)) if y_class==1 else np.log(max(1-proba, 0.05)) for proba, y_class in zip(self.predict_proba(X), y)]))

            if loglikelihood:
                return loglikelihood_result

        if optimization_algorithm == 'IWLS':
            
            for i in range(self.n_iterations):
                for X_batch, y_batch in zip(X_batched, y_batched):
                    y_pred = sigmoid(np.dot(X_batch, self.weights) + self.bias)

                    X_batch = np.hstack([X_batch, np.ones([X_batch.shape[0],1])])

                    weights = y_pred * (1 - y_pred)
                    X_weighted = X_batch * np.sqrt(weights[:, np.newaxis])
        
                    y_weighted = y_batch - y_pred
                    
                    try:
                        dw = np.linalg.solve(np.dot(X_weighted.T, X_weighted), np.dot(X_weighted.T, y_weighted))
                    except np.linalg.LinAlgError:
                        dw = np.linalg.solve(np.dot(X_weighted.T, X_weighted) + 0.1 * np.eye(X_weighted.shape[1]), np.dot(X_weighted.T, y_weighted))
                    
                    dw *= self.learning_rate
                    self.weights += dw[:-1]
                    self.bias += dw[-1]

                current_loss = self.calculate_loss(X, y)

                if self.stop_condition(previous_loss, current_loss):
                    if n == 10:
                        # print(f"IWLS stopping at iteration {i}")
                        if loglikelihood:
                            return loglikelihood_result
                        return
                    else:
                        n+=1
                else:
                    n = 0
                previous_loss = current_loss

                    # if np.linalg.norm(dw) < self.tol:
                    #     print(f"IWLS stopping at iteration {i}")
                    #     return
                if loglikelihood:
                   loglikelihood_result.append(np.sum([np.log(max(proba, 0.05)) if y_class==1 else np.log(max(1-proba, 0.05)) for proba, y_class in zip(self.predict_proba(X), y)]))

            if loglikelihood:
                return loglikelihood_result


        # if optimization_algorithm == 'IWLS2':
        #     X_batched = batch_data(X, batch_size)
        #     y_batched = batch_data(y, batch_size)
            
        #     for _ in range(self.n_iterations):
        #         for X_batch, y_batch in zip(X_batched, y_batched):
        #             y_pred = sigmoid(np.dot(X_batch, self.weights) + self.bias)

        #             X_batch = np.hstack([X_batch, np.ones([X_batch.shape[0],1])])
        #             a = np.dot(X_batch.T, y_batch - y_pred)
        #             b = np.dot(np.dot(X_batch.T, np.diag(y_batch - y_pred)), X_batch)
        #             b = np.linalg.inv(b)
        #             change = np.dot(a, b)
        #             # if np.linalg.norm(change) > tol:
        #             self.weights += change[:-1]
        #             self.bias += change[-1]

        if optimization_algorithm == 'ADAM':
            beta1 = 0.9
            beta2 = 0.999
            epsilon = 1e-8

            m = np.zeros(n_features + 1)  # First moment estimate
            v = np.zeros(n_features + 1)  # Second moment estimate
            t = 0
            for t in range(1, self.n_iterations+1):
                for X_batch, y_batch in zip(X_batched, y_batched):
                    y_pred = sigmoid(np.dot(X_batch, self.weights) + self.bias)
                    X_batch = np.hstack([X_batch, np.ones([X_batch.shape[0],1])])
                    norm = n_samples if batch_size == -1 else batch_size
                    if batch_size == -1:
                        pass
                    gradient = np.dot(X_batch.T, (y_pred - y_batch)) / norm # derivative of log-loss with respect to weights Log_loss = -1/n * sum_over_i (yi * log(pi) + (1 - yi) * log(1 - pi) ), pi = 1 / (1 + exp(-Beta * xi) ), therefore d/dBeta * Log_loss = 1/n * X.T * (y_pred - y)
                    m = beta1 * m + (1 - beta1) * gradient
                    v = beta2 * v + (1 - beta2) * (gradient ** 2)

                    m_hat = m / (1 - beta1 ** t)
                    v_hat = v / (1 - beta2 ** t)

                    update = self.learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
                    self.weights -= update[:-1]
                    self.bias -= update[-1]

                current_loss = self.calculate_loss(X, y)

                if self.stop_condition(previous_loss, current_loss):
                    if n == 10:
                        # print(f"ADAM stopping at iteration {t}")
                        if loglikelihood:
                            return loglikelihood_result
                        return
                    else:
                        n+=1
                else:
                    n = 0

                previous_loss = current_loss

                    # if np.linalg.norm(update) < self.tol:
                    #     print(f"ADAM stopping at iteration {t}")
                    #     return
                if loglikelihood:
                   loglikelihood_result.append(np.sum([np.log(max(proba, 0.05)) if y_class==1 else np.log(max(1-proba, 0.05)) for proba, y_class in zip(self.predict_proba(X), y)]))

            if loglikelihood:
                return loglikelihood_result

        # code for whole dataset gradient
        # for i in range(self.n_iterations):
        #     y_pred = sigmoid(np.dot(X, self.weights) + self.bias)

        #     dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
        #     db = (1 / n_samples) * np.sum(y_pred - y)

        #     self.weights -= self.learning_rate * dw
        #     self.bias -= self.learning_rate * db

    def predict_proba(self, X_input, interaction_model=False):
        X = deepcopy(X_input)
        if interaction_model:
            X = add_data_interaction(X_input)
        y_pred = sigmoid(np.dot(X, self.weights) + self.bias)

        return y_pred
    
    def predict(self, X_input, interaction_model=False):
        X = deepcopy(X_input)
        if interaction_model:
            X = add_data_interaction(X_input)
        y_pred = sigmoid(np.dot(X, self.weights) + self.bias)
        
        return [0 if y < 0.5 else 1 for y in y_pred]
    
    def calculate_loss(self, X, y):
        y_pred = self.predict(X)
        return mean_absolute_error(y, y_pred)
    
    def stop_condition(self, prev_loss, loss):
        loss_change = loss - prev_loss
        return loss_change < self.tol