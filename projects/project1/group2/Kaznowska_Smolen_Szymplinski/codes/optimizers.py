import numpy as np
from utils import *
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split



class Optimizer():
    def __init__(self,
                 random_state: int = None,
                 max_iter: int = 100,
                 learning_rate: float = 1e-3,
                 early_stopping: bool = True,
                 convergence_stopping: bool = True,
                 log_like_eval: bool = False,
                 n_iter_no_change: int = 5,
                 epsilon_change: float = 1e-2,
                 val_size: float = 0.2
                ) -> None:
        
        self.random_state = random_state
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.early_stopping = early_stopping
        self.convergence_stopping = convergence_stopping
        self.log_like_eval = log_like_eval
        self.n_iter_no_change = n_iter_no_change
        self.epsilon_change = epsilon_change
        self.val_size = val_size


    def _train_setup(self, lr_model, X, y):
        if self.early_stopping:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=self.val_size,
                                                            random_state=self.random_state)
            early_stop = EarlyStopping(self.n_iter_no_change, self.epsilon_change)
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None
            early_stop = None

        results_holder = ResultsHolder(lr_model)
        convergence_checker = ConvergenceChecker(500, self.epsilon_change)

        return X_train, X_val, y_train, y_val, results_holder, convergence_checker, early_stop


    def _iteration_end_evaluation(self, lr_model, results_holder, convergence_checker, early_stop, X_train, X_val, y_train, y_val):
        # Evaluate model on train data for convergence check
        if self.log_like_eval:
            p = lr_model.predict_proba(X_train)
            results_holder.evaluate_log_likelihood(p, y_train)

        if self.convergence_stopping:
            results_holder.evaluate_train(X_train, y_train)
            if convergence_checker(results_holder.train_loss):
                return True
        
        # Early stopping
        if self.early_stopping:
            results_holder.evaluate_val(X_val, y_val)
            if early_stop(lr_model, results_holder.val_loss[-1]):
                return True
        
        return False


    def train(self, lr_model, X, y, evaluate):
        pass



class SGD(Optimizer):
    def train(self, lr_model, X, y) -> ResultsHolder:
        X_train, X_val, y_train, y_val, results_holder, convergence_checker, early_stop = self._train_setup(lr_model, X, y)
                
        for _ in tqdm(range(self.max_iter), desc='Model training', position=0):
            # Iter over all observations
            X_train, y_train = shuffle(X_train, y_train, random_state=self.random_state)
            for x_el, y_el in zip(np.array(X_train), y_train):
                # Calculating gradients
                linear_comb = np.sum(x_el*lr_model.coef) + lr_model.intercept
                grad_intercept = (sigmoid(linear_comb) - y_el)
                grad_weights = grad_intercept*x_el
                
                # Updating linear regression model parameters
                lr_model.coef = lr_model.coef - self.learning_rate*grad_weights
                lr_model.intercept = lr_model.intercept - self.learning_rate*grad_intercept
            
            if self._iteration_end_evaluation(lr_model, results_holder, convergence_checker, early_stop, X_train, X_val, y_train, y_val):
                break

        return results_holder



class IWLS(Optimizer):
    def train(self, lr_model, X, y):
        X_train, X_val, y_train, y_val, results_holder, convergence_checker, early_stop = self._train_setup(lr_model, X, y)

        # Add column of ones
        n  = len(X_train)
        X_train = np.column_stack((np.ones(n), X_train))

        #initate p all equal 0.5 
        p = np.ones(n) * 0.5
        for _ in tqdm(range(self.max_iter), desc='Model training', position=0):
            #caluclate z and W
            z = logit(p) + (y_train - p) *  logit_derivative(p)
            W = np.diag(1 / (logit_derivative(p)**2 * p*(1-p)))
            
            #get new beta and p
            beta =  get_beta(X_train, W, z)
            p =  np.vectorize(sigmoid)(X_train @ beta)

            #update linear regression model parameters
            lr_model.intercept = beta[0]
            lr_model.coef = beta[1:]

            if self._iteration_end_evaluation(lr_model, results_holder, convergence_checker, early_stop, X_train[:, 1:], X_val, y_train, y_val):
                break
        
        return results_holder
                


class Adam(Optimizer):
    def __init__(self, 
                 random_state: int = None, 
                 max_iter: int = 1000, 
                 learning_rate: float = 1e-2, 
                 early_stopping: bool = True, 
                 convergence_stopping: bool = True,
                 log_like_eval: bool = False,
                 n_iter_no_change: int = 5, 
                 val_size: float = 0.2,
                 b1: float = 0.9,
                 b2: float = 0.999) -> None:
        super().__init__(random_state, max_iter, learning_rate, early_stopping, convergence_stopping, log_like_eval, n_iter_no_change, val_size)
        self.b1 = b1
        self.b2 = b2

    def train(self, lr_model, X, y):
        X_train, X_val, y_train, y_val, results_holder, convergence_checker, early_stop = self._train_setup(lr_model, X, y)

        # Initialize parameters
        n_samples, n_features = X_train.shape
        epsilon = 1e-8

        #theta = [lr_model.intercept]
        #theta.extend(lr_model.coef)
        #theta = np.array(theta) # bias + weights
        theta = np.zeros(n_features + 1)
        m = np.zeros(n_features + 1) #momentum
        v = np.zeros(n_features + 1) #RMSProp
        
        # Add bias
        X_train = np.column_stack((np.ones(n_samples), X_train))

        # without batching (maybe to change?)
        for iter in tqdm(range(self.max_iter), desc='Model training', position=0):
            t = iter+1

            # Compute gradient
            z = np.dot(X_train, theta)
            h = np.vectorize(sigmoid)(z)
            gradient = np.dot(X_train.T, (h - y_train)) / len(y_train)

            # Update moments estimates
            m = self.b1 * m + (1 - self.b1) * gradient
            v = self.b2 * v + (1 - self.b2) * (gradient ** 2)

            # Correct biases
            m_hat = m / (1 - self.b1 ** t)
            v_hat = v / (1 - self.b2 ** t)

            # Update parameters
            theta -= self.learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

            lr_model.intercept = theta[0]
            lr_model.coef = theta[1:]

            if self._iteration_end_evaluation(lr_model, results_holder, convergence_checker, early_stop, X_train[:, 1:], X_val, y_train, y_val):
                break

        return results_holder
