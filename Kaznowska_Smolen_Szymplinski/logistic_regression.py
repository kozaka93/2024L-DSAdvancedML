import numpy as np
from utils import *

class LogisticRegression(object):
    
    def __init__(self,
                 optimizer: object,
                 interactions: bool = False 
                ) -> None:
        
        self.optimizer = optimizer
        self.intercept = None
        self.coef = None
        self.classes = None
        self.interactions = interactions
        
    def fit(self, X, y):
        
       
        
        if self.interactions:
            X = add_interactions(X)
            
        if len(np.unique(y)) != 2:
            raise Exception("There should be exactly two classes.")
        if len(X) != len(y):
            raise Exception("Length of 'X' is not equal length od 'y'.")
        
        self.classes = np.unique(y)
        self.coef = np.random.random(X.shape[1])
        self.intercept = np.random.random()
            
        return self.optimizer.train(self, X, y)
    
    def predict_proba(self, X):
        if self.interactions:
            X = add_interactions(X)
        
        linear_combination = X @ self.coef + self.intercept
        return np.vectorize(sigmoid)(linear_combination)
    
    def predict(self, X):
        return (self.predict_proba(X) >= 0.5)*1