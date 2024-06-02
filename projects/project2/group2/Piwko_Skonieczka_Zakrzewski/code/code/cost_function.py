import numpy as np
from sklearn.metrics import precision_score

def custom_cost_function(features, y_test, y_pred):
    """
    Calculates profit - custom metric for evaluation of models that calculates 
    potential revenue based on correctly predicted positve cases with penalty from 
    number of features.

    Parameters:
        features (list): List of features used in model training
        y_test (list, pd.series, np.array): true labels
        y_pred (list, pd.series, np.array): predicted labels
    """
    precision = precision_score(y_test, y_pred)
    numb_of_features = len(features)
    return 10*1000*precision - 200*numb_of_features