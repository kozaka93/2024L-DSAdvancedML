import numpy as np
import pandas as pd
from scipy.io import arff
from statsmodels.stats.outliers_influence import variance_inflation_factor
pd.set_option('future.no_silent_downcasting', True)
np.seterr(divide='ignore', invalid='ignore')


def load_data(file_path, remove_collinearity=True):
    data = arff.loadarff(file_path)
    if file_path in ['./data/phoneme.arff', './data/ozone-level.arff', './data/QSAR-biodegradation.arff',
                     './data/breast-cancer-wisconsin.arff', './data/eeg-eye-state.arff']:
        X, y = transform_data(data)
    elif file_path == './data/spam.arff':
        X, y = transform_data(data, b'1', b'0', 'class')
    elif file_path == './data/steel-plates-fault.arff':
        X, y = transform_data(data, b'1', b'2', 'Class')
    elif file_path == './data/diabetes.arff':
        X, y = transform_data(data, b'tested_positive', b'tested_negative', 'class')
    elif file_path == './data/transplant.arff':
        X, y = transform_data(data, b'P', b'N', 'binaryClass')
    else:
        raise ValueError('Wrong file path.')

    if remove_collinearity:
        X = remove_multicollinearity(X)
    return X, y



def transform_data(data, replace1=b'1', replace0=b'2', target_name='Class'):
    df = pd.DataFrame(data[0])
    df = df.replace(replace1, 1).infer_objects(copy=False)
    df = df.replace(replace0, 0).infer_objects(copy=False)
    y = df[target_name]
    X = df.drop(target_name, axis=1)
    return X, y


def remove_multicollinearity(X, threshold=10):
    while True:
        vif = calculate_vif(X)
        max_vif = vif['VIF'].max()
        if max_vif > threshold:
            max_vif_index = vif['VIF'].idxmax()
            X = X.drop([vif.loc[max_vif_index, 'columns']], axis=1)
        else:
            break
    return X


def calculate_vif(X):
    vif = pd.DataFrame()
    vif["columns"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif



