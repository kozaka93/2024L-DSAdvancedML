import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd


def remove_nan_rows_xy(X, y):
    """
    A function to remove nan variables. If the row contain nan in features
    or in target then whole row is removed from the dataset.
    (i.e. row from features and corresponding row in labels).
    Additionally, it prints number of removed rows.

    Parameters
    ----------
    X : array-like
       A matrix of features
    y : array-like
       A vector of labels

    Returns
    -------
    X_clean : numpy.ndarray
       A matrix of features with removed nan
    y_clean : numpy.ndarray
       A vector of target with removed nan
    """

    X = pd.DataFrame(X)
    y = pd.DataFrame(y)
    df = pd.concat([X, y], axis=1)
    df_clean = df.dropna()
    X_clean = df_clean.iloc[:, :-1].to_numpy()
    y_clean = df_clean.iloc[:, -1].to_numpy()
    num_removed_rows = X.shape[0] - X_clean.shape[0]
    print(f"Number of rows removed: {num_removed_rows}")
    return X_clean, y_clean


def preprocess_algerian_forest_fires(X, y):
    """
    The function to preprocess Algerian forest fires.
    Additional preprocessing is needed as
    target variables (fire, not fire) contains additional white spaces)

    Parameters
    ----------
    X : array-like
       A matrix of features
    y : array-like
       A vector of labels

    Returns
    -------
    X : numpy.ndarray
       Preprocessed matrix of features
    y : numpy.ndarray
       Preprocessed vector of labels
    """
    y = y.astype('str')
    y = np.char.strip(y)
    le = LabelEncoder()
    y = le.fit_transform(y)
    X[:, 0] = le.fit_transform(X[:, 0])
    return X.astype(np.float32), y


def preprocess_waveform_database_generator_version_1(X, y):
    """
    The function to preprocess waveform_database_generator_version_1.
    Additional preprocessing is needed: subset 2 classes

    Parameters
    ----------
    X : array-like
       A matrix of features
    y : array-like
       A vector of labels

    Returns
    -------
    X : numpy.ndarray
       Preprocessed matrix of features
    y : numpy.ndarray
       Preprocessed vector of labels
    """
    class_values = np.array([0, 1])
    selected_records = np.isin(y, class_values)
    X = X[selected_records]
    y = y[selected_records]
    return X, y


def remove_collinear_variables(X, t):
    """
    The function to remove collinear columns

    Parameters
    ----------
    X : array-like
       A matrix of features
    t : float
       Remove column when the Pearson correlation is higher than t

    Returns
    -------
    X_new : numpy.ndarray
       Preprocessed matrix of features
    len(highly_correlated) : int
       Number of removed columns
    """
    correlations = np.abs(np.corrcoef(X, rowvar=False))
    mask = np.triu(np.ones_like(correlations, dtype=bool), k=1)
    highly_correlated = np.unique(np.where(mask & (correlations > t))[1])
    X_new = np.delete(X, highly_correlated, axis=1)
    return len(highly_correlated), X_new
