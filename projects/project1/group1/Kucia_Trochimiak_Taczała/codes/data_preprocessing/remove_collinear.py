import pandas as pd
import numpy as np

def remove_collinear_variables(X, threshold=0.7):
    """
    Remove collinear variables from a DataFrame and list the dropped columns with their indices and max correlation coefficients.

    Parameters:
    X (pd.DataFrame): DataFrame containing the features.
    threshold (float): The correlation coefficient value above which 
                       the function will consider two features collinear.

    Returns:
    pd.DataFrame: DataFrame with collinear variables removed.
    List: List of tuples containing the index, name of the dropped columns and their maximum correlation coefficients.
    """
    # Calculate the correlation matrix
    corr_matrix = X.corr().abs()

    # The matrix is symmetric so we need to extract the upper triangle matrix without the diagonal (k = 1)
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    # Initialize list to store dropped columns and their max correlation
    dropped_info = []

    # Find features with correlation greater than the threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    for column in to_drop:
        max_corr = upper[column].max()
        column_index = X.columns.get_loc(column)
        dropped_info.append((column_index, column, max_corr))

    # Drop features 
    X_reduced = X.drop(columns=to_drop)

    return X_reduced