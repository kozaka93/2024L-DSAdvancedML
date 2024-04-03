import numpy as np
import pandas as pd

from data.utils import get_openml_df


def get_biodeg() -> pd.DataFrame:
    """
    https://www.openml.org/search?type=data&sort=runs&status=active&id=1494
    
    41 Variables
    
    Target: 0 - not ready biodegradable; 1 - ready biodegradable 
    """
    df = get_openml_df('qsar-biodeg')
    # convert 1/2 to 0/1
    df['Target'] = df['Target'].map({'1': 0, '2': 1})
    return df


def get_breast_cancer() -> pd.DataFrame:
    """
    https://www.openml.org/search?type=data&sort=runs&status=active&id=1510
    
    30 Variables
    
    Target: prognosis (benign (0) or malignant (1)) 
    """
    df = get_openml_df('wdbc')
    # convert 1/2 to 0/1
    df['Target'] = df['Target'].map({'1': 0, '2': 1})
    return df


def get_philippine() -> pd.DataFrame:
    """
    https://www.openml.org/search?type=data&sort=runs&status=active&id=41145

    308 Variables
    
    Target: 0/1. 
    Meanings behind the variables and labels are unknown - the identity of the dataset is hidden
    """
    df = get_openml_df('philippine')
    df['Target'] = df['Target'].astype(np.int32)
    return df 


def get_spambase() -> pd.DataFrame:
    """
    https://www.openml.org/search?type=data&sort=runs&status=active&id=44

    57 Variables
    
    Target: 0 - not spam; 1 - spam 
    """
    df = get_openml_df('spambase')
    df['Target'] = df['Target'].astype(np.int32)
    return df

def get_arythmia() -> pd.DataFrame:
    """
    https://www.openml.org/search?type=data&status=active&id=5

    206 Variables 

    The dataset contains information about heartbeats and their classification.

    Returns:
    - df: pandas DataFrame containing the Cardiac Arrhythmia dataset.

    Target: 0 - arythmia, 1 - normal
    We've grouped the target variable. In the initial dataset, there was 
    differentiation among various diseases.
    """

    df = get_openml_df(5) 
    

    df['Target'] = df['Target'].apply(lambda x: '0' if x != '1' else x)\
        .map({'1': 1, '0': 0})

    # delete category variables
    df = df.select_dtypes(exclude=['category']) 
    df['Target'] = df['Target'].astype('category')

    return df


def get_auto_price() -> pd.DataFrame:
    """

    https://www.openml.org/search?type=data&sort=runs&status=active&id=756&fbclid=IwAR3ZnzDET_8tdFfj_yQ6UCEoQC6y4J8ZvfDhtW7G9YnOwc7D-oFqNjXOf-s

    15 Variables

    The dataset contains information about auto Price.

    Returns:
    - df: pandas DataFrame S

    Target: classifying all instances with a lower target value as positive ('P') and all others as negative ('N').
    Transformed with mapping 'P': 1, 'N': 0
    """

    df = get_openml_df(756)
    df['Target'] = df['Target'].map({'P': 1, 'N': 0})

    return df


