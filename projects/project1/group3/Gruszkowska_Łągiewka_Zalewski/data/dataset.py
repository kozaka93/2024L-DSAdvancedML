from ucimlrepo import fetch_ucirepo
import json
import numpy as np
from sklearn.preprocessing import LabelEncoder


def get_dataset(dataset_name: str):
    """
    Parameters
    ----------
    dataset_name: name of dataset, list of possible dataset defined in dataset_names.json

    Returns
    -------
    X : numpy.ndarray
       Matrix of features
    y.flatten() : numpy.ndarray
       A vector of features encoded as 0 and 1
    """
    mapping = json.load(open('data/dataset_names.json'))
    dataset = fetch_ucirepo(id=mapping[dataset_name])
    X = np.array(dataset.data.features)
    y = np.array(dataset.data.targets).ravel()
    if dataset_name not in ['algerian_forest_fires', 'waveform_database_generator_version_1']:
        le = LabelEncoder()
        y = le.fit_transform(y)
    return X, y.flatten()
