import openml
import numpy as np
from sklearn.preprocessing import LabelEncoder
from typing import Tuple, List

def read_data(id: int) -> Tuple[np.array, np.array]:
    """
    Loads data using OpenML client and splits it into features and target matrices.
    """
    dataset = openml.datasets.get_dataset(id)
    df, _, _, _ = dataset.get_data(dataset_format="dataframe")
    numerical_cols = df.select_dtypes(include='number').columns
    target_col = df.select_dtypes(exclude='number').columns
    X = df[numerical_cols].to_numpy()
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[target_col])
    return X, y

def list_data(id_list: List[int]) -> List[Tuple[np.array, np.array]]:
    """
    Creates a list of pairs of features and target matrices (X, y).
    """
    data_list = []
    for id in id_list:
        data_list.append(read_data(id))
    return data_list