from itertools import combinations

import numpy as np
from scipy.io import arff


class DataLoader:
    def __init__(self, product=False):
        self.data = {}
        self.small_datasets = ["banknote", "kin8nm", "phoneme"]
        self.large_datasets = [
            "elevators",
            "jm1",
            "kdd_JapaneseVowels",
            "mfeat-karhunen",
            "mfeat-zernike",
            "pc1",
        ]
        self.supported_datasets = self.small_datasets + self.large_datasets
        self.product = product

    def __getitem__(self, item):
        if item not in self.supported_datasets:
            raise ValueError(f"Dataset {item} is not supported")

        if item not in self.data:
            with open(f"data/{item}.arff", "r", encoding="utf-8") as f:
                data, _ = arff.loadarff(f)
            data = np.array(data.tolist(), dtype=object)

            x = data[:, :-1].astype(float)

            y_counts = np.unique(data[:, -1], return_counts=True)
            majority_class = y_counts[1].argmax()
            y = np.array([0 if _y == y_counts[0][majority_class] else 1 for _y in data[:, -1]], dtype=float)

            y = y[~np.isnan(x).any(axis=1)]
            x = x[~np.isnan(x).any(axis=1)]

            if self.product:
                cols = []
                for col1, col2 in combinations(range(x.shape[1]), 2):
                    cols.append(x[:, col1] * x[:, col2])
                x = np.column_stack((x, *cols))

            collinear_features = np.where(np.abs(np.corrcoef(x, rowvar=False)) > 0.75)
            collinear_features = np.unique(collinear_features[0][collinear_features[0] != collinear_features[1]])
            x = np.delete(x, collinear_features, axis=1)

            sparse_counts = np.isclose(x, 0, atol=1e-4).sum(axis=0)
            sparse_features = np.where(sparse_counts / x.shape[0] > 0.85)[0]
            x = np.delete(x, sparse_features, axis=1)

            self.data[item] = (x, y)

        return self.data[item]

    def get_supported_datasets(self):
        return self.supported_datasets

    def get_small_supported_datasets(self):
        return self.small_datasets

    def get_large_supported_datasets(self):
        return self.large_datasets
