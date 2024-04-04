from ucimlrepo import fetch_ucirepo

import torch
import numpy as np
import pandas as pd
from base import PreprocessData
from remove_collinear import remove_collinear_variables


class AlgerianForestFires(PreprocessData):
    def __init__(self, data_dir: str):
        super().__init__(
            data_dir,
            "AlgerianForestFires",
            """This dataset contains information about algerian forest fires""",
        )

    def load_and_transform(self) -> tuple[torch.Tensor, torch.Tensor]:
        algerian_forest_fires = fetch_ucirepo(id=547)
        X = algerian_forest_fires.data.features.copy()  # Creating a copy of the features dataframe
        y = algerian_forest_fires.data.targets.copy()   # Creating a copy of the targets dataframe

        # Map string labels to numerical values using .loc
        y["Classes  "] = y["Classes  "].str.strip()
        y.loc[:, "Classes  "] = y["Classes  "].map({"fire": 1, "not fire": 0})

        # Drop rows in y with NaN and keep track of the dropped indices
        dropped_indices = y[y["Classes  "].isna()].index
        y.dropna(subset=["Classes  "], inplace=True)

        # Ensure X is aligned with y by dropping the same rows
        X = X.drop(dropped_indices)
        X.loc[:, "region"] = X["region"].map({"Sidi-Bel Abbes": 0, "Bejaia": 1})
        X.loc[:, "FWI"] = pd.to_numeric(X["FWI"], errors="coerce")
        X.dropna(subset=["FWI"], inplace=True)
        X.loc[:, "DC"] = pd.to_numeric(X["DC"], errors="coerce")
        X.dropna(subset=["DC"], inplace=True)
        X.reset_index(drop=True, inplace=True)
        X = remove_collinear_variables(X, threshold=0.7)
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.int)
        y_tensor = y_tensor.view(-1)
        print(X_tensor.size(), y_tensor.size())
        return X_tensor, y_tensor


if __name__ == "__main__":
    algerian_forest_fires = AlgerianForestFires("C:/Users/filip/Downloads/archive (2)/")
    algerian_forest_fires.load_and_transform()
    algerian_forest_fires.upload_data()
