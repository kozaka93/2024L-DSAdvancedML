import torch

from base import PreprocessData

import pandas as pd
from ucimlrepo import fetch_ucirepo
from remove_collinear import remove_collinear_variables


class HeartDisease(PreprocessData):
    def __init__(self, data_dir: str):
        super().__init__(
            data_dir,
            "HeartDisease",
            """This dataset contains information about heart disease.""",
        )

    def load_and_transform(self) -> tuple[torch.Tensor, torch.Tensor]:
        heart_disease = fetch_ucirepo(id=45)
        X = pd.DataFrame(heart_disease.data.features)
        y = X["fbs"]
        integer_variables = ["age", "trestbps", "chol", "thalach", "oldpeak"]
        X = X[integer_variables]
        X = X.astype("float32")
        X = remove_collinear_variables(X, threshold=0.7)
        X = torch.tensor(X.values, dtype=torch.float32)
        y = torch.tensor(y.values, dtype=torch.int)
        print(X.size(), y.size())
        return X, y


if __name__ == "__main__":
    heart = HeartDisease("data")
    heart.load_and_transform()
    heart.upload_data()
