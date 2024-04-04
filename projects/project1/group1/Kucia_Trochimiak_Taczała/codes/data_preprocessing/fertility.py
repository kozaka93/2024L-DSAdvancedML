from ucimlrepo import fetch_ucirepo

import torch
from remove_collinear import remove_collinear_variables
from base import PreprocessData


class Fertility(PreprocessData):
    def __init__(self, data_dir: str):
        super().__init__(
            data_dir,
            "Fertility",
            """This dataset contains information about fertility""",
        )

    def load_and_transform(self) -> tuple[torch.Tensor, torch.Tensor]:
        fertility = fetch_ucirepo(id=244)
        X = fertility.data.features
        y = fertility.data.targets
        y["diagnosis"] = y["diagnosis"].map({"N": 1, "O": 0})
        X = remove_collinear_variables(X, threshold=0.7)
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.int)
        y_tensor = y_tensor.view(-1)
        print(X_tensor.size(), y_tensor.size())
        return X_tensor, y_tensor


if __name__ == "__main__":
    fertility = Fertility("data")
    fertility.load_and_transform()
    fertility.upload_data()
