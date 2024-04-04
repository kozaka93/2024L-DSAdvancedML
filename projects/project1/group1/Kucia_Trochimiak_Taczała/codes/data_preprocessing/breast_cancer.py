from ucimlrepo import fetch_ucirepo

import torch

from base import PreprocessData
from remove_collinear import remove_collinear_variables


class BreastCancer(PreprocessData):
    def __init__(self, data_dir: str):
        super().__init__(
            data_dir,
            "BreastCancer",
            """This dataset contains information about breast cancer""",
        )

    def load_and_transform(self) -> tuple[torch.Tensor, torch.Tensor]:
        breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)
        X = breast_cancer_wisconsin_diagnostic.data.features
        y = breast_cancer_wisconsin_diagnostic.data.targets
        y["Diagnosis"] = y["Diagnosis"].map({"M": 1, "B": 0})
        X = remove_collinear_variables(X, threshold=0.7)
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.int)
        y_tensor = y_tensor.view(-1)
        print(X_tensor.size(), y_tensor.size())
        return X_tensor, y_tensor


if __name__ == "__main__":
    breast_cancer = BreastCancer("C:/Users/filip/Desktop/PW/Dane_AML/")
    breast_cancer.load_and_transform()
    breast_cancer.upload_data()
