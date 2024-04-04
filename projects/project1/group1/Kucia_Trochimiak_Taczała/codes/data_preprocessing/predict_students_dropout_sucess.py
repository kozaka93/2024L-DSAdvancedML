from ucimlrepo import fetch_ucirepo

import torch

from base import PreprocessData
from remove_collinear import remove_collinear_variables


class PredictStudentsDropoutSuccess(PreprocessData):
    def __init__(self, data_dir: str):
        super().__init__(
            data_dir,
            "PredictStudentsDropoutSuccess",
            """This dataset contains information about predicting students dropout and success.""",
        )

    def load_and_transform(self) -> tuple[torch.Tensor, torch.Tensor]:
        predict_students_dropout_and_academic_success = fetch_ucirepo(id=697)

        X = predict_students_dropout_and_academic_success.data.features
        y = predict_students_dropout_and_academic_success.data.targets
        y["Target"] = y["Target"].map({"Dropout": 0, "Graduate": 1, "Enrolled": 1})
        X = remove_collinear_variables(X, threshold=0.7)
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.int)
        y_tensor = y_tensor.view(-1)
        print(X_tensor.size(), y_tensor.size())
        return X_tensor, y_tensor


if __name__ == "__main__":
    predict_students_dropout_success = PredictStudentsDropoutSuccess("data")
    predict_students_dropout_success.load_and_transform()
    predict_students_dropout_success.upload_data()
