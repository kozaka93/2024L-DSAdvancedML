import torch
from base import PreprocessData

import pandas as pd
import os
from remove_collinear import remove_collinear_variables


class AirlinePassengerSatisfaction(PreprocessData):
    def __init__(self, data_dir: str):
        super().__init__(
            data_dir,
            "AirlinePassengerSatisfaction",
            """This dataset contains information about satisfaction. Satisfaction is important""",
        )

    def load_and_transform(self) -> tuple[torch.Tensor, torch.Tensor]:
        aps_train = pd.read_csv(os.path.join(self.data_dir, "aps_train.csv"))
        aps_test = pd.read_csv(os.path.join(self.data_dir, "aps_test.csv"))
        aps = pd.concat([aps_train, aps_test], ignore_index=True)

        X, y = aps.drop(columns=["satisfaction"]), aps["satisfaction"]

        X = X.drop(columns=["Unnamed: 0"])
        X = X.drop(columns=["id"])
        X["Gender"] = X["Gender"].map({"Female": 0, "Male": 1})
        X["Customer Type"] = X["Customer Type"].map(
            {"Loyal Customer": 1, "disloyal Customer": 0}
        )
        X["Type of Travel"] = X["Type of Travel"].map(
            {"Personal Travel": 0, "Business travel": 1}
        )
        X["Class"] = X["Class"].map({"Eco": 0, "Eco Plus": 1, "Business": 2})

        y = y.map({"satisfied": 1, "neutral or dissatisfied": 0})
        X = remove_collinear_variables(X, threshold=0.7)
        X_tensor = torch.tensor(X.values, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.int)
        y_tensor = y_tensor.view(-1)
        print(X_tensor.size(), y_tensor.size())
        return X_tensor, y_tensor


if __name__ == "__main__":
    current_dir = os.getcwd()
    print("Current directory:", current_dir)
    airline_passenger_satisfaction = AirlinePassengerSatisfaction("C:/Users/filip/Downloads/archive (2)/")
    airline_passenger_satisfaction.load_and_transform()
    airline_passenger_satisfaction.upload_data()
