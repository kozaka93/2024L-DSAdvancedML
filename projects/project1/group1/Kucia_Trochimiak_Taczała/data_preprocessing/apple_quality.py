import torch

from base import PreprocessData

import pandas as pd
import os
from remove_collinear import remove_collinear_variables


class AppleQuality(PreprocessData):
    def __init__(self, data_dir: str):
        super().__init__(
            data_dir,
            "AppleQuality",
            """This dataset contains information about various 
            attributes of a set of fruits, providing insights into their characteristics.""",
        )

    def load_and_transform(self) -> tuple[torch.Tensor, torch.Tensor]:
        df = pd.read_csv(os.path.join(self.data_dir, "apple_quality.csv"))
        df = df.drop(df.index[-1])
        df = df.drop(columns=df.columns[0])
        X, y = df.drop(columns=["Quality"]), df["Quality"]
        y = y.map({"good": 1, "bad": 0})
        X = X.astype("float32")
        X = remove_collinear_variables(X, threshold=0.7)
        X = torch.tensor(X.values, dtype=torch.float32)
        y = torch.tensor(y.values, dtype=torch.int)
        print(X.size(), y.size())
        return X, y


if __name__ == "__main__":
    current_dir = os.getcwd()
    print("Current directory:", current_dir)
    apple_quality = AppleQuality("C:/Users/filip/Desktop/PW/Dane_AML/")
    apple_quality.load_and_transform()
    apple_quality.upload_data()
