import torch

from base import PreprocessData
from remove_collinear import remove_collinear_variables
import pandas as pd
import os


class WaterQuality(PreprocessData):
    def __init__(self, data_dir: str):
        super().__init__(
            data_dir,
            "WaterQuality",
            """This dataset contains information about water quality""",
        )

    def load_and_transform(self) -> tuple[torch.Tensor, torch.Tensor]:
        df = pd.read_csv(os.path.join(self.data_dir, "waterQuality1.csv"))
        df = df[df['is_safe'] != '#NUM!']
        X, y = df.drop(columns=["is_safe"]), df["is_safe"]
        X['ammonia'] = pd.to_numeric(X['ammonia'], errors='coerce')
        y = y.astype(int)
        X = remove_collinear_variables(X, threshold=0.7)
        X = torch.tensor(X.values, dtype=torch.float32)
        y = torch.tensor(y.values, dtype=torch.int)
        print(X.size(), y.size())
        return X, y


if __name__ == "__main__":
    water_quality = WaterQuality("C:/Users/filip/Desktop/PW/Dane_AML/")
    water_quality.load_and_transform()
    water_quality.upload_data()
