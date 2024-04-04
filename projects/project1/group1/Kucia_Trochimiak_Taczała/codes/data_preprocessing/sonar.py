import torch

from base import PreprocessData
from remove_collinear import remove_collinear_variables
import pandas as pd
from scipy.io import arff
import os


class Sonar(PreprocessData):
    def __init__(self, data_dir: str):
        super().__init__(
            data_dir,
            "Sonar",
            """This dataset contains information about mining""",
        )

    def load_and_transform(self) -> tuple[torch.Tensor, torch.Tensor]:
        data, meta = arff.loadarff(os.path.join(self.data_dir, "dataset_40_sonar.arff"))
        df = pd.DataFrame(data)
        df = df.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
        X, y = df.drop(columns=["Class"]), df["Class"]
        y = y.map({"Rock": 1, "Mine": 0})
        X = X.astype("float32")
        X = remove_collinear_variables(X, threshold=0.7)
        X = torch.tensor(X.values, dtype=torch.float32)
        y = torch.tensor(y.values, dtype=torch.int)
        print(X.size(), y.size())
        return X, y


if __name__ == "__main__":
    current_dir = os.getcwd()
    print("Current directory:", current_dir)
    sonar = Sonar("data")
    sonar.load_and_transform()
    sonar.upload_data()
