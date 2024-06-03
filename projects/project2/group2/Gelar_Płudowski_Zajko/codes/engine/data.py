import json
from pathlib import Path

import numpy as np
import pandas as pd

from .constants import BORUTA_FEATURES_PATH, RFE_RANKING_PATH


class DataProvider:
    """Wrapper for preprocessed data"""

    def __init__(
        self,
        data_path: Path,
        boruta_features_path: Path = BORUTA_FEATURES_PATH,
        rfe_ranking_path: Path = RFE_RANKING_PATH,
    ):
        """
        Args:
            data_path (Path): path to preprocessed data
            boruta_features_path (Path, optional): path to boruta's algorithm selection.
                Defaults to BORUTA_FEATURES_PATH.
            rfe_ranking_path (Path, optional): Path to RFE ranking. Defaults to RFE_RANKING_PATH.
        """
        self.boruta_features_path = boruta_features_path
        self.rfe_ranking_path = rfe_ranking_path
        with open(boruta_features_path) as f:
            self.boruta_features = json.load(f)["boruta_features"]
        with open(rfe_ranking_path) as f:
            self.rfe_ranking = json.load(f)
            self.rfe_ranking = {
                name: np.argsort(ranks).tolist()
                for name, ranks in self.rfe_ranking.items()
            }

        self.df_train = pd.read_csv(data_path / "train.csv")
        self.X_train, self.y_train = self.__preprocess_data(self.df_train)
        self.df_valid = pd.read_csv(data_path / "valid.csv")
        self.X_valid, self.y_valid = self.__preprocess_data(self.df_valid)
        self.df_test = pd.read_csv(data_path / "test.csv")
        self.X_test, self.y_test = self.__preprocess_data(self.df_test)

    def __preprocess_data(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.Series]:
        X, y = df.iloc[:, self.boruta_features], df.iloc[:, -1]
        return X, y

    @property
    def train_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """Return training data

        Returns:
            tuple[pd.DataFrame, pd.Series]: X and y
        """
        return self.X_train, self.y_train

    @property
    def validation_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """Return validation data

        Returns:
            tuple[pd.DataFrame, pd.Series]: X and y
        """
        return self.X_valid, self.y_valid

    @property
    def test_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """Return test data

        Returns:
            tuple[pd.DataFrame, pd.Series]: X and y
        """
        return self.X_test, self.y_test

    @property
    def joined_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """Return joined train, valid, and test data

        Returns:
            tuple[pd.DataFrame, pd.Series]: X and y
        """
        return pd.concat([self.X_train, self.X_valid, self.X_test]), pd.concat(
            [self.y_train, self.y_valid, self.y_test]
        )
