"""
datasets.py

Preprocessing for all datasets.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from datasets.dataset_model import Dataset
from datasets.preprocess_helpers import one_hot_encode
from scipy.io import arff
from sklearn import preprocessing

DATA_DIR = Path("../data")


class Booking(Dataset):
    """Preprocessing for booking.csv dataset."""

    def __init__(self):
        super().__init__(
            name="booking",
            filename=DATA_DIR / "booking.csv",
            target_colname="booking status",
        )

    def load_and_preprocess(self) -> pd.DataFrame:
        """Loads and preprocesses the Booking dataset."""
        booking = pd.read_csv(self.filename).drop(
            ["Booking_ID", "date of reservation"], axis=1
        )
        booking["market segment type"] = 1 * (
            booking["market segment type"] == "Online"
        )
        booking["booking status"] = 1 * (booking["booking status"] == "Canceled")
        label_encoder = preprocessing.LabelEncoder()
        booking["room type"] = label_encoder.fit_transform(booking["room type"])
        booking = one_hot_encode(booking)
        self.df = booking
        return booking


class Churn(Dataset):
    """Preprocessing for churn.csv dataset."""

    def __init__(self) -> None:
        super().__init__(
            name="churn", filename=DATA_DIR / "churn.csv", target_colname="Target"
        )

    def load_and_preprocess(self) -> pd.DataFrame:
        """Loads and preprocesses the Churn dataset."""
        churn = pd.read_csv(self.filename)
        churn["FrequentFlyer"] = 1 * (churn["FrequentFlyer"] == "Yes")
        churn["BookedHotelOrNot"] = 1 * (churn["BookedHotelOrNot"] == "Yes")
        churn["AccountSyncedToSocialMedia"] = 1 * (
            churn["AccountSyncedToSocialMedia"] == "Yes"
        )
        churn.loc[churn["AnnualIncomeClass"] == "Low Income", "AnnualIncomeClass"] = 0
        churn.loc[
            churn["AnnualIncomeClass"] == "Middle Income", "AnnualIncomeClass"
        ] = 1
        churn.loc[churn["AnnualIncomeClass"] == "High Income", "AnnualIncomeClass"] = 2
        churn.AnnualIncomeClass = churn.AnnualIncomeClass.astype(int)
        self.df = churn
        return churn


class Diabetes(Dataset):
    """Preprocessing for diabetes.arff dataset."""

    def __init__(self) -> None:
        super().__init__(
            name="diabetes",
            filename=DATA_DIR / "diabetes.arff",
            target_colname="class",
        )

    def load_and_preprocess(self) -> pd.DataFrame:
        """Loads and preprocesses the Diabetes dataset."""
        df = pd.DataFrame(arff.loadarff(self.filename)[0])
        str_df = df.select_dtypes([object]).astype(str)
        df[str_df.columns] = str_df
        df["class"] = df["class"].apply(lambda x: 1 if x == "tested_positive" else 0)
        self.df = df
        return df


class Employee(Dataset):
    """Preprocessing for employee.csv dataset."""

    def __init__(self) -> None:
        super().__init__(
            name="employee",
            filename=DATA_DIR / "employee.csv",
            target_colname="LeaveOrNot",
        )

    def load_and_preprocess(self) -> pd.DataFrame:
        """Loads and preprocesses the Employee dataset."""
        df = pd.read_csv(self.filename)
        df["EducationBachelors"] = 1 * (df["Education"] == "Bachelors")
        df["EducationMasters"] = 1 * (df["Education"] == "Masters")
        df["Gender"] = df["Gender"].map({"Female": 1, "Male": 0})
        df["EverBenched"] = df["EverBenched"].map({"No": 0, "Yes": 1})
        df.drop(["Education", "City"], axis=1, inplace=True)
        self.df = df
        return df


class Challenger(Dataset):
    """Preprocessing for challenger_lol.csv dataset."""

    def __init__(self) -> None:
        super().__init__(
            name="challenger_lol",
            filename=DATA_DIR / "challenger_lol.csv",
            target_colname="blueWins",
        )

    def load_and_preprocess(self) -> pd.DataFrame:
        """Loads and preprocesses the Challenger dataset."""
        df = pd.read_csv(self.filename)
        df.drop(["blueFirstBlood", "redFirstBlood", "gameId"], axis=1, inplace=True)
        for col in ["blue", "red"]:
            for lane in ["BOT_LANE", "MID_LANE", "TOP_LANE"]:
                df[f"{col}FirstTowerLane_{lane}"] = df[f"{col}FirstTowerLane"].apply(
                    lambda x: int(lane in x)
                )
            for dragon in ["AIR_DRAGON", "WATER_DRAGON", "FIRE_DRAGON", "EARTH_DRAGON"]:
                df[f"{col}DragnoType_{dragon}"] = df[f"{col}DragnoType"].apply(
                    lambda x: int(dragon in x)
                )
            df.drop(f"{col}FirstTowerLane", axis=1, inplace=True)
            df.drop(f"{col}DragnoType", axis=1, inplace=True)
        self.df = df
        return df


class Jungle(Dataset):
    """Preprocessing for jungle_chess.arff dataset."""

    def __init__(self) -> None:
        super().__init__(
            name="jungle",
            filename=DATA_DIR / "jungle_chess.arff",
            target_colname="class",
        )

    def load_and_preprocess(self) -> pd.DataFrame:
        """Loads and preprocesses the Jungle dataset."""
        df = arff.loadarff(self.filename)
        df = pd.DataFrame(df[0])
        str_df = df.select_dtypes([object])
        str_df = str_df.stack().str.decode("utf-8").unstack()
        for col in str_df:
            df[col] = str_df[col]
        df = df[df["class"] != "d"]
        df[
            ["highest_strength", "closest_to_den", "fastest_to_den", "class"]
        ] = df.copy()[
            ["highest_strength", "closest_to_den", "fastest_to_den", "class"]
        ].applymap(
            lambda x: int(x == "w")
        )
        df = pd.concat(
            [
                df,
                pd.get_dummies(
                    df[["white_piece0_advanced", "black_piece0_advanced"]],
                    drop_first=True,
                )
                * 1,
            ],
            axis=1,
        )
        df.drop(
            [
                "white_piece0_advanced",
                "black_piece0_advanced",
                "white_piece0_in_water",
                "black_piece0_in_water",
            ],
            axis=1,
            inplace=True,
        )
        df = df.apply(pd.to_numeric)
        self.df = df
        return df


class Ionosphere(Dataset):
    """Preprocessing for ionosphere.data dataset."""

    def __init__(self) -> None:
        super().__init__(
            name="ionosphere",
            filename=DATA_DIR / "ionosphere.data",
            target_colname="class",
        )

    def load_and_preprocess(self) -> pd.DataFrame:
        """Loads and preprocesses the Ionosphere dataset."""
        df = pd.read_csv(self.filename, header=None)
        df = df.rename(columns={34: "class"})
        df["class"] = df["class"].map({"g": 0, "b": 1})
        self.df = df
        return df


class Water(Dataset):
    """Preprocessing for water_quality.csv dataset."""

    def __init__(self) -> None:
        super().__init__(
            name="water",
            filename=DATA_DIR / "water_quality.csv",
            target_colname="is_safe",
        )
        self.additional_preprocess = Water._impute_water

    def load_and_preprocess(self) -> pd.DataFrame:
        """Loads and preprocesses the Water dataset."""
        water = pd.read_csv(self.filename)
        self.df = water
        return water

    @staticmethod
    def _replace_water_nans(df: pd.DataFrame) -> pd.DataFrame:
        """Imputation of #NUM! values in water_quality data frame."""
        df["ammonia"] = df["ammonia"].replace("#NUM!", -100)
        df["ammonia"] = df["ammonia"].astype(float)
        df["ammonia"] = df["ammonia"].replace(
            -100, df.loc[df["ammonia"] != -100, "ammonia"].mean()
        )

        df["is_safe"] = df["is_safe"].replace("#NUM!", -100)
        df["is_safe"] = df["is_safe"].astype(int)
        return df

    @staticmethod
    def _impute_water(
        water_train: pd.DataFrame, water_test: pd.DataFrame
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Data imputation in water_quality data frame using column dominant from training set."""
        water_train = Water._replace_water_nans(water_train)
        water_test = Water._replace_water_nans(water_test)

        if np.mean(water_train.loc[water_train["is_safe"] != -100, "is_safe"]) > 0.5:
            dominant = 1
        else:
            dominant = 0

        water_train["is_safe"] = water_train["is_safe"].replace(-100, dominant)
        water_test["is_safe"] = water_test["is_safe"].replace(-100, dominant)

        return water_train, water_test


class Seeds(Dataset):
    """Preprocessing for seeds.txt dataset."""

    def __init__(self) -> None:
        super().__init__(
            name="seeds", filename=DATA_DIR / "seeds.txt", target_colname="class"
        )

    def load_and_preprocess(self) -> pd.DataFrame:
        """Loads and preprocesses the Seeds dataset."""
        cols = [
            "A",
            "P",
            "C",
            "kernel_length",
            "kernel_width",
            "asymmetry_coef",
            "kernel_groove_length",
            "class",
        ]
        df = pd.read_csv(self.filename, sep=r"\s+", header=None, names=cols)
        # combine classes 1, 3 (similar) and 2 (different) based on pairplot
        df["class"] = df["class"].map({1: 0, 2: 1, 3: 0})
        self.df = df
        return df


class Sonar(Dataset):
    """Preprocessing for sonar.data dataset."""

    def __init__(self) -> None:
        super().__init__(
            name="sonar", filename=DATA_DIR / "sonar.data", target_colname="class"
        )

    def load_and_preprocess(self) -> pd.DataFrame:
        """Loads and preprocesses the Sonar dataset."""
        df = pd.read_csv(self.filename, header=None)
        df = df.rename(columns={60: "class"})
        df["class"] = df["class"].map({"R": 0, "M": 1})
        self.df = df
        return df
