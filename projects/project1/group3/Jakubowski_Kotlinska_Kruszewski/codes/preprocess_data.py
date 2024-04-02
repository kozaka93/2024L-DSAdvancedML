import argparse
import pickle
import numpy as np
import pandas as pd
from typing import List, Tuple
from utils import list_data


def impute_data(data_list: List[Tuple[np.array, np.array]]):
    """
    Fills in missing data by replacing it with the average value of the corresponding column.
    """
    for i, (X, y) in enumerate(data_list):
        X = pd.DataFrame(X)
        y = pd.DataFrame(y)
        missing_X = X.isnull().sum().sum()
        missing_y = y.isnull().sum().sum()

        if missing_X > 0 or missing_y > 0:
            print(f"Dataset {i+1} has missing values:")
            print(f"Missing values in X: {missing_X}")
            print(f"Missing values in y: {missing_y}")
            X = X.fillna(X.mean())
            y = y.fillna(y.mean())
            data_list[i] = (X.to_numpy(), (y.to_numpy()).flatten())
        else:
            print(f"Dataset {i+1} has no missing values.")


def remove_correlated_columns(df: pd.DataFrame, threshold: float = 0.8):
    """
    Removes features with correlation coefficient higher than 0.8.
    """
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    df.drop(columns=to_drop, inplace=True)
    return df


def main(args):
    data_list = list_data(args.data_list)

    impute_data(data_list)

    for i, (X, y) in enumerate(data_list):
        print(f"Checking dataset {i+1} for highly correlated columns...")
        X = pd.DataFrame(X)
        y = pd.DataFrame(y, columns=["y"])
        data = pd.concat([X, y], axis=1)

        X_cleaned = remove_correlated_columns(data.drop(columns=["y"]), threshold=0.8)
        y_cleaned = data["y"]

        data_list[i] = (X_cleaned.to_numpy(), (y_cleaned.to_numpy()).flatten())
        print(f"{X.shape[1] - X_cleaned.shape[1]} highly correlated columns removed.")

    # Save data_list to a file - you can load it later using pickle.load()
    with open("data_list.pkl", "wb") as f:
        pickle.dump(data_list, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-list",
        type=list,
        default=[37, 1462, 871, 752, 1120, 23512, 23517, 979, 1487],
        help="List of datasets from openML to preprocess",
    )

    args = parser.parse_args()
    main(args)
