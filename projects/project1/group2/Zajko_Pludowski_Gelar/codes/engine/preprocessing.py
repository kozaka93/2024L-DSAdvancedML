import numpy as np
import pandas as pd
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


def get_generic_preprocessing() -> Pipeline:
    cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "one-hot",
                OneHotEncoder(
                    sparse_output=False, handle_unknown="ignore", drop="first"
                ),
            ),
        ]
    )

    num_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    pipeline = Pipeline(
        [
            (
                "transformers",
                make_column_transformer(
                    (
                        cat_pipeline,
                        make_column_selector(
                            dtype_include=("object", "category")
                        ),
                    ),
                    (
                        num_pipeline,
                        make_column_selector(dtype_include=np.number),
                    ),
                ),
            )
        ]
    )

    return pipeline


def eliminate_correlated_values(
    X: pd.DataFrame, treshold: float = 5.0
) -> None:
    X = X.assign(const=1)  # faster than add_constant from statsmodels
    variables = list(range(X.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [
            variance_inflation_factor(X.iloc[:, variables].values, ix)
            for ix in range(X.iloc[:, variables].shape[1])
        ]
        vif = vif[:-1]  # don't let the constant be removed in the loop.
        maxloc = vif.index(max(vif))
        if max(vif) > treshold:
            del variables[maxloc]
            dropped = True

    print("Remaining variables:")
    print(X.columns[variables[:-1]])
    return X.iloc[:, variables[:-1]]


def ensure_last_target(df: pd.DataFrame, target_name: str) -> pd.DataFrame:
    if df.columns[-1] == target_name:
        return df

    colnames = df.columns.to_list()

    target_idx = colnames.index(target_name)
    order = [i for i in range(len(colnames))]

    order[target_idx], order[len(colnames) - 1] = (
        order[len(colnames) - 1],
        order[target_idx],
    )

    reodered_columns = [colnames[i] for i in order]

    df = df[reodered_columns]
    return df
