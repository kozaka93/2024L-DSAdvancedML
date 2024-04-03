import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import Tuple


def drop_colinear(df: pd.DataFrame, th: int) -> pd.DataFrame:
    while True:
        vif = pd.DataFrame()
        vif["Variable"] = df.columns
        vif["VIF"] = [
            variance_inflation_factor(df.values, i) for i in range(df.shape[1])
        ]
        vif = vif[vif.VIF > th]
        if vif.empty:
            break
        to_drop = vif[vif.VIF == vif.VIF.max()]["Variable"].values[0]
        df = df.drop(columns=to_drop)
    return df


def preprocess(df: pd.DataFrame, th: int) -> Tuple[pd.DataFrame, pd.Series]:
    print(f"Missing data: {df.isna().sum().sum() / df.size * 100:.2f}%")
    df.dropna(inplace=True)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X = drop_colinear(X, th)
    return X, y
