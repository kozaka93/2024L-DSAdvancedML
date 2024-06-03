from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector
from tqdm import trange

from utils import *


def run_tests_mean(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    max_features: int,
    th: float,
) -> None:
    to_drop = drop_colinear(X_train, th)
    X_train_base = X_train.drop(columns=to_drop)
    X_test_base = X_test.drop(columns=to_drop)

    tmp = pd.concat([X_train_base, y_train], axis=1)
    tmp0 = tmp[tmp["label"] == 0]
    tmp1 = tmp[tmp["label"] == 1]
    diff = (tmp0.mean() - tmp1.mean() / tmp.mean()).abs()
    diff = diff.sort_values(ascending=False)
    y_train, y_test = y_train.values, y_test.values

    results = []
    for num in trange(1, max_features + 1):

        X_train = X_train_base[diff.index[:num]]
        X_test = X_test_base[diff.index[:num]]
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        results = test_models(results, X_train, X_test, y_train, y_test, num)

    save_results(results, "../results/results_mean.csv", "mean")
