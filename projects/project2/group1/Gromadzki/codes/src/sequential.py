from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector
from tqdm import trange

from utils import *


def run_tests_seq(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    max_features: int,
    th: float,
) -> None:
    to_drop = drop_colinear(X_train, th)
    X_train = X_train.drop(columns=to_drop)
    X_test = X_test.drop(columns=to_drop)

    scaler = StandardScaler()
    X_train_base = scaler.fit_transform(X_train)
    X_test_base = scaler.transform(X_test)
    y_train, y_test = y_train.values, y_test.values

    results = []
    for num in trange(1, max_features + 1):

        model = LogisticRegression()
        selector = SequentialFeatureSelector(
            model, n_features_to_select=num, direction="forward", n_jobs=-1
        )
        selector.fit(X_train_base, y_train)

        X_train = X_train_base[:, selector.support_]
        X_test = X_test_base[:, selector.support_]

        results = test_models(results, X_train, X_test, y_train, y_test, num)

    save_results(results, "../results/results_seq.csv", "seq")
