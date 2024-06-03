from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from tqdm import trange

from utils import *


def run_tests_sfm(
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

        model = RandomForestClassifier()
        selector = SelectFromModel(model, max_features=num, threshold=-np.inf)
        selector.fit(X_train_base, y_train)

        X_train = X_train_base[:, selector.get_support()]
        X_test = X_test_base[:, selector.get_support()]

        results = test_models(results, X_train, X_test, y_train, y_test, num)

    save_results(results, "../results/results_sfm.csv", "sfm")
