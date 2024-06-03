from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE, mutual_info_classif
from tqdm import trange

from utils import *


def run_tests_rfe(
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

    mi_scores = mutual_info_classif(X_train, y_train)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X_train.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    top_features = mi_scores.head(50).index

    scaler = StandardScaler()
    X_train_base = scaler.fit_transform(X_train[top_features])
    X_test_base = scaler.transform(X_test[top_features])
    y_train, y_test = y_train.values, y_test.values

    results = []
    for num in trange(1, max_features + 1):

        model = LogisticRegression()
        rfe = RFE(model, n_features_to_select=num)
        rfe.fit(X_train_base, y_train)

        X_train = X_train_base[:, rfe.support_]
        X_test = X_test_base[:, rfe.support_]

        results = test_models(results, X_train, X_test, y_train, y_test, num)

    save_results(results, "../results/results_rfe.csv", "rfe")
