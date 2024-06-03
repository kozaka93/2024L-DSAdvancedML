import pandas as pd
from sklearn.model_selection import train_test_split

from corr import run_tests_corr
from sfm import run_tests_sfm
from k_best import run_tests_kbest
from mean import run_tests_mean
from rfe import run_tests_rfe
from sequential import run_tests_seq

from utils import get_data

SEED = 1337
MAX_FEATURES = 25
TH = 0.9
SPLIT = 0.2

def main(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    max_features: int,
    th: float,
) -> None:
    run_tests_corr(X_train, X_test, y_train, y_test, max_features, th)
    run_tests_sfm(X_train, X_test, y_train, y_test, max_features, th)
    run_tests_kbest(X_train, X_test, y_train, y_test, max_features, th)
    run_tests_mean(X_train, X_test, y_train, y_test, max_features, th)
    run_tests_rfe(X_train, X_test, y_train, y_test, max_features, th)
    run_tests_seq(X_train, X_test, y_train, y_test, max_features, th)


if __name__ == "__main__":
    df, y = get_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=SPLIT, random_state=SEED
    )

    main(X_train, X_test, y_train, y_test, MAX_FEATURES, TH)
