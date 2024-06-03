from pathlib import Path

from .utils import make_paths

DATA_PATH = Path("data/preprocessed")
BORUTA_FEATURES_PATH = Path("data/features/boruta.json")
RFE_RANKING_PATH = Path("data/features/rfe_ranking.json")
RESULTS_PATH = Path("results")
make_paths(RESULTS_PATH)

RETURN_ON_CORRECT_CLIENT = 10
COST_OF_FEATURE = 200
CLIENTS_IN_TEST_SET = 5000
CLIENTS_TO_SELECT_IN_TEST_SET = 1000

# Conversion of possible solver penalty pairs to mitigate
# issue with non-supported conditional categorical variables.
LOGISTIC_SOLVER_PENALTY_PAIRS = [
    ("liblinear", "l1"),
    ("liblinear", "l2"),
    ("saga", "elasticnet"),
    ("saga", "l1"),
    ("saga", "l2"),
    ("saga", None),
    ("lbfgs", None),
    ("lbfgs", "l2"),
    ("newton-cg", None),
    ("newton-cg", "l2"),
    ("newton-cholesky", None),
    ("newton-cholesky", "l2"),
    ("sag", None),
    ("sag", "l2"),
]
