import numpy as np
import pytest

from src.logistic_regression import LogisticRegression


@pytest.fixture
def log_reg():
    return LogisticRegression()


def test_add_interactions(log_reg):
    X = np.array([[1, 2, 3], [3, 4, 5], [5, 6, 7]])
    log_reg.add_interactions = True
    X_with_interactions = log_reg._add_interactions(X)
    expected_result = np.array(
        [[1, 2, 3, 2, 3, 6], [3, 4, 5, 12, 15, 20], [5, 6, 7, 30, 35, 42]]
    )
    np.testing.assert_array_equal(X_with_interactions, expected_result)
