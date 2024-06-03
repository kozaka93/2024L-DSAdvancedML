from sklearn.neural_network import MLPClassifier
import pandas as pd
from evaluation import evaluate
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


def mlp_grid(X_train, y_train, grid) -> None:
    """
    Perform a grid search for the best hyperparameters of a MLP model.
    Args:
        X_train: A numpy array of features
        y_train: A numpy array of labels
        grid: A dictionary of hyperparameters to test
    """
    mlp = MLPClassifier(random_state=42)
    grid_search = GridSearchCV(mlp, grid, scoring=evaluate, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    df = pd.DataFrame(grid_search.cv_results_)
    df.to_csv("data/mlp_final.csv")
    print("Best parameter (CV score=%0.3f):" % grid_search.best_score_)
    print(grid_search.best_params_)


def svm_grid(X_train, y_train, grid) -> None:
    """
    Perform a grid search for the best hyperparameters of a SVM model.
    Args:
        X_train: A numpy array of features
        y_train: A numpy array of labels
        grid: A dictionary of hyperparameters to test
    """
    svc = SVC(random_state=42)
    grid_search = GridSearchCV(svc, grid, scoring=evaluate, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    df = pd.DataFrame(grid_search.cv_results_)
    df.to_csv("data/svm_final.csv")
    print("Best parameter (CV score=%0.3f):" % grid_search.best_score_)
    print(grid_search.best_params_)


def main():
    X_train = pd.read_csv("data/x_train.txt", header=None, delim_whitespace=True)
    y_train = pd.read_csv("data/y_train.txt", header=None, delim_whitespace=True)
    X_train = X_train[[101, 102, 103, 105]]
    X_train_scaled = StandardScaler().fit_transform(X_train)

    svm_grid = {
        "C": [0.1, 1, 10],
        "coef0": [1.7, 1.75, 1.8, 1.9, 2],
        "gamma": [1, 0.1, 0.01, 0.001, 0.0001],
        "kernel": ["poly"],
        "degree": [2, 3],
        "probability": [True],
        "shrinking": [True, False],
    }

    svm_grid(X_train_scaled, y_train, svm_grid)

    mlp_grid = {
        "hidden_layer_sizes": [
            (100,),
            (100, 100),
            (200, 200),
            (100, 100, 100),
            (100, 100, 100, 100),
        ],
        "activation": ["relu", "tanh", "logistic"],
        "solver": ["adam"],
        "alpha": [0.0001, 0.001, 0.01, 0.1],
        "learning_rate": ["constant", "adaptive"],
        "early_stopping": [True],
        "max_iter": [1000],
    }
    mlp_grid(X_train_scaled, y_train, mlp_grid)


if __name__ == "__main__":
    main()
