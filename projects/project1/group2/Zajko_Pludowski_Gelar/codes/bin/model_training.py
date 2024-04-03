from functools import partial
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import typer
from loguru import logger
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.tree import DecisionTreeClassifier
from typing_extensions import Annotated

from engine.models import LogisticRegression
from engine.optimizers import ADAMOptimizer, IWLSOptimizer, SGDOptimizer

app = typer.Typer()

RESULTS: pd.DataFrame = pd.DataFrame()
CONVERGENCE_RESULTS: pd.DatetimeTZDtype = pd.DataFrame()


def get_data(data_path: str | Path) -> Dict[str, pd.DataFrame]:
    root = Path(data_path)
    data_paths = root.glob("*.csv")

    data = {}

    for data_path in data_paths:
        df = pd.read_csv(data_path)
        data[data_path.stem] = df

    return data


def log_metric(
    model_name: str, data_name: str, score: float, fold: int
) -> None:
    row = pd.DataFrame(
        {
            "model": [model_name],
            "data": [data_name],
            "score": [score],
            "fold": [fold],
        }
    )
    global RESULTS
    RESULTS = pd.concat([RESULTS, row])


def log_coveregence(
    optimizer_name: str,
    data_name: str,
    training_history: List[float],
    fold: int,
) -> None:
    batch = pd.DataFrame(data=training_history, columns=["loglik"])
    batch["optimizer"] = optimizer_name
    batch["data"] = data_name
    batch["fold"] = fold
    batch["iter"] = np.arange(len(training_history))

    global CONVERGENCE_RESULTS

    CONVERGENCE_RESULTS = pd.concat([CONVERGENCE_RESULTS, batch])


def evaluate_model(
    model: any, y_test: np.ndarray, X_test: np.ndarray
) -> float:
    y_pred = model.predict(X_test)
    score = balanced_accuracy_score(y_test, y_pred)
    return score


@app.command(help="evaluate model on selected datasets")
def main(
    max_iter: Annotated[
        int, typer.Option(..., help="max iterations in optimizers")
    ] = 500
) -> None:
    logger.info("Start loading data")

    small_data = get_data("data/small")
    big_data = get_data("data/big")

    sklearn_models = {
        "LDA": LinearDiscriminantAnalysis,
        "QDA": QuadraticDiscriminantAnalysis,
        "tree": DecisionTreeClassifier,
        "forest": RandomForestClassifier,
    }

    optimizers = {
        "SGD": partial(SGDOptimizer, **{"num_batches": 1}),
        "IWLS": partial(IWLSOptimizer, **{}),
        "ADAM": partial(ADAMOptimizer, **{"num_batches": 1}),
    }

    for data_name, data in (small_data | big_data).items():
        logger.info(f"Start trainig on data: {data_name}")

        X, y = data.iloc[:, :-1], data.iloc[:, -1]
        X, y = X.to_numpy(), y.to_numpy()
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        for idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            for model_name, Model in sklearn_models.items():
                logger.info(f"{model_name} - {idx}")

                model = Model()
                model.fit(X_train, y_train)
                score = evaluate_model(model, y_test, X_test)

                log_metric(model_name, data_name, score, idx)

            for optmizer_name, Optimizer in optimizers.items():
                logger.info(f"{optmizer_name} - {idx}")

                optimizer = Optimizer()
                model = LogisticRegression()
                optimizer.optimize(model, X_train, y_train, max_iter=max_iter)
                score = evaluate_model(model, y_test, X_test)

                log_metric(optmizer_name, data_name, score, idx)
                log_coveregence(
                    optmizer_name, data_name, optimizer.metric_history, idx
                )

                if data_name in small_data.keys():
                    logger.info(f"{optmizer_name} - {idx} - with interactions")

                    optimizer = Optimizer()
                    model = LogisticRegression()
                    optimizer.optimize(
                        model,
                        X_train,
                        y_train,
                        use_iteractions=True,
                        max_iter=max_iter,
                    )
                    score = evaluate_model(model, y_test, X_test)

                    log_metric(
                        optmizer_name, f"{data_name}-iteractions", score, idx
                    )
                    log_coveregence(
                        optmizer_name, data_name, optimizer.metric_history, idx
                    )

    global CONVERGENCE_RESULTS
    global RESULTS

    CONVERGENCE_RESULTS.to_csv("results/convergence_results.csv", index=False)
    RESULTS.to_csv("results/results.csv", index=False)


if __name__ == "__main__":
    main()
