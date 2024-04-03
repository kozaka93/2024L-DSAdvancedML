import os

import matplotlib.pyplot as plt
import numpy as np
import openml
import pandas as pd
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from ucimlrepo import fetch_ucirepo

from src.logistic_regression import LogisticRegression
from src.prepare_datasets import prepare_data

DATASETS = {
    "ucl": {
        # big
        151: ["class", {"R": 1, "M": 0}, False],
    },
    "openml": {
        # small
        974: ["binaryClass", {"N": 0, "P": 1}, False],
        969: ["binaryClass", {"N": 0, "P": 1}, False],  # iris
        1462: ["Class", {1: 0, 2: 1}, True],
        # big
        849: ["binaryClass", {"N": 0, "P": 1}, False],
        1547: ["Class", {"class1": 0, "class2": 1}, False],
        1510: ["Class", {1: 0, 2: 1}, True],
        833: ["binaryClass", {"N": 0, "P": 1}, False],  # - sgd plot with spikes
        879: ["binaryClass", {"N": 0, "P": 1}, False],
    },
}

classifiers = {
    "Logistic Regression (SGD) with interactions": {
        "add_interactions": True,
        "learning_rate": 0.001,
        "max_iter": 500,
        "tolerance": 1e-5,
        "optimizer": "sgd",
    },
    "Logistic Regression (SGD)": {
        "add_interactions": False,
        "learning_rate": 0.001,
        "max_iter": 500,
        "tolerance": 1e-5,
        "optimizer": "sgd",
    },
    "Logistic Regression (Adam) with interactions": {
        "add_interactions": True,
        "learning_rate": 0.01,
        "max_iter": 500,
        "tolerance": 1e-5,
        "optimizer": "adam",
    },
    "Logistic Regression (Adam)": {
        "add_interactions": False,
        "learning_rate": 0.01,
        "max_iter": 500,
        "tolerance": 1e-5,
        "optimizer": "adam",
    },
    "Logistic Regression (IRLS) with interactions": {
        "add_interactions": True,
        "learning_rate": 0.01,
        "max_iter": 500,
        "tolerance": 1e-5,
        "optimizer": "irls",
    },
    "Logistic Regression (IRLS)": {
        "add_interactions": False,
        "learning_rate": 0.01,
        "max_iter": 500,
        "tolerance": 1e-5,
        "optimizer": "irls",
    },
    "Linear Discriminant Analysis": LinearDiscriminantAnalysis,
    "Quadratic Discriminant Analysis": QuadraticDiscriminantAnalysis,
    "Decision Tree": DecisionTreeClassifier,
    "Random Forest": RandomForestClassifier,
}


def plot_log_likelihood(
    dataset_id,
    source,
    log_likelihood_history_sgd,
    log_likelihood_history_adam,
    log_likelihood_history_irls,
):
    max_iters_sgd = max(len(history) for history in log_likelihood_history_sgd)
    max_iters_adam = max(len(history) for history in log_likelihood_history_adam)
    max_iters_irls = max(len(history) for history in log_likelihood_history_irls)
    plt.figure()
    for history in log_likelihood_history_sgd:
        plt.plot(
            range(len(history)),
            history,
            color="blue",
            alpha=0.2,
        )
    for history in log_likelihood_history_adam:
        plt.plot(
            range(len(history)),
            history,
            color="green",
            alpha=0.2,
        )
    for history in log_likelihood_history_irls:
        plt.plot(
            range(len(history)),
            history,
            color="red",
            alpha=0.2,
        )

    avg_sgd = np.nanmean(
        [
            history + [np.nan] * (max_iters_sgd - len(history))
            for history in log_likelihood_history_sgd
        ],
        axis=0,
    )
    avg_adam = np.nanmean(
        [
            history + [np.nan] * (max_iters_adam - len(history))
            for history in log_likelihood_history_adam
        ],
        axis=0,
    )
    avg_irls = np.nanmean(
        [
            history + [np.nan] * (max_iters_irls - len(history))
            for history in log_likelihood_history_irls
        ],
        axis=0,
    )

    plt.plot(
        range(len(avg_sgd)),
        avg_sgd,
        color="blue",
        label="Average LR (SGD)",
    )
    plt.plot(
        range(len(avg_adam)),
        avg_adam,
        color="green",
        label="Average LR (Adam)",
    )
    plt.plot(
        range(len(avg_irls)),
        avg_irls,
        color="red",
        label="Average LR (IRLS)",
    )

    plt.legend()
    plt.xlabel("Number of iterations")
    plt.ylabel("Log-Likelihood")
    plt.title(f"Log-Likelihood values after each iteration for Dataset Id={dataset_id}")
    plt.ylim(-0.8, 0)
    directory = "experiments/log_likelihood"
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(f"{directory}/{source}_{dataset_id}.png")


def compare_with_different_classifiers(no_iters=5, test_size=0.2):
    # Compare the classification performance of logistic regression (try all 3 methods: IWLS, SGD, ADAM) and LDA, QDA, Decision tree and Random Forest.
    for source, datasets in DATASETS.items():
        for id, dataset in datasets.items():
            results = []
            print(f"{source} dataset, id={id}:")
            target_column, mapping, cast_to_int = dataset
            if source == "ucl":
                dataset = fetch_ucirepo(id=id).data
                X = dataset.features
                y = dataset.targets
                df = pd.concat([X, y], axis=1)
            else:
                df = openml.datasets.get_dataset(id).get_data()[0]
            X, y = prepare_data(df, cast_to_int, mapping, target_column)
            log_likelihood_history = {"adam": [], "sgd": [], "irls": []}
            for name, params_or_model in list(classifiers.items()):
                accuracy = []
                for j in range(no_iters):
                    if "Logistic Regression" in name:
                        model = LogisticRegression(**params_or_model)
                    else:
                        model = params_or_model()
                    print(f"Fitting model: {name}")

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=j
                    )
                    np.random.seed(j)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    accuracy = balanced_accuracy_score(y_test, y_pred)
                    results.append(
                        {
                            "Dataset": f"{source}_{id}",
                            "Run": j + 1,
                            "Classifier": name,
                            "Balanced_Accuracy": accuracy,
                        }
                    )
                    if "Logistic Regression" in name:
                        optimizer = params_or_model["optimizer"]

                        log_history = model.get_log_likelihood()
                        log_likelihood_history[optimizer].append(log_history)
            plot_log_likelihood(
                id,
                source,
                log_likelihood_history["sgd"],
                log_likelihood_history["adam"],
                log_likelihood_history["irls"],
            )
            results = pd.DataFrame(results)
            results.to_csv(f"experiments/results/{source}_{id}.csv", index=False)
    return


if __name__ == "__main__":
    compare_with_different_classifiers(5)
