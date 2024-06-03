import json
import time
from itertools import product

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def get_data():
    X = pd.read_csv("../data/x_train.txt", sep=" ", header=None)
    y = pd.read_csv("../data/y_train.txt", header=None)
    scaler = StandardScaler()

    X = scaler.fit_transform(X)
    y = y.values.ravel()

    return X, y


def get_param_combinations(param_dict):
    value_prod = list(product(*param_dict.values()))
    keys = param_dict.keys()
    return [dict(zip(keys, values)) for values in value_prod]


def param_json_to_str(param_json):
    if type(param_json).__name__ == "function":
        return param_json.__name__
    return param_json


def get_params_json(params):
    params_mapped = {k: param_json_to_str(v) for k, v in params.items()}
    return json.dumps(params_mapped).replace('"', "'")


def save_results(results, filename):
    df = pd.DataFrame(
        results,
        columns=[
            "feature_selector",
            "feature_selector_params",
            "classifier",
            "classifier_params",
            "n_features",
            "accuracy",
            "accuracy_std",
            "accuracy_top_20pc",
            "elapsed_time",
        ],
    )
    df.to_csv(f"../results/{filename}.csv", index=False)


def experiment(
    X,
    y,
    fs_cls,
    fs_kwargs,
    clf_cls,
    clf_kwargs,
    n_features,
    k_param_name,
    requires_estimator,
    train_test_seeds,
):
    # Run experiment
    start = time.time()
    accs, accs_top_20pc = _experiment_internal(
        X,
        y,
        fs_cls,
        fs_kwargs,
        clf_cls,
        clf_kwargs,
        n_features,
        k_param_name,
        requires_estimator,
        train_test_seeds,
    )
    elapsed = time.time() - start
    elapsed = elapsed / len(train_test_seeds)

    acc = accs.mean()
    acc_std = accs.std()
    acc_top_20pc = accs_top_20pc.mean()

    result = (
        fs_cls.__name__,
        get_params_json(fs_kwargs),
        clf_cls.__name__,
        get_params_json(clf_kwargs),
        n_features,
        acc,
        acc_std,
        acc_top_20pc,
        elapsed,
    )

    return result


def _experiment_internal(
    X,
    y,
    fs_cls,
    fs_kwargs,
    clf_cls,
    clf_kwargs,
    n_features,
    k_param_name,
    requires_estimator,
    train_test_seeds,
):
    clf = clf_cls(**clf_kwargs)

    fs_kwargs = {
        k_param_name: n_features,
        **fs_kwargs,
    }
    if requires_estimator:
        feature_selector = fs_cls(estimator=clf, **fs_kwargs)
    else:
        feature_selector = fs_cls(**fs_kwargs)

    accs = []
    accs_top_20pc = []

    for seed in train_test_seeds:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            stratify=y,
            random_state=seed,
        )

        # Feature selection
        X_train = feature_selector.fit_transform(X_train, y_train)
        X_test = feature_selector.transform(X_test)

        # Training
        clf.fit(X_train, y_train)

        # Prediction
        pred = clf.predict(X_test)

        proba_1 = clf.predict_proba(X_test)[:, 1]
        proba_1 = np.array([proba_1, y_test]).T
        proba_1 = proba_1[proba_1[:, 0].argsort()][::-1]

        # Evaluation
        acc = accuracy_score(y_test, pred)
        top_20pc = proba_1[: int(len(proba_1) * 0.2)]
        acc_top_20pc = accuracy_score(top_20pc[:, 1], np.round(top_20pc[:, 0]))

        print(seed, acc)

        accs.append(acc)
        accs_top_20pc.append(acc_top_20pc)

    return np.array(accs), np.array(accs_top_20pc)
