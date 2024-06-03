import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from typing import List, Tuple, Union

SEED = 1337

np.random.seed(SEED)
tf.random.set_seed(SEED)


def get_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_table("../data/x_train.txt", sep=" ", header=None)
    y = pd.read_table("../data/y_train.txt", sep=" ", header=None, names=["label"])[
        "label"
    ]
    return df, y


def drop_colinear(df: pd.DataFrame, th: float) -> pd.DataFrame:
    cols = []
    while True:
        cor = df.corr().abs()
        cor = cor.map(lambda x: 0 if x == 1 else x)
        if cor.max().max() < th:
            break
        max_index = cor.stack().idxmax()
        max_row, max_col = max_index
        df = df.drop(columns=max_col)
        cols.append(max_col)
    return cols


def get_callback(patience: int) -> tf.keras.callbacks.Callback:
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=patience, mode="max", restore_best_weights=True
    )
    return early_stopping


def custom_metric(y_true: np.ndarray, y_pred: np.ndarray, num: int) -> int:
    idx = np.argsort(y_pred[:, 1])[-200:]  # 200 from 0.2 * 1000 (test size)
    corr = y_true[idx].sum()
    return corr * 5 * 10 - num * 200


def eval_model(y_true: np.ndarray, y_pred: np.ndarray, num: int) -> Tuple[float, int]:
    metric = custom_metric(y_true, y_pred, num)
    y_pred = np.argmax(y_pred, axis=1)
    acc = accuracy_score(y_true, y_pred)
    return acc, metric


def test_models(
    results: List[List[Union[int, str, float]]],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    num: int,
) -> List[List[Union[int, str, float]]]:
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input(shape=(num,)),
            tf.keras.layers.Dense(num, activation="relu"),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(2, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=["accuracy"],
    )
    model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=32,
        validation_data=(X_test, y_test),
        callbacks=get_callback(4),
        verbose=0,
    )
    y_pred = model.predict(X_test, verbose=0)
    acc, metric = eval_model(y_test, y_pred, num)
    results.append([num, "NN", acc, metric])

    model = LogisticRegression(random_state=SEED)
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)
    acc, metric = eval_model(y_test, y_pred, num)
    results.append([num, "LR", acc, metric])

    model = XGBClassifier(random_state=SEED)
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)
    acc, metric = eval_model(y_test, y_pred, num)
    results.append([num, "XGB", acc, metric])

    model = RandomForestClassifier(random_state=SEED)
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)
    acc, metric = eval_model(y_test, y_pred, num)
    results.append([num, "RF", acc, metric])

    model = SVC(probability=True, random_state=SEED)
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)
    acc, metric = eval_model(y_test, y_pred, num)
    results.append([num, "SVM", acc, metric])

    model = LinearDiscriminantAnalysis()
    model.fit(X_train, y_train)
    y_pred = model.predict_proba(X_test)
    acc, metric = eval_model(y_test, y_pred, num)
    results.append([num, "LDA", acc, metric])

    return results


def save_results(
    results: List[List[Union[int, str, float]]], path: str, method: str
) -> None:
    results = pd.DataFrame(
        results, columns=["num_features", "model", "accuracy", "metric"]
    )
    results["method"] = method
    results.sort_values("metric", ascending=False, inplace=True)
    results.to_csv(path, index=False)
