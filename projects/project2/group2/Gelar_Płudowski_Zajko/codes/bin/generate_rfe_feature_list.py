import json
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import ElasticNet
from tqdm import tqdm

MODELS: Dict[str, Any] = {
    "random_forest": RandomForestClassifier(n_jobs=5, random_state=1),
    "elastic_net": ElasticNet(alpha=0, random_state=1, max_iter=10_000),
}


def load_boruta_features(path: str) -> np.ndarray:
    with open(path, "r") as f:
        features = json.load(f)

    return features["boruta_features"]


def main() -> None:
    """
    Generate order of impotance of the Boruta features. The order is obtained by using
    RFE alogrithm with models selected in variable `MODELS`.
    """
    df = pd.read_csv("data/preprocessed/train.csv")
    boruta_features = load_boruta_features("data/features/boruta.json")

    X, y = df.iloc[:, :-1], df.iloc[:, -1]

    X_filtered = X.iloc[:, boruta_features]

    results_ranking = {}

    for model_name, model in (p_bar := tqdm(MODELS.items())):
        p_bar.set_description(model_name)

        selector = RFE(estimator=model, n_features_to_select=1)
        selector = selector.fit(X_filtered, y)
        results_ranking[model_name] = selector.ranking_.tolist()

    json_object = json.dumps(results_ranking, indent=4)

    with open("data/features/rfe_ranking.json", "w") as outfile:
        outfile.write(json_object)


if __name__ == "__main__":
    main()
