import json

import numpy as np
import pandas as pd
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier


def main(train_df_path: str = "data/preprocessed/train.csv") -> None:
    """Generate list of indicies recognized by the Buruta alogirthm
    as relevant in the task. Results are saved in the file

    Args:
        train_df_path (str, optional): Path to preprocessed data.
            Defaults to "data/preprocessed/train.csv".
    """

    df = pd.read_csv(train_df_path).values
    X, y = df[:, :-1], df[:, -1]

    rf = RandomForestClassifier(
        n_jobs=-1, class_weight="balanced", max_depth=5
    )

    feat_selector = BorutaPy(
        rf, n_estimators="auto", verbose=2, random_state=1
    )

    feat_selector.fit(X, y)

    best_features_idx = np.where(feat_selector.ranking_ == 1)[0].tolist()

    json_object = json.dumps({"boruta_features": best_features_idx}, indent=4)

    with open("data/features/boruta.json", "w") as outfile:
        outfile.write(json_object)


if __name__ == "__main__":
    main()
