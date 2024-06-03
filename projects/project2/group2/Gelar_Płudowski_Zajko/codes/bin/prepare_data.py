from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main(
    raw_data_path: Path = Path("data/raw"),
    preprocessed_data_path: Path = Path("data/preprocessed"),
    seed: int = 123,
) -> None:
    """Prepare data for the experiments. Include train-test-val split and column renaming.

    Args:
        raw_data_path (Path, optional): Path to raw data. Defaults to Path("data/raw").
        preprocessed_data_path (Path, optional): Path to save preprocessed data. Defaults to Path("data/preprocessed").
        seed (int, optional): Random seed to perform reproducible split. Defaults to 123.
    """
    preprocessed_data_path = Path(preprocessed_data_path)
    if not preprocessed_data_path.exists():
        preprocessed_data_path.mkdir(parents=True, exist_ok=True)

    X_train = pd.read_csv(raw_data_path / "x_train.txt", sep=" ", header=None)
    y_train = pd.read_csv(raw_data_path / "y_train.txt", sep=" ", header=None)
    X_test = pd.read_csv(raw_data_path / "x_test.txt", sep=" ", header=None)

    y_train.columns = pd.Index(["target"])

    df = X_train.join(y_train)

    df_train, df_val = train_test_split(
        df, test_size=0.3, random_state=seed, stratify=y_train
    )

    df_val, df_test = train_test_split(
        df_val,
        test_size=1 / 3,
        random_state=seed,
        stratify=df_val.iloc[:, -1],
    )

    df_train, df_val, df_test = (
        pd.DataFrame(df_train),
        pd.DataFrame(df_val),
        pd.DataFrame(df_test),
    )

    df_train.columns = [
        *list(map(lambda x: f"col_{x}", df_train.columns[:-1])),
        "target",
    ]
    df_val.columns = [
        *list(map(lambda x: f"col_{x}", df_val.columns[:-1])),
        "target",
    ]
    df_test.columns = [
        *list(map(lambda x: f"col_{x}", df_test.columns[:-1])),
        "target",
    ]
    X_test.columns = pd.Index(list(map(lambda x: f"col_{x}", X_test.columns)))

    df_train.to_csv(preprocessed_data_path / "train.csv", index=False)
    df_val.to_csv(preprocessed_data_path / "valid.csv", index=False)
    df_test.to_csv(preprocessed_data_path / "test.csv", index=False)
    X_test.to_csv(preprocessed_data_path / "test_final.csv", index=False)


if __name__ == "__main__":
    main()
