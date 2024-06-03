import os

import pandas as pd
import numpy as np
import wandb

from project_config import config, ArtifactType, JobType


def upload_data() -> None:
    """
    Uploads the data to the specified Weights & Biases project.
    """
    # Load the data
    x_train_df = pd.read_csv(
        os.path.join(config.data_folder, "x_train.txt"),
        header=None,
        sep=r"\s+",
    )
    y_train_df = pd.read_csv(
        os.path.join(config.data_folder, "y_train.txt"),
        header=None,
        sep=r"\s+",
    )
    x_test_df = pd.read_csv(
        os.path.join(config.data_folder, "x_test.txt"),
        header=None,
        sep=r"\s+",
    )

    # Upload the data
    artifact = wandb.Artifact(
        name=config.dataset_artifact_name,
        type=ArtifactType.DATASET.value,
    )

    # save as npy file to artifact
    with artifact.new_file("x_train.npy", mode="wb") as file:
        np.save(file, x_train_df.values)

    with artifact.new_file("y_train.npy", mode="wb") as file:
        np.save(file, y_train_df.values.reshape(-1))

    with artifact.new_file("x_test.npy", mode="wb") as file:
        np.save(file, x_test_df.values)

    # upload to W&B
    with wandb.init(
        project=config.project,
        entity=config.entity,
        job_type=JobType.UPLOAD_DATA.value,
    ) as run:
        run.log_artifact(artifact)


if __name__ == "__main__":
    upload_data()
