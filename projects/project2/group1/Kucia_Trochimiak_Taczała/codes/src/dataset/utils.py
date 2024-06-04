import os
from glob import glob

from lightning.pytorch.loggers import WandbLogger
import numpy as np

from project_config import config


def download_data(wandb_logger: WandbLogger) -> tuple[np.ndarray, np.ndarray]:
    data_artifact = wandb_logger.use_artifact(f'{config.dataset_artifact_name}:latest')
    data_dir = data_artifact.download()
    x = np.load(os.path.join(data_dir, 'x_train.npy'))
    y = np.load(os.path.join(data_dir, 'y_train.npy'))
    return x, y


def get_cached_data(artifacts_path: str) -> tuple[np.ndarray, np.ndarray]:
    folders = sorted(glob(
        os.path.join(artifacts_path, f'{config.dataset_artifact_name}*')
    ))[::-1]
    if len(folders) == 0:
        raise FileNotFoundError('No cached data found.')

    x = np.load(os.path.join(folders[0], 'x_train.npy'))
    y = np.load(os.path.join(folders[0], 'y_train.npy'))
    return x, y


def get_cached_data_with_test(artifacts_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    folders = sorted(glob(
        os.path.join(artifacts_path, f'{config.dataset_artifact_name}*')
    ))[::-1]
    if len(folders) == 0:
        raise FileNotFoundError('No cached data found.')

    x = np.load(os.path.join(folders[0], 'x_train.npy'))
    y = np.load(os.path.join(folders[0], 'y_train.npy'))
    x_test = np.load(os.path.join(folders[0], 'x_test.npy'))
    return x, y, x_test
