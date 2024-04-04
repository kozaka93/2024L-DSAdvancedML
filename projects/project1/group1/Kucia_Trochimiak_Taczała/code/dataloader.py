from torch.utils.data import TensorDataset, DataLoader
import os
import torch
from wandb.sdk.wandb_config import Config as WandbConfig
import wandb

from create_interactons import create_interactions


class DataloaderModule:
    train_dataset: TensorDataset
    test_dataset: TensorDataset

    def __init__(
        self,
        logger: wandb.sdk.wandb_run.Run,
        data_name: str,
        config: WandbConfig,
    ):
        super().__init__()
        self.logger = logger
        self.config = config
        self.data_name = data_name
        self.num_features: int | None = None

    @property
    def num_workers(self) -> int:
        return getattr(self.config, "num_workers", 4)

    def prepare_data(self):
        data_artifact = self.logger.use_artifact(f"{self.data_name}:latest")
        data_dir = data_artifact.download()
        X, y = torch.load(os.path.join(data_dir, f"data.pt"))
        areInteractions = getattr(self.config, "interactions", False)
        if areInteractions:
            X = create_interactions(X)

        self.num_features = X.shape[1]
        dataset = TensorDataset(X, y)

        dataset_size = len(dataset)
        test_size = int(dataset_size * self.config.test_split)
        train_size = dataset_size - test_size
        self.train_dataset, self.test_dataset = torch.utils.data.random_split(
            dataset, [train_size, test_size]
        )

    def train_dataloader(self):
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )
        return train_dataloader

    def test_dataloader(self):
        test_dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.num_workers,
        )
        return test_dataloader
