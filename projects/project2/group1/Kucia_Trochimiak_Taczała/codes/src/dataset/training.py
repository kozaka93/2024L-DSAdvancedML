from torch.utils.data import TensorDataset, DataLoader
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
import torch
import os
import sys

from .utils import download_data


class TrainingDataset(pl.LightningDataModule):
    def __init__(
        self,
        wandb_logger: WandbLogger,
        batch_size: int,
        features: list[int],
        test_size: float = 0.2,
        random_seed: int = 42,
    ):
        super().__init__()
        self.logger = wandb_logger
        self.batch_size = batch_size
        self.features = features
        self.test_size = test_size
        self.random_seed = random_seed

        self.train: TensorDataset | None = None
        self.test: TensorDataset | None = None

    @property
    def data_loader_kwargs(self) -> dict:
        data = {}
        # if sys.platform in ["linux", "darwin"]:
        # data["num_workers"] = min(
        # len(os.sched_getaffinity(0)), 8
        # )  # num of cpu cores
        return data

    def prepare_data(self) -> None:
        x, y = download_data(self.logger)
        x = x[:, self.features]
        dataset = TensorDataset(
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.float32),
        )
        train_split = 1 - self.test_size

        self.train, self.test = torch.utils.data.random_split(
            dataset,
            [train_split, self.test_size],
            generator=torch.Generator().manual_seed(self.random_seed),
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            **self.data_loader_kwargs,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            **self.data_loader_kwargs,
        )
