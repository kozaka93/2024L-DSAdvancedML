from typing import Type

from lightning.pytorch.loggers import WandbLogger
import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor

from dataset.training import TrainingDataset
from models.neural_network import FCModel
from models.lightning import LightningModel
from models.kan import KANModel

from run_config import RunConfig


def prepare_session(
    config: RunConfig,
    wandb_logger: WandbLogger,
) -> tuple[pl.Trainer, LightningModel, TrainingDataset]:

    data = TrainingDataset(
        wandb_logger,
        config.batch_size,
        config.features,
        config.test_size,
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    trainer = pl.Trainer(
        logger=wandb_logger,
        log_every_n_steps=10,
        max_epochs=config.epochs,
        callbacks=[lr_monitor]
    )

    model = FCModel(**config.model_params)
    pl_model = LightningModel(
        model,
        config.model_name,
        len(config.features),
        config.optimizer,
        config.optimizer_params,
        config.scheduler_factor,
        config.scheduler_patience,
    )

    return trainer, pl_model, data
