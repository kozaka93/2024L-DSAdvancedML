from lightning.pytorch.loggers import WandbLogger
from .prepare_session import prepare_session
from run_config import RunConfig


def train(config: dict, wandb_logger: WandbLogger) -> None:
    trainer, pl_model, data = prepare_session(RunConfig.from_dict(config), wandb_logger)
    data.prepare_data()
    trainer.fit(pl_model, data)
    pl_model.load_best_model()
    trainer.test(pl_model, data)
