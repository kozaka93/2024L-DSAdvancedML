import argparse

import numpy as np
from sklearn.model_selection import train_test_split
from lightning.pytorch.loggers import WandbLogger
import wandb

from model_definitions import BaseModel
from project_config import config
from configs.sweep_run_config import SweepRunConfig
from dataset.utils import download_data


def perform_experiment(
    model: BaseModel, x: np.ndarray, y: np.ndarray, test_size: float = 0.2, repeat: int = 5
) -> tuple[float, float, float, float]:
    """
    Perform an experiment with the given model and data.
    """
    results = []

    for _ in range(repeat):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
        model.fit(x_train, y_train)
        results.append(model.calculate_gain(x_test, y_test))

    return np.mean(results), np.std(results), np.max(results), np.min(results)


def run_agent():
    wandb_logger = WandbLogger(project=config.project, entity=config.entity)
    sweep_config = SweepRunConfig.from_dict(wandb_logger.experiment.config.as_dict())

    x, y = download_data(wandb_logger)
    x_selected_features = x[:, sweep_config.features]
    model = sweep_config.model_type(**sweep_config.model_params)
    mean, std, max_gain, min_gain = perform_experiment(
        model=model,
        x=x_selected_features,
        y=y,
        test_size=sweep_config.test_size,
        repeat=sweep_config.repeat,
    )

    wandb_logger.experiment.log(
        {
            "gain_mean": mean,
            "gain_std": std,
            "gain_max": max_gain,
            "gain_min": min_gain,
        }
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run a wandb agent to execute an experiment."
    )

    parser.add_argument(
        "sweep_id",
        type=str,
        help="Sweep ID provided by sweep.py",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="Number of experiments to perform. Can be None to run indefinitely."
    )

    # Parse the arguments
    args = parser.parse_args()

    wandb.agent(
        args.sweep_id, run_agent, count=args.count, project=config.project,
        entity=config.entity
    )
