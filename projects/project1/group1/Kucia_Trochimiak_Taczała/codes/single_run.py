import wandb
import yaml
import argparse
import os
from dotenv import load_dotenv

from trainer import Trainer
from model import LogisticRegression
from dataloader import DataloaderModule
import optimizers

# loading variables from .env file
load_dotenv()

PROJECT = os.getenv("PROJECT")
ENTITY = os.getenv("ENTITY")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a wandb experiment using a YAML configuration file."
    )

    # Add an argument for the YAML file path
    parser.add_argument(
        "yaml_file",
        type=str,
        help="Path to the YAML configuration file for the experiment "
        "(assume that parent directory is configs_experiment).",
    )

    # Parse the arguments
    args = parser.parse_args()

    with open(
        os.path.join("configs_experiment", f"{args.yaml_file}.yaml"), "r"
    ) as file:
        experiment_config = yaml.safe_load(file)

    with wandb.init(project=PROJECT, entity=ENTITY, config=experiment_config) as run:
        config = wandb.config
        dataloader = DataloaderModule(run, config.data_name, config)
        dataloader.prepare_data()
        model = LogisticRegression(dataloader.num_features)
        optimizer = getattr(optimizers, config.optimizer)(model)

        trainer = Trainer(model, dataloader, optimizer, log_wandb=True)
        trainer.train(config.epochs)
        trainer.test()
        trainer.evaluate_models(config.optimizer)
