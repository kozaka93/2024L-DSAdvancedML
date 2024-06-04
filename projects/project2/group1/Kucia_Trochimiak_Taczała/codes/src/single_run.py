import argparse
import os
import yaml
import wandb

from project_config import config, JobType
from experiment import run_agent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a wandb experiment using a YAML configuration file."
    )

    # Add an argument for the YAML file path
    parser.add_argument(
        "yaml_file",
        type=str,
        help="Path to the YAML configuration file for the experiment "
        "(assume that parent directory is src/configs/single_runs).",
    )

    # Parse the arguments
    args = parser.parse_args()

    with open(os.path.join("configs", "single_runs", f"{args.yaml_file}.yaml"), "r") as file:
        experiment_config = yaml.safe_load(file)

    with wandb.init(
        project=config.project,
        entity=config.entity,
        job_type=JobType.TRAINING.value,
        config=experiment_config,
    ) as run:
        run_agent()
