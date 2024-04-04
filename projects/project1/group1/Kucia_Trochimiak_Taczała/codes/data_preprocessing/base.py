from abc import ABC, abstractmethod
import torch
import os
import wandb
from dotenv import load_dotenv

# loading variables from .env file
load_dotenv()

PROJECT = os.getenv("PROJECT")


class PreprocessData(ABC):
    def __init__(self, data_dir: str, data_name: str, data_description: str):
        self.data_dir = data_dir
        assert os.path.isdir(data_dir), f"Data directory {data_dir} not found"
        self.data_name = data_name
        self.data_description = data_description

    @abstractmethod
    def load_and_transform(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Load the data from local drive and transform the data into
        two tensors: X (features) and y (labels)
        :return: tuple of X and y tensors
        """

    def upload_data(self):
        with wandb.init(project=PROJECT, job_type="load-data") as run:
            artifact = wandb.Artifact(
                self.data_name,
                type="dataset",
                description=self.data_description,
            )

            with artifact.new_file("data.pt", mode="wb") as file:
                # Save the tensors to a file
                torch.save(self.load_and_transform(), file)

            # Upload to W&B
            run.log_artifact(artifact)
