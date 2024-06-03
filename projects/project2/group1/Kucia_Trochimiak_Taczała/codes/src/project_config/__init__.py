import os

from .data_class import Config
from .enums import ArtifactType, JobType


config = Config.from_yaml(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
)
