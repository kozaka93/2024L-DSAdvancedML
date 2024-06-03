from enum import Enum


class JobType(Enum):
    UPLOAD_DATA = "upload-data"
    TRAINING = "training"


class ArtifactType(Enum):
    DATASET = "dataset"
    MODEL = "model"
