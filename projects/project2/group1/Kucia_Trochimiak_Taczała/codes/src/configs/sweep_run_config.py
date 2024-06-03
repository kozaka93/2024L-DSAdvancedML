from dataclasses import dataclass
from typing import Type

import model_definitions


@dataclass
class SweepRunConfig:
    model_type: Type[model_definitions.BaseModel]
    model_params: dict
    features: list[int]
    test_size: float = 0.2
    repeat: int = 5

    @classmethod
    def from_dict(cls, data: dict) -> "SweepRunConfig":
        """
        Create class instance from dictionary.
        """
        model_params = {key: data.pop(key) for key in data.pop("model_params", {})}
        model_type = getattr(model_definitions, data.pop("model_type"))

        return cls(
            model_params=model_params,
            model_type=model_type,
            **data,
        )
