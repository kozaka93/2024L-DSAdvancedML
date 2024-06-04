from dataclasses import dataclass
from typing import Type

import torch


@dataclass
class RunConfig:
    batch_size: int
    features: list[int]
    epochs: int
    model_params: dict
    model_name: str
    optimizer: Type[torch.optim.Optimizer]
    optimizer_params: dict
    scheduler_factor: float | None = None
    scheduler_patience: int | None = None
    test_size: float = 0.2

    @classmethod
    def from_dict(cls, data: dict) -> "RunConfig":
        """
        Create class instance from dictionary.
        """
        model_params = {
            key: data.pop(key)
            for key in data.pop("model_params")
        }
        model_params["input_size"] = len(data["features"])

        optimizer = getattr(torch.optim, data.pop("optimizer"))
        optimizer_params = {
            key: data.pop(key)
            for key in data.pop("optimizer_params")
        }

        if "beta1" in optimizer_params and "beta2" in optimizer_params:
            optimizer_params["betas"] = (
                optimizer_params.pop("beta1"),
                optimizer_params.pop("beta2"),
            )

        return cls(
            model_params=model_params,
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            **data,
        )
