import torch

from .base import BaseGrad


class SGD(BaseGrad):
    def step(self, grad: torch.Tensor, _param_name: str) -> torch.Tensor:
        return self.lr * grad
