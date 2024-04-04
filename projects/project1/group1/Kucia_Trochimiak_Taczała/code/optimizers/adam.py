import torch

from .base import BaseGrad


class ADAM(BaseGrad):
    def __init__(
        self,
        model,
        lr: float = 0.001,
        beta1: float = 0.999,
        beta2: float = 0.9,
        epsilon: float = 1e-8,
    ):
        super().__init__(model, lr)
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        self.first_moment = {"beta0": 0, "beta1": torch.zeros_like(self.model.beta1)}
        self.second_moment = {"beta0": 0, "beta1": torch.zeros_like(self.model.beta1)}

    def step(self, grad: torch.Tensor, param_name: str):
        self.first_moment[param_name] = (
            self.beta1 * self.first_moment[param_name] + (1 - self.beta1) * grad
        )
        self.second_moment[param_name] = self.beta2 * self.second_moment[param_name] + (
            1 - self.beta2
        ) * (grad**2)

        m_hat = self.first_moment[param_name] / (1 - self.beta1**self.step_count)
        v_hat = self.second_moment[param_name] / (1 - self.beta2**self.step_count)

        update = self.lr * m_hat / (torch.sqrt(v_hat) + self.epsilon)
        return update
