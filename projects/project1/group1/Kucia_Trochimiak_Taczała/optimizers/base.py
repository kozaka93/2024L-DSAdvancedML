from abc import ABC, abstractmethod

import torch


class Base(ABC):
    def __init__(self, model):
        self.model = model
        self.step_count = 0

    @staticmethod
    def log_likelihood(X, y, beta):
        logits = torch.matmul(X, beta)
        return -torch.sum(logits * y - torch.log(1 + torch.exp(logits)))

    @abstractmethod
    def backprop(self, X: torch.Tensor, y: torch.Tensor, y_hat: torch.Tensor):
        """
        Perform backpropagation - update weights of the model.
        When you overwrite this method, you should call super().backprop at the beginning.
        :param X: input features
        :param y: true output
        :param y_hat: predicted output
        """
        self.step_count += 1


class BaseGrad(Base, ABC):
    def __init__(self, model, lr: float = 0.001):
        super().__init__(model)
        self.lr = lr

    def beta_1_grad(
        self, X: torch.Tensor, y: torch.Tensor, y_hat: torch.Tensor
    ) -> torch.Tensor:
        return torch.mean(X.T * (y - y_hat), dim=1)

    def beta_0_grad(self, y: torch.Tensor, y_hat: torch.Tensor) -> torch.Tensor:
        return torch.mean(y - y_hat)

    @abstractmethod
    def step(self, grad: torch.Tensor, param_name: str) -> torch.Tensor:
        """
        Give a parameter update for one training step.
        :param grad: gradient of the parameter
        :param param_name: name of the parameter
        :return: weights update delta
        """

    def backprop(self, X: torch.Tensor, y: torch.Tensor, y_hat: torch.Tensor):
        super().backprop(X, y, y_hat)
        self.model.beta0 += self.step(self.beta_0_grad(y, y_hat), "beta0")
        self.model.beta1 += self.step(self.beta_1_grad(X, y, y_hat), "beta1")
