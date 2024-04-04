import torch
import torch.nn.functional as F


class LogisticRegression:
    def __init__(self, num_features: int):
        self.num_features = num_features

        self.beta0 = torch.zeros(1)
        self.beta1 = torch.zeros(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The bias term (self.beta0) will automatically be broadcasted correctly.
        return torch.sigmoid(torch.matmul(x, self.beta1) + self.beta0)

    def loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Clamp y_hat to ensure values are within [0, 1]
        y_hat_clamped = torch.clamp(y_hat, 0, 1)
        y = torch.clamp(y.float(), 0, 1)

        # Ensure y is a float tensor and has the same shape as y_hat_clamped
        y = y.float().view_as(y_hat_clamped)

        return F.binary_cross_entropy(y_hat_clamped, y)
