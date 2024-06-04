from itertools import combinations

import torch
from torch import nn


class FCModel(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_layer_sizes: list[int],
        activation: str,
        dropouts: list[float],
        interactions: bool = False,
    ):
        """
        Basic fully connected neural network model with dropout.
        """
        if (
            len(hidden_layer_sizes) != len(dropouts)
            or len(hidden_layer_sizes) == 0
            or len(dropouts) == 0
        ):
            raise ValueError(
                "hidden_layer_sizes and dropouts must have the same length > 0"
            )

        self.feature_combinations = []
        if interactions:
            self.feature_combinations = list(combinations(range(input_size), 2))

        super().__init__()
        self.activation = getattr(nn, activation)
        self.fc = nn.Sequential(
            self.linear_module(input_size + len(self.feature_combinations), hidden_layer_sizes[0], dropouts[0]),
            *[
                self.linear_module(
                    hidden_layer_sizes[i],
                    hidden_layer_sizes[i + 1],
                    dropouts[i + 1],
                )
                for i in range(len(hidden_layer_sizes) - 1)
            ],
            nn.Linear(hidden_layer_sizes[-1], 1),
            nn.Sigmoid(),
        )

    def linear_module(
        self,
        in_features: int,
        out_features: int,
        dropout: float,
    ):
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            self.activation(),
            nn.Dropout(dropout),
        )

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        if len(self.feature_combinations) > 0:
            interactions = torch.cat(
                [
                    src[:, i].unsqueeze(1) * src[:, j].unsqueeze(1)
                    for i, j in self.feature_combinations
                ],
                dim=1,
            )
            src = torch.cat([src, interactions], dim=1)
        return self.fc(src)
