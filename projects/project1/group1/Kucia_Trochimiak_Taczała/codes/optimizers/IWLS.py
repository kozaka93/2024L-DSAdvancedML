import torch
from .base import Base
import numpy as np
from numpy.linalg import inv
from abc import ABC, abstractmethod


class IWLS(Base):
    def calc_pi(self, X, beta):
        exp = torch.exp(torch.matmul(X, beta))
        return exp / (1 + exp)

    def log_likelihood(self, X, y, beta):
        logits = torch.matmul(X, beta)
        return -torch.sum(logits * y - torch.log(1 + torch.exp(logits)))

    def backprop(self, X: torch.Tensor, y: torch.Tensor, y_hat: torch.Tensor):
        super().backprop(
            X, y, y_hat
        )  # Increment step_count and perform any other base setup

        # Adjust dimensions for intercept term
        intercept = torch.ones((X.shape[0], 1), dtype=X.dtype, device=X.device)
        X_augmented = torch.hstack((intercept, X))

        # Calculate beta vector (intercept + weights)
        beta0 = self.model.beta0.view(1)

        # Ensure beta1 is a 1D tensor
        beta1 = self.model.beta1

        # Concatenate beta0 and beta1
        beta = torch.cat([beta0, beta1], 0)

        # Calculate pi using the calc_pi method
        pi = self.calc_pi(X_augmented, beta)

        # Iterative Weighted Least Squares (IWLS) update rule
        weights = pi * (1 - pi)
        weights = torch.clamp(weights, min=1e-10)
        W = torch.diag_embed(weights)

        eta = torch.matmul(X_augmented, beta)
        h_prime_eta = pi * (1 - pi)
        z = eta + (y - pi) / torch.clamp(h_prime_eta, min=1e-5)

        # Perform the IWLS update of beta
        try:
            XtWX = torch.matmul(X_augmented.T, torch.matmul(W, X_augmented))
            XtWz = torch.matmul(X_augmented.T, torch.matmul(W, z))
            XtWX_inv = torch.linalg.inv(
                XtWX + 1e-10 * torch.eye(X_augmented.shape[1], device=X.device)
            )
            beta_new = torch.matmul(XtWX_inv, XtWz)

            # Update model weights
            self.model.beta0 = torch.unsqueeze(beta_new[0], 0)
            self.model.beta1 = beta_new[1:]

        except RuntimeError as e:
            print("An error occurred during IWLS optimization:", e)
