import numpy as np
import torch
import torch.nn as nn


def r2_weighted(y_true: np.array, y_pred: np.array, sample_weight: np.array) -> float:
    r2 = 1 - np.average((y_pred - y_true) ** 2, weights=sample_weight) / (
        np.average((y_true) ** 2, weights=sample_weight) + 1e-38
    )
    return r2


def r2_weighted_torch(
    y_true: torch.Tensor, y_pred: torch.Tensor, sample_weights: torch.Tensor
) -> torch.Tensor:
    numerator = torch.sum(sample_weights * (y_pred - y_true) ** 2)
    denominator = torch.sum(sample_weights * (y_true) ** 2) + 1e-38
    r2 = 1 - numerator / denominator
    return r2


class WeightedR2Loss(nn.Module):
    def __init__(self, epsilon: float = 1e-38) -> None:
        super(WeightedR2Loss, self).__init__()
        self.epsilon = epsilon

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:

        numerator = torch.sum(weights * (y_pred - y_true) ** 2)
        denominator = torch.sum(weights * (y_true) ** 2) + 1e-38
        loss = numerator / denominator
        return loss
