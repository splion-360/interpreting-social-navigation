"""File description: Bivariate Gaussian trajectory loss utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class Gaussian2DParameters:
    """Constrained bivariate Gaussian parameters.

    Attributes:
        mu_x: X-coordinate mean.
        mu_y: Y-coordinate mean.
        sigma_x: Positive X-coordinate standard deviation.
        sigma_y: Positive Y-coordinate standard deviation.
        rho: Correlation coefficient in `(-1, 1)`.
    """

    mu_x: Tensor
    mu_y: Tensor
    sigma_x: Tensor
    sigma_y: Tensor
    rho: Tensor


def gaussian_2d_parameters(
    outputs: Tensor,
    min_sigma: float = 1e-3,
    max_abs_rho: float = 0.999,
) -> Gaussian2DParameters:
    """Transform raw network outputs into valid Gaussian parameters.

    Args:
        outputs: Tensor shaped `[..., 5]` containing `mu_x`, `mu_y`,
            raw `sigma_x`, raw `sigma_y`, and raw `rho`.
        min_sigma: Small positive lower bound added after softplus.
        max_abs_rho: Absolute bound applied after tanh to avoid singular covariance.

    Returns:
        Constrained bivariate Gaussian parameters.
    """

    mu_x, mu_y, raw_sigma_x, raw_sigma_y, raw_rho = outputs.unbind(dim=-1)
    return Gaussian2DParameters(
        mu_x=mu_x,
        mu_y=mu_y,
        sigma_x=F.softplus(raw_sigma_x) + min_sigma,
        sigma_y=F.softplus(raw_sigma_y) + min_sigma,
        rho=torch.clamp(torch.tanh(raw_rho), min=-max_abs_rho, max=max_abs_rho),
    )


def bivariate_gaussian_nll(
    outputs: Tensor,
    targets: Tensor,
    *,
    mask: Tensor | None = None,
    reduction: str = "mean",
    eps: float = 1e-6,
) -> Tensor:
    """Compute bivariate Gaussian negative log likelihood.

    Args:
        outputs: Raw Gaussian predictions shaped `[..., 5]`.
        targets: Target coordinates shaped `[..., 2]`.
        mask: Optional boolean tensor matching `outputs.shape[:-1]`.
        reduction: One of `mean`, `sum`, or `none`.
        eps: Numeric floor for `1 - rho^2`.

    Returns:
        Reduced negative log likelihood.
    """

    params = gaussian_2d_parameters(outputs)
    target_x, target_y = targets.unbind(dim=-1)

    norm_x = (target_x - params.mu_x) / params.sigma_x
    norm_y = (target_y - params.mu_y) / params.sigma_y
    one_minus_rho_sq = torch.clamp(1 - params.rho.square(), min=eps)
    z = norm_x.square() + norm_y.square() - 2 * params.rho * norm_x * norm_y
    nll = (
        torch.log(
            torch.tensor(2.0 * torch.pi, device=outputs.device, dtype=outputs.dtype)
        )
        + torch.log(params.sigma_x)
        + torch.log(params.sigma_y)
        + 0.5 * torch.log(one_minus_rho_sq)
        + z / (2 * one_minus_rho_sq)
    )

    if mask is not None:
        if reduction == "none":
            return torch.where(mask, nll, torch.zeros_like(nll))
        nll = nll[mask]

    if reduction == "mean":
        return nll.mean()
    if reduction == "sum":
        return nll.sum()
    if reduction == "none":
        return nll
    raise ValueError(f"unknown reduction: {reduction}")


def bivariate_gaussian_horizon_nll(
    outputs: Tensor,
    targets: Tensor,
    *,
    observation_length: int,
    mask: Tensor | None = None,
) -> Tensor:
    """Compute Gaussian NLL only over the prediction horizon.

    Args:
        outputs: Raw Gaussian predictions for shifted targets shaped
            `[time - 1, nodes, 5]`.
        targets: Shifted target coordinates shaped `[time - 1, nodes, 2]`.
        observation_length: Number of observed frames before prediction begins.
        mask: Optional boolean mask shaped `[time - 1, nodes]`.

    Returns:
        Mean negative log likelihood over prediction targets.
    """

    horizon_start = observation_length - 1
    horizon_mask = mask[horizon_start:] if mask is not None else None
    return bivariate_gaussian_nll(
        outputs[horizon_start:],
        targets[horizon_start:],
        mask=horizon_mask,
    )
