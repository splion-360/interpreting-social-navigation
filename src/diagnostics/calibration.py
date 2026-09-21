"""File description: Horizon-wise calibration measurements for Gaussian forecasts."""

import numpy as np
import torch

from loss import gaussian_2d_parameters


def calibration_profile(
    gaussian_outputs: np.ndarray,
    target: np.ndarray,
    *,
    coordinate_scale: np.ndarray,
) -> list[dict[str, float | int]]:
    """Measure Gaussian calibration independently at each future step.

    Args:
        gaussian_outputs: Raw parameters shaped `[time, nodes, 5]`.
        target: Normalized target positions shaped `[time, nodes, 2]`.
        coordinate_scale: Pixel scale shaped `[2]`.

    Returns:
        Coverage, Mahalanobis distance, and pixel sigma for every future step.
    """

    params = gaussian_2d_parameters(
        torch.from_numpy(gaussian_outputs.astype(np.float32))
    )
    target_tensor = torch.from_numpy(target.astype(np.float32))
    dx = target_tensor[..., 0] - params.mu_x
    dy = target_tensor[..., 1] - params.mu_y
    one_minus_rho_sq = torch.clamp(1.0 - params.rho.square(), min=1e-6)
    mahalanobis_sq = (
        (dx / params.sigma_x).square()
        + (dy / params.sigma_y).square()
        - 2.0 * params.rho * dx * dy / (params.sigma_x * params.sigma_y)
    ) / one_minus_rho_sq
    scale = torch.from_numpy(coordinate_scale.astype(np.float32))
    profile = []
    for step in range(gaussian_outputs.shape[0]):
        step_distance = mahalanobis_sq[step]
        profile.append(
            {
                "horizon_step": step + 1,
                "coverage_50": float((step_distance <= 1.38629436).float().mean()),
                "coverage_95": float((step_distance <= 5.99146455).float().mean()),
                "mahalanobis_sq": float(step_distance.mean()),
                "sigma_x_px": float((params.sigma_x[step] * scale[0]).mean()),
                "sigma_y_px": float((params.sigma_y[step] * scale[1]).mean()),
            }
        )
    return profile
