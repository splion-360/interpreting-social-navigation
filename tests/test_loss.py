"""File description: Tests for Gaussian trajectory loss utilities."""

import torch

from loss import bivariate_gaussian_nll, gaussian_2d_parameters


def test_bivariate_gaussian_nll_is_finite_and_differentiable() -> None:
    outputs = torch.zeros((2, 4, 36, 5), requires_grad=True)
    targets = torch.ones((2, 4, 36, 2))

    loss = bivariate_gaussian_nll(outputs, targets)
    loss.backward()

    assert torch.isfinite(loss)
    assert outputs.grad is not None
    assert torch.isfinite(outputs.grad).all()


def test_gaussian_parameter_transform_keeps_sigma_and_rho_valid() -> None:
    outputs = torch.tensor([[[[0.0, 0.0, -100.0, 100.0, 20.0]]]])

    params = gaussian_2d_parameters(outputs)

    assert torch.all(params.sigma_x > 0)
    assert torch.all(params.sigma_y > 0)
    assert torch.all(params.rho < 1)
    assert torch.all(params.rho > -1)


def test_bivariate_gaussian_nll_applies_mask_before_reduction() -> None:
    outputs = torch.zeros((1, 2, 1, 5))
    targets = torch.tensor([[[[0.0, 0.0]], [[100.0, 100.0]]]])
    mask = torch.tensor([[[True], [False]]])

    masked = bivariate_gaussian_nll(outputs, targets, mask=mask)
    first_only = bivariate_gaussian_nll(outputs[:, :1], targets[:, :1])

    assert torch.allclose(masked, first_only)


def test_bivariate_gaussian_nll_preserves_shape_with_masked_none_reduction() -> None:
    outputs = torch.zeros((1, 2, 1, 5))
    targets = torch.zeros((1, 2, 1, 2))
    mask = torch.tensor([[[True], [False]]])

    loss = bivariate_gaussian_nll(outputs, targets, mask=mask, reduction="none")

    assert loss.shape == (1, 2, 1)
    assert loss[0, 0, 0] > 0
    assert loss[0, 1, 0] == 0
