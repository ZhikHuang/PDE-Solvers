from __future__ import annotations

import math

import torch


def exact_solution(x: torch.Tensor) -> torch.Tensor:
    """Exact solution u(x, y) = sin(pi x) sin(pi y)."""
    return torch.sin(math.pi * x[:, 0:1]) * torch.sin(math.pi * x[:, 1:2])


def rhs(x: torch.Tensor) -> torch.Tensor:
    """Right-hand side f for -Delta u = f."""
    return 2.0 * math.pi**2 * exact_solution(x)


def boundary_value(x: torch.Tensor) -> torch.Tensor:
    """Dirichlet boundary value. It is zero for this manufactured solution."""
    return torch.zeros((x.shape[0], 1), device=x.device, dtype=x.dtype)


def relative_l2_error(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(pred - truth) / torch.linalg.norm(truth)
