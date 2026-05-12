from __future__ import annotations

import torch


def sample_interior(n: int, device: torch.device) -> torch.Tensor:
    """Uniform samples from the unit square interior."""
    return torch.rand((n, 2), device=device)


def sample_boundary(n: int, device: torch.device) -> torch.Tensor:
    """Uniform samples from the four sides of the unit square."""
    counts = [n // 4] * 4
    for i in range(n % 4):
        counts[i] += 1

    samples = []

    y = torch.rand((counts[0], 1), device=device)
    samples.append(torch.cat([torch.zeros_like(y), y], dim=1))

    y = torch.rand((counts[1], 1), device=device)
    samples.append(torch.cat([torch.ones_like(y), y], dim=1))

    x = torch.rand((counts[2], 1), device=device)
    samples.append(torch.cat([x, torch.zeros_like(x)], dim=1))

    x = torch.rand((counts[3], 1), device=device)
    samples.append(torch.cat([x, torch.ones_like(x)], dim=1))

    return torch.cat(samples, dim=0)


def make_grid(n: int, device: torch.device) -> torch.Tensor:
    """Create an n by n grid on [0, 1]^2."""
    xs = torch.linspace(0.0, 1.0, n, device=device)
    yy, xx = torch.meshgrid(xs, xs, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
