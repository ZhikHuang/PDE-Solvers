from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    """Small tanh MLP for approximating the PDE solution."""

    def __init__(self, in_dim: int = 2, hidden: int = 64, layers: int = 4) -> None:
        super().__init__()
        if layers < 2:
            raise ValueError("layers must be at least 2")

        modules: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(layers - 2):
            modules.extend([nn.Linear(hidden, hidden), nn.Tanh()])
        modules.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*modules)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
