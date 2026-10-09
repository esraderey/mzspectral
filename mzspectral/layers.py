"""Capa factorizada de referencia W = U·diag(S)·V, sin dependencias fuera de PyTorch.

(c) 2026 MSC Star Team (Esraderey, Raul Cruz Acosta). Licencia MIT (ver LICENSE).

Es la forma de tres factores de MZTrain (ZFactorizedLinear, Invencion 1 de su PRIOR_ART.md) en su
version minima: U (out x r), S (r), V (r x in) como parametros de primer orden; forward eficiente
((x·V^T)·S)·U^T sin materializar W; init por SVD truncada de una matriz Kaiming-uniforme con escala
de energia (EPSI: los S retenidos se escalan por ||W||_F/||W_r||_F). Sirve para usar ZSpectron3 sin
MZTrain; con MZTrain, ZSpectron3 acepta directamente ZFactorizedLinear (mismos nombres U, S, V).
"""
from __future__ import annotations

import math

import torch
from torch import nn


class FactorizedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = False, epsi: bool = True):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        if not isinstance(rank, int) or rank < 1 or in_features < 1 or out_features < 1:
            raise ValueError(f"rank, in_features y out_features deben ser >= 1 (rank={rank!r})")
        self.rank = min(rank, in_features, out_features)
        self.U = nn.Parameter(torch.empty(out_features, self.rank))
        self.S = nn.Parameter(torch.empty(self.rank))
        self.V = nn.Parameter(torch.empty(self.rank, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.epsi = epsi
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        W = torch.empty(self.out_features, self.in_features)
        nn.init.kaiming_uniform_(W, a=math.sqrt(5))           # el init de nn.Linear
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        r = self.rank
        alpha = (W.norm() / S[:r].norm()) if self.epsi else torch.tensor(1.0)
        self.U.copy_(U[:, :r])
        self.S.copy_(S[:r] * alpha)
        self.V.copy_(Vh[:r])

    def weight(self) -> torch.Tensor:
        """W = U·diag(S)·V (solo para lecturas; el forward no lo materializa)."""
        return (self.U * self.S) @ self.V

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = ((x @ self.V.T) * self.S) @ self.U.T
        return y if self.bias is None else y + self.bias

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}, bias={self.bias is not None}"
