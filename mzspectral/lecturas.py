"""Lecturas espectrales de un modelo (CPU, double): las que T22 uso para leer el efecto del optimizador.

(c) 2026 MSC Star Team (Esraderey, Raul Cruz Acosta). Licencia MIT (ver LICENSE).
"""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from .optim import es_factorizada


@torch.no_grad()
def rango_estable(W: torch.Tensor) -> float:
    """||W||_F^2 / ||W||_2^2: cuantas direcciones "cuentan" en W."""
    sv = torch.linalg.svdvals(W.detach().double())
    e = sv.pow(2)
    return float(e.sum() / e.max().clamp_min(1e-300))


@torch.no_grad()
def normas(m: nn.Module) -> Dict[str, dict]:
    """Por modulo (nombre): factorizados -> sigma_A = ||U·diag(S)||_2, sigma_B = ||diag(S)·V||_2,
    ||U||_2, ||V||_2, rango estable de W = U·diag(S)·V, ortogonalidad de U y V (rms fuera de la
    diagonal de las Gram normalizadas); nn.Linear -> ||W||_2, rango estable y Frobenius."""
    out = {}
    for nombre, mod in m.named_modules():
        if es_factorizada(mod):
            U, S, V = mod.U.detach().double(), mod.S.detach().double(), mod.V.detach().double()
            Un, Vn = U / U.norm(dim=0).clamp_min(1e-300), V / V.norm(dim=1).clamp_min(1e-300)[:, None]   # una columna nula no da NaN
            gU, gV = Un.T @ Un, Vn @ Vn.T
            r = gU.shape[0]
            off = lambda g, r=r: float((g - torch.eye(r, dtype=g.dtype, device=g.device)).pow(2).sum().div(max(1, r * (r - 1))).sqrt())
            out[nombre] = {"tipo": "factorizada", "sigma_A": float(torch.linalg.matrix_norm(U * S, ord=2)),
                           "sigma_B": float(torch.linalg.matrix_norm(S[:, None] * V, ord=2)),
                           "norma2_U": float(torch.linalg.matrix_norm(U, ord=2)), "norma2_V": float(torch.linalg.matrix_norm(V, ord=2)),
                           "rango_estable_W": rango_estable((U * S) @ V), "ortog_U": off(gU), "ortog_V": off(gV),
                           "S_abs": [float(S.abs().min()), float(S.abs().mean()), float(S.abs().max())]}
        elif isinstance(mod, nn.Linear):
            W = mod.weight.detach().double()
            out[nombre] = {"tipo": "densa", "norma2_W": float(torch.linalg.matrix_norm(W, ord=2)),
                           "rango_estable_W": rango_estable(W), "frobenius_W": float(W.norm())}
    return out
