"""Optimizadores espectrales: ZMuon para matrices densas y ZSpectron3 para factores U·diag(S)·V.

(c) 2026 MSC Star Team (Esraderey, Raul Cruz Acosta). Licencia MIT (ver LICENSE).

Mecanica comun a los dos (Jordan et al. 2024, "Muon"; Liu et al. 2025, "Muon is scalable",
arXiv 2502.16982):
- momento con Nesterov: buf <- mu*buf + g; u = g + mu*buf (mu = 0.95);
- ortogonalizacion de u con 5 iteraciones de Newton-Schulz (coeficientes 3.4445, -4.7750, 2.0315;
  normalizacion de Frobenius con eps 1e-7; traspuesta si hay mas filas que columnas), en fp32;
- escala RMS: la actualizacion ortogonalizada se multiplica por 0.2*sqrt(max(m, n)), con lo que su
  RMS por entrada es ~0.2*lr, la de AdamW: lr y weight decay se expresan en unidades de AdamW;
- weight decay desacoplado: W <- W*(1 - lr*wd).

ZSpectron3 (Spectron, arXiv 2602.12429, Algoritmo 1, escrito para un producto W = A·B, llevado a
tres factores): a primer orden dW = dU·diag(S)·V + U·diag(S)·dV, asi que las normas que acotan la
actualizacion del producto son ||diag(S)·V||_2 para el paso de U y ||U·diag(S)||_2 para el de V.
Se definen sigma_A := ||U·diag(S)||_2 y sigma_B := ||diag(S)·V||_2 (iteracion de potencia con
arranque caliente, leidas ANTES del paso) y U, V reciben el momento ortogonalizado por
1/(sigma_A + sigma_B + 1). S (1-D) no es de este optimizador: va en AdamW, como el embedding, las
normas y los sesgos (regla de Muon: lo que no es matriz, a AdamW). El weight decay no lleva el
factor 1/(sigma_A + sigma_B + 1) (forma publicada).

Evidencia (T22 = E9 de RFC-CALIDAD-1, MZTrain; ver EVIDENCIA.md): sobre la linea base sana de
T20, a 18 000 pasos y tres semillas, ZMuon mejora al denso 0.066 BPC y ZSpectron3 al factorizado
0.063 BPC frente a AdamW, con la mitad de estado de optimizador y 0 inestabilidades hasta LR 9.6e-3.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence

import torch
from torch import nn
from torch.optim import Optimizer

MOMENTO, NESTEROV, RMS = 0.95, True, 0.2
WD, BETAS, EPS = 0.01, (0.9, 0.95), 1e-8          # el AdamW "sano" de MZTrain (T20)
K_NS, EPS_NS, NS_ABC = 5, 1e-7, (3.4445, -4.7750, 2.0315)
K_POT_INIT, K_POT = 10, 1                         # iteraciones de potencia: al arrancar y por paso
SEMILLA_POT = 3000                                # generador propio de los vectores de potencia: 3000 + semilla


@torch.no_grad()
def newton_schulz5(G: torch.Tensor, steps: int = K_NS, eps: float = EPS_NS) -> torch.Tensor:
    """Ortogonalizacion aproximada de G (Jordan et al.): X con valores singulares ~1 (0.7-1.2)."""
    a, b, c = NS_ABC
    X = G.float()
    X = X / (X.norm() + eps)
    alto = X.size(0) > X.size(1)
    if alto:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if alto:
        X = X.T
    return X


@torch.no_grad()
def potencia(W: torch.Tensor, u: torch.Tensor, k: int):
    """k >= 1 iteraciones de potencia desde u (unitario, dim W.shape[0]); (sigma, u) con
    sigma = u^T W v ~ ||W||_2 (estimacion por debajo), tensor 0-d en el dispositivo de W."""
    v = None
    for _ in range(max(1, k)):
        v = W.T @ u
        v = v / (v.norm() + 1e-30)
        u = W @ v
        u = u / (u.norm() + 1e-30)
    return u @ (W @ v), u


def escala_rms(p: torch.Tensor, rms: float) -> float:
    return rms * math.sqrt(max(p.shape))


def _momento_ortogonal(p: torch.Tensor, st: dict, g: dict) -> torch.Tensor:
    if "buf" not in st:
        st["buf"] = torch.zeros_like(p)
    buf = st["buf"]
    buf.mul_(g["momentum"]).add_(p.grad)
    u = p.grad.add(buf, alpha=g["momentum"]) if g["nesterov"] else buf
    return newton_schulz5(u)


class ZMuon(Optimizer):
    """Muon para matrices 2-D: W <- W*(1 - lr*wd) - lr*rms*sqrt(max(m,n))*NS5(momento Nesterov)."""

    def __init__(self, params: Iterable[torch.Tensor], lr: float, momentum: float = MOMENTO, nesterov: bool = NESTEROV,
                 weight_decay: float = WD, rms: float = RMS):
        params = list(params)
        if not params or any(p.dim() != 2 for p in params):
            raise ValueError("ZMuon solo recibe matrices 2-D (al menos una)")
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay, rms=rms))

    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                O = _momento_ortogonal(p, self.state[p], g)
                p.mul_(1.0 - g["lr"] * g["weight_decay"]).add_(O, alpha=-g["lr"] * escala_rms(p, g["rms"]))


def es_factorizada(mod: nn.Module) -> bool:
    """Un modulo con parametros U (out x r), S (r) y V (r x in): la forma de tres factores."""
    U, S, V = (getattr(mod, a, None) for a in ("U", "S", "V"))
    return (all(isinstance(t, nn.Parameter) for t in (U, S, V)) and U.dim() == 2 and S.dim() == 1 and V.dim() == 2
            and U.shape[1] == S.shape[0] == V.shape[0])


class ZSpectron3(Optimizer):
    """Spectron adaptado a U·diag(S)·V. `capas`: modulos con parametros U, S, V (es_factorizada).
    U y V reciben lr*rms*sqrt(max(m,n))*NS5(momento)/(sigma_A + sigma_B + 1); S no es suyo."""

    def __init__(self, capas: Iterable[nn.Module], lr: float, momentum: float = MOMENTO, nesterov: bool = NESTEROV,
                 weight_decay: float = WD, rms: float = RMS, gen: Optional[torch.Generator] = None):
        self.capas = list(capas)
        if not self.capas or not all(es_factorizada(c) for c in self.capas):
            raise ValueError("ZSpectron3 recibe capas con parametros U (out x r), S (r) y V (r x in)")
        params = [p for c in self.capas for p in (c.U, c.V)]
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay, rms=rms))
        self.gen = gen if gen is not None else torch.Generator(device="cpu").manual_seed(SEMILLA_POT)

    def _sigma(self, st: dict, W: torch.Tensor) -> torch.Tensor:
        if "u_pot" not in st:
            u = torch.randn(W.shape[0], generator=self.gen, dtype=torch.float32)
            st["u_pot"] = (u / u.norm()).to(device=W.device, dtype=W.dtype)
            k = K_POT_INIT
        else:
            k = K_POT
        s, st["u_pot"] = potencia(W, st["u_pot"], k)
        return s

    @torch.no_grad()
    def sigmas(self) -> List[tuple]:
        """[(sigma_A, sigma_B)] por capa con las u guardadas (sin avanzarlas): para lecturas."""
        out = []
        for c in self.capas:
            stU, stV = self.state[c.U], self.state[c.V]
            if "u_pot" not in stU or "u_pot" not in stV:
                out.append((None, None))
                continue
            A, B = c.U * c.S, c.S[:, None] * c.V
            out.append((float(potencia(A, stU["u_pot"], 1)[0]), float(potencia(B, stV["u_pot"], 1)[0])))
        return out

    @torch.no_grad()
    def step(self, closure=None):
        g = self.param_groups[0]
        for c in self.capas:
            U, S, V = c.U, c.S, c.V
            if U.grad is None and V.grad is None:
                continue
            stU, stV = self.state[U], self.state[V]
            sA = self._sigma(stU, U * S)            # U·diag(S): (out, r)
            sB = self._sigma(stV, S[:, None] * V)   # diag(S)·V: (r, in)
            f = 1.0 / (sA + sB + 1.0)                # tensor 0-d: el paso no sincroniza GPU->CPU
            stU["sigmas"] = torch.stack([sA, sB]).detach()
            for p, st in ((U, stU), (V, stV)):
                if p.grad is None:
                    continue
                O = _momento_ortogonal(p, st, g)
                p.mul_(1.0 - g["lr"] * g["weight_decay"]).sub_(O * (g["lr"] * escala_rms(p, g["rms"]) * f))


class Doble:
    """Optimizador compuesto: AdamW (embedding, normas, sesgos y S) + los espectrales (ZMuon para
    las matrices densas, ZSpectron3 para las factorizadas). Los espectrales dan su paso ANTES que
    AdamW para leer sigma con la S de antes del paso. param_groups: primero los de AdamW (los
    diagnosticos leen betas de param_groups[0]); state: la union."""

    def __init__(self, adam: Optimizer, espectrales: Dict[str, Optimizer]):
        if not espectrales:
            raise ValueError("Doble necesita al menos un optimizador espectral")
        self.adam, self.espectrales = adam, dict(espectrales)

    @property
    def param_groups(self) -> list:
        return self.adam.param_groups + [g for o in self.espectrales.values() for g in o.param_groups]

    @property
    def state(self) -> dict:
        out = dict(self.adam.state)
        for o in self.espectrales.values():
            out.update(o.state)
        return out

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.adam.zero_grad(set_to_none=set_to_none)
        for o in self.espectrales.values():
            o.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        for o in self.espectrales.values():
            o.step()
        self.adam.step()

    def fijar_lr(self, lr: float, lr_adam: Optional[float] = None) -> None:
        """LR del espectral y, si se da, del grupo AdamW (si no, el AdamW sigue a lr*mult)."""
        for o in self.espectrales.values():
            for g in o.param_groups:
                g["lr"] = lr
        for g in self.adam.param_groups:
            g["lr"] = lr_adam if lr_adam is not None else lr * g.get("mult", 1.0)

    def state_dict(self) -> dict:
        return {"tipo": "mzspectral", "adamw": self.adam.state_dict(),
                "espectral": {k: o.state_dict() for k, o in self.espectrales.items()}}

    def load_state_dict(self, d: dict) -> None:
        if d.get("tipo") != "mzspectral" or set(d.get("espectral", {})) != set(self.espectrales):
            raise ValueError(f"estado incompatible: {d.get('tipo')!r} con {sorted(d.get('espectral', {}))}")
        self.adam.load_state_dict(d["adamw"])
        for k, o in self.espectrales.items():
            o.load_state_dict(d["espectral"][k])


def particion(m: nn.Module, excluir: Sequence[nn.Module] = ()) -> dict:
    """{"capas": modulos factorizados (ZSpectron3), "matrices": pesos 2-D de nn.Linear (ZMuon),
    "adam": el resto (embeddings, normas, sesgos, S y lo excluido)}; cada parametro en un solo sitio.
    `excluir`: modulos cuyas matrices van a AdamW (p. ej. una cabeza no atada)."""
    capas, matrices, ids = [], [], set()
    ex = {id(x) for x in excluir}
    for mod in m.modules():
        if id(mod) in ex:
            continue
        if es_factorizada(mod):
            capas.append(mod)
            ids.update((id(mod.U), id(mod.V)))
        elif isinstance(mod, nn.Linear) and mod.weight.dim() == 2:
            matrices.append(mod.weight)
            ids.add(id(mod.weight))
    adam = [p for p in m.parameters() if id(p) not in ids]
    return {"capas": capas, "matrices": matrices, "adam": adam}


def construir(m: nn.Module, lr: float, lr_adam: Optional[float] = None, betas=BETAS, eps: float = EPS, wd: float = WD,
              momentum: float = MOMENTO, nesterov: bool = NESTEROV, rms: float = RMS, seed: int = 0,
              excluir: Sequence[nn.Module] = ()) -> Doble:
    """Doble para `m`: ZMuon en los nn.Linear, ZSpectron3 en las capas U·diag(S)·V, AdamW en el
    resto. `lr` es el LR espectral (unidades de AdamW); `lr_adam` el del grupo AdamW (por defecto
    igual a `lr`; MZTrain usa el LR calibrado de AdamW para ese grupo). El grupo AdamW lleva
    `mult = lr_adam/lr` para calendarios que escalen todos los grupos por igual."""
    lr_adam = lr if lr_adam is None else lr_adam
    part = particion(m, excluir)
    if not part["capas"] and not part["matrices"]:
        raise ValueError("el modelo no tiene matrices densas ni capas factorizadas")
    adam = torch.optim.AdamW([{"params": part["adam"], "mult": lr_adam / lr}], lr=lr_adam, betas=betas, eps=eps, weight_decay=wd)
    esp: Dict[str, Optimizer] = {}
    if part["matrices"]:
        esp["muon"] = ZMuon(part["matrices"], lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=wd, rms=rms)
    if part["capas"]:
        esp["spectron"] = ZSpectron3(part["capas"], lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=wd, rms=rms,
                                     gen=torch.Generator(device="cpu").manual_seed(SEMILLA_POT + seed))
    return Doble(adam, esp)
