"""Optimizadores espectrales: ZMuon para matrices densas y ZSpectron3 para factores U·diag(S)·V.

(c) 2026 MSC Star Team (Esraderey, Raul Cruz Acosta). Licencia MIT (ver LICENSE).

Mecanica comun a los dos (Jordan et al. 2024, "Muon"; Liu et al. 2025, "Muon is scalable",
arXiv 2502.16982):
- momento con Nesterov: buf <- mu*buf + g; u = g + mu*buf (mu = 0.95);
- ortogonalizacion de u con 5 iteraciones de Newton-Schulz (coeficientes 3.4445, -4.7750, 2.0315;
  normalizacion de Frobenius con eps 1e-7; traspuesta si hay mas filas que columnas), en fp32.
  No es una ortogonalizacion exacta: el mapa escalar phi(s) = 3.4445 s - 4.7750 s^3 + 2.0315 s^5
  deja los valores singulares en la banda [0.68, 1.13] cuando s_i/||G||_F >~ 2e-3; por debajo, las
  direcciones pequenas se quedan pequenas (~485 s). Es el mismo comportamiento que el Muon de
  referencia;
- escala RMS: la actualizacion ortogonalizada se multiplica por 0.2*sqrt(max(m, n)); su RMS por
  entrada queda en ~0.17-0.20*lr (0.2 exacto solo con una ortogonal exacta), del orden de la de
  AdamW: lr y weight decay se expresan en unidades de AdamW;
- weight decay desacoplado POR PARAMETRO: cada matriz o factor hace p <- p*(1 - lr*wd). En el denso
  eso es W*(1 - lr*wd); en el factorizado decaen U y V (con lr) y S (con lr_adam, en AdamW), asi
  que el producto W = U·diag(S)·V decae por (1 - lr*wd)^2 (1 - lr_adam*wd): con los LR calibrados
  en T22 (4.8e-3 y 3e-4) son 9.9e-5 por paso al pico frente a 1.2e-5 en el denso con ZMuon, y a
  lo largo de 18 000 pasos W se encoge por 0.41 frente a 0.90 (ver EVIDENCIA.md, "Limites").

ZSpectron3 (Spectron, arXiv 2602.12429, Algoritmo 1, escrito para un producto W = A·B, llevado a
tres factores): a primer orden dW = dU·diag(S)·V + U·diag(S)·dV + U·diag(dS)·V. Los dos primeros
terminos los acotan ||diag(S)·V||_2 (paso de U) y ||U·diag(S)||_2 (paso de V): se definen
sigma_A := ||U·diag(S)||_2 y sigma_B := ||diag(S)·V||_2 (iteracion de potencia con arranque
caliente, leidas ANTES del paso) y U, V reciben el momento ortogonalizado por 1/(sigma_A +
sigma_B + 1). El tercer termino lo mueve AdamW sobre S (1-D: como el embedding, las normas y los
sesgos, regla de Muon), sin ese factor y escalando con ||U||_2 ||V||_2; con lr_adam << lr pesa poco
(en T22, ~5 % de la cota de U y V) y crece si lr_adam se acerca a lr o si las normas de U y V
crecen. El weight decay no lleva el factor 1/(sigma_A + sigma_B + 1) (forma publicada).

Evidencia (T22 = E9 de RFC-CALIDAD-1, MZTrain; ver EVIDENCIA.md): sobre la linea base sana de
T20, a 18 000 pasos y tres semillas, ZMuon mejora al denso 0.066 BPC y ZSpectron3 al factorizado
0.063 BPC frente a AdamW, con la mitad de estado de optimizador y 0 inestabilidades hasta LR 9.6e-3.
En el factorizado cambian a la vez el optimizador y el decaimiento efectivo (x11): la ablacion que
los separa esta pendiente.
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
    """Ortogonalizacion aproximada de G (Jordan et al.): valores singulares en [0.68, 1.13] para las
    direcciones con s_i/||G||_F >~ 2e-3; las mas pequenas quedan pequenas. G = 0 -> 0."""
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
    sigma = u^T W v <= ||W||_2 (estimacion por debajo; el error normal es < 1 %), tensor 0-d en el
    dispositivo de W: el paso no sincroniza GPU->CPU. Si W·v = 0, u sale nulo y sigma 0: el
    llamador (ZSpectron3._sigma) lo detecta y repone un vector de reserva."""
    v = None
    for _ in range(max(1, k)):
        v = W.T @ u
        v = v / (v.norm() + 1e-30)
        u = W @ v
        u = u / (u.norm() + 1e-30)
    return u @ (W @ v), u


def escala_rms(p: torch.Tensor, rms: float) -> float:
    return rms * math.sqrt(max(p.shape))


def _validar(lr: float, momentum: float, weight_decay: float, rms: float) -> None:
    if not (isinstance(lr, (int, float)) and lr > 0):
        raise ValueError(f"lr debe ser > 0 (recibido {lr!r})")
    if not 0.0 <= momentum < 1.0:
        raise ValueError(f"momentum debe estar en [0, 1) (recibido {momentum!r})")
    if weight_decay < 0:
        raise ValueError(f"weight_decay debe ser >= 0 (recibido {weight_decay!r})")
    if rms <= 0:
        raise ValueError(f"rms debe ser > 0 (recibido {rms!r})")


def _momento_ortogonal(p: torch.Tensor, st: dict, g: dict) -> torch.Tensor:
    if "buf" not in st:
        st["buf"] = torch.zeros_like(p)
    buf = st["buf"]
    buf.mul_(g["momentum"]).add_(p.grad)
    u = p.grad.add(buf, alpha=g["momentum"]) if g["nesterov"] else buf
    return newton_schulz5(u)


def _con_closure(closure):
    if closure is None:
        return None
    with torch.enable_grad():
        return closure()


class ZMuon(Optimizer):
    """Muon para matrices 2-D: W <- W*(1 - lr*wd) - lr*rms*sqrt(max(m,n))*NS5(momento Nesterov)."""

    def __init__(self, params: Iterable[torch.Tensor], lr: float, momentum: float = MOMENTO, nesterov: bool = NESTEROV,
                 weight_decay: float = WD, rms: float = RMS):
        params = list(params)
        _validar(lr, momentum, weight_decay, rms)
        if not params or any(p.dim() != 2 for p in params):
            raise ValueError("ZMuon solo recibe matrices 2-D (al menos una)")
        if len({id(p) for p in params}) != len(params):
            raise ValueError("ZMuon recibio el mismo parametro dos veces")
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay, rms=rms))

    @torch.no_grad()
    def step(self, closure=None):
        loss = _con_closure(closure)
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                O = _momento_ortogonal(p, self.state[p], g)
                p.mul_(1.0 - g["lr"] * g["weight_decay"]).add_(O, alpha=-g["lr"] * escala_rms(p, g["rms"]))
        return loss


def es_factorizada(mod: nn.Module) -> bool:
    """Un modulo con parametros propios U (out x r), S (r) y V (r x in): la forma de tres factores."""
    U, S, V = (mod._parameters.get(a) if hasattr(mod, "_parameters") else None for a in ("U", "S", "V"))
    return (all(isinstance(t, nn.Parameter) for t in (U, S, V)) and U.dim() == 2 and S.dim() == 1 and V.dim() == 2
            and U.shape[1] == S.shape[0] == V.shape[0])


class ZSpectron3(Optimizer):
    """Spectron adaptado a U·diag(S)·V. `capas`: modulos con parametros U, S, V (es_factorizada).
    U y V reciben lr*rms*sqrt(max(m,n))*NS5(momento)/(sigma_A + sigma_B + 1); S no es suyo.
    Un solo grupo de parametros: add_param_group despues de construir lanza error (las capas se
    recorren por `capas`, no por los grupos)."""

    def __init__(self, capas: Iterable[nn.Module], lr: float, momentum: float = MOMENTO, nesterov: bool = NESTEROV,
                 weight_decay: float = WD, rms: float = RMS, gen: Optional[torch.Generator] = None):
        _validar(lr, momentum, weight_decay, rms)
        self.capas = list(capas)
        if not self.capas or not all(es_factorizada(c) for c in self.capas):
            raise ValueError("ZSpectron3 recibe capas con parametros U (out x r), S (r) y V (r x in)")
        if len({id(c) for c in self.capas}) != len(self.capas) or len({id(p) for c in self.capas for p in (c.U, c.V)}) != 2 * len(self.capas):
            raise ValueError("ZSpectron3 recibio una capa o un factor dos veces")
        params = [p for c in self.capas for p in (c.U, c.V)]
        self._cerrado = False
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay, rms=rms))
        self._cerrado = True
        self.gen = gen if gen is not None else torch.Generator(device="cpu").manual_seed(SEMILLA_POT)

    def add_param_group(self, param_group: dict) -> None:
        if getattr(self, "_cerrado", False):
            raise NotImplementedError("ZSpectron3 recorre sus capas, no sus grupos: construye otro optimizador para capas nuevas")
        super().add_param_group(param_group)

    # deepcopy / pickle: Optimizer.__getstate__ solo guarda defaults, state y param_groups
    def __getstate__(self) -> dict:
        d = super().__getstate__()
        d.update(capas=self.capas, gen=self.gen, _cerrado=self._cerrado)
        return d

    def __setstate__(self, d: dict) -> None:
        super().__setstate__(d)                       # Optimizer.load_state_dict lo llama solo con state y param_groups
        self.capas = d.get("capas", getattr(self, "capas", None))
        self.gen = d.get("gen", getattr(self, "gen", None))
        self._cerrado = d.get("_cerrado", getattr(self, "_cerrado", True))

    def _unitario(self, n: int, W: torch.Tensor) -> torch.Tensor:
        u = torch.randn(n, generator=self.gen, dtype=torch.float32)
        return (u / u.norm()).to(device=W.device, dtype=W.dtype)

    def _sigma(self, st: dict, W: torch.Tensor) -> torch.Tensor:
        if "u_pot" not in st:
            st["u_pot"] = self._unitario(W.shape[0], W)
            st["u_res"] = torch.roll(st["u_pot"], 1)       # reserva unitaria sin consumir el generador (misma secuencia que T22)
            k = K_POT_INIT
        else:
            k = K_POT
        if st["u_pot"].dtype != W.dtype or st["u_pot"].device != W.device:
            st["u_pot"], st["u_res"] = st["u_pot"].to(device=W.device, dtype=W.dtype), st["u_res"].to(device=W.device, dtype=W.dtype)
        s, u = potencia(W, st["u_pot"], k)
        malo = u.norm() < 0.5                         # u es unitario salvo colapso (W·v = 0): se repone la reserva
        st["u_pot"] = torch.where(malo, st["u_res"], u)
        return torch.where(malo, torch.zeros_like(s), s)

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
        loss = _con_closure(closure)
        g = self.param_groups[0]
        for c in self.capas:
            U, S, V = c.U, c.S, c.V
            if U.grad is None and V.grad is None:
                continue
            stU, stV = self.state[U], self.state[V]
            sA = self._sigma(stU, U * S)            # U·diag(S): (out, r)
            sB = self._sigma(stV, S[:, None] * V)   # diag(S)·V: (r, in)
            f = 1.0 / (sA + sB + 1.0)                # tensor 0-d: el paso no sincroniza
            stU["sigmas"] = torch.stack([sA, sB]).detach()
            for p, st in ((U, stU), (V, stV)):
                if p.grad is None:
                    continue
                O = _momento_ortogonal(p, st, g)
                p.mul_(1.0 - g["lr"] * g["weight_decay"]).sub_(O * (g["lr"] * escala_rms(p, g["rms"]) * f))
        return loss

    def state_dict(self) -> dict:
        d = super().state_dict()
        d["gen_state"] = self.gen.get_state()          # para que una capa que reciba su primer gradiente tras el checkpoint reanude identica
        return d

    def load_state_dict(self, state_dict: dict) -> None:
        d = dict(state_dict)
        gen_state = d.pop("gen_state", None)
        super().load_state_dict(d)
        if gen_state is not None:
            self.gen.set_state(gen_state)


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

    def step(self, closure=None):
        loss = _con_closure(closure)
        for o in self.espectrales.values():
            o.step()
        self.adam.step()
        return loss

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
        """Comprueba todo el estado antes de tocar nada (tipo, espectrales, grupos y numero de
        parametros por grupo); solo despues carga, para no dejar un optimizador a medias."""
        if not isinstance(d, dict) or d.get("tipo") != "mzspectral" or set(d.get("espectral", {})) != set(self.espectrales):
            raise ValueError(f"estado incompatible: {getattr(d, 'get', lambda k: None)('tipo')!r} con {sorted(d.get('espectral', {})) if isinstance(d, dict) else '?'}")
        pares = [(self.adam, d["adamw"])] + [(o, d["espectral"][k]) for k, o in self.espectrales.items()]
        for o, sd in pares:
            if not isinstance(sd, dict) or "state" not in sd or "param_groups" not in sd:
                raise ValueError("estado incompleto: faltan 'state' o 'param_groups'")
            if len(sd["param_groups"]) != len(o.param_groups) or any(len(a["params"]) != len(b["params"]) for a, b in zip(sd["param_groups"], o.param_groups)):
                raise ValueError("estado incompatible: grupos o numero de parametros distintos")
        for o, sd in pares:
            o.load_state_dict(sd)


def _parametro_propio(mod: nn.Module, nombre: str) -> Optional[nn.Parameter]:
    """El Parameter `nombre` del modulo si es suyo y no esta parametrizado (weight_norm, etc.)."""
    p = mod._parameters.get(nombre) if hasattr(mod, "_parameters") else None
    return p if isinstance(p, nn.Parameter) else None


def particion(m: nn.Module, excluir: Sequence[nn.Module] = ()) -> dict:
    """{"capas": modulos factorizados (ZSpectron3), "matrices": pesos 2-D propios de nn.Linear
    (ZMuon), "adam": el resto}; cada parametro en un solo sitio aunque este compartido. Van a AdamW:
    embeddings, normas, sesgos, S, los pesos parametrizados, los pesos de nn.Linear atados a un
    nn.Embedding (cabeza atada) y todo lo que cuelgue de un modulo de `excluir` (recursivo)."""
    ex = set()
    for x in excluir:
        ex.update(id(sub) for sub in x.modules())
    embebidos = {id(mod.weight) for mod in m.modules() if isinstance(mod, nn.Embedding) and isinstance(getattr(mod, "weight", None), nn.Parameter)}
    capas, matrices, ids = [], [], set()
    for mod in m.modules():
        if id(mod) in ex:
            continue
        if es_factorizada(mod):
            if id(mod.U) in ids or id(mod.V) in ids:
                continue                                   # factores compartidos: una sola vez
            capas.append(mod)
            ids.update((id(mod.U), id(mod.V)))
        elif isinstance(mod, nn.Linear):
            w = _parametro_propio(mod, "weight")
            if w is None or w.dim() != 2 or id(w) in ids or id(w) in embebidos:
                continue
            matrices.append(w)
            ids.add(id(w))
    adam = [p for p in m.parameters() if id(p) not in ids]        # m.parameters() ya deduplica los compartidos
    return {"capas": capas, "matrices": matrices, "adam": adam}


def construir(m: nn.Module, lr: float, lr_adam: Optional[float] = None, betas=BETAS, eps: float = EPS, wd: float = WD,
              momentum: float = MOMENTO, nesterov: bool = NESTEROV, rms: float = RMS, seed: int = 0,
              excluir: Sequence[nn.Module] = ()) -> Doble:
    """Doble para `m`: ZMuon en los nn.Linear, ZSpectron3 en las capas U·diag(S)·V, AdamW en el
    resto. `lr` es el LR espectral (unidades de AdamW); `lr_adam` el del grupo AdamW (por defecto
    igual a `lr`; MZTrain usa el LR calibrado de AdamW para ese grupo). El grupo AdamW lleva
    `mult = lr_adam/lr` para calendarios que escalen todos los grupos por igual."""
    _validar(lr, momentum, wd, rms)
    lr_adam = lr if lr_adam is None else lr_adam
    if not (isinstance(lr_adam, (int, float)) and lr_adam > 0):
        raise ValueError(f"lr_adam debe ser > 0 (recibido {lr_adam!r})")
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
