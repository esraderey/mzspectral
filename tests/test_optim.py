"""mzspectral en CPU, sin MZTrain: Newton-Schulz, potencia, ZMuon, ZSpectron3, particion, Doble,
construir y reanudacion identica desde state_dict."""
from __future__ import annotations

import math
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import torch
from torch import nn

import mzspectral as Z
from mzspectral import optim as E


class Modelito(nn.Module):
    """Embedding + LayerNorm + una densa + una factorizada + una densa con sesgo: todos los casos."""

    def __init__(self, v=50, d=24, r=6):
        super().__init__()
        self.tok = nn.Embedding(v, d)
        self.ln = nn.LayerNorm(d)
        self.densa = nn.Linear(d, 2 * d, bias=False)
        self.fact = Z.FactorizedLinear(2 * d, d, rank=r)
        self.salida = nn.Linear(d, v, bias=True)

    def forward(self, x):
        h = self.ln(self.tok(x))
        h = self.fact(torch.relu(self.densa(h)))
        return self.salida(h)


def _loss(m, x):
    return nn.functional.cross_entropy(m(x).flatten(0, 1), x.flatten())


# ------------------------------------------------------------ piezas
def test_newton_schulz_ortogonaliza_en_la_direccion_polar():
    g = torch.Generator().manual_seed(1)
    for shape in ((40, 12), (12, 40), (30, 30)):
        G = torch.randn(*shape, generator=g)
        O = E.newton_schulz5(G)
        sv = torch.linalg.svdvals(O)
        assert O.shape == G.shape and O.dtype == torch.float32 and 0.6 < sv.min() and sv.max() < 1.3, shape
        U, _, Vh = torch.linalg.svd(G, full_matrices=False)
        assert torch.nn.functional.cosine_similarity(O.flatten(), (U @ Vh).flatten(), dim=0) > 0.95
        assert torch.allclose(E.newton_schulz5(3.0 * G), O, atol=1e-5)
    assert torch.equal(E.newton_schulz5(torch.zeros(5, 3)), torch.zeros(5, 3))


def test_potencia_estima_la_norma_espectral_por_debajo():
    g = torch.Generator().manual_seed(2)
    u1, v1, u2, v2 = (torch.randn(n, generator=g) for n in (30, 12, 30, 12))
    W = 3.0 * torch.outer(u1 / u1.norm(), v1 / v1.norm()) + 1.0 * torch.outer(u2 / u2.norm(), v2 / v2.norm()) + 0.05 * torch.randn(30, 12, generator=g)
    exacta = float(torch.linalg.matrix_norm(W, ord=2))
    u = torch.randn(30, generator=g)
    s10, u10 = E.potencia(W, u / u.norm(), E.K_POT_INIT)
    assert torch.is_tensor(s10) and s10.dim() == 0
    assert abs(float(s10) - exacta) / exacta < 0.01 and float(s10) <= exacta + 1e-6
    assert abs(float(E.potencia(W, u10, 1)[0]) - exacta) / exacta < 1e-3


def test_muon_y_spectron_coinciden_con_la_formula_a_mano():
    g = torch.Generator().manual_seed(11)
    lr, wd, mu = 3e-3, 0.01, E.MOMENTO
    p = nn.Parameter(torch.randn(24, 10, generator=g))
    q = p.detach().clone()
    opt = Z.ZMuon([p], lr=lr, weight_decay=wd)
    buf = torch.zeros_like(q)
    for _ in range(3):
        G = torch.randn(24, 10, generator=g)
        p.grad = G.clone()
        opt.step()
        buf = mu * buf + G
        q = q * (1 - lr * wd) - lr * E.RMS * math.sqrt(24) * E.newton_schulz5(G + mu * buf)
        assert torch.allclose(p.detach(), q, atol=1e-6)
    torch.manual_seed(12)
    lay = Z.FactorizedLinear(20, 16, rank=6)
    U, S, V = (t.detach().clone() for t in (lay.U, lay.S, lay.V))
    sp = Z.ZSpectron3([lay], lr=lr, weight_decay=wd, gen=torch.Generator().manual_seed(5))
    bU, bV = torch.zeros_like(U), torch.zeros_like(V)
    for _ in range(3):
        GU, GV = torch.randn(16, 6, generator=g), torch.randn(6, 20, generator=g)
        lay.U.grad, lay.V.grad = GU.clone(), GV.clone()
        sp.step()
        sA, sB = (float(x) for x in sp.state[lay.U]["sigmas"])
        eA, eB = float(torch.linalg.matrix_norm(U * S, ord=2)), float(torch.linalg.matrix_norm(S[:, None] * V, ord=2))
        assert 0.9 * eA <= sA <= eA * 1.001 and 0.9 * eB <= sB <= eB * 1.001
        f = 1.0 / (sA + sB + 1.0)
        bU, bV = mu * bU + GU, mu * bV + GV
        U = U * (1 - lr * wd) - lr * E.RMS * math.sqrt(16) * f * E.newton_schulz5(GU + mu * bU)
        V = V * (1 - lr * wd) - lr * E.RMS * math.sqrt(20) * f * E.newton_schulz5(GV + mu * bV)
        assert torch.allclose(lay.U.detach(), U, atol=1e-6) and torch.allclose(lay.V.detach(), V, atol=1e-6)
        assert torch.equal(lay.S.detach(), S)                                 # S no es de ZSpectron3
    (a, b), = sp.sigmas()
    assert 0.9 * eA <= a <= eA * 1.001 and 0.9 * eB <= b <= eB * 1.001
    with pytest.raises(ValueError):
        Z.ZMuon([nn.Parameter(torch.zeros(3))], lr=1e-3)
    with pytest.raises(ValueError):
        Z.ZSpectron3([nn.Linear(3, 3)], lr=1e-3)


def test_muon_con_gradiente_nulo_solo_decae():
    q = nn.Parameter(torch.ones(6, 4))
    o = Z.ZMuon([q], lr=0.1, weight_decay=0.5)
    q.grad = torch.zeros(6, 4)
    o.step()
    assert torch.allclose(q.detach(), torch.full((6, 4), 0.95))


def test_spectron_acota_la_actualizacion_del_producto():
    torch.manual_seed(4)
    lay = Z.FactorizedLinear(20, 16, rank=6)
    U0, S0, V0 = (t.detach().clone() for t in (lay.U, lay.S, lay.V))
    opt = Z.ZSpectron3([lay], lr=5e-3, weight_decay=0.0, gen=torch.Generator().manual_seed(9))
    lay(torch.randn(7, 20)).pow(2).sum().backward()
    opt.step()
    W0, W1 = (U0 * S0) @ V0, lay.weight().detach()
    sA, sB = float(torch.linalg.matrix_norm(U0 * S0, ord=2)), float(torch.linalg.matrix_norm(S0[:, None] * V0, ord=2))
    cota = 5e-3 * max(E.escala_rms(lay.U, E.RMS), E.escala_rms(lay.V, E.RMS)) * 1.3 * (sA + sB) / (sA + sB + 1)
    assert 0 < float(torch.linalg.matrix_norm(W1 - W0, ord=2)) <= cota


# ------------------------------------------------------------ particion, construir, Doble
def test_particion_cubre_cada_parametro_una_vez():
    m = Modelito()
    part = Z.particion(m)
    ids = [id(p) for p in part["adam"]] + [id(p) for p in part["matrices"]] + [id(p) for c in part["capas"] for p in (c.U, c.V)]
    assert sorted(ids) == sorted(id(p) for p in m.parameters()) and len(ids) == len(set(ids))
    assert part["capas"] == [m.fact] and {id(p) for p in part["matrices"]} == {id(m.densa.weight), id(m.salida.weight)}
    assert any(p is m.tok.weight for p in part["adam"]) and any(p is m.fact.S for p in part["adam"]) and any(p is m.salida.bias for p in part["adam"])
    part2 = Z.particion(m, excluir=[m.salida])
    assert {id(p) for p in part2["matrices"]} == {id(m.densa.weight)} and any(p is m.salida.weight for p in part2["adam"])
    with pytest.raises(ValueError):
        Z.construir(nn.Sequential(nn.Embedding(5, 3), nn.LayerNorm(3)), lr=1e-3)


def test_construir_doble_dos_relojes_estado_y_orden_del_paso(monkeypatch):
    torch.manual_seed(13)
    m = Modelito()
    opt = Z.construir(m, lr=2.4e-3, lr_adam=3e-4, seed=17)
    assert isinstance(opt, Z.Doble) and set(opt.espectrales) == {"muon", "spectron"} and "betas" in opt.param_groups[0]
    assert opt.param_groups[0]["lr"] == 3e-4 and opt.param_groups[0]["mult"] == pytest.approx(3e-4 / 2.4e-3)
    assert all(g["lr"] == 2.4e-3 for g in opt.param_groups[1:]) and all(g["weight_decay"] == E.WD for g in opt.param_groups)
    opt.fijar_lr(1e-3)
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-3 * 3e-4 / 2.4e-3) and all(g["lr"] == 1e-3 for g in opt.param_groups[1:])
    opt.fijar_lr(2.4e-3, 3e-4)
    # orden del paso: las sigma guardadas son las de la S de ANTES del paso de AdamW
    monkeypatch.setattr(E, "potencia", lambda W, u, k: (torch.linalg.matrix_norm(W, ord=2), u))
    opt.fijar_lr(2.4e-3, 0.5)
    U0, S0, V0 = (t.detach().clone() for t in (m.fact.U, m.fact.S, m.fact.V))
    x = torch.randint(50, (2, 8))
    opt.zero_grad(set_to_none=True)
    _loss(m, x).backward()
    opt.step()
    S1 = m.fact.S.detach()
    sA, sB = (float(v) for v in opt.state[m.fact.U]["sigmas"])
    eA0, eB0 = float(torch.linalg.matrix_norm(U0 * S0, ord=2)), float(torch.linalg.matrix_norm(S0[:, None] * V0, ord=2))
    eA1, eB1 = float(torch.linalg.matrix_norm(U0 * S1, ord=2)), float(torch.linalg.matrix_norm(S1[:, None] * V0, ord=2))
    assert abs(eA1 - eA0) > 0.1 * eA0 and abs(eB1 - eB0) > 0.1 * eB0
    assert abs(sA - eA0) < 1e-4 * eA0 and abs(sB - eB0) < 1e-4 * eB0
    # estado: todos los parametros tienen estado; round trip identico
    assert len(opt.state) == sum(1 for _ in m.parameters())
    d = opt.state_dict()
    assert d["tipo"] == "mzspectral" and set(d["espectral"]) == {"muon", "spectron"}
    m2 = Modelito()
    opt2 = Z.construir(m2, lr=2.4e-3, lr_adam=3e-4, seed=17)
    opt2.load_state_dict(d)
    for p, q in zip(m.parameters(), m2.parameters()):
        for k, v in opt.state[p].items():
            assert torch.equal(v, opt2.state[q][k]) if torch.is_tensor(v) else v == opt2.state[q][k], k
    with pytest.raises(ValueError):
        Z.construir(nn.Sequential(nn.Linear(4, 4)), lr=1e-3).load_state_dict(d)


def test_reanudacion_identica_desde_state_dict():
    def tren(corte=None, pasos=12):
        torch.manual_seed(21)
        m = Modelito()
        opt = Z.construir(m, lr=2.4e-3, lr_adam=3e-4, seed=17)
        g = torch.Generator().manual_seed(1017)
        for t in range(pasos):
            if corte is not None and t == corte:
                sd = {"m": m.state_dict(), "opt": opt.state_dict()}
                torch.manual_seed(99)
                m = Modelito()
                opt = Z.construir(m, lr=2.4e-3, lr_adam=3e-4, seed=17)
                m.load_state_dict(sd["m"])
                opt.load_state_dict(sd["opt"])
            x = torch.randint(50, (4, 8), generator=g)
            opt.zero_grad(set_to_none=True)
            _loss(m, x).backward()
            opt.step()
        return m
    m1, m2 = tren(), tren(corte=5)
    assert all(torch.equal(a, b) for a, b in zip(m1.parameters(), m2.parameters()))


def test_entrena_y_baja_la_perdida():
    torch.manual_seed(3)
    m = Modelito()
    opt = Z.construir(m, lr=1e-2, lr_adam=1e-2)
    g = torch.Generator().manual_seed(4)
    x = torch.randint(50, (8, 8), generator=g)
    l0 = float(_loss(m, x).detach())
    for _ in range(200):
        opt.zero_grad(set_to_none=True)
        _loss(m, x).backward()
        opt.step()
    assert float(_loss(m, x).detach()) < 0.6 * l0 and all(torch.isfinite(p).all() for p in m.parameters())


def test_layers_y_lecturas():
    torch.manual_seed(5)
    lay = Z.FactorizedLinear(32, 24, rank=8)
    assert lay.U.shape == (24, 8) and lay.S.shape == (8,) and lay.V.shape == (8, 32) and Z.es_factorizada(lay)
    un, vn = lay.U.norm(dim=0), lay.V.norm(dim=1)
    assert torch.allclose(un, torch.ones(8), atol=1e-5) and torch.allclose(vn, torch.ones(8), atol=1e-5)
    x = torch.randn(5, 32)
    assert torch.allclose(lay(x), x @ lay.weight().T, atol=1e-5)
    W = torch.empty(24, 32)
    nn.init.kaiming_uniform_(W, a=math.sqrt(5))
    assert abs(float(lay.weight().detach().norm()) / float(W.norm()) - 1) < 0.25   # EPSI: energia de Frobenius del orden de la densa
    n = Z.normas(Modelito())
    assert n["fact"]["tipo"] == "factorizada" and n["fact"]["ortog_U"] < 1e-4 and 1 <= n["fact"]["rango_estable_W"] <= 6
    assert n["densa"]["tipo"] == "densa" and 1 < n["densa"]["rango_estable_W"] <= 24 and "salida" in n and "tok" not in n
    assert Z.__version__ == "0.1.0" and Z.Muon is Z.ZMuon and Z.Spectron3 is Z.ZSpectron3
