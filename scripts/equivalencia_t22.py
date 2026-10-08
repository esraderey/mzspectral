"""Equivalencia bit a bit entre mzspectral (standalone) y el codigo sellado del experimento T22 de
MZTrain (D:/mztrain-evidencia-T22/mztrain_t22/espectral.py, SHA-256 e2f01afa...): mismos factores,
mismos gradientes y mismas semillas de potencia -> mismas actualizaciones, en las capas sueltas y
en el GPT de mztrain montado como en T22. Solo LEE de T22, T20 y mztrain; no corre nada en GPU.
Requiere esas carpetas en D: (si faltan, termina con aviso y codigo 3).

(c) 2026 MSC Star Team. Licencia MIT (ver LICENSE).
Uso: D:/mztrain/.venv/Scripts/python.exe scripts/equivalencia_t22.py
"""
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.dont_write_bytecode = True
RUTAS = ("D:/mzspectral", "D:/mztrain-evidencia-T22", "D:/mztrain-evidencia-T20")
if not all(os.path.isdir(p) for p in RUTAS):
    print("faltan carpetas de referencia:", [p for p in RUTAS if not os.path.isdir(p)])
    sys.exit(3)
for p in RUTAS:
    sys.path.insert(0, p)
import torch  # noqa: E402
from torch import nn  # noqa: E402
from mztrain.layers import ZFactorizedLinear  # noqa: E402
from mztrain_t22 import espectral as T22  # noqa: E402
import mzspectral as Z  # noqa: E402

g = torch.Generator().manual_seed(2026)
# --- Spectron sobre una ZFactorizedLinear de mztrain (la capa real de T22)
torch.manual_seed(7)
lay_a = ZFactorizedLinear(40, 32, rank=8, bias=False, init_method="svd")
torch.manual_seed(7)
lay_b = ZFactorizedLinear(40, 32, rank=8, bias=False, init_method="svd")
assert all(torch.equal(x, y) for x, y in zip(lay_a.parameters(), lay_b.parameters()))
oa = T22.Spectron([lay_a], lr=4.8e-3, gen=torch.Generator().manual_seed(3017))
ob = Z.ZSpectron3([lay_b], lr=4.8e-3, gen=torch.Generator().manual_seed(3017))
for t in range(5):
    GU, GV = torch.randn(32, 8, generator=g), torch.randn(8, 40, generator=g)
    lay_a.U.grad, lay_a.V.grad = GU.clone(), GV.clone()
    lay_b.U.grad, lay_b.V.grad = GU.clone(), GV.clone()
    oa.step()
    ob.step()
    assert torch.equal(lay_a.U, lay_b.U) and torch.equal(lay_a.V, lay_b.V), f"Spectron difiere en el paso {t}"
    assert torch.equal(oa.state[lay_a.U]["sigmas"], ob.state[lay_b.U]["sigmas"])
# --- Muon sobre una matriz densa
torch.manual_seed(8)
wa = nn.Parameter(torch.randn(48, 36))
wb = nn.Parameter(wa.detach().clone())
ma, mb = T22.Muon([wa], lr=1.2e-3), Z.ZMuon([wb], lr=1.2e-3)
for t in range(5):
    G = torch.randn(48, 36, generator=g)
    wa.grad, wb.grad = G.clone(), G.clone()
    ma.step()
    mb.step()
    assert torch.equal(wa, wb), f"Muon difiere en el paso {t}"
# --- Doble completo sobre el GPT de mztrain (misma particion que T22: build de t22 frente a construir)
import t22  # noqa: E402  (parchea t20.build; no corre nada)
import t20  # noqa: E402
from mztrain_t20 import sano  # noqa: E402

t20.ARMS.update({"Fspec": {"d": 48, "lin": "fact", "r": 8, "opt": "spectron"}, "Dsmuon": {"d": 24, "lin": "dense", "opt": "muon"}})
t20.L, t20.H, t20.SEQ = 2, 4, 16
sano.WARMUP = 6
for arm, lr, lr_adam in (("Fspec", 4.8e-3, 3e-4), ("Dsmuon", 1.2e-3, 6e-4)):
    spec = {"etapa": "A", "arm": arm, "lr": lr, "beta2": 0.95, "seed": 17, "steps": 60, "key": "x"}
    m_a, o_a, cal = t22.build(spec, 50, "cpu")
    torch.manual_seed(17)
    m_b = type(m_a)(50, 16, t20.ARMS[arm]["d"], 2, 4, t20._lin(t20.ARMS[arm]))
    sano.init_sano(m_b, 17)
    assert all(torch.equal(x, y) for x, y in zip(m_a.parameters(), m_b.parameters()))
    o_b = Z.construir(m_b, lr=lr, lr_adam=lr_adam, seed=17)
    o_b.fijar_lr(cal.lr(0))        # mismo LR de arranque que el calendario de T20 (rampa 0.1*pico): mult*base
    assert [h["lr"] for h in o_a.param_groups] == [h["lr"] for h in o_b.param_groups], arm
    gen = torch.Generator().manual_seed(1017)
    for t in range(8):
        x = torch.randint(50, (4, 16), generator=gen)
        for m, o in ((m_a, o_a), (m_b, o_b)):
            o.zero_grad(set_to_none=True)
            _, loss = m(x, x)
            loss.backward()
            o.step()
        assert all(torch.equal(p, q) for p, q in zip(m_a.parameters(), m_b.parameters())), f"{arm}: difiere en el paso {t}"
    print(arm, "identico en 8 pasos del GPT de mztrain")
print("EQUIVALENCIA BIT A BIT: OK (Spectron 5 pasos, Muon 5 pasos, Fspec y Dsmuon 8 pasos)")
