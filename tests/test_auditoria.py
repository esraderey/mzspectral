"""Anclas de los hallazgos de la auditoria del 2026-10-08 (INFORME.md e INFORME-MATEMATICAS.md en
.forja/auditoria-2026-10-08): cada test reproduce el defecto corregido en 0.1.1."""
from __future__ import annotations

import copy
import json
import os
import subprocess
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import pytest
import torch
from torch import nn

import mzspectral as Z
from mzspectral import optim as E

SELLO = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "sello.py")


class Atado(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok = nn.Embedding(10, 4)
        self.cabeza = nn.Linear(4, 10, bias=False)
        self.cabeza.weight = self.tok.weight
        self.oculta = nn.Sequential(nn.Linear(4, 4, bias=False), nn.Linear(4, 4, bias=False))
        self.fact = Z.FactorizedLinear(4, 4, rank=2)


# ------------------------------------------------------------ particion (PER-LOG-002, PER-LOG-003, PER-ROB-003)
def test_parametro_compartido_entra_una_sola_vez():
    a, b = nn.Linear(6, 6, bias=False), nn.Linear(6, 6, bias=False)
    b.weight = a.weight
    part = Z.particion(nn.Sequential(a, b))
    assert sum(p is a.weight for p in part["matrices"]) == 1 and len(part["adam"]) == 0
    with pytest.raises(ValueError, match="dos veces"):
        Z.ZMuon([a.weight, a.weight], lr=1e-3)
    lay = Z.FactorizedLinear(6, 6, rank=2)
    with pytest.raises(ValueError, match="dos veces"):
        Z.ZSpectron3([lay, lay], lr=1e-3)


def test_cabeza_atada_al_embedding_va_a_adamw_y_excluir_es_recursivo():
    m = Atado()
    part = Z.particion(m)
    assert not any(p is m.tok.weight for p in part["matrices"]) and any(p is m.tok.weight for p in part["adam"])
    assert {id(p) for p in part["matrices"]} == {id(m.oculta[0].weight), id(m.oculta[1].weight)} and part["capas"] == [m.fact]
    part2 = Z.particion(m, excluir=[m.oculta])
    assert part2["matrices"] == [] and all(any(p is q for q in part2["adam"]) for p in m.oculta.parameters())
    part3 = Z.particion(m, excluir=[m])
    assert part3["matrices"] == [] and part3["capas"] == [] and len(part3["adam"]) == len(list(m.parameters()))


def test_linear_parametrizada_va_a_adamw():
    wn = torch.nn.utils.parametrizations.weight_norm(nn.Linear(4, 4, bias=False))
    m = nn.Sequential(wn, nn.Linear(4, 4, bias=False))
    part = Z.particion(m)
    assert len(part["matrices"]) == 1 and part["matrices"][0] is m[1].weight and len(part["adam"]) == 2
    opt = Z.construir(m, lr=1e-3)
    x = torch.randn(3, 4)
    opt.zero_grad(set_to_none=True)
    m(x).pow(2).sum().backward()
    opt.step()


# ------------------------------------------------------------ validacion y closure (PER-ROB-001, PER-ROB-002, PER-ROB-004, PER-ROB-005)
def test_hiperparametros_invalidos_y_rank_cero():
    m = nn.Sequential(nn.Linear(4, 4, bias=False))
    for kw in (dict(lr=0.0), dict(lr=-1e-3), dict(lr=1e-3, lr_adam=0.0), dict(lr=1e-3, wd=-0.1), dict(lr=1e-3, momentum=1.0), dict(lr=1e-3, rms=0.0)):
        with pytest.raises(ValueError):
            Z.construir(m, **kw)
    with pytest.raises(ValueError):
        Z.ZMuon([m[0].weight], lr=0.0)
    with pytest.raises(ValueError):
        Z.ZSpectron3([Z.FactorizedLinear(4, 4, rank=2)], lr=1e-3, momentum=1.5)
    with pytest.raises(ValueError):
        Z.FactorizedLinear(4, 4, rank=0)
    with pytest.raises(ValueError):
        Z.FactorizedLinear(4, 4, rank=1.5)


def test_closure_se_evalua_y_se_devuelve():
    m = nn.Sequential(nn.Linear(4, 4, bias=False), Z.FactorizedLinear(4, 4, rank=2))
    opt = Z.construir(m, lr=1e-3)
    x = torch.randn(3, 4)
    vistos = []

    def closure():
        opt.zero_grad(set_to_none=True)
        loss = m(x).pow(2).sum()
        loss.backward()
        vistos.append(float(loss.detach()))
        return loss

    out = opt.step(closure)
    assert len(vistos) == 1 and float(out.detach()) == vistos[0]
    lay = Z.FactorizedLinear(4, 4, rank=2)
    sp = Z.ZSpectron3([lay], lr=1e-3)
    assert float(sp.step(lambda: (setattr(lay.U, "grad", torch.ones(4, 2)), setattr(lay.V, "grad", torch.ones(2, 4)), torch.tensor(7.0))[-1])) == 7.0
    with pytest.raises(NotImplementedError):
        sp.add_param_group({"params": [nn.Parameter(torch.zeros(2, 2))]})


# ------------------------------------------------------------ sigma, dtype, deepcopy, reanudacion (PER-LOG-001, PER-ROB-007, PER-REC-003, PER-REC-001)
def test_factor_nulo_al_arrancar_no_deja_sigma_en_cero():
    lay = Z.FactorizedLinear(8, 6, rank=3)
    with torch.no_grad():
        lay.S.zero_()                                   # U·diag(S) = 0 en el primer paso: el vector de potencia colapsaria
    sp = Z.ZSpectron3([lay], lr=1e-2)
    lay.U.grad, lay.V.grad = torch.randn(6, 3), torch.randn(3, 8)
    sp.step()
    assert float(sp.state[lay.U]["sigmas"].abs().sum()) == 0.0 and abs(float(sp.state[lay.U]["u_pot"].norm()) - 1) < 1e-5   # reserva repuesta
    with torch.no_grad():
        lay.S.fill_(1.0)
    lay.U.grad, lay.V.grad = torch.randn(6, 3), torch.randn(3, 8)
    sp.step()
    sA, sB = (float(v) for v in sp.state[lay.U]["sigmas"])
    assert sA > 0.5 and sB > 0.5                       # sigma vuelve a leerse


def test_cambio_de_dtype_tras_el_primer_paso():
    lay = Z.FactorizedLinear(8, 6, rank=3)
    sp = Z.ZSpectron3([lay], lr=1e-2)
    lay.U.grad, lay.V.grad = torch.randn(6, 3), torch.randn(3, 8)
    sp.step()
    lay.double()
    lay.U.grad, lay.V.grad = torch.randn(6, 3, dtype=torch.float64), torch.randn(3, 8, dtype=torch.float64)
    sp.step()
    assert sp.state[lay.U]["u_pot"].dtype == torch.float64 and torch.isfinite(lay.U).all()


def test_deepcopy_conserva_capas_y_generador():
    lay = Z.FactorizedLinear(8, 6, rank=3)
    sp = Z.ZSpectron3([lay], lr=1e-2)
    sp2 = copy.deepcopy(sp)
    assert len(sp2.capas) == 1 and sp2.capas[0] is not lay and torch.equal(sp2.gen.get_state(), sp.gen.get_state())
    sp2.capas[0].U.grad, sp2.capas[0].V.grad = torch.randn(6, 3), torch.randn(3, 8)
    sp2.step()                                         # la copia puede dar pasos
    assert "sigmas" in sp2.state[sp2.capas[0].U]


def test_reanudacion_identica_cuando_una_capa_recibe_su_primer_gradiente_tras_el_checkpoint():
    def modelo():
        torch.manual_seed(5)
        return nn.ModuleDict({"a": Z.FactorizedLinear(6, 6, rank=2), "b": Z.FactorizedLinear(6, 6, rank=2)})

    def tren(corte):
        m = modelo()
        opt = Z.construir(m, lr=2e-3, seed=3)
        g = torch.Generator().manual_seed(11)
        for t in range(6):
            if t == corte:
                sd = {"m": m.state_dict(), "opt": opt.state_dict()}
                m = modelo()
                opt = Z.construir(m, lr=2e-3, seed=99)   # otra semilla: solo el estado guardado debe mandar
                m.load_state_dict(sd["m"])
                opt.load_state_dict(sd["opt"])
            x = torch.randn(4, 6, generator=g)
            opt.zero_grad(set_to_none=True)
            (m["a"](x).pow(2).sum() + (m["b"](x).pow(2).sum() if t >= 3 else 0)).backward()   # "b" recibe su primer gradiente en t=3
            opt.step()
        return m
    m1, m2 = tren(corte=None), tren(corte=2)
    assert all(torch.equal(p, q) for p, q in zip(m1.parameters(), m2.parameters()))


def test_load_state_dict_rechaza_estados_incompatibles_sin_tocar_nada():
    m = nn.Sequential(nn.Linear(4, 4, bias=False), Z.FactorizedLinear(4, 4, rank=2))
    opt = Z.construir(m, lr=1e-3)
    d = opt.state_dict()
    malo = copy.deepcopy(d)
    malo["espectral"]["muon"]["param_groups"][0]["params"] = [0, 1]      # un parametro de mas
    with pytest.raises(ValueError, match="incompatible"):
        opt.load_state_dict(malo)
    malo = copy.deepcopy(d)
    malo["espectral"]["extra"] = {}
    with pytest.raises(ValueError, match="incompatible"):
        opt.load_state_dict(malo)
    with pytest.raises(ValueError):
        opt.load_state_dict({"tipo": "mzspectral", "adamw": {}, "espectral": {"muon": {}, "spectron": {}}})
    opt.load_state_dict(d)


def test_normas_con_columna_nula_no_da_nan():
    lay = Z.FactorizedLinear(6, 6, rank=3)
    with torch.no_grad():
        lay.U[:, 0] = 0
    n = Z.normas(nn.Sequential(lay))["0"]
    assert all(torch.isfinite(torch.tensor(v)).all() for k, v in n.items() if k != "tipo")


# ------------------------------------------------------------ sello (PER-SEG-001, 003, 004, 005, 006)
def _sello(args, root):
    return subprocess.run([sys.executable, SELLO, *args, "--root", str(root)], capture_output=True, text=True)


def test_sello_ve_el_bytecode_y_los_artefactos(tmp_path):
    (tmp_path / "paq").mkdir()
    (tmp_path / "paq" / "x.py").write_text("a = 1\n")
    (tmp_path / "LICENSE").write_text("MIT\n")
    (tmp_path / ".github").mkdir()
    (tmp_path / ".github" / "ci.yml").write_text("run: x\n")
    assert _sello(["sellar"], tmp_path).returncode == 0
    sellados = [f["path"] for f in json.loads((tmp_path / "SEAL.json").read_text())["files"]]
    assert sellados == [".github/ci.yml", "LICENSE", "paq/x.py"]                 # los directorios ocultos tambien
    assert _sello(["verificar"], tmp_path).returncode == 0
    (tmp_path / "paq" / "__pycache__").mkdir()
    (tmp_path / "paq" / "__pycache__" / "x.cpython-313.pyc").write_bytes(b"\x00bytecode")
    r = _sello(["verificar"], tmp_path)
    assert r.returncode == 1 and "__pycache__/x.cpython-313.pyc" in r.stdout      # un .pyc plantado delata el arbol
    assert _sello(["sellar"], tmp_path).returncode == 2                           # y no se puede sellar sucio
    (tmp_path / "paq" / "__pycache__" / "x.cpython-313.pyc").unlink()
    (tmp_path / "paq" / "__pycache__").rmdir()
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "lib.py").write_text("b = 2\n")
    assert _sello(["verificar"], tmp_path).returncode == 1 and _sello(["sellar"], tmp_path).returncode == 2


def test_sello_tolera_crlf_y_contrasta_el_manifest(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\ny = 2\n", newline="\n")
    assert _sello(["sellar"], tmp_path).returncode == 0
    (tmp_path / "a.py").write_bytes(b"x = 1\r\ny = 2\r\n")                       # clon con autocrlf
    assert _sello(["verificar"], tmp_path).returncode == 0
    (tmp_path / "a.py").write_bytes(b"x = 1\r\ny = 3\r\n")
    assert _sello(["verificar"], tmp_path).returncode == 1
    (tmp_path / "a.py").write_bytes(b"x = 1\ny = 2\n")
    man = (tmp_path / "MANIFEST.sha256").read_text()
    (tmp_path / "MANIFEST.sha256").write_text(man.replace("a.py", "b.py"))
    r = _sello(["verificar"], tmp_path)
    assert r.returncode == 1 and "MANIFEST.sha256 MAL" in r.stdout


def test_merkle_distingue_la_hoja_duplicada_y_la_firma_ausente_no_es_presente(tmp_path):
    sys.path.insert(0, os.path.dirname(SELLO))
    import sello
    h = [{"path": p, "sha256": "ab" * 32} for p in ("a", "b", "c")]
    assert sello._merkle(h) != sello._merkle(h + [h[-1]]) and sello._merkle(h) != sello._merkle(h[:2])
    (tmp_path / "a.py").write_text("x = 1\n")
    assert _sello(["sellar"], tmp_path).returncode == 0
    s = json.loads((tmp_path / "SEAL.json").read_text())
    assert s["signature"] is None and "ausente" in _sello(["verificar"], tmp_path).stdout
    s["signature"], s["signature_public_key"], s["signature_algorithm"] = "00" * 64, "11" * 32, "Ed25519"
    (tmp_path / "SEAL.json").write_text(json.dumps(s))
    r = _sello(["verificar"], tmp_path)
    assert r.returncode == 1 and ("INVALIDA" in r.stdout or "NO verificable" in r.stdout)   # nunca "presente" sin comprobar
