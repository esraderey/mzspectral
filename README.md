<!--
  © 2026 MSC Star Team (Esraderey, Raúl Cruz Acosta).
  Distribuido bajo licencia MIT (ver LICENSE).
-->

# mzspectral — optimizadores espectrales para matrices densas y factorizadas v0.1.0

> **ZMuon** (Muon con la escala RMS de Moonshot) para matrices densas y **ZSpectron3** (Spectron
> adaptado a la forma de tres factores U·diag(S)·V de MZTrain) para matrices factorizadas, en un
> optimizador compuesto de dos relojes con AdamW para lo que no es matriz. Sin dependencias fuera
> de PyTorch. Con su efecto **medido, preregistrado y sellado** antes de extraerse como tecnología.

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Authorship](https://img.shields.io/badge/Authorship-MSC%20Star%20Team-blueviolet.svg)](AUTHORSHIP.md)
[![Prior Art](https://img.shields.io/badge/Prior%20Art-Published-orange.svg)](PRIOR_ART.md)
[![Evidencia](https://img.shields.io/badge/evidencia-T22%20preregistrado-informational.svg)](EVIDENCIA.md)
[![Sealed](https://img.shields.io/badge/SHA--256%2B512%2BMerkle-Sealed-success.svg)](SEAL.json)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

---

## La verdad en tres líneas

1. **Mejora a los dos tipos de modelo por igual.** Sobre la línea base sana de MZTrain (AdamW
   calibrado, recorte, rampa + coseno), a 18 000 pasos y tres semillas, ZMuon baja el BPC del
   denso 0,066 y ZSpectron3 el del factorizado 0,063, con **la mitad de estado de optimizador** y
   **cero inestabilidades** en 13 corridas hasta LR 9,6e-3 ([EVIDENCIA.md](EVIDENCIA.md)).
2. **No cierra el hueco a iso-parámetro.** El factorizado con ZSpectron3 queda por debajo del
   denso entrenado con AdamW (3/3 semillas), pero el denso con ZMuon vuelve a ganar por el mismo
   margen de antes (+0,054 frente a +0,051). Si tu modelo cabe denso, entrena denso con ZMuon.
3. **Medido a una escala** (13,5M parámetros, carácter, fp32, RTX 4060). Fuera de ahí es
   extrapolación: la receta está declarada para que la midas tú.

## Instalación

```bash
pip install -e D:\mzspectral
```

Requiere Python ≥ 3.9 y PyTorch ≥ 2.0. Tests: `python -m pytest tests` (CPU, unos segundos).

## Uso

```python
import torch
from torch import nn
import mzspectral as Z

modelo = ...                                   # cualquier nn.Module: nn.Linear -> ZMuon;
                                               # capas con U, S, V (FactorizedLinear o
                                               # ZFactorizedLinear de MZTrain) -> ZSpectron3;
                                               # embeddings, normas, sesgos y S -> AdamW
opt = Z.construir(modelo, lr=2.4e-3, lr_adam=3e-4, wd=0.01, seed=17)

for paso, (x, y) in enumerate(lotes):
    opt.zero_grad(set_to_none=True)
    perdida = modelo(x, y)
    perdida.backward()
    torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)   # la línea base sana recorta a 1,0
    opt.step()
    opt.fijar_lr(calendario(paso))             # el grupo AdamW sigue con lr*mult (mult = lr_adam/lr)

torch.save({"modelo": modelo.state_dict(), "opt": opt.state_dict()}, "ckpt.pt")   # reanudación bit a bit
```

- `lr` es el LR del optimizador espectral **en unidades de AdamW** (escala RMS 0,2·√max(m, n)):
  la rejilla de AdamW sirve de punto de partida. En T22 los óptimos fueron 1,2e-3 (denso) y
  4,8e-3 (factorizado) con el grupo AdamW a su LR calibrado (6e-4 y 3e-4).
- `Doble` no hereda de `torch.optim.Optimizer`: con un calendario propio usa `fijar_lr`; para
  otros, escribe `g["lr"]` en `opt.param_groups` (el primero es el de AdamW y lleva `mult`).
- `Z.ZMuon(params, lr)` y `Z.ZSpectron3(capas, lr)` se pueden usar sueltos.
- `Z.normas(modelo)` da σ_A, σ_B, ‖U‖₂, ‖V‖₂, rango estable y ortogonalidad por capa.

## Qué hace exactamente

Común a los dos (Jordan et al. 2024; Liu et al. 2025): momento con Nesterov (μ = 0,95),
ortogonalización del momento con 5 iteraciones de Newton-Schulz (3,4445, −4,7750, 2,0315; fp32),
actualización × 0,2·√max(m, n), weight decay desacoplado W ← W·(1 − η·wd).

**ZSpectron3** (Spectron, arXiv:2602.12429, llevado a tres factores): σ_A = ‖U·diag(S)‖₂ y
σ_B = ‖diag(S)·V‖₂ por iteración de potencia con arranque caliente (10 iteraciones al arrancar,
1 por paso), leídas antes del paso; U ← U·(1 − η·wd) − η·0,2·√max(out, r)·NS5(u_U)/(σ_A + σ_B + 1)
y V igual. S va en AdamW. Detalles, elecciones y sus consecuencias: [PRIOR_ART.md](PRIOR_ART.md)
y el preregistro de T22.

## Procedencia y marcas

| Documento | Contenido |
|---|---|
| [`LICENSE`](LICENSE) | MIT, © 2026 MSC Star Team. |
| [`AUTHORSHIP.md`](AUTHORSHIP.md) | Titularidad, origen y declaración honesta del proceso (asistencia de IA bajo dirección de los titulares). |
| [`NOTICE.md`](NOTICE.md) | Métodos de terceros adaptados (Muon, Moonshot, Spectron) y PyTorch. |
| [`PRIOR_ART.md`](PRIOR_ART.md) | Qué no se reivindica y qué sí: la adaptación a tres factores, el compuesto de dos relojes y la evidencia. |
| [`TRADEMARK.md`](TRADEMARK.md) | Marcas de los titulares y uso nominativo de «Muon» y «Spectron». |
| [`EVIDENCIA.md`](EVIDENCIA.md) | El experimento T22 de MZTrain: números, protocolo, hashes y equivalencia bit a bit con el código sellado. |
| [`SEAL.json`](SEAL.json) + [`MANIFEST.sha256`](MANIFEST.sha256) | Sello de integridad (SHA-256 + SHA-512 + Merkle); firma Ed25519 pendiente de los titulares. |

```bash
python scripts/sello.py verificar
```
