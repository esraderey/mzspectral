"""mzspectral — optimizadores espectrales para matrices densas y factorizadas U·diag(S)·V.

(c) 2026 MSC Star Team (Esraderey, Raul Cruz Acosta). Distribuido bajo licencia MIT (ver LICENSE).
Tecnologia extraida del experimento T22 (E9 de RFC-CALIDAD-1) de MZTrain: ver EVIDENCIA.md.

- ZMuon:      Muon (Jordan et al. 2024; escala RMS de Liu et al. 2025) para matrices 2-D.
- ZSpectron3: Spectron (arXiv 2602.12429) adaptado a la forma de tres factores U·diag(S)·V.
- Doble:      optimizador compuesto: AdamW para lo que no es matriz (embedding, normas, sesgos, S)
              y el espectral para las matrices; un solo step/zero_grad/state_dict.
- construir:  reparte los parametros de un modelo y monta el Doble.
- FactorizedLinear: capa de referencia W = U·diag(S)·V con init SVD y escala de energia (EPSI).
"""
from .optim import (BETAS, EPS, MOMENTO, NESTEROV, RMS, WD, Doble, ZMuon, ZSpectron3, construir, es_factorizada,
                    newton_schulz5, particion, potencia)
from .layers import FactorizedLinear
from .lecturas import normas, rango_estable

__version__ = "0.1.0"
__all__ = ["ZMuon", "ZSpectron3", "Doble", "construir", "particion", "es_factorizada", "newton_schulz5", "potencia",
           "FactorizedLinear", "normas", "rango_estable", "MOMENTO", "NESTEROV", "RMS", "WD", "BETAS", "EPS", "__version__"]
Muon = ZMuon
Spectron3 = ZSpectron3
