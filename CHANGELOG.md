# Changelog

Todos los cambios notables en mzspectral se documentan en este archivo.

El formato está basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.0.0/),
y este proyecto adhiere a [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-10-08

### Primera extracción como tecnología standalone
- `ZMuon`: Muon (momento Nesterov 0,95, Newton-Schulz de 5 iteraciones, escala RMS 0,2·√max(m,n),
  weight decay desacoplado) para matrices 2-D.
- `ZSpectron3`: Spectron adaptado a la forma de tres factores U·diag(S)·V: σ_A = ‖U·diag(S)‖₂ y
  σ_B = ‖diag(S)·V‖₂ por iteración de potencia con arranque caliente, leídas antes del paso; paso
  de U y V dividido por (σ_A + σ_B + 1); S en AdamW.
- `Doble` + `construir`: reparto automático de parámetros (densas → ZMuon, factorizadas →
  ZSpectron3, resto → AdamW con `mult` para calendarios de dos relojes); `state_dict` completo,
  reanudación bit a bit; el espectral da su paso antes que AdamW.
- `FactorizedLinear`: capa de referencia sin dependencias (init SVD con escala de energía).
- `normas` / `rango_estable`: lecturas espectrales de un modelo.
- Equivalencia bit a bit con el código sellado del experimento T22 de MZTrain
  (`mztrain_t22/espectral.py`, SHA-256 `e2f01afa…`), verificada el 2026-10-08 (ver EVIDENCIA.md).
- 10 tests en CPU.
