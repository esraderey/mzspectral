# Changelog

Todos los cambios notables en mzspectral se documentan en este archivo.

El formato está basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.0.0/),
y este proyecto adhiere a [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-10-08

### Auditoría del 2026-10-08 (maestranza + peritaje + crisol + revisión matemática; `.forja/auditoria-2026-10-08`)
- Sello: cubre todo el árbol (también directorios ocultos); `sellar` se niega con bytecode, builds o
  egg-info presentes y `verificar` los delata como ficheros nuevos; hashes con finales de línea
  normalizados (más `.gitattributes` con LF); raíz de Merkle con ruta en la hoja y sin duplicar la
  impar; contraste de `MANIFEST.sha256`; firma Ed25519 verificada contra claves de confianza (o
  declarada no verificable), nunca «presente» sin comprobar; `firmar` y `keygen`.
- `particion`: un parámetro compartido entra una sola vez; la cabeza atada a un `nn.Embedding` va a
  AdamW; `excluir` es recursivo; los pesos parametrizados (`weight_norm`) van a AdamW.
- `ZSpectron3`: vector de reserva si la iteración de potencia colapsa con un factor nulo (σ no
  queda atascada en 0); estado del generador en `state_dict` (reanudación idéntica aunque una capa
  reciba su primer gradiente tras el checkpoint); `deepcopy`/pickle conservan capas y generador;
  tolera cambio de dtype o dispositivo; `add_param_group` posterior lanza error.
- `ZMuon`, `ZSpectron3`, `Doble`: `step(closure)` evalúa y devuelve la pérdida; validación de
  `lr`, `lr_adam`, `momentum`, `weight_decay` y `rms`; `Doble.load_state_dict` comprueba todo antes
  de cargar.
- `FactorizedLinear`: `rank` < 1 lanza error. `normas`: sin NaN con columnas nulas; `eye` en el
  dispositivo del modelo.
- `equivalencia_t22.py`: comprobaciones explícitas (valen con `python -O`) y verificación del
  SHA-256 del código sellado de T22 antes de comparar.
- Documentación (revisión matemática): el diferencial del producto lleva tres términos (el de S lo
  mueve AdamW, sin el factor espectral); el weight decay decae cada factor, y en el factorizado el
  producto decae por (1 − lr·wd)²(1 − lr_adam·wd): en T22, 8,2× el del denso por paso y un
  encogimiento acumulado de W de 0,41 frente a 0,90–0,95 en los otros brazos, declarado como límite
  de la atribución; Newton-Schulz deja la banda [0,68, 1,13] solo para σᵢ/‖G‖_F ≳ 2e-3; RMS medido
  0,17–0,20·lr.
- `.gitignore` (bytecode, builds, `.idea`, `.forja`). 10 anclas nuevas de tests.

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
