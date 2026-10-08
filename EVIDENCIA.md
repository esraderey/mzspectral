# EVIDENCIA — de dónde sale mzspectral y qué está medido

**Experimento de origen:** T22 = E9 de RFC-CALIDAD-1 (MZTrain), carpeta `D:\mztrain-evidencia-T22`.
**Fecha de ejecución:** 2026-10-07 17:48 – 2026-10-08 01:00 (smoke + cadena), RTX 4060.
**Preregistro sellado antes de correr:** `PREREGISTRO-T22.sha256`, bloque
`# v1 2026-10-07T17:45:30-0600` (59 ficheros: harness, reglas, tests, el harness sellado de T20
del que hereda la línea base, el corpus y el paquete mztrain). Revisión independiente (G4) antes
del sello; cierre de forja COMPLETA; sin desviaciones de protocolo.

## Hashes de procedencia (SHA-256, calculados el 2026-10-08)

| Artefacto de T22 | SHA-256 |
|---|---|
| `mztrain_t22/espectral.py` (el código del que deriva `mzspectral/optim.py`) | `e2f01afaf44fd91727875d396aa4684d4b7dc3606f458319859e45f47b9075f2` |
| `PREREGISTRO-T22.md` | `1953dde0a01903e789b6bb224e36a9fee80f6474ccf4e423459747336e4e2c6d` |
| `PREREGISTRO-T22.sha256` (el sello) | `f1f080ce1a7e501482a48747332c69e13934ddd11676484379908e57b97737a0` |
| `t22_results.json` (todas las corridas, curvas y checkpoints) | `041e8219e8e31303de002b85bc9dd9e836f664e57d215ca681776336e6358099` |
| `t22_decisiones.json` (decisiones de las reglas) | `222c10d58f319ffd614326988fccc6288fdcd3ce6717bc47016147b127f57df1` |
| `T22-VEREDICTO.md` | `9641b3d5eee7898fc16d9c638bd1db54d319df0be479cadcbe402ccfc36a29d8` |

**Equivalencia bit a bit** entre este paquete y el código sellado de T22, verificada el
2026-10-08 con `scripts/equivalencia_t22.py` (misma capa `ZFactorizedLinear` de mztrain, mismos
gradientes y semillas de potencia: ZSpectron3 = Spectron de T22 en 5 pasos; ZMuon = Muon de T22 en
5 pasos; `construir` = `t22.build` sobre el GPT de mztrain en los brazos Fspec y Dsmuon, 8 pasos
cada uno, parámetros idénticos). Lo que T22 midió es, por tanto, lo que este paquete hace.

## Montaje

GPT de 8 bloques, 8 cabezas, pre-LN, cabeza atada; corpus char-WT103 (120M caracteres,
vocabulario de WT-2 + `<otro>`, validación WT-2); lote 16 × 128, fp32; semillas 17/29/43. Línea
base sana ★ de T20 en los cuatro brazos: embedding N(0, 0,02²), AdamW β = (0,9, 0,95), eps 1e-8,
wd 0,01, recorte global 1,0, rampa de 200 pasos + coseno a 0.

| Brazo | Modelo (parámetros) | Optimizador de las matrices | Grupo AdamW |
|---|---|---|---|
| Ds★ (T20) | denso d=368 (13 472 112) | AdamW, LR 6e-4 | todo |
| F★ (T20) | factorizado d=768, r=128 (13 570 816) | AdamW, LR 3e-4 | todo |
| Dsmuon (T22) | el mismo denso, mismo init | **ZMuon**, LR 1,2e-3 (calibrado por caminata) | tok, pos, LayerNorm a 6e-4 |
| Fspec (T22) | el mismo factorizado, mismo init | **ZSpectron3**, LR 4,8e-3 (calibrado por caminata) | tok, pos, LayerNorm, S a 3e-4 |

## Resultado (BPC de validación en el último paso; finales a 18 000 pasos)

| Brazo | s17 | s29 | s43 | media |
|---|---|---|---|---|
| Dsmuon | 1,6228 | 1,6259 | 1,6216 | **1,6234** |
| Fspec | 1,6749 | 1,6774 | 1,6786 | 1,6770 |
| Ds★ | 1,6847 | 1,6919 | 1,6919 | 1,6895 |
| F★ | 1,7413 | 1,7368 | 1,7421 | 1,7401 |

| Diferencia pareada por semilla | s17 | s29 | s43 | media |
|---|---|---|---|---|
| Ds★ − Dsmuon (lo que gana el denso con ZMuon) | +0,0619 | +0,0660 | +0,0703 | **+0,0661** |
| F★ − Fspec (lo que gana el factorizado con ZSpectron3) | +0,0664 | +0,0594 | +0,0635 | **+0,0631** |
| Fspec − Dsmuon (hueco a iso-parámetro con la receta espectral) | +0,0521 | +0,0515 | +0,0570 | +0,0535 |
| F★ − Ds★ (hueco a iso-parámetro con AdamW) | +0,0567 | +0,0449 | +0,0502 | +0,0506 |
| Fspec − Ds★ (factorizado espectral frente a denso AdamW) | −0,0097 | −0,0145 | −0,0133 | −0,0125 |

Etiqueta preregistrada de T22: **RECORTA** (Fspec < F★ en 3/3 por ≥ 0,02; no VICTORIA porque
pierde con el mejor denso en 3/3). MUON AYUDA AL DENSO: sí. CIERRA LA MITAD DEL HUECO: no.

Calibración (6000 pasos, semilla 17): Fspec 1,2e-3 → 1,8645; 2,4e-3 → 1,8262; **4,8e-3 →
1,8149**; 9,6e-3 → 1,8353 (F★ 1,9172). Dsmuon 6e-4 → 1,7780; **1,2e-3 → 1,7496**; 2,4e-3 →
1,7628 (Ds★ 1,8526).

Otros hechos: 0 NaN y 0 corridas degeneradas en 13 corridas; estado del optimizador 53 MiB
(Dsmuon) y 56 MiB (Fspec) frente a 103 MiB con AdamW; VRAM igual a la de Ds y F; dispersión
entre semillas 0,004; el factorizado con ZSpectron3 llega al BPC final de F★ en el paso 12 500
y al de Ds★ en el 15 500; el denso con ZMuon llega al final de Ds★ en el 11 500. Lecturas en CPU:
el rango estable de la matriz efectiva del factorizado sube de 28 / 24 / 9 / 16 (qkv / proj / fc1
/ fc2, de 128) a 36 / 26 / 41 / 39; en el denso, de 30–35 / 36 / 16 / 33 a 106–119 / 72 / 117 /
114; la parte visible del gradiente y la alineación escritura→lectura suben; U algo más ortogonal
y V algo menos que con AdamW.

## Coste y límites

- Tiempo por paso (con freno térmico, no comparable entre brazos): Fspec ≈ 2× F★ y Dsmuon ≈ 1,6×
  Ds★, por los lanzamientos de núcleos pequeños de Newton-Schulz y de la iteración de potencia,
  no por FLOPs. ZSpectron3 no sincroniza GPU→CPU (σ como tensor 0-d).
- Una escala, una tarea, fp32. El grupo AdamW no se recalibró junto al espectral. Spectron está
  adaptado (S en AdamW, Nesterov, escala RMS, weight decay sin amortiguar), no es la receta
  publicada al pie de la letra. Todo ello está declarado en el preregistro y en `PRIOR_ART.md`.

## Dónde está todo

`D:\mztrain-evidencia-T22\`: `PREREGISTRO-T22.md`, `PREREGISTRO-T22.sha256`, `t22_results.json`,
`t22_decisiones.json`, `t22_analysis.json`, `diag_t22.json`, `T22-VEREDICTO.md`,
`DESVIACIONES-T22.md`, `.forja/revisor.json`, checkpoints en `ckpt/`. Experimentos previos del
mismo programa: T20 (E1, línea base sana) y T21 (E5, curva de forma).
