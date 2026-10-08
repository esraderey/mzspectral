# PUBLICACIÓN DEFENSIVA — PRIOR ART

**Obra:** mzspectral — Optimizadores espectrales para matrices densas y factorizadas U·diag(S)·V
**Titulares:** MSC Star Team (Esraderey, Raúl Cruz Acosta)
**Fecha de publicación de este documento:** 2026-10-08
**Fecha cierta anterior:** 2026-10-07 17:45 (bloque v1 de `PREREGISTRO-T22.sha256` en
`D:\mztrain-evidencia-T22`, que incluye el hash de `mztrain_t22/espectral.py`, el código del que
deriva este paquete; ver `EVIDENCIA.md`)
**Sellado criptográfico:** ver `SEAL.json` y `MANIFEST.sha256`

---

## Finalidad

Documento publicado con efectos de **publicación defensiva** y **fecha cierta** sobre la
adaptación y la receta abajo descritas: constituir estado de la técnica oponible a solicitudes
de patente posteriores que reivindiquen lo mismo o equivalentes obvios, y documentar la
paternidad de la aportación de los titulares. Las descripciones permiten a un experto
reproducir cada técnica desde el código fuente (`mzspectral/optim.py`).

## Lo que NO se reivindica (estado de la técnica previo, de terceros)

- **Muon** (K. Jordan et al., 2024): momento con Nesterov y ortogonalización del momento por
  iteraciones de Newton-Schulz con los coeficientes (3,4445, −4,7750, 2,0315).
- **Escala RMS y weight decay de Muon** (J. Liu et al., 2025, arXiv:2502.16982): multiplicar la
  actualización ortogonalizada por 0,2·√max(m, n) para que su RMS iguale la de AdamW; weight
  decay desacoplado; AdamW para los parámetros que no son matrices.
- **Spectron** (arXiv:2602.12429, 2026): para un producto W = A·B, momento por factor
  ortogonalizado y paso η/(σ_A + σ_B + 1) con σ la norma espectral de cada factor estimada por
  iteración de potencia, que acota ‖ΔW‖₂ por η.
- **Forma de tres factores U·diag(S)·V con S entrenable** (MZTrain, mismos titulares, 2025:
  Invención 1 de su `PRIOR_ART.md`).

## Aportación 1 — Spectron sobre tres factores con la ganancia S en un reloj aparte

**Fecha de primera fijación:** 2026-10-07 (sellada). **Archivo:** `mzspectral/optim.py`
(`ZSpectron3`).

Spectron está escrito para dos factores A·B. En la forma U·diag(S)·V (S entrenable) no existe
un par (A, B) con W = A·B que conserve a S como parámetro. La adaptación de los titulares:

1. A primer orden ΔW = ΔU·diag(S)·V + U·diag(S)·ΔV; las normas que acotan la actualización del
   producto son ‖diag(S)·V‖₂ para el paso de U y ‖U·diag(S)‖₂ para el de V. Se definen
   **σ_A := ‖U·diag(S)‖₂ y σ_B := ‖diag(S)·V‖₂** (no ‖U‖₂ y ‖V‖₂, que ignorarían la escala que
   lleva S), estimadas por iteración de potencia con arranque caliente, **leídas antes del paso
   y con la S de antes del paso**, y U y V reciben su momento ortogonalizado dividido por
   (σ_A + σ_B + 1).
2. **S no entra en el optimizador espectral**: es 1-D, actúa como ganancia espectral y se
   entrena con AdamW en un reloj separado (su propio LR, el calibrado para AdamW), junto con el
   embedding, las normalizaciones y los sesgos. El espectral da su paso antes que AdamW en la
   misma iteración.
3. La cota publicada ‖ΔW‖₂ ≤ η pasa, con la escala RMS, a η·0,2·√max(m, n) a primer orden, y el
   weight decay se deja sin el factor 1/(σ_A + σ_B + 1) (forma publicada), lo que se declara
   como elección con su consecuencia (mayor razón decaimiento/paso en U y V).

## Aportación 2 — Optimizador compuesto de dos relojes con reparto automático

**Fecha de primera fijación:** 2026-10-07 (sellada). **Archivo:** `mzspectral/optim.py`
(`Doble`, `particion`, `construir`).

Un único objeto con `step`, `zero_grad`, `param_groups`, `state`, `state_dict` y
`load_state_dict` que reparte los parámetros de cualquier `nn.Module` por su forma (capas con
U, S, V → ZSpectron3; `nn.Linear` → ZMuon; el resto → AdamW), conserva en el grupo AdamW el
multiplicador `mult = lr_adam/lr` para calendarios que escalen todos los grupos por igual, y
garantiza reanudación bit a bit desde su estado (momentos, vectores de potencia y sigmas
incluidos). El denso y el factorizado reciben así la misma receta con un solo punto de
calibración (el LR espectral).

## Aportación 3 — Evidencia preregistrada del efecto de la receta

**Fecha:** 2026-10-07/08. **Fuente:** experimento T22 (E9 de RFC-CALIDAD-1) de MZTrain, con
preregistro sellado antes de correr, reglas de decisión mecánicas y revisión independiente;
resumen y hashes en `EVIDENCIA.md`.

Sobre un GPT de 13,5M parámetros (8 bloques, char-WT103, lote 16 × 128, fp32, RTX 4060), a
18 000 pasos y tres semillas: ZMuon mejora al denso 0,066 BPC frente a AdamW calibrado, y
ZSpectron3 mejora al factorizado 0,063 BPC frente a AdamW calibrado, con la mitad de estado de
optimizador, sin una inestabilidad en 13 corridas hasta LR 9,6e-3, y sin cerrar el hueco a
iso-parámetro entre factorizado y denso (+0,054 frente a +0,051). Esta medida, con su protocolo,
es parte de la publicación defensiva: fija lo que la receta hace y lo que no hace.

## Alcance

La aportación es la adaptación y la receta compuesta, no los algoritmos de terceros citados. La
evidencia es a una escala y en una tarea; su extrapolación no se reivindica.
