# DECLARACIÓN DE AUTORÍA Y TITULARIDAD

**Obra:** mzspectral — Optimizadores espectrales para matrices densas y factorizadas U·diag(S)·V
**Versión declarada:** 0.1.0
**Fecha de primera fijación:** 2026-10-07 (código de `mztrain_t22/espectral.py`, sellado en el
preregistro de T22 el 2026-10-07 a las 17:45, antes de cualquier corrida) y 2026-10-08 (extracción
como paquete standalone)
**Fecha de esta declaración:** 2026-10-08
**Titular del copyright:** © 2026 MSC Star Team — Distribuido bajo licencia MIT

---

## 1. Autores y atribución

| Autor | Rol | Identificación |
|-------|-----|----------------|
| **Esraderey** | Co-titular; arquitecto de MZTrain y de la forma de tres factores U·diag(S)·V a la que se adapta el optimizador | msc.framework@gmail.com |
| **Raúl Cruz Acosta** | Co-titular; dirección del programa experimental (RFC-CALIDAD-1, T20–T22), decisión de extraer la tecnología y de su marcado | msc.framework@gmail.com |

Los titulares son **co-titulares en partes iguales (50/50)** de los derechos
patrimoniales y morales sobre la obra, salvo pacto escrito en contrario.

## 2. Origen y proceso (declaración honesta)

1. Los **algoritmos base** son de terceros y se citan en `NOTICE.md` y `PRIOR_ART.md`:
   Muon (K. Jordan et al., 2024), su escala RMS y weight decay (J. Liu et al., 2025,
   arXiv:2502.16982) y Spectron (arXiv:2602.12429, 2026). mzspectral no reivindica esos
   algoritmos.
2. Lo que es **propio** (ver `PRIOR_ART.md`): la adaptación de Spectron a la forma de tres
   factores con S entrenable como ganancia espectral en un reloj separado (AdamW), la elección
   de σ_A = ‖U·diag(S)‖₂ y σ_B = ‖diag(S)·V‖₂ como normas que acotan la actualización del
   producto a primer orden, el optimizador compuesto de dos relojes y la **evidencia
   experimental preregistrada** que fija el efecto de la receta (T22 de MZTrain: `EVIDENCIA.md`).
3. **Asistencia de IA.** La implementación, los tests y la redacción de esta versión se
   produjeron con asistencia de un sistema de IA (Claude, Anthropic) bajo la dirección de los
   titulares, a partir de una especificación preregistrada y sellada antes de correr
   (`PREREGISTRO-T22.md` del experimento T22), con revisión adversarial independiente (G4) y
   verificación mecánica (tests en CPU, equivalencia bit a bit con el código sellado). Las
   decisiones de diseño con efecto en el resultado (qué adaptar, qué dejar en AdamW, la escala
   RMS, el weight decay sin amortiguar) están declaradas en ese preregistro y asumidas por los
   titulares. Esta declaración NO afirma que el código haya sido escrito a mano por los autores.
4. No se ha copiado código de los trabajos citados; las implementaciones de referencia de Muon
   son públicas y los coeficientes de Newton-Schulz son los publicados por sus autores.

## 3. Firmas

Esta declaración surte efecto frente a terceros cuando la firmen los titulares (abajo) o cuando
el sello `SEAL.json` lleve la firma Ed25519 de una clave publicada por ellos.

- Esraderey: ______________________  fecha: __________
- Raúl Cruz Acosta: ______________________  fecha: __________
