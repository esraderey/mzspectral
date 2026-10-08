# NOTICE — Avisos y atribuciones

mzspectral — Optimizadores espectrales para matrices densas y factorizadas U·diag(S)·V
© 2026 MSC Star Team (Esraderey, Raúl Cruz Acosta).

Este software se distribuye bajo la licencia **MIT** (ver [`LICENSE`](LICENSE)):
permite uso, copia, modificación, redistribución y uso comercial, con la única
condición de conservar el aviso de copyright y el texto de la licencia. La autoría
se documenta en [`AUTHORSHIP.md`](AUTHORSHIP.md); lo que es adaptación de métodos
publicados y lo que es aportación propia, en [`PRIOR_ART.md`](PRIOR_ART.md); la
evidencia experimental de la que procede, en [`EVIDENCIA.md`](EVIDENCIA.md).

## 1. Métodos de terceros que este software implementa o adapta

mzspectral **no incorpora código** de estos trabajos: implementa sus algoritmos a
partir de su descripción publicada y los cita como origen. Los nombres «Muon» y
«Spectron» se usan de forma nominativa para identificar los métodos.

| Método | Autores / origen | Qué se toma |
|---|---|---|
| Muon | K. Jordan et al., 2024 (optimizador para capas ocultas de redes; implementación de referencia pública) | momento con Nesterov, ortogonalización del momento con 5 iteraciones de Newton-Schulz y los coeficientes (3,4445, −4,7750, 2,0315) |
| «Muon is Scalable for LLM Training» | J. Liu et al., 2025, arXiv:2502.16982 (Moonshot AI) | escala RMS 0,2·√max(m, n) de la actualización ortogonalizada, weight decay desacoplado, AdamW para lo que no es matriz |
| Spectron | arXiv:2602.12429, 2026 | paso de cada factor dividido por (σ_A + σ_B + 1), σ por iteración de potencia con arranque caliente; cota de la norma espectral de la actualización del producto |
| ZFactorizedLinear (forma de tres factores U·diag(S)·V con S entrenable) | MZTrain, MSC Star Team (mismos titulares), 2025 | la forma de la capa a la que se adapta Spectron; `FactorizedLinear` de este paquete es su versión mínima de referencia |

## 2. Componentes de terceros en tiempo de ejecución

| Componente | Versión mínima | Licencia | Fuente |
|---|---|---|---|
| PyTorch | ≥ 2.0 | BSD-3-Clause | https://github.com/pytorch/pytorch |

PyTorch no se redistribuye como parte de mzspectral; se referencia por dependencia.

## 3. Marcas

«mzspectral», «ZMuon», «ZSpectron3», «MZTrain» y «MSC Star Team» son marcas de MSC
Star Team; su uso se rige por [`TRADEMARK.md`](TRADEMARK.md). «Muon» y «Spectron»
designan métodos publicados por terceros y se mencionan con fines descriptivos;
PyTorch, NVIDIA y CUDA son marcas de sus titulares.

## 4. Verificación de integridad

- `MANIFEST.sha256` — hashes SHA-256 por archivo.
- `SEAL.json` — doble hash (SHA-256 + SHA-512), raíz de Merkle, marca de tiempo y
  campos de firma Ed25519 (pendientes de la firma de los titulares).

```bash
python scripts/sello.py verificar
```

## 5. Contacto

msc.framework@gmail.com
