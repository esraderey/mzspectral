"""Sello de integridad de mzspectral: MANIFEST.sha256 + SEAL.json (SHA-256 + SHA-512 + raiz de
Merkle), con el mismo formato que el sello de MZTrain (scripts/seal.py de MZTrain). La firma
Ed25519 de autoria la ponen los titulares con su clave privada (fuera de este repositorio):
`python D:\\mztrain\\scripts\\seal.py sign --root D:\\mzspectral --key <clave>` regenera el sello
firmado con el mismo formato.

(c) 2026 MSC Star Team (Esraderey, Raul Cruz Acosta). Licencia MIT (ver LICENSE).

  python scripts/sello.py sellar      genera MANIFEST.sha256 y SEAL.json
  python scripts/sello.py verificar   comprueba hashes, raiz de Merkle y hash canonico; informa de la firma
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

INCLUDE_SUFFIXES = {".py", ".pyi", ".md", ".rst", ".txt", ".toml", ".yaml", ".yml", ".json", ".cfg", ".ini", ".in"}
INCLUDE_NAMES = {"LICENSE", "NOTICE", "MANIFEST.in", ".gitignore"}
EXCLUDE_DIRS = {".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".tox",
                "build", "dist", ".eggs", "node_modules", "mzspectral.egg-info", ".claude", ".forja"}
EXCLUDE_FILES = {"MANIFEST.sha256", "SEAL.json"}
EXCLUDE_SUFFIXES = {".pt", ".pth", ".bin", ".pyc", ".so", ".dll", ".png", ".jpg", ".pdf", ".zip", ".gz", ".whl", ".egg"}


def _h(path: Path, algo: str, chunk: int = 1 << 20) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _incluir(p: Path, root: Path) -> bool:
    rel = p.relative_to(root)
    if any(part in EXCLUDE_DIRS for part in rel.parts[:-1]):
        return False
    if p.name in EXCLUDE_FILES or p.suffix.lower() in EXCLUDE_SUFFIXES:
        return False
    return p.suffix.lower() in INCLUDE_SUFFIXES or p.name in INCLUDE_NAMES


def _ficheros(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDE_DIRS)
        for fn in sorted(filenames):
            p = Path(dirpath) / fn
            if _incluir(p, root):
                yield p


def _entradas(root: Path) -> list:
    out = []
    for p in _ficheros(root):
        out.append({"path": p.relative_to(root).as_posix(), "sha256": _h(p, "sha256"), "sha512": _h(p, "sha512"),
                    "size": p.stat().st_size})
    return sorted(out, key=lambda e: e["path"])


def _merkle(hojas: list) -> str:
    """Raiz de Merkle sobre SHA-256: pares concatenados (hex -> bytes); hoja impar se duplica."""
    nivel = [bytes.fromhex(h) for h in hojas] or [hashlib.sha256(b"").digest()]
    while len(nivel) > 1:
        if len(nivel) % 2:
            nivel.append(nivel[-1])
        nivel = [hashlib.sha256(nivel[i] + nivel[i + 1]).digest() for i in range(0, len(nivel), 2)]
    return nivel[0].hex()


def _canonico(entradas: list) -> bytes:
    return json.dumps(entradas, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _sello(root: Path) -> dict:
    entradas = _entradas(root)
    canon = _canonico(entradas)
    return {
        "seal_version": "1.0",
        "work": {"name": "mzspectral", "long_name": "Optimizadores espectrales para matrices densas y factorizadas U·diag(S)·V",
                 "version": "0.1.0", "package": "mzspectral"},
        "owner": {"team": "MSC Star Team",
                  "authors": [{"name": "Esraderey", "role": "co-titular", "contact": "msc.framework@gmail.com"},
                              {"name": "Raul Cruz Acosta", "role": "co-titular", "contact": "msc.framework@gmail.com"}],
                  "copyright": "(c) 2026 MSC Star Team.", "license": "MIT (ver LICENSE)"},
        "origen": {"experimento": "T22 = E9 de RFC-CALIDAD-1 (MZTrain)", "carpeta": "D:/mztrain-evidencia-T22",
                   "sello_t22": "# v1 2026-10-07T17:45:30-0600",
                   "espectral_py_sha256": "e2f01afaf44fd91727875d396aa4684d4b7dc3606f458319859e45f47b9075f2"},
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tool": {"name": "scripts/sello.py", "python": platform.python_version(), "platform": sys.platform},
        "algorithms": ["SHA-256", "SHA-512", "Merkle (SHA-256)"],
        "file_count": len(entradas), "total_bytes": sum(e["size"] for e in entradas),
        "merkle_root_sha256": _merkle([e["sha256"] for e in entradas]),
        "files_canonical_sha256": hashlib.sha256(canon).hexdigest(),
        "files_canonical_sha512": hashlib.sha512(canon).hexdigest(),
        "files": entradas,
        "signature_algorithm": None, "signature_public_key": None, "signature": None,
        "signature_status": "pendiente: firma Ed25519 de los titulares (seal.py sign de MZTrain)",
    }


def sellar(root: Path) -> int:
    s = _sello(root)
    (root / "SEAL.json").write_text(json.dumps(s, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    lineas = ["# mzspectral MANIFEST.sha256", "# (c) 2026 MSC Star Team. Distribuido bajo licencia MIT.",
              f"# generated_at_utc = {s['generated_at_utc']}", f"# merkle_root_sha256 = {s['merkle_root_sha256']}",
              f"# file_count = {s['file_count']}"] + [f"{e['sha256']}  {e['path']}" for e in s["files"]]
    (root / "MANIFEST.sha256").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print(f"[sellar] {s['file_count']} ficheros, {s['total_bytes']} bytes, merkle {s['merkle_root_sha256'][:16]}...; firma: {s['signature_status']}")
    return 0


def verificar(root: Path) -> int:
    p = root / "SEAL.json"
    if not p.exists():
        print("[verificar] falta SEAL.json", file=sys.stderr)
        return 2
    esperado = json.loads(p.read_text(encoding="utf-8"))
    actual = _entradas(root)
    ea = {e["path"]: e for e in esperado["files"]}
    aa = {e["path"]: e for e in actual}
    faltan = sorted(set(ea) - set(aa))
    nuevos = sorted(set(aa) - set(ea))
    cambiados = sorted(k for k in set(ea) & set(aa) if (ea[k]["sha256"], ea[k]["sha512"]) != (aa[k]["sha256"], aa[k]["sha512"]))
    merkle_ok = _merkle([e["sha256"] for e in actual]) == esperado.get("merkle_root_sha256")
    canon_ok = hashlib.sha256(_canonico(actual)).hexdigest() == esperado.get("files_canonical_sha256")
    ok = not (faltan or nuevos or cambiados) and merkle_ok and canon_ok
    print(f"[verificar] integridad: {'OK' if ok else 'FALLA'} (faltan {faltan}, nuevos {nuevos}, cambiados {cambiados}, "
          f"merkle {'ok' if merkle_ok else 'MAL'}, canonico {'ok' if canon_ok else 'MAL'})")
    print(f"[verificar] firma de autoria: {esperado.get('signature_status') if not esperado.get('signature') else 'presente (verificar con seal.py de MZTrain)'}")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="sello de integridad de mzspectral")
    ap.add_argument("cmd", choices=("sellar", "verificar"))
    ap.add_argument("--root", default=None)
    a = ap.parse_args(argv)
    root = Path(a.root).resolve() if a.root else Path(__file__).resolve().parent.parent
    return sellar(root) if a.cmd == "sellar" else verificar(root)


if __name__ == "__main__":
    sys.exit(main())
