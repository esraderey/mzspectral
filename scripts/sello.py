"""Sello de integridad y firma de autoria de mzspectral.

(c) 2026 MSC Star Team (Esraderey, Raul Cruz Acosta). Licencia MIT (ver LICENSE).

  python scripts/sello.py sellar               genera MANIFEST.sha256 y SEAL.json (sin firma)
  python scripts/sello.py verificar            comprueba hashes, raiz de Merkle, hash canonico,
                                               MANIFEST.sha256 y, si la hay, la firma Ed25519
  python scripts/sello.py firmar --key K.pem   sella y firma con la clave privada Ed25519 del titular
  python scripts/sello.py keygen --out K.pem   genera un par Ed25519 (la privada FUERA del repositorio)

Cobertura (hallazgos PER-SEG-001/003/005/006 de la auditoria 2026-10-08): se sella TODO fichero
del arbol salvo `.git`, los artefactos de auditoria y editor (`.forja`, `.idea`), las caches de
herramientas y los dos ficheros del sello. `__pycache__`, `*.pyc`, `*.pyd`, `*.so`, `build/`,
`dist/` y `*.egg-info` NO se excluyen: si existen al sellar, `sellar` se niega (hay que limpiar); si
aparecen despues, `verificar` los ve como "nuevos" y falla. Los ficheros de texto se hashean con
los finales de linea normalizados a LF (y `.gitattributes` fija LF), asi un clon con autocrlf
verifica igual. La raiz de Merkle enlaza ruta y hash de cada hoja y no duplica la hoja impar.
La firma Ed25519 cubre el sello entero (sin los campos de firma) y se verifica contra
TRUSTED_PUBLIC_KEYS; sin `cryptography` instalado, `verificar` lo dice y devuelve 1 si hay firma.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import sys
from pathlib import Path

try:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
    HAY_CRYPTO = True
except Exception:          # pragma: no cover
    HAY_CRYPTO = False

# Claves publicas Ed25519 de confianza (hex, 32 bytes): las publican los titulares por un canal
# externo. La primera es la de MZTrain (scripts/seal.py de MZTrain).
TRUSTED_PUBLIC_KEYS = [
    "1c1765ef1730ea1173e33cebd6a909c86ae8de5dba326e60fb6e6ee5d44a4eb9",
]
EXCLUDE_DIRS = {".git", ".forja", ".idea", ".pytest_cache", ".mypy_cache", ".ruff_cache", ".benchmarks", ".tox"}
EXCLUDE_FILES = {"MANIFEST.sha256", "SEAL.json"}
SUCIO_DIRS = {"__pycache__", "build", "dist", ".eggs"}
SUCIO_SUFIJOS = {".pyc", ".pyo", ".pyd", ".so", ".dll", ".egg"}
TEXTO = {".py", ".pyi", ".md", ".rst", ".txt", ".toml", ".yaml", ".yml", ".json", ".cfg", ".ini", ".in", ".gitignore",
         ".gitattributes", ".sha256"}
TEXTO_NOMBRES = {"LICENSE", "NOTICE", ".gitignore", ".gitattributes", "MANIFEST.in"}


def _es_texto(p: Path) -> bool:
    return p.suffix.lower() in TEXTO or p.name in TEXTO_NOMBRES


def _bytes(p: Path) -> bytes:
    b = p.read_bytes()
    return b.replace(b"\r\n", b"\n") if _es_texto(p) else b


def _h(p: Path, algo: str) -> str:
    return hashlib.new(algo, _bytes(p)).hexdigest()


def _ficheros(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in EXCLUDE_DIRS)
        for fn in sorted(filenames):
            p = Path(dirpath) / fn
            if p.name in EXCLUDE_FILES and p.parent == root:
                continue
            yield p


def _sucios(root: Path) -> list:
    out = []
    for p in _ficheros(root):
        rel = p.relative_to(root)
        if any(part in SUCIO_DIRS or part.endswith(".egg-info") for part in rel.parts[:-1]) or p.suffix.lower() in SUCIO_SUFIJOS:
            out.append(rel.as_posix())
    return out


def _entradas(root: Path) -> list:
    out = []
    for p in _ficheros(root):
        out.append({"path": p.relative_to(root).as_posix(), "sha256": _h(p, "sha256"), "sha512": _h(p, "sha512"),
                    "size": len(_bytes(p))})
    return sorted(out, key=lambda e: e["path"])


def _merkle(entradas: list) -> str:
    """Raiz de Merkle sobre SHA-256: hoja = sha256(ruta \\0 sha256_hex); pares concatenados; una hoja
    impar sube sin duplicarse (asi [a,b,c] y [a,b,c,c] no coinciden)."""
    nivel = [hashlib.sha256(e["path"].encode("utf-8") + b"\0" + e["sha256"].encode("ascii")).digest() for e in entradas]
    if not nivel:
        return hashlib.sha256(b"").hexdigest()
    while len(nivel) > 1:
        sig = [hashlib.sha256(nivel[i] + nivel[i + 1]).digest() for i in range(0, len(nivel) - 1, 2)]
        if len(nivel) % 2:
            sig.append(nivel[-1])
        nivel = sig
    return nivel[0].hex()


def _canonico(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _version(root: Path) -> str:
    try:
        m = re.search(r'__version__\s*=\s*"([^"]+)"', (root / "mzspectral" / "__init__.py").read_text(encoding="utf-8"))
        return m.group(1) if m else "?"
    except OSError:
        return "?"


CAMPOS_FIRMA = ("signature_algorithm", "signature_public_key", "signature", "signature_status")


def _bytes_a_firmar(sello: dict) -> bytes:
    return _canonico({k: v for k, v in sello.items() if k not in CAMPOS_FIRMA})


def _sello(root: Path) -> dict:
    entradas = _entradas(root)
    canon = _canonico(entradas)
    return {
        "seal_version": "1.1",
        "work": {"name": "mzspectral", "long_name": "Optimizadores espectrales para matrices densas y factorizadas U·diag(S)·V",
                 "version": _version(root), "package": "mzspectral"},
        "owner": {"team": "MSC Star Team",
                  "authors": [{"name": "Esraderey", "role": "co-titular", "contact": "msc.framework@gmail.com"},
                              {"name": "Raul Cruz Acosta", "role": "co-titular", "contact": "msc.framework@gmail.com"}],
                  "copyright": "(c) 2026 MSC Star Team.", "license": "MIT (ver LICENSE)"},
        "origen": {"experimento": "T22 = E9 de RFC-CALIDAD-1 (MZTrain)", "carpeta": "D:/mztrain-evidencia-T22",
                   "sello_t22": "# v1 2026-10-07T17:45:30-0600",
                   "espectral_py_sha256": "e2f01afaf44fd91727875d396aa4684d4b7dc3606f458319859e45f47b9075f2"},
        "generated_at_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tool": {"name": "scripts/sello.py", "python": platform.python_version(), "platform": sys.platform},
        "algorithms": ["SHA-256", "SHA-512", "Merkle (SHA-256, hoja = ruta+hash)", "texto con finales de linea LF"],
        "file_count": len(entradas), "total_bytes": sum(e["size"] for e in entradas),
        "merkle_root_sha256": _merkle(entradas),
        "files_canonical_sha256": hashlib.sha256(canon).hexdigest(),
        "files_canonical_sha512": hashlib.sha512(canon).hexdigest(),
        "files": entradas,
        "signature_algorithm": None, "signature_public_key": None, "signature": None,
        "signature_status": "ausente: pendiente de la firma Ed25519 de los titulares (sello.py firmar --key)",
    }


def _escribir(root: Path, s: dict) -> None:
    (root / "SEAL.json").write_text(json.dumps(s, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    lineas = ["# mzspectral MANIFEST.sha256", "# (c) 2026 MSC Star Team. Distribuido bajo licencia MIT.",
              f"# generated_at_utc = {s['generated_at_utc']}", f"# merkle_root_sha256 = {s['merkle_root_sha256']}",
              f"# file_count = {s['file_count']}"] + [f"{e['sha256']}  {e['path']}" for e in s["files"]]
    (root / "MANIFEST.sha256").write_text("\n".join(lineas) + "\n", encoding="utf-8", newline="\n")


def sellar(root: Path, firma: dict = None) -> int:
    sucios = _sucios(root)
    if sucios:
        print(f"[sellar] ERROR: limpia antes de sellar (bytecode, builds o egg-info): {sucios[:8]}{' ...' if len(sucios) > 8 else ''}", file=sys.stderr)
        return 2
    s = _sello(root)
    if firma:
        s.update(firma)
    _escribir(root, s)
    print(f"[sellar] {s['file_count']} ficheros, {s['total_bytes']} bytes, merkle {s['merkle_root_sha256'][:16]}...; firma: {s['signature_status']}")
    return 0


def firmar(root: Path, key_path: Path) -> int:
    if not HAY_CRYPTO:
        print("[firmar] ERROR: falta 'cryptography' (pip install cryptography)", file=sys.stderr)
        return 2
    if not key_path.exists():
        print(f"[firmar] ERROR: no existe la clave privada {key_path}", file=sys.stderr)
        return 2
    priv = serialization.load_pem_private_key(key_path.read_bytes(), password=None)
    if not isinstance(priv, Ed25519PrivateKey):
        print("[firmar] ERROR: la clave privada no es Ed25519", file=sys.stderr)
        return 2
    if _sucios(root):
        return sellar(root)
    s = _sello(root)
    pub_hex = priv.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()
    sig_hex = priv.sign(_bytes_a_firmar(s)).hex()
    s.update(signature_algorithm="Ed25519", signature_public_key=pub_hex, signature=sig_hex,
             signature_status="firmado" + ("" if pub_hex in TRUSTED_PUBLIC_KEYS else " (clave NO listada en TRUSTED_PUBLIC_KEYS)"))
    _escribir(root, s)
    print(f"[firmar] firmado Ed25519; clave publica {pub_hex}")
    if pub_hex not in TRUSTED_PUBLIC_KEYS:
        print("[firmar] AVISO: anade la clave publica a TRUSTED_PUBLIC_KEYS y publicala por un canal externo")
    return 0


def keygen(out_path: Path) -> int:
    if not HAY_CRYPTO:
        print("[keygen] ERROR: falta 'cryptography' (pip install cryptography)", file=sys.stderr)
        return 2
    priv = Ed25519PrivateKey.generate()
    out_path.write_bytes(priv.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption()))
    pub_hex = priv.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()
    print(f"[keygen] clave privada: {out_path} (GUARDALA FUERA del repositorio)\n[keygen] clave publica (hex): {pub_hex}")
    return 0


def _firma_estado(s: dict) -> tuple:
    """(ok, mensaje): ok es None si no hay firma, False si es invalida o no verificable, True si vale."""
    sig, pub = s.get("signature"), s.get("signature_public_key")
    if not sig:
        return None, "ausente (el sello prueba integridad, no autoria)"
    if not HAY_CRYPTO:
        return False, "presente pero NO verificable aqui (falta cryptography)"
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(pub)).verify(bytes.fromhex(sig), _bytes_a_firmar(s))
    except Exception:
        return False, "INVALIDA"
    if pub not in TRUSTED_PUBLIC_KEYS:
        return False, f"valida pero con una clave NO listada en TRUSTED_PUBLIC_KEYS ({pub[:16]}...)"
    return True, f"valida (clave de confianza {pub[:16]}...)"


def _manifest(root: Path) -> dict:
    p = root / "MANIFEST.sha256"
    if not p.exists():
        return None
    out = {}
    for ln in p.read_text(encoding="utf-8").splitlines():
        if ln and not ln.startswith("#"):
            h, _, ruta = ln.partition("  ")
            out[ruta] = h
    return out


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
    merkle_ok = _merkle(actual) == esperado.get("merkle_root_sha256")
    canon_ok = hashlib.sha256(_canonico(actual)).hexdigest() == esperado.get("files_canonical_sha256") and len(actual) == esperado.get("file_count")
    man = _manifest(root)
    man_ok = man is not None and man == {e["path"]: e["sha256"] for e in esperado["files"]}
    integridad = not (faltan or nuevos or cambiados) and merkle_ok and canon_ok and man_ok
    print(f"[verificar] integridad: {'OK' if integridad else 'FALLA'} (faltan {faltan}, nuevos {nuevos}, cambiados {cambiados}, "
          f"merkle {'ok' if merkle_ok else 'MAL'}, canonico {'ok' if canon_ok else 'MAL'}, MANIFEST.sha256 {'ok' if man_ok else 'MAL o ausente'})")
    ok_firma, msg = _firma_estado(esperado)
    print(f"[verificar] firma de autoria: {msg}")
    return 0 if integridad and ok_firma is not False else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="sello de integridad y firma de mzspectral")
    ap.add_argument("cmd", choices=("sellar", "verificar", "firmar", "keygen"))
    ap.add_argument("--root", default=None)
    ap.add_argument("--key", default=None, help="clave privada Ed25519 (PEM) para firmar")
    ap.add_argument("--out", default=None, help="ruta de la clave privada que genera keygen")
    a = ap.parse_args(argv)
    root = Path(a.root).resolve() if a.root else Path(__file__).resolve().parent.parent
    if a.cmd == "sellar":
        return sellar(root)
    if a.cmd == "verificar":
        return verificar(root)
    if a.cmd == "firmar":
        if not a.key:
            print("[firmar] falta --key", file=sys.stderr)
            return 2
        return firmar(root, Path(a.key))
    if not a.out:
        print("[keygen] falta --out", file=sys.stderr)
        return 2
    return keygen(Path(a.out))


if __name__ == "__main__":
    sys.exit(main())
