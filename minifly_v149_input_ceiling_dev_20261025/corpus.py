"""V149 corpus: the V148 record and split rules (imported unchanged) plus the V149 main text, pinned by SHA-256.

Texts come from NLTK's copy of Project Gutenberg (archive SHA-256 2d3c3ab5...d611). The three V148 texts are read from
the V148 folder; the new main text is stored here.
"""
import hashlib
import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
V148 = HERE.parent / "minifly_v148_pretrain_probe_dev_20261025"
_spec = importlib.util.spec_from_file_location("v148_corpus", V148 / "corpus.py")
_v148 = importlib.util.module_from_spec(_spec)
sys.modules["v148_corpus"] = _v148
_spec.loader.exec_module(_v148)

EOS, records, symbols, split, prefixes = _v148.EOS, _v148.records, _v148.symbols, _v148.split, _v148.prefixes
TEXTS = {   # name: (path, sha256)
    "buster": (V148 / "data" / "burgess-busterbrown.txt", _v148.TEXTS["buster"][1]),
    "bryant": (V148 / "data" / "bryant-stories.txt", _v148.TEXTS["bryant"][1]),
    "brown": (HERE / "data" / "chesterton-brown.txt", "67027976205b4b30e367c704807f56c8ffac3e8ea6037d993e57ab1fe28447ec"),
}


def load(name):
    path, digest = TEXTS[name]
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise SystemExit(f"corpus {name}: SHA-256 mismatch")
    return raw
