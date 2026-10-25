"""V153 corpus: V148's record and split rules (V148 corpus.py loaded unchanged under its own name) and the V153 texts,
pinned by SHA-256. All texts come from NLTK's copy of Project Gutenberg (archive SHA-256 2d3c3ab5...d611).
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

EOS, records, symbols, split = _v148.EOS, _v148.records, _v148.symbols, _v148.split
TEXTS = {   # name: (path, sha256, role)
    "buster": (V148 / "data" / "burgess-busterbrown.txt", _v148.TEXTS["buster"][1], "C2 dev"),
    "thursday": (HERE / "data" / "chesterton-thursday.txt",
                 "8b431a2a8db84089cd6f83fa66610c3b12c04114609df6e90137c0c58f2c5493", "C3 rehearsal (fresh)"),
    "persuasion": (HERE / "data" / "austen-persuasion.txt",
                   "cec097657cbfeb62aafe178973f514fa07ab66a5e28ea025ebd5d51ca870e40f", "S6 main, only if granted"),
}


def load(name):
    path, digest, _ = TEXTS[name]
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise SystemExit(f"corpus {name}: SHA-256 mismatch")
    return raw
