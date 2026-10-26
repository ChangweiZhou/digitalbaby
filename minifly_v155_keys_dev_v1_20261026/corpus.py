"""V155 corpus: V148's record and split rules (V148 corpus.py loaded unchanged under its own name) and the V155 text
pairs, pinned by SHA-256. All texts come from NLTK's copy of Project Gutenberg (archive SHA-256 2d3c3ab5...d611).

Text A (keys learned): 40 KB taught, 5 KB calibration, 10 KB held-out.
Text B (carry-over):   10 KB taught (readout only), 5 KB calibration, 10 KB held-out.
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

EOS, records, split = _v148.EOS, _v148.records, _v148.split
D = HERE / "data"
TEXTS = {   # name: (file, sha256, role)
    "thursday": (D / "chesterton-thursday.txt", "8b431a2a8db84089cd6f83fa66610c3b12c04114609df6e90137c0c58f2c5493",
                 "A for C0 testbed and C2 dev (used before: V153 rehearsal, headroom check)"),
    "edgeworth": (D / "edgeworth-parents.txt", "543b2162782f839b9917b88956acbd7507df37c543286c16c228b36dda0cf950",
                  "B for C2 dev (used before: V149 rehearsal 2)"),
    "emma": (D / "austen-emma.txt", "71b0923d75fcf2f1faa897b22da8de8e2476d7478a2b22388eb58a3420e00215",
             "A for C3 rehearsal (unused)"),
    "ball": (D / "chesterton-ball.txt", "c6b6d4e8c09a3f49a09f37f4e47663d4d6f048b62d86c70ccafdfd8f27059a58",
             "B for C3 rehearsal (unused)"),
    "persuasion": (D / "austen-persuasion.txt", "cec097657cbfeb62aafe178973f514fa07ab66a5e28ea025ebd5d51ca870e40f",
                   "A for S6 official, only if granted (unused)"),
    "moby_dick": (D / "melville-moby_dick.txt", "dc2466ee3c2e6e2ea1f86233d142e4b5578fdc1d6f0ca657bee7572202a980c7",
                  "B for S6 official, only if granted (unused)"),
}
SIZES = {"A": (40, 5, 10), "B": (10, 5, 10)}
PAIRS = {"testbed": ("thursday", None), "technical": ("thursday", "edgeworth"), "dev": ("thursday", "edgeworth"),
         "rehearsal": ("emma", "ball"), "official": ("persuasion", "moby_dick")}


def load(name):
    path, digest, _ = TEXTS[name]
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise SystemExit(f"corpus {name}: SHA-256 mismatch")
    return raw


def splits(name, role):
    return split(records(load(name)), *SIZES[role])
