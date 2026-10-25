"""Write LOCK.json for the V149 main run: runtime closure hashes (from sys.modules after importing the runner), the
native assets, corpus files and the scorer, the settings, and the decision rule and predictions. python make_lock.py
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import run  # noqa: E402,F401  (imports corpus, states, readouts, ladder, fastmem and the native stores)

SETTINGS = json.loads((HERE / "cycles" / "SETTINGS.json").read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


mods = {Path(m.__file__).resolve() for m in list(sys.modules.values())
        if getattr(m, "__file__", None) and str(Path(m.__file__).resolve()).startswith(str(ROOT))}
mods.discard(Path(__file__).resolve())
assets = [ROOT / "minifly_a_v3/package/FULL151_CANONICAL_B.npz",
          ROOT / "minifly_a_v3/package/FULL151_BIRTH_FINGERPRINT.json",
          ROOT / "minifly_cls_dev_20261017/OUTPUT_CODEBOOK.json",
          HERE / "summarize.py",                                   # scores the decision and the predictions
          HERE / "data" / "chesterton-brown.txt",
          ROOT / "minifly_v148_pretrain_probe_dev_20261025/data/burgess-busterbrown.txt",
          ROOT / "minifly_v148_pretrain_probe_dev_20261025/data/bryant-stories.txt",
          HERE / "data" / "edgeworth-parents.txt"]
closure = {str(p.relative_to(ROOT)): sha(p) for p in sorted(mods) + assets}
lock = {
    "locked": "2026-10-25",
    "design": "minifly_v149_input_ceiling_design_20261025/DESIGN.md with the cycle decisions in cycles/CYCLE2.md and CYCLE3.md",
    "main_text": "brown",
    "settings": SETTINGS,
    "decision": {"rule": "C = R1 held bits/symbol on PERM-obs; B = bigram held bits/symbol (40 KB). "
                         "Readout checks fail -> CANNOT_TEST_AS_SPECIFIED; C >= B - margin -> CONTEXT_FIRST; "
                         "else OUTPUT_FIRST", "margin_bits": 0.10},
    "predictions": {
        "P1": "C lies between unigram and B - 0.10 (CONTEXT_FIRST); C < B - 0.10 weakens it",
        "P2": "R1 on PERM-last within 0.15 of bigram and PERM-x not better than PERM-last by more than 0.10; "
              "PERM-x beating PERM-last by more than 0.10 weakens it",
        "P3": "R2 at least 0.30 worse than R1 on PERM-obs; within 0.30 weakens it",
        "scoring": "summarize.py (hashed in closure_sha256), applied once to results/main",
        "dev_informed_expectation_not_a_prediction": "FE0-x R1 beat bigram on the dev texts by 0.33 (Buster), "
                                                     "0.06 (Bryant) and 0.20 (Edgeworth); text-dependent",
    },
    "python": sys.version.split()[0],
    "packages": {m: sys.modules[m].__version__ for m in ("numpy", "scipy") if m in sys.modules},
    "closure_sha256": closure,
}
(HERE / "LOCK.json").write_text(json.dumps(lock, indent=1) + "\n")
print(len(closure), "files locked")
