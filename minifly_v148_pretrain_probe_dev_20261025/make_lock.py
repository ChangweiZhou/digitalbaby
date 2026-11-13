"""Write LOCK.json for the V148 main run: runtime closure hashes (from sys.modules after importing the runner),
loaded assets, corpus files, settings and the operationalised predictions. python make_lock.py
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import run  # noqa: E402,F401  (imports corpus, ladder, probe, fastmem and the native stores)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


mods = {Path(m.__file__).resolve() for m in list(sys.modules.values())
        if getattr(m, "__file__", None) and str(Path(m.__file__).resolve()).startswith(str(ROOT))}
mods.discard(Path(__file__).resolve())
assets = [ROOT / "minifly_a_v3/package/FULL151_CANONICAL_B.npz",
          ROOT / "minifly_a_v3/package/FULL151_BIRTH_FINGERPRINT.json",
          ROOT / "minifly_cls_dev_20261017/OUTPUT_CODEBOOK.json",
          HERE / "summarize.py"] + sorted((HERE / "data").glob("*.txt"))  # summarize.py scores the predictions
closure = {str(p.relative_to(ROOT)): sha(p) for p in sorted(mods) + assets}
lock = {
    "locked": "2026-10-25",
    "design": "minifly_v148_pretrain_probe_design_20261025/DESIGN.md, with the cycle-2 protocol decisions (cycles/CYCLE2.md)",
    "main_text": "alice",
    "settings": {"prefixes": "0,5,10,20,40", "passes": "1,2,4", "calib_kb": 5, "held_kb": 10, "gen": 50},
    "predictions": {
        "P1": "held bits/byte at 40 KB (pass 1) is at most the bigram's and not below PPM-3; below the best of PPM "
              "orders 3-5 = weakened (better than predicted); above bigram = not as predicted (worse)",
        "P2": "at 40 KB (pass 1), taught bits/byte on BOTH the last and the first 5 KB of the train pool is at least "
              "P2_gap_bits below held bits/byte (last 5 KB: something memorised; first 5 KB: not erased)",
        "P2_gap_bits": 0.25,
        "P3": "MiniFly's held improvement from 5 KB to 40 KB (pass 1) is smaller than the improvement of every PPM "
              "rung (orders 2-5) over the same range",
        "scoring": "summarize.py (hashed in closure_sha256), applied once to results/main",
        "dev_informed_expectation_not_a_prediction": "cycle 2 suggests P1 fails on the worse side (MiniFly worse than bigram)",
    },
    "python": sys.version.split()[0],
    "packages": {m: sys.modules[m].__version__ for m in ("numpy", "scipy") if m in sys.modules},
    "closure_sha256": closure,
}
(HERE / "LOCK.json").write_text(json.dumps(lock, indent=1) + "\n")
print(len(closure), "files locked")
