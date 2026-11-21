"""S(d,0) in window A2 for every distance d from 1 to 11, from a run's recorded window (E63's step1_read with its list of
distances extended; nothing else changed). Usage: python3 depth_read.py <run name> ..."""
import sys, json
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE / "TISSUE_STEP1")]
import step1_read as R
import tissue_step1 as T
R.DS = tuple(range(1, 12))
def depth(name):
    f = HERE / "TISSUE_EVENT" / "runs" / name
    rec = json.loads((f / "receipt.json").read_text())
    st, _ = T.stream(rec["seed"], rec["stream"]["length"])
    z = np.load(f / "A2.npz")
    X, w0, dist = z["x"], int(z["positions"][0]), z["distance"]
    sc = R.cell_scores(X, st, w0, X.shape[0], dist, rec["seed"])
    out = {}
    for d in R.DS:
        v = sc.get(str(d), {}).get("by_lag", {}).get("0") if isinstance(sc.get(str(d)), dict) else None
        out[d] = (v["S"] if v else None, int((dist == d).sum()))
    return out
if __name__ == "__main__":
    for n in sys.argv[1:]:
        o = depth(n)
        print(n.ljust(34), " ".join(f"d{d}:{'  –  ' if s is None else f'{s:.3f}'}({c})" for d, (s, c) in o.items()))
