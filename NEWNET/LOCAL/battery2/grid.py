"""Battery v2, scaling part (25 Nov 2026): network size and data on separate axes, two seeds, one fit per learner.
  L = a + b*log2(size) + c*log2(data)   (b: per doubling of size; c: per doubling of data), plus time per byte for each factor.
Held-out: the middle 10,000 bytes of Moby Dick, never read. Each grid point is a fresh learner reading the first D bytes.
Learners:
  net13 (positive control; non-compliant, used only to check that this grid recovers a known size term): t26_calib.py run as a
        subprocess, size = cells per column M (16, 32, 64), 7-bit bytes as in its original runs.
  lt7:  size = scale of tissue / inhibitory / relay cell counts (1, 2, 4), 8-bit bytes.
Usage: python3 grid.py net13|lt7 OUT.jsonl [workers]   -> one JSON line per point; then: python3 grid.py fit OUT.jsonl"""
import sys, os, json, time, subprocess, itertools
import numpy as np
from concurrent.futures import ThreadPoolExecutor
NEW = "/Users/pencilbard/Downloads/digitalbaby_newnet/NEWNET"; HERE = os.path.dirname(os.path.abspath(__file__))
MOBY = NEW + "/../minifly_v155_keys_dev_v1_20261026/data/melville-moby_dick.txt"
SIZES, DATA, SEEDS = (1, 2, 4), (16000, 32000, 64000), (1, 2)
T26 = HERE + "/t26_mac.py"
NET13 = "net=13 L=16 cC=64 joint=1 mesh=1 mesh_base=0 mesh_dag=5 c_k_win=4 c_th_on=0.9 c_th_match=0.75"


def point_net13(s, D, seed):
    args = f"stream=moby T={D} NT=10000 {NET13} M={16 * s} seed={seed}".split()
    t = time.time(); r = subprocess.run(["python3", T26] + args, cwd=NEW, capture_output=True, text=True,
                                        env=dict(os.environ, OMP_NUM_THREADS="1", PYTHONPATH=NEW))
    j = json.loads([l for l in r.stdout.splitlines() if l.startswith("{")][-1]); sec = time.time() - t
    return dict(learner="net13", size=16 * 64 * 16 * s, s=s, D=D, seed=seed, nats=j["nats"]["temp"], secs=round(sec, 1), us_per_byte=round(1e6 * sec / (D + 10000), 1))


def point_lt7(s, D, seed):
    code = f"""
import sys, time, json, numpy as np; sys.path.insert(0, {HERE + '/../lt7'!r}); from lt7 import Net7
x = np.frombuffer(open({MOBY!r}, 'rb').read(), np.uint8); m = len(x) // 2; held = x[m:m + 10000]
n = Net7(dict(ntis=2048, ninh=512, nrly=512, seed={seed}), scale={s}); t = time.time(); n.run(x[:{D}]); tr = time.time() - t
print(json.dumps(dict(nats=n.test(held)[0], train_us_per_byte=1e6 * tr / {D})))"""
    t = time.time(); r = subprocess.run(["python3", "-c", code], capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS="1"))
    j = json.loads([l for l in r.stdout.splitlines() if l.startswith("{")][-1]); sec = time.time() - t
    return dict(learner="lt7", size=(2048 + 512 + 2 * 512) * s, s=s, D=D, seed=seed, nats=round(j["nats"], 4), secs=round(sec, 1), us_per_byte=round(j["train_us_per_byte"], 1))


def fit(path):
    R = [json.loads(l) for l in open(path)]
    for lr in sorted({r["learner"] for r in R}):
        P = [r for r in R if r["learner"] == lr]; n = np.log2([r["size"] for r in P]); d = np.log2([r["D"] for r in P]); y = np.array([r["nats"] for r in P])
        X = np.c_[np.ones_like(n), n, d]; w, *_ = np.linalg.lstsq(X, y, rcond=None); res = y - X @ w; r2 = 1 - res @ res / ((y - y.mean()) @ (y - y.mean()))
        cov = (res @ res) / max(len(y) - 3, 1) * np.linalg.inv(X.T @ X); se = np.sqrt(np.diag(cov))
        # seed noise: mean absolute difference between the two seeds at the same point
        pairs = {}; [pairs.setdefault((r["s"], r["D"]), []).append(r["nats"]) for r in P]
        sn = np.mean([abs(v[0] - v[1]) for v in pairs.values() if len(v) == 2]) if any(len(v) == 2 for v in pairs.values()) else float("nan")
        tb = np.log2([r["us_per_byte"] for r in P]); wt, *_ = np.linalg.lstsq(X, tb, rcond=None)
        print(f"{lr}: {len(P)} points  size term b {w[1]:+.4f} (se {se[1]:.4f})  data term c {w[2]:+.4f} (se {se[2]:.4f})  R2 {r2:.3f}  seed gap {sn:.4f}")
        print(f"   time per byte exponent: in size {wt[1]:+.2f}, in data {wt[2]:+.2f}")
        for (s, D), v in sorted(pairs.items()): print(f"   s={s} D={D // 1000}k  nats {np.round(v, 4).tolist()}")


if __name__ == "__main__":
    if sys.argv[1] == "fit": fit(sys.argv[2]); sys.exit()
    lr, out = sys.argv[1], sys.argv[2]; W = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    fn = point_net13 if lr == "net13" else point_lt7
    done = set()
    if os.path.exists(out): done = {(j["learner"], j["s"], j["D"], j["seed"]) for j in map(json.loads, open(out))}
    jobs = [p for p in itertools.product(SIZES, DATA, SEEDS) if (lr, *p) not in done]
    jobs.sort(key=lambda p: -p[0] * p[1])                     # costliest first
    def run(p):
        r = fn(*p)
        with open(out, "a") as f: f.write(json.dumps(r) + "\n")
        print(json.dumps(r), flush=True)
    with ThreadPoolExecutor(W) as ex: list(ex.map(run, jobs))
