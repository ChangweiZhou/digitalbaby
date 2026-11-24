"""Search rounds over general dimensions of the local tissue (24 Nov 2026). Candidates are sampled from SPACE (or listed in a file),
run through ltb.py in parallel (one process each, OMP_NUM_THREADS=1), results appended to OUT (one JSON line per candidate).
Usage: python3 search.py OUT N_CANDIDATES SEED [WORKERS] [battery args...]   |   python3 search.py OUT --file CANDS.txt [WORKERS] [...]"""
import sys, os, json, subprocess, random
from concurrent.futures import ThreadPoolExecutor

SPACE = dict(R=[1, 2, 3, 4, 6, 8], C=[128, 256, 512], M=[8, 16, 32], S=[6, 10, 16, 24], P=[4, 6, 8, 12], B=[16, 32, 64, 128],
             KB=[1, 2, 4, 8], NG=[4, 6, 8], FO=[8, 12, 16, 24], FOP=[8, 12, 16, 24, 32], NO=[8, 12, 16, 24], FOO=[8, 12, 24],
             lat=[0, 1], td=[0, 1], outr=[0, 1], graded=[0, 1], rho_p=[0.05, 0.1, 0.2, 0.3], rho_d=[0.05, 0.1, 0.15, 0.25, 0.4],
             rho_o=[0.1, 0.25, 0.5], eta=[0.1, 0.2, 0.4], eta_m=[0.0, 0.02, 0.05, 0.1], th_on=[0.5, 0.75, 0.9], th_m=[0.3, 0.5, 0.7],
             eta_o=[0.05, 0.1, 0.2, 0.4], w_new=[0.3, 0.4, 0.55])
here = os.path.dirname(os.path.abspath(__file__))


def sample(rng):
    return {k: rng.choice(v) for k, v in SPACE.items()}


def run(job):
    name, opts, extra = job
    o = " ".join(f"{k}={v}" for k, v in opts.items())
    r = subprocess.run(["python3", "ltb.py", name, o] + extra, cwd=here, capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS="1"))
    line = [l for l in r.stdout.splitlines() if l.startswith("{")]
    res = json.loads(line[-1]) if line else dict(name=name, opts=opts, error=r.stderr[-500:])
    with open(OUT, "a") as f: f.write(json.dumps(res) + "\n")
    return res


if __name__ == "__main__":
    OUT = sys.argv[1]
    if sys.argv[2] == "--file":
        cands = [json.loads(l) for l in open(sys.argv[3]) if l.strip()]; rest = sys.argv[4:]
        jobs = [(c["name"], c["opts"]) for c in cands]
    else:
        n, seed = int(sys.argv[2]), int(sys.argv[3]); rest = sys.argv[4:]; rng = random.Random(seed)
        jobs = [(f"r{seed}_{i:03d}", sample(rng)) for i in range(n)]
    W = int(rest[0]) if rest and "=" not in rest[0] else 10
    extra = [a for a in rest if "=" in a]
    done = set()
    if os.path.exists(OUT): done = {json.loads(l)["name"] for l in open(OUT) if l.strip()}
    jobs = [(nm, o, extra) for nm, o in jobs if nm not in done]
    with ThreadPoolExecutor(W) as ex: list(ex.map(run, jobs))
