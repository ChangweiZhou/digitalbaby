"""Round over cell kinds (24 Nov 2026): candidates of lt2 (K populations) sampled from general dimensions; each runs the same battery
as ltb.py (life over overlapping streams -> cl; moby ladder with populations x s and bytes x s -> slope). Ranges of firing share,
fan-in, segment size, expansion and patches follow the measured ranges in 'Quantitative Neural Wiring Rules' (Gemini, 24 Nov).
Usage: python3 search_pop.py OUT N SEED [WORKERS]   (one process per candidate)  |  python3 search_pop.py --one JSON [T NT T0]"""
import sys, os, json, random, time, subprocess
import numpy as np
from concurrent.futures import ThreadPoolExecutor
here = os.path.dirname(os.path.abspath(__file__))


def sample(rng):
    K = rng.choice([1, 2, 2, 3, 3, 4])
    pops = []
    for q in range(K):
        tau = rng.choice([(1, 1), (1, 8), (2, 64), (16, 512)])
        p = dict(n=rng.choice([1024, 2048, 4096, 8192]), FP=rng.choice([4, 6, 8, 12]), FD=rng.choice([0, 10, 20, 30]), G=10,
                 B=rng.choice([32, 64, 128]), a=rng.choice([0.02, 0.05, 0.1, 0.15]), lp=rng.choice([0, 1]), ld=rng.choice([0, 1]),
                 ad=rng.choice([0, 1]), inp=1 if q == 0 else rng.choice([0, 0, 1]), rd=rng.choice([0, 1]), np=rng.choice([0, 1, 3, 7]),
                 tlo=tau[0], thi=tau[1], rho=rng.choice([0.05, 0.1, 0.2, 0.3]), sth=rng.choice([0.4, 0.6, 0.8]), dg=rng.choice([0.5, 1.0, 2.0]))
        earlier = list(range(q))
        p["prox"] = [j for j in earlier if rng.random() < 0.7] or (earlier[-1:] if not p["inp"] else [])
        if not p["inp"] and not p["prox"]: p["inp"] = 1
        p["dist"] = [j for j in range(K) if (rng.random() < (0.7 if j == q else 0.4))]
        if p["FD"] and not p["dist"]: p["dist"] = [q]
        pops.append(p)
    if not any(p["rd"] for p in pops): pops[-1]["rd"] = 1
    g = dict(NL=rng.choice([256, 512, 1024]), KIN=rng.choice([8, 16, 32]), FOI=rng.choice([16, 32]), FO=rng.choice([8, 16, 32]),
             NO=rng.choice([16, 32]), FOO=rng.choice([16, 32]), NGP=rng.choice([4, 6]), NGD=rng.choice([6, 8, 10]),
             eta=rng.choice([0.1, 0.2, 0.4]), eta_m=rng.choice([0.0, 0.02, 0.05, 0.1]), w_new=rng.choice([0.3, 0.4, 0.55]),
             eta_o=rng.choice([0.05, 0.1, 0.2, 0.4]), orep=rng.choice([0.0, 0.02, 0.1]), rho_o=rng.choice([0.25, 0.5]))
    return dict(g=g, pops=pops)


def battery(cand, T=64000, NT=10000, T0=32000):
    from lt2 import Net2
    from ltb import S, held
    out = {}; t0 = time.time()
    net = Net2(cand); order = ["shak", "emma", "code", "moby", "shak"]; tests = ["shak", "emma", "code", "moby"]; pos = {k: 0 for k in S}; mat = []
    for k in order:
        net.run(S[k][pos[k]:pos[k] + T]); pos[k] += T; mat.append([round(net.test(held(j, NT))[0], 4) for j in tests])
    end = dict(zip(tests, mat[-1]))
    out.update(cl=round(float(np.mean(mat[-1])), 4), mat=mat, forget={k: round(end[k] - mat[order.index(k)][tests.index(k)], 4) for k in ["emma", "code"]},
               audit=net.audit(), life_s=round(time.time() - t0, 1))
    xs, ys = [], []
    for s in (1, 2, 4):
        n2 = Net2(cand, scale=s); n2.run(S["moby"][:T0 * s]); ys.append(n2.test(held("moby", NT))[0]); xs.append(np.log2(s))
    out.update(slope=round(float(np.polyfit(xs, ys, 1)[0]), 4), ladder=[round(v, 4) for v in ys], secs=round(time.time() - t0, 1))
    return out


def run(job):
    name, cand = job
    r = subprocess.run(["python3", "search_pop.py", "--one", json.dumps(cand)], cwd=here, capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS="1"))
    line = [l for l in r.stdout.splitlines() if l.startswith("{")]
    res = dict(name=name, cand=cand, **(json.loads(line[-1]) if line else dict(error=r.stderr[-600:])))
    with open(OUT, "a") as f: f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    if sys.argv[1] == "--one":
        print(json.dumps(battery(json.loads(sys.argv[2]), *[int(a) for a in sys.argv[3:]])), flush=True); sys.exit()
    OUT, n, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); W = int(sys.argv[4]) if len(sys.argv) > 4 else 10
    rng = random.Random(seed); jobs = [(f"p{seed}_{i:03d}", sample(rng)) for i in range(n)]
    done = {json.loads(l)["name"] for l in open(OUT)} if os.path.exists(OUT) else set()
    with ThreadPoolExecutor(W) as ex: list(ex.map(run, [j for j in jobs if j[0] not in done]))
