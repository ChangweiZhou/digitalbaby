#!/usr/bin/env python3
"""nano/ref2, PLAN_2.md section 3: the systems of cells against the network's seeds, byte by byte, from the kept files.
    python3 compare.py            # writes res/compare.json and prints the tables
d = loss of the cells - loss of the network (negative: the cells are lower). Standard errors: as if bytes were independent, and from the
means of the 111 whole blocks of 1,000 bytes (the last 540 bytes are left out of that error only)."""
import glob, json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); R = os.path.join(HERE, "res"); N = 111540; BL = 1000
ld = lambda f: np.fromfile(os.path.join(R, f), np.float32).astype(np.float64)
def stat(d, sub=None):
    """mean of d on the chosen bytes, its two standard errors."""
    idx = np.arange(N) if sub is None else sub; v = d[idx]; nb = N // BL
    bm = np.array([d[idx[(idx >= b * BL) & (idx < (b + 1) * BL)]].mean() for b in range(nb)])
    return dict(n=int(len(v)), mean=float(v.mean()), se_independent=float(v.std(ddof=1) / np.sqrt(len(v))), se_blocks=float(bm.std(ddof=1) / np.sqrt(nb)), blocks=nb,
                share_of_blocks_below_zero=float((bm < 0).mean()))
seeds = sorted(int(os.path.basename(f)[5:-4]) for f in glob.glob(os.path.join(R, "nll_s*.f32"))); nets = {s: ld(f"nll_s{s}.f32") for s in seeds}
cells = {f"{n}_{c}": ld(f"cells_{n}_{c}.f32") for n in ("T2", "big", "all") for c in ("cell1", "cell2") if os.path.exists(os.path.join(R, f"cells_{n}_{c}.f32"))}
for a in list(nets.values()) + list(cells.values()): assert len(a) == N and np.isfinite(a).all()
e8 = np.arange(0, N, 8); out = dict(seeds=seeds, n=N, every_eighth=int(len(e8)), block=BL)
out["network"] = {str(s): dict(all=float(v.mean()), every_eighth=float(v[e8].mean())) for s, v in nets.items()}
if len(seeds) > 1:
    m = np.array([v.mean() for v in nets.values()]); m8 = np.array([v[e8].mean() for v in nets.values()]); mean_net = np.mean(list(nets.values()), 0)
    out["network_over_seeds"] = dict(all=dict(mean=float(m.mean()), sd=float(m.std(ddof=1)), lowest=float(m.min()), highest=float(m.max())),
                                     every_eighth=dict(mean=float(m8.mean()), sd=float(m8.std(ddof=1)), lowest=float(m8.min()), highest=float(m8.max())),
                                     every_eighth_minus_all=stat(np.where(np.arange(N) % 8 == 0, mean_net, np.nan), e8)["mean"] - float(mean_net.mean()))
    out["seed_against_seed"] = {f"{a}-{b}": dict(all=stat(nets[a] - nets[b]), every_eighth=stat(nets[a] - nets[b], e8)) for i, a in enumerate(seeds) for b in seeds[i + 1:]}
    nets["mean_of_seeds"] = mean_net
out["cells"] = {k: dict(all=float(v.mean()), every_eighth=float(v[e8].mean())) for k, v in cells.items()}
out["cells_minus_network"] = {k: {str(s): dict(all=stat(v - w), every_eighth=stat(v - w, e8)) for s, w in nets.items()} for k, v in cells.items()}
json.dump(out, open(os.path.join(R, "compare.json"), "w"), indent=1)
print("network, nats a byte:", {s: (round(v["all"], 4), round(v["every_eighth"], 4)) for s, v in out["network"].items()}, "(all bytes, every eighth)")
if "network_over_seeds" in out: print("over seeds:", json.dumps(out["network_over_seeds"]))
print("\n| cells | against | bytes | cells | network | d = cells - network | s.e. independent | s.e. blocks | d / s.e. blocks |\n|---|---|---|---|---|---|---|---|---|")
for k, v in out["cells_minus_network"].items():
    for s, r in v.items():
        for part in ("all", "every_eighth"):
            q = r[part]; c = out["cells"][k][part]; print(f"| {k} | {s} | {part} ({q['n']}) | {c:.4f} | {c - q['mean']:.4f} | {q['mean']:+.4f} | {q['se_independent']:.4f} | {q['se_blocks']:.4f} | {q['mean'] / q['se_blocks']:+.1f} |")
if "seed_against_seed" in out:
    print("\n| seed - seed | d all | s.e. blocks | d every eighth | s.e. blocks |\n|---|---|---|---|---|")
    for k, v in out["seed_against_seed"].items(): print(f"| {k} | {v['all']['mean']:+.4f} | {v['all']['se_blocks']:.4f} | {v['every_eighth']['mean']:+.4f} | {v['every_eighth']['se_blocks']:.4f} |")
