import sys, json, statistics as st
B = sys.argv[1]; sys.path.insert(0, B)
import ptread, numpy as np
SEEDS = (73129, 93683, 116089)
red = lambda row: row["scores"]["summary"]["reduction_score"]
print("== sample standard deviation of E40's rows 1,024, 2,048, 4,096, 8,192; change 1,024 -> 8,192 and 4,096 -> 8,192")
for rule in "SB":
    for s in SEEDS:
        r = {x["updates"]: x for x in json.load(open(f"{B}/rec/e40_lived_{rule}_{s}.json"))["rows"]}
        v = [red(r[u]) for u in (1024, 2048, 4096, 8192)]
        e = {x["updates"]: x for x in json.load(open(f"{B}/rec/e48_{rule}_{s}_r16.json"))["rows"]}
        rmid_t = (red(e[512]) + red(e[1024])) / 2; rmid_f = (red(r[512]) + red(r[1024])) / 2
        print(rule, s, f"sd {st.stdev(v):.3f}  x sqrt(2/5) = {st.stdev(v) * (2/5) ** .5:.2f}; 1024->8192 {v[3]-v[0]:+.2f}; 4096->8192 {v[3]-v[2]:+.2f}; hint at 1024 {red(e[1024]) - red(r[1024]):.3f}; rise 512->1024 {red(e[1024]) - red(e[512]):.3f}; R-mid term {rmid_t:.2f} free {rmid_f:.2f} diff {rmid_t - rmid_f:+.2f}")
print("== smallest and largest max|w| per tensor over the six lived networks (tolerance of checks 3 and 4)")
mx = {}
for rule in "SB":
    for s in SEEDS:
        sd = ptread.load(f"{B}/w/life_{rule}_{s}.pt")
        for k, v in sd.items():
            if k != "book.weight": mx.setdefault(k, []).append(float(np.abs(v).max()))
for k, v in mx.items():
    print(f"{k}: {min(v):.4f} .. {max(v):.4f}; 0.00025 x smallest max = {0.00025 * min(v):.2e}")
print("float32 spacing below 1: 5.96e-8; in [1, 2): 1.19e-7")
# simulate check 3's arithmetic in float32 on one tensor: p0*(c) + d  against p0 + d
rng = np.random.default_rng(0)
sd = ptread.load(f"{B}/w/life_B_93683.pt")
worst = 0
for k, p0 in sd.items():
    if k == "book.weight": continue
    d = (rng.choice([-1.0, 1.0], size=p0.shape) * 0.001 * rng.uniform(0.2, 1.0, size=p0.shape)).astype(np.float32)   # a first Adam step: about the rate in size
    free = (p0 * np.float32(0.99975)).astype(np.float32) + d
    wd0 = p0 + d
    dev = np.abs((wd0 - free) - np.float32(0.00025) * p0).max()
    worst = max(worst, float(dev))
print("simulated largest deviation of (free-wd0 - free) from 0.00025 x start, float32, network B 93683:", f"{worst:.2e}")
