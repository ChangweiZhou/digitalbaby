"""For each P/C rule of the t28 life: its (answer position i, source lag) pairs (lag = 4 + i - j for W[i] = w[j]; the inserted constant
has none), and how many earlier P/C rules used the same pair ('prior uses'); share of the rule's copied positions whose pair was used
before. Then, per net/cue/seed, least squares f100 ~ a + b*rule + c*share_seen (and f100 ~ share_seen alone) over P and C rules.
Usage: python3 t28_overlap.py [learnset_runs/res.jsonl]"""
import sys, json, numpy as np
sys.argv = [sys.argv[0]] + sys.argv[1:]
import importlib.util
spec = importlib.util.spec_from_file_location("ls", "t28_learnset.py")
src = open("t28_learnset.py").read().split("if LIST:")[0]
g = {}; exec(compile("import sys\nsys.argv=['x']\n" + src, "t28", "exec"), g)
LIFE = g["LIFE"]; name = g["name"]
def pairs(R):
    if R["kind"] == "M": return []
    out = [(i, 4 + i - j) for i, j in enumerate(R["pi"])]
    if R["kind"] == "C":
        q = R["q"]; out = [(i if i < q else i + 1, 4 + (i if i < q else i + 1) - j) for i, (_, _) in enumerate(out) for j in [R["pi"][i]]]
    return out
seen = {}; info = {}
for R in LIFE:
    ps = pairs(R)
    if ps:
        prior = [seen.get(p, 0) for p in ps]; info[R["idx"]] = dict(name=name(R), kind=R["kind"], pairs=ps, prior=prior, share=sum(x > 0 for x in prior) / len(ps))
        for p in ps: seen[p] = seen.get(p, 0) + 1
print("| rule | name | (i, lag) pairs | prior uses | share seen |\n|---|---|---|---|---|")
for k, v in info.items(): print(f"| {k} | {v['name']} | {' '.join(map(str, v['pairs']))} | {v['prior']} | {v['share']:.2f} |")
rows = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "learnset_runs/res.jsonl")]
print("\n| net | cue | seed | kinds | f100 ~ rule: slope x20 | f100 ~ share: slope | f100 ~ rule + share: rule x20 / share | f500 same |\n|---|---|---|---|---|---|---|---|")
for r in rows:
    if r["net"] == "store" and r["seed"] != 1: continue
    for kinds in ("P", "C", "PC"):
        bb = [b for b in r["blocks"] if b["tag"] == "learn" and b["kind"] in kinds]
        x = np.array([b["rule"] for b in bb], float); s = np.array([info[b["rule"]]["share"] for b in bb]); out = []
        for m in ("f100", "f500"):
            y = np.array([b[m] for b in bb])
            b1 = np.polyfit(x, y, 1)[0]; b2 = np.polyfit(s, y, 1)[0] if s.std() > 0 else float("nan")
            A = np.c_[np.ones_like(x), x, s]; c = np.linalg.lstsq(A, y, rcond=None)[0]
            out.append((b1, b2, c))
        (a1, a2, ac), (_, _, bc) = out
        print(f"| {r['net']} | {r['cue']} | {r['seed']} | {kinds} | {20 * a1:+.2f} | {a2:+.2f} | {20 * ac[1]:+.2f} / {ac[2]:+.2f} | {20 * bc[1]:+.2f} / {bc[2]:+.2f} |")
