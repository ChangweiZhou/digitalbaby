"""Summary of t28_learnset.py lives (learnset_runs/res.jsonl): per net / cue the per-rule curve, least-squares slopes over the rule
number by kind (P, M, C, all), seed spread, revisit savings, keeping probes and bits. Prints markdown.
Usage: python3 t28_summary.py [learnset_runs/res.jsonl]"""
import sys, json, numpy as np
from collections import defaultdict
rows = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "learnset_runs/res.jsonl")]
def label(r):
    o = r["opts"]
    return {"12rc": "net12_rc", "12": "net12", "10": "K8 net10", "store": "storage"}[r["net"]]
G = defaultdict(dict)
for r in rows: G[(label(r), r["cue"])][r["seed"]] = r
KINDS = ("P", "M", "C", "all")
def slope(x, y):
    x, y = np.array(x, float), np.array(y, float); b = np.polyfit(x, y, 1); return b[0], b[1] + b[0] * x.min(), b[1] + b[0] * x.max()
def L(v): return 1501 if v is None else v
order = ["net12", "net12_rc", "K8 net10", "storage"]
for cue in (1, 0):
    print(f"\n## cue={cue}\n")
    for nl in order:
        if (nl, cue) not in G: continue
        S = G[(nl, cue)]; seeds = sorted(S)
        print(f"### {nl}, cue={cue}, seeds {seeds}\n")
        print("| rule | name | kind | L80 | L50 | f20 | f100 | f500 | last500 |\n|---|---|---|---|---|---|---|---|---|")
        lb = [[b for b in S[s]["blocks"] if b["tag"] == "learn"] for s in seeds]
        for i in range(20):
            bs = [x[i] for x in lb]
            fmt = lambda k: " / ".join("-" if b[k] is None else str(b[k]) for b in bs)
            print(f"| {i + 1} | {bs[0]['name']} | {bs[0]['kind']} | {fmt('L80')} | {fmt('L50')} | {fmt('f20')} | {fmt('f100')} | {fmt('f500')} | {fmt('last500')} |")
        print("\nSlopes per rule number (least squares; per seed; fitted first -> last of the kind's rules):\n")
        print("| kind | f100 slope (x20) | f100 fitted first -> last | f500 slope (x20) | L80 slope (lines/rule) | L80 not reached | L50 slope |\n|---|---|---|---|---|---|---|")
        for k in KINDS:
            cells = defaultdict(list)
            for x in lb:
                bb = [b for b in x if k == "all" or b["kind"] == k]; idx = [b["rule"] for b in bb]
                s1, a1, z1 = slope(idx, [b["f100"] for b in bb]); cells["f100"].append(f"{20 * s1:+.3f}"); cells["fl"].append(f"{a1:.2f}->{z1:.2f}")
                cells["f500"].append(f"{20 * slope(idx, [b['f500'] for b in bb])[0]:+.3f}")
                cells["L80"].append(f"{slope(idx, [L(b['L80']) for b in bb])[0]:+.1f}"); cells["nr"].append(f"{sum(b['L80'] is None for b in bb)}/{len(bb)}")
                cells["L50"].append(f"{slope(idx, [L(b['L50']) for b in bb])[0]:+.1f}")
            print(f"| {k} | " + " | ".join(" / ".join(cells[c]) for c in ("f100", "fl", "f500", "L80", "nr", "L50")) + " |")
        print("\nRevisit (500 lines; f100 / L80 first block -> revisit) and keeping probes (learning off; mean over the 20 rules):\n")
        for s, x in zip(seeds, lb):
            rv = [b for b in S[s]["blocks"] if b["tag"] == "revisit"]
            t = "; ".join(f"r{b['rule']} {x[b['rule'] - 1]['f100']}->{b['f100']} / {x[b['rule'] - 1]['L80'] or '-'}->{b['L80'] or '-'}" for b in rv)
            p20, pe = S[s]["probe_after20"], S[s]["probe_end"]
            pk = lambda p, k: np.mean([p[str(b["rule"])] for b in x if k == "all" or b["kind"] == k])
            print(f"- seed {s}: {t}. Probe after rule 20 (P / M / C / all): " + " / ".join(f"{pk(p20, k):.2f}" for k in KINDS)
                  + "; at end: " + " / ".join(f"{pk(pe, k):.2f}" for k in KINDS) + f". Bits total {S[s]['bits']['total']:,} (answer {S[s]['bits']['answer']:,},"
                  f" input {S[s]['bits']['input']:,}); {S[s]['s']} s.")
        print()
