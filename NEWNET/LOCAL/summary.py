"""Table of a search round: best by cl (continual-learning score, nats, lower better) and by slope (nats per doubling, more negative better),
with the Pareto front. Usage: python3 summary.py ROUND.jsonl [TOP]"""
import sys, json
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
top = int(sys.argv[2]) if len(sys.argv) > 2 else 15
ok = [r for r in rows if "cl" in r and "slope" in r]
print(f"{len(rows)} lines, {len(ok)} complete, {len(rows) - len(ok)} failed")
def fmt(r):
    o = r["opts"]; a = r["audit"]
    return (f"{r['name']:>10} cl {r['cl']:.3f} slope {r['slope']:+.3f} ladder {'/'.join(f'{v:.2f}' for v in r['ladder'])} "
            f"forget e{r['forget']['emma']:+.2f} c{r['forget']['code']:+.2f} fanin {a['fanin']} {r['secs']:.0f}s | "
            + " ".join(f"{k}={v}" for k, v in o.items()))
print("\nbest cl:"); [print(fmt(r)) for r in sorted(ok, key=lambda r: r["cl"])[:top]]
print("\nbest slope:"); [print(fmt(r)) for r in sorted(ok, key=lambda r: r["slope"])[:top]]
front = [r for r in ok if not any(q["cl"] <= r["cl"] and q["slope"] <= r["slope"] and (q["cl"] < r["cl"] or q["slope"] < r["slope"]) for q in ok)]
print("\nPareto front:"); [print(fmt(r)) for r in sorted(front, key=lambda r: r["cl"])]
