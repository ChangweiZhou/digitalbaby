# The tables of where.py: one per tissue (U, M), rows d (and the pool), columns lags 0..4, (a) and (b) side by side, the three seeds as
# three numbers; '*' after a seed's number where it exceeds C0(lag) + 0.02 of that seed. Then the mean and largest outputs by d, and one
# sentence per lag. usage: python3 where_table.py <U|M> -> where/TABLE_<U|M>.md
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
tis_ = sys.argv[1]; SEEDS = (73129, 93683, 116089); LAGS = "01234"
R = {s: json.load(open(os.path.join(HERE, 'where', f'{tis_}_{s}.json'))) for s in SEEDS}
names = []
for s in SEEDS:
    for r in R[s]['rows']:
        if r['d'] not in names and r['d'] != 'unreached':
            names.append(r['d'])
names = sorted([n for n in names if n != 'pool'], key=int) + ['pool']
row = lambda s, n: next((r for r in R[s]['rows'] if r['d'] == n), None)
def cell(g, n, L):
    out = []
    for s in SEEDS:
        r = row(s, n); v = None if r is None else r[g][L]
        if v is None:
            out.append('—'); continue
        out.append(f"{v['S']:.3f}" + ('*' if v['S'] > R[s]['c0'][L] + 0.02 else ''))
    return ' / '.join(out)
title = {'U': 'U: rule fixed, no control', 'M': 'M: rule fixed, gain=2 rho=1 gmax=4', 'UL': 'U over 262,144 bytes (window: the last 4,096 bytes)', 'ML': 'M over 262,144 bytes (window: the last 4,096 bytes)'}[tis_]
CLT = 'CLT_262144.json' if tis_.endswith('L') else 'CLT.json'
md = [f"## {title}", "", ("Window A2 (bytes 12,288 to 16,383) of 16,384 bytes." if not tis_.endswith("L") else "Window: bytes 258,048 to 262,143 of 262,144.") + " Each entry: seeds 73129 / 93683 / 116089. (a): the readings tool (floor 0.01). "
      "(b): no floor, each cell divided by its own standard deviation. `*`: above C0(lag) + 0.02 of that seed. `—`: no cell at that distance in that seed. "
      "pool: 128 sampled from every cell that is not a feedback cell.", "",
      "C0(lag), lags 0..4: " + "; ".join(f"seed {s}: " + ", ".join(f"{R[s]['c0'][L]:.4f}" for L in LAGS) for s in SEEDS), "",
      ("C<L(lag), lags 1..4 (every nearer byte, `where_clt.py`): " + "; ".join(f"seed {s_}: " + ", ".join(f"{v:.4f}" for v in json.load(open(os.path.join(HERE, 'where', CLT)))[str(s_)].values()) for s_ in SEEDS)
       if os.path.exists(os.path.join(HERE, 'where', CLT)) else ""), "",
      "| d | cells | " + " | ".join(f"lag {L} (a) | lag {L} (b)" for L in LAGS) + " |", "|" + "---|" * (2 + 2 * len(LAGS))]
for n in names:
    cells = ' / '.join(str(row(s, n)['cells']) if row(s, n) else '0' for s in SEEDS)
    md.append(f"| {n} | {cells} | " + " | ".join(f"{cell('a', n, L)} | {cell('b', n, L)}" for L in LAGS) + " |")
md += ["", "Mean and largest raw output of the cells at d over the window (seeds 73129 / 93683 / 116089):", "", "| d | mean output | largest output | share of outputs above 0.01 |", "|---|---|---|---|"]
for n in names:
    rs = [row(s, n) for s in SEEDS]
    md.append(f"| {n} | " + " / ".join('—' if r is None else f"{r['mean_output']:.2e}" for r in rs) + " | " + " / ".join('—' if r is None else f"{r['max_output']:.2e}" for r in rs)
              + " | " + " / ".join('—' if r is None else f"{r['share_above_001']:.3f}" for r in rs) + " |")
md += ["", "Under (b), at which distances the byte at each lag is carried above C0(lag) + 0.02:", ""]
sent = {}
for L in LAGS:
    per = {s: [n for n in names if n != 'pool' and row(s, n) and row(s, n)['b'][L] and row(s, n)['b'][L]['S'] > R[s]['c0'][L] + 0.02] for s in SEEDS}
    allsame = len({tuple(v) for v in per.values()}) == 1
    common = [n for n in names if n != 'pool' and all(n in per[s] for s in SEEDS)]
    pool = [s for s in SEEDS if row(s, 'pool')['b'][L]['S'] > R[s]['c0'][L] + 0.02]
    if L == "0":
        hi5 = {s: [n for n in names if n != 'pool' and row(s, n) and row(s, n)['b'][L] and row(s, n)['b'][L]['S'] >= 0.5] for s in SEEDS}
        md.append("- Lag 0: C0(0) = 1.000, so no score can exceed C0(0) + 0.02. Under (b), S(d, 0) is at least 0.5 at " + "; ".join(
            f"seed {s}: d = {', '.join(hi5[s]) if hi5[s] else 'none'}" for s in SEEDS) + f" (the seeds {'agree' if len({tuple(v) for v in hi5.values()}) == 1 else 'do not agree'}).")
        sent[L] = dict(at_least_half=hi5); continue
    txt = (f"- Lag {L}: " + "; ".join(f"seed {s}: d = {', '.join(per[s]) if per[s] else 'none'}" for s in SEEDS)
           + f". The three seeds {'agree' if allsame else 'do not agree'}" + (f"; common to all three: d = {', '.join(common)}" if common and not allsame else "")
           + f". The pool is above the level in {len(pool)} of 3 seeds.")
    md.append(txt); sent[L] = dict(per_seed=per, agree=allsame, common=common, pool_above=pool)
open(os.path.join(HERE, 'where', f'TABLE_{tis_}.md'), 'w').write("\n".join(md) + "\n")
json.dump(sent, open(os.path.join(HERE, 'where', f'SENTENCES_{tis_}.json'), 'w'))
print("\n".join(md))
