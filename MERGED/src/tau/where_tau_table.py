# Tables of where_tau.py: python3 where_tau_table.py <prefix, e.g. M_q16> -> where_tau/TABLE_<prefix>.md (and SENT_<prefix>.json)
# One row per distance d (and the pool), lags 0..4, (a) the readings tool (floor 0.01) and (b) no floor, each cell divided by its own
# standard deviation; seeds 73129 / 93683 / 116089. Marks after a seed's number: '*' above C0(lag) + 0.02; '+' above C<L(lag) + 0.02
# (C<L from where/CLT.json, the stream of window A2; for lag 1 C<1 = C0). Then the mean output by d, the divergence after 512 bytes,
# and per lag the distances where (b) is above C<L(lag) + 0.02 (lag >= 1), and for lag 0 where S(d, 0) >= 0.5.
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
pre = sys.argv[1]; SEEDS = (73129, 93683, 116089); LAGS = "01234"
R = {s: json.load(open(os.path.join(HERE, 'where_tau', f'{pre}_{s}.json'))) for s in SEEDS}
CLT = json.load(open(os.path.join(HERE, 'where', 'CLT.json')))
clt = lambda s, L: R[s]['c0'][L] if L in ('0', '1') else CLT[str(s)][L]
names = []
for s in SEEDS:
    for r in R[s]['rows']:
        if r['d'] not in names and r['d'] not in ('unreached', 'pool'):
            names.append(r['d'])
names = sorted(names, key=int) + ['pool']
row = lambda s, n: next((r for r in R[s]['rows'] if r['d'] == n), None)
def mark(s, L, v):
    m = ''
    if L != '0' and v > R[s]['c0'][L] + 0.02:
        m += '*'
    if L != '0' and v > clt(s, L) + 0.02:
        m += '+'
    return m
def cell(g, n, L):
    out = []
    for s in SEEDS:
        r = row(s, n); v = None if r is None else r[g][L]
        out.append('—' if v is None else f"{v['S']:.3f}{mark(s, L, v['S'])}")
    return ' / '.join(out)
md = [f"## {pre}: {R[SEEDS[0]]['args']}", "", "Window A2 (bytes 12,288 to 16,383) of 16,384 bytes. Each entry: seeds 73129 / 93683 / 116089. (a): the readings tool (floor 0.01). "
      "(b): no floor, each cell divided by its own standard deviation. `*`: above C0(lag) + 0.02; `+`: above C<L(lag) + 0.02 (for lag 1 the two are one level). "
      "pool: 128 sampled from every cell that is not a feedback cell.", "",
      "Time constants of the cells (count by value), seed 73129: " + ", ".join(f"{k}: {v}" for k, v in R[SEEDS[0]]['tau_counts'].items()), "",
      "Divergence of two copies (one cell's potential changed by 1e-9, 4,608 bytes before the end), largest output difference after 1 / 4 / 16 / 64 / 256 / 512 bytes: "
      + "; ".join(f"seed {s}: " + " / ".join(f"{R[s]['divergence'][k]:.1e}" for k in ('1', '4', '16', '64', '256', '512')) for s in SEEDS), "",
      "| d | cells | " + " | ".join(f"lag {L} (a) | lag {L} (b)" for L in LAGS) + " |", "|" + "---|" * (2 + 2 * len(LAGS))]
for n in names:
    md.append(f"| {n} | " + ' / '.join(str(row(s, n)['cells']) if row(s, n) else '0' for s in SEEDS) + " | " + " | ".join(f"{cell('a', n, L)} | {cell('b', n, L)}" for L in LAGS) + " |")
md += ["", "| d | mean output | share of outputs above 0.01 | mean tau of the cells at d |", "|---|---|---|---|"]
for n in names:
    rs = [row(s, n) for s in SEEDS]
    md.append(f"| {n} | " + " / ".join('—' if r is None else f"{r['mean_output']:.2e}" for r in rs) + " | " + " / ".join('—' if r is None else f"{r['share_above_001']:.3f}" for r in rs)
              + " | " + " / ".join(f"{R[s]['tau_by_distance'].get(n, float('nan')):.1f}" if n != 'pool' else '—' for s in SEEDS) + " |")
md += ["", "Under (b):", ""]
sent = {}
for L in LAGS:
    if L == '0':
        per = {s: [n for n in names if n != 'pool' and row(s, n) and row(s, n)['b'][L] and row(s, n)['b'][L]['S'] >= 0.5] for s in SEEDS}
        d124 = {s: [round(row(s, n)['b']['0']['S'], 3) if row(s, n) else None for n in ('1', '2', '4')] for s in SEEDS}
        md.append("- Lag 0: S(d, 0) >= 0.5 at " + "; ".join(f"seed {s}: d = {', '.join(per[s]) or 'none'}" for s in SEEDS) + ". S(1, 0), S(2, 0), S(4, 0): "
                  + "; ".join(f"seed {s}: {d124[s]}" for s in SEEDS) + ".")
        sent[L] = dict(at_least_half=per, d124=d124); continue
    per = {s: [n for n in names if n != 'pool' and row(s, n) and row(s, n)['b'][L] and row(s, n)['b'][L]['S'] > clt(s, L) + 0.02] for s in SEEDS}
    per0 = {s: [n for n in names if n != 'pool' and row(s, n) and row(s, n)['b'][L] and row(s, n)['b'][L]['S'] > R[s]['c0'][L] + 0.02] for s in SEEDS}
    common = [n for n in names if n != 'pool' and all(n in per[s] for s in SEEDS)]
    pool = [s for s in SEEDS if row(s, 'pool')['b'][L]['S'] > clt(s, L) + 0.02]
    md.append(f"- Lag {L}: above C<L + 0.02 at " + "; ".join(f"seed {s}: d = {', '.join(per[s]) or 'none'}" for s in SEEDS)
              + f" (all three: {', '.join(common) or 'none'}); the pool in {len(pool)} of 3 seeds. Above C0 + 0.02 only: "
              + "; ".join(f"seed {s}: d = {', '.join(x for x in per0[s] if x not in per[s]) or 'none'}" for s in SEEDS) + ".")
    sent[L] = dict(above_clt=per, common=common, pool=pool, above_c0=per0)
open(os.path.join(HERE, 'where_tau', f'TABLE_{pre}.md'), 'w').write("\n".join(md) + "\n")
json.dump(sent, open(os.path.join(HERE, 'where_tau', f'SENT_{pre}.json'), 'w'))
print("\n".join(md))
