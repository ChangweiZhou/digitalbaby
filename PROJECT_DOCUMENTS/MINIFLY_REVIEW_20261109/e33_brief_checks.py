"""Checks for the audit of the E33 brief (commit 00fad472). Read-only: receipts on the default branch; nothing is trained.

a. S-bal and the answer-only sweep birth, row by row, from their receipts (the receipts' own add-89 panel): accuracy at 12 and 20 letters.
b. S-bal's loss per position by fine role at the start and the end, and the error each class carries per item at the end.
c. For each of the brief's arms: the non-answer weight it carries and the error per item it carries, as S-bal ends.
d. Which reference hashes and weight files exist.
e. Every clean pattern over the eight arms in which adding targets never helps, and the outcome row of the brief it falls under.
"""
import itertools, json, subprocess
import numpy as np
D = 'origin/codex/minifly-consolidated-20261023'; P = 'MOD97_LOCAL_RULE_20261106/exp45/'; SEEDS = (73129, 93683, 116089)
def load(p):
    r = subprocess.run(['git', 'show', f'{D}:{P}{p}'], capture_output=True); return json.loads(r.stdout) if r.returncode == 0 else None
pct = lambda d: round(100 * d['exact'] / d['n'], 1)

print('a. accuracy by saved row, receipts\' own add-89 panel: at 12 letters | at 20 letters (mean of the last three rows in brackets)')
for name, pat in (('S-bal (E28)', 'e28/e28_dev_S-bal_{s}.json'), ('answer-only sweep birth (E26 S)', 'e26/e26_dev_S_{s}.json')):
    for s in SEEDS:
        c = load(pat.format(s=s))['curve']; a12 = [pct(r['sealed']['add89']['12']) for r in c]; a20 = [pct(r['sealed']['add89']['20']) for r in c]
        print(f"   {name} {s}: {a12} ({np.mean(a12[-3:]):.1f}) | {a20} ({np.mean(a20[-3:]):.1f})")
d = load('e28/e28_dev_S-bal_73129.json')
print('b. S-bal 73129, loss per position by fine role (mean, positions) in the first and the last block of 512 updates')
for i in (0, -1):
    fr = d['role_curve'][i]['by_fine_role']; print('   block ending', d['role_curve'][i]['block_end'], {k: (round(v['mean'], 3), v['positions']) for k, v in fr.items()})
fr = d['role_curve'][-1]['by_fine_role']; items = fr['code']['positions']
CL = {'D': ('distractor',), 'O': ('operand',), 'S': ('code', 'operator', 'equals')}
n_other = sum(fr[k]['positions'] for ks in CL.values() for k in ks)
share = {c: sum(fr[k]['positions'] for k in ks) / n_other for c, ks in CL.items()}
err = {c: sum(fr[k]['mean'] * fr[k]['positions'] for k in ks) / items for c, ks in CL.items()}
print('   share of non-answer positions:', {c: round(v, 3) for c, v in share.items()}, '| error per item at the end, nats:', {c: round(v, 2) for c, v in err.items()},
      '| of S, the task code alone:', round(fr['code']['mean'], 2))
print('c. each arm: non-answer weight (0.5 x share of non-answer positions) and error per item at the end of S-bal')
ARMS = ['S', 'O', 'D', 'OS', 'DS', 'DO', 'DOS']
for a in ARMS:
    print(f"   A+{a:3s}: weight {0.5 * sum(share[c] for c in a):.3f} | error per item {sum(err[c] for c in a):5.1f} nats")
print(f"   A+D with D's terms halved (the brief's A+c-light for c = D, weight equal to S's): error per item {err['D'] * share['S'] / share['D']:.1f} nats")
dry = load('e28_qual/dry/e28_dev_S-bal_dry2048_73129.json')
print('d. dry reference tensors_sha256:', dry['tensors_sha256'][:16], '| S-bal 73129 recorded row hashes:', [(r['step'], r['tensors_sha256'][:8]) for r in d['curve']])
ls = subprocess.run(['git', 'ls-tree', '-r', '--name-only', D, 'MOD97_LOCAL_RULE_20261106/exp45/e28', 'MOD97_LOCAL_RULE_20261106/exp45/e26'], capture_output=True, text=True).stdout.split()
print('   weight files on the default branch: S-bal 73129', [f.split('/')[-1] for f in ls if 'S-bal_73129' in f and f.endswith('.pt')], '| E26 S 73129', [f.split('/')[-1] for f in ls if 'e26_dev_S_73129' in f and f.endswith('.pt')])

print('e. clean patterns in which adding targets never helps (A extrapolates, A+DOS fails, every arm either extrapolates or fails), and the brief\'s outcome row')
subsets = ['', 'D', 'O', 'S', 'DO', 'DS', 'OS', 'DOS']
def outcome(fails):           # fails: set of arms that fail; the rest extrapolate
    if '' in fails or 'DOS' not in fails: return 0
    for c in 'DOS':
        if all((a in fails) == (c in a) for a in subsets): return 1
    if fails == set(subsets) - {''}: return 2
    if not ({'D', 'O', 'S'} & fails): return 3
    return 4
seen = {}
nonempty = subsets[1:]
for r in range(1, 4):
    for mins in itertools.combinations(nonempty, r):
        if any(set(a) < set(b) or set(b) < set(a) for a, b in itertools.combinations(mins, 2)): continue      # minimal failing sets: an antichain
        fails = frozenset(a for a in nonempty if any(set(m) <= set(a) for m in mins))
        seen[fails] = (mins, outcome(set(fails)))
for fails, (mins, o) in sorted(seen.items(), key=lambda kv: (kv[1][1], len(kv[0]))):
    print(f"   fails as soon as the targets include {' or '.join('+'.join(m) for m in mins):22s} -> outcome {o} | failing arms: {', '.join('A+' + a for a in subsets if a in fails)}")
