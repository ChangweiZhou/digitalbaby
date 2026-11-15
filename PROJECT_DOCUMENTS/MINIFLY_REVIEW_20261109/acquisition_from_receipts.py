"""Acquisition and retention read from the E26 receipts (15 November 2026).  No forward passes.

For each arm (S sweep, B backprop, K the E31 life with Kolen-Pollack feedback) and seed: the birth curve (thousand items seen; held-out exact answers on add and
sub at 12 letters) and, after every stage of the life, the same score for the birth family (mod 89, never rehearsed
in the life), for the first life family (mod 97) and the third (mod 73), with the items of each family seen so far.

Run from the repository root:  python3 -I PROJECT_DOCUMENTS/MINIFLY_REVIEW_20261109/acquisition_from_receipts.py
"""
import json, subprocess, sys
REF = 'origin/codex/minifly-consolidated-20261023'; P = 'MOD97_LOCAL_RULE_20261106/exp45/'
def load(name): return json.loads(subprocess.run(['git', 'show', f'{REF}:{P}{name}'], capture_output=True, check=True).stdout)
pc = lambda d: 100 * d['exact'] / d['n']
hard = lambda sc, m: (pc(sc[f'add{m}']['12']) + pc(sc[f'sub{m}']['12'])) / 2
for arm in 'SBK':                                   # K: the E31 life with Kolen-Pollack feedback, started from the sweep birth
    for seed in (73129, 93683, 116089):
        b = load(f"e26/e26_dev_{'S' if arm == 'K' else arm}_{seed}.json"); l = load(f'e31/e31_life_K_{seed}.json' if arm == 'K' else f'e26/e26_life_{arm}_{seed}.json')
        birth = [(r['step'] * 64 // 1000, round(hard(r['sealed'], 89))) for r in b['curve']]
        st = l['stages']
        rows = lambda s, fam: st[s].get('exposure_rows_tokens', {}).get(f'joint/{fam}', [0])[0]
        cum = {f: 0 for f in ('F1', 'F2', 'F3')}; life = []
        for s in range(len(st)):
            for f in cum: cum[f] += rows(s, f)
            sc = st[s]['scores']
            life.append(dict(stage=st[s].get('task') or 'start', add_sub_89=round(hard(sc, 89)), add_sub_97=round(hard(sc, 97)), add_sub_73=round(hard(sc, 73)),
                             max89=round(pc(sc['max89']['12'])), F1k=cum['F1'] // 1000, F3k=cum['F3'] // 1000))
        print(f"arm {arm} seed {seed}")
        print("  birth (4 ops mod 89, half the pairs taught): (thousand items seen, add/sub held) ", birth)
        for r in life: print("  life", r)
