"""Final independent primary bound inversion and complete receipt audit."""
import hashlib
import json
import math
from scipy.optimize import brentq
from scipy.stats import binom
from compact_bridge import ROOT, ARMS
from compact_fixture import SCIENCE_WORLDS, make_world
from compact_audit import read, audit_receipt, paired_private_equal
from compact_integrity import verify
from compact_storage import atomic_json


def upper(k, n, a):
    if k == n: return 1.
    if k == 0: return -math.expm1(math.log(a) / n)
    return float(brentq(lambda p: binom.cdf(k, n, p) - a, 0., 1., xtol=1e-14))


def lower(k, n, a):
    if k == 0: return 0.
    if k == n: return math.exp(math.log(a) / n)
    return float(brentq(lambda p: binom.sf(k - 1, n, p) - a, 0., 1., xtol=1e-14))


def main():
    identity = verify(); spec = json.loads((ROOT / 'SPEC.json').read_text())
    summary = json.loads((ROOT / 'results/SUMMARY.json').read_text())
    primary = {a: {s: [] for s in ('old', 'new', 'revision')} for a in ARMS}
    manifest = {}; total_calls = 0
    for wid in SCIENCE_WORLDS:
        world = make_world(wid); pair = []
        for arm in ARMS:
            path = ROOT / 'results/science/receipts' / f'{wid}_{arm}.json.gz'
            doc = read(path); result = audit_receipt(doc, identity); total_calls += result['actual_store_calls']; pair.append(doc)
            if doc['world'] != wid or doc['arm'] != arm or doc['engineering_only'] or doc['logical_lives'] != 4:
                raise ValueError('science identity/level')
            if doc['peak_rss_bytes'] > spec['max_worker_rss_bytes'] or doc['worker_s'] > spec['max_job_seconds']:
                raise ValueError('completed worker resource violation')
            rows = doc['branches']['W']['probes'][-1]['rows']
            for stage, item in world['primary_items'].items():
                row = next(r for r in rows if r['stage'] == stage and r['item'] == item)
                expected = world['sets'][stage][item]['outcome']
                primary[arm][stage].append(int(row['prediction']['emitted'] == expected))
            manifest[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
        paired_private_equal(*pair)
    decisions = []
    for s in ('old', 'new', 'revision'):
        bs, cs = primary[ARMS[0]][s], primary[ARMS[1]][s]
        losses = sum(b > c for b, c in zip(bs, cs)); hits = sum(cs)
        hi = upper(losses, len(cs), .05 / 6); lo = lower(hits, len(cs), .05 / 6)
        published = summary['gates'][s]
        if losses != published['paired_loss_worlds'] or hits != published['selected_cue_correct']:
            raise ValueError('primary score recomputation')
        if abs(hi - published['loss_probability_upper']) > 1e-11 or abs(lo - published['candidate_accuracy_lower']) > 1e-11:
            raise ValueError('independent interval inversion')
        decisions.append(hi < .05 and lo > .9)
    if all(decisions) != summary['adopt_compact']: raise ValueError('decision differs')
    atomic_json(ROOT / 'results/SCIENCE_MANIFEST.json', manifest)
    atomic_json(ROOT / 'results/FINAL_AUDIT.json', {'verdict': 'PASS', 'identity': identity,
                'receipts': len(manifest), 'actual_store_calls': total_calls,
                'independent_binomial_inversion_matches': True, 'private_trajectory_parity': True,
                'adopt_compact': summary['adopt_compact']})
    print(json.dumps({'verdict': 'PASS', 'receipts': len(manifest)}))


if __name__ == '__main__': main()
