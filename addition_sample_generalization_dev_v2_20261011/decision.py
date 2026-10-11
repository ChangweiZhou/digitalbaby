"""Locked score and statistical equations. No technical result selects an arm."""
import math
import numpy as np
from scipy.stats import t
import config
from collections import Counter, defaultdict


def weights():
    x = np.log2(np.asarray(config.CHECKPOINTS)+80)
    result = np.zeros(6)
    for i in range(5):
        amount = (x[i+1]-x[i])/(2*(x[-1]-x[0]))
        result[i] += amount
        result[i+1] += amount
    return result


def metrics(receipt):
    result = {}
    for history, branch in receipt['branches'].items():
        old = [branch['forks'][str(n)]['final_old_held']['accuracy'] for n in config.CHECKPOINTS]
        new = [branch['forks'][str(n)]['final_new_held']['accuracy'] for n in config.CHECKPOINTS]
        immediate = [branch['forks'][str(n)]['immediate_old_held']['accuracy'] for n in config.CHECKPOINTS]
        result[history] = {'persistent_old_held': old, 'new_held': new,
                           'immediate_old_held': immediate,
                           'score': float(weights() @ np.asarray(old)),
                           'final_old_held': old[-1], 'final_new_held': new[-1]}
    result['causal_N'] = result['W']['score']-result['N_OLD']['score']
    result['causal_shuffled'] = result['W']['score']-result['SHUFFLED']['score']
    result['technical_only'] = True
    return result


def countermodels(receipt):
    """Actual observed examples only; true values are used solely for scoring."""
    import worlds
    master = receipt['master']
    result = {}
    for n in config.CHECKPOINTS:
        present = [r for r in master['old'][:n]+master['new'] if r['target'] is not None]
        counts = Counter(r['target'] for r in present)
        left, right = defaultdict(Counter), defaultdict(Counter)
        table = {}
        for row in present:
            a, b = row['pair']
            left[a][row['target']] += 1
            right[b][row['target']] += 1
            table[(a, b)] = row['target']
        mode = lambda c: min(range(9), key=lambda y: (-c[y], y))
        fallback = mode(counts)
        mean = sum(r['target'] for r in present)/len(present) if present else 0.
        rounded = min(range(9), key=lambda y: ((y-mean)**2, y))
        totals = defaultdict(list)
        for query in worlds.queries(master, 'old', False):
            scores = defaultdict(list)
            for a, b in query['pairs']:
                if (a, b) in table or (b, a) in table:
                    raise AssertionError('exact/canonical held-pair exposure')
                # Identity-ranked function cannot access the latent numeral map.
                alphabet = master['old_alphabet']
                ranked = alphabet.index(a)+alphabet.index(b)
                estimates = {'exact_key_mode_fallback': fallback,
                             'canonical_key_mode_fallback': fallback,
                             'target_mode': fallback, 'rounded_target_mean': rounded,
                             'left_only': mode(left[a]), 'right_only': mode(right[b]),
                             'fixed_symbol_rank': ranked, 'zero_birth': 0}
                for name, y in estimates.items():
                    scores[name].append(y == query['truth_private'])
            for name, values in scores.items():
                totals[name].append(sum(values)/len(values))
        result[str(n+80)] = {name: sum(values)/len(values) for name, values in totals.items()}
    return result


def lower_bound(strata):
    values = [np.asarray(x, dtype=float) for x in strata]
    if len(values) != 2 or any(len(x) != 32 for x in values):
        raise ValueError('64 primary worlds, 32 per noise stratum required')
    if any(not np.isfinite(x).all() or np.any(np.abs(x) > 1) for x in values):
        raise ValueError('bounded paired world scores')
    components = [.25*np.var(x, ddof=1)/len(x) for x in values]
    mean = sum(float(x.mean()) for x in values)/2
    variance = sum(components)
    if variance <= 1e-24:
        penalty = math.sqrt(2*math.log(20)/64)
        return {'mean': mean, 'lower': mean-penalty, 'method': 'Hoeffding',
                'worlds': 64}
    df = variance**2/sum(v*v/31 for v in components)
    return {'mean': mean, 'lower': mean-float(t.ppf(.95, df))*math.sqrt(variance),
            'df': df, 'method': 'stratified paired Student', 'worlds': 64}


def synthetic_suite():
    zero = lower_bound([np.zeros(32), np.zeros(32)])
    if zero['lower'] >= 0 or zero['method'] != 'Hoeffding':
        raise AssertionError('degenerate interval falsely passes')
    positive = lower_bound([np.ones(32)*.5, np.ones(32)*.5])
    negative = lower_bound([-np.ones(32)*.5, -np.ones(32)*.5])
    if positive['lower'] <= 0 or negative['lower'] >= 0:
        raise AssertionError('positive/negative analytic direction')
    varying = lower_bound([np.linspace(.1, .3, 32), np.linspace(.2, .4, 32)])
    if varying['method'] != 'stratified paired Student' or varying['worlds'] != 64:
        raise AssertionError('statistical units')
    rejected = 0
    for bad in ([np.zeros(320), np.zeros(320)], [np.full(32, np.nan), np.zeros(32)]):
        try:
            lower_bound(bad)
        except ValueError:
            rejected += 1
        else:
            raise AssertionError('invalid statistical input accepted')
    if not np.all(weights() > 0) or abs(weights().sum()-1) > 1e-14:
        raise AssertionError('sample curve integration')
    return {'verdict': 'PASS', 'zero_variance': zero,
            'positive': positive, 'varying': varying, 'rejected_invalid_cases': rejected,
            'science_endpoints_used': 0, 'selected_candidate': None}
