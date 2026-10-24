"""Reference SCORES / verdict / headline for corpus rows (documentation copies; see refcalc.py)."""
import math
from collections import Counter

import base_items as BI
import refcalc as RC

TAUGHT = [(BI.prompt(a, b), BI.truth(a, b)) for (a, b) in BI.TAUGHT_PAIRS]
T_SET = set(BI.TAUGHT_PAIRS)
T_ORBITS = set(tuple(sorted(p)) for p in BI.TAUGHT_PAIRS)


def taught_has(a, b):
    return (a, b) in T_SET or (b, a) in T_SET


def stratum(a, b):
    ans = str(a + b)
    if a == 0 or b == 0 or ans in (str(a), str(b)):
        return 'S0'
    if tuple(sorted((a, b))) in T_ORBITS:
        return 'S1'
    firsts = set(x for x, _ in T_SET) | set(y for _, y in T_SET)
    seconds = firsts
    if a not in firsts or b not in seconds or len(ans) > max(len(t) for _, t in TAUGHT):
        return 'S4'
    return 'S3' if a + b >= 10 else 'S2'


def determined(a, b):
    vals = range(0, 10)
    for a2 in vals:
        for b2 in vals:
            if a2 != a and b2 != b and taught_has(a, b2) and taught_has(a2, b) and taught_has(a2, b2):
                return True
    return False


def _ab(row):
    return RC.parse(RC.dec(row['prompt_hex']))


def cell_rows(rows, world, arm, pool, ckpt):
    return [r for r in rows if r['world'] == world and r['arm'] == arm and r['pool'] == pool
            and r['checkpoint'] == ckpt]


def rate(cell, counted_only=False):
    if counted_only:
        cell = [r for r in cell if stratum(*_ab(r)) in ('S2', 'S3', 'S4') and determined(*_ab(r))]
    return sum(1 for r in cell if RC.is_exact(r)) / float(len(cell)) if cell else 0.0


def per_world(rows, chance=0.05):
    worlds = sorted(set(r['world'] for r in rows), key=lambda w: (len(w), w))
    out = []
    for w in worlds:
        cand = cell_rows(rows, w, 'W', 'HELD', 'final')
        lad = RC.ladder(TAUGHT, cand, 'HELD')
        names, best = RC.best(lad)
        npt = rate(cell_rows(rows, w, 'NPT', 'HELD', 'final'), True)
        # SPEC 9.5 as amended (20.6): the decision comparator also holds the declared shuf and frozen arms
        # (comparator_arms names NPT only); G (SPEC 9.6) keeps NPT, the ladder and chance
        ctrl = max(rate(cell_rows(rows, w, a, 'HELD', 'final'), True) for a in ('SHUF', 'FROZEN'))
        v = rate(cand, True)
        old_end = rate(cell_rows(rows, w, 'W', 'OLD', 'old_end'))
        old_fin = rate(cell_rows(rows, w, 'W', 'OLD', 'final'))
        old_lad = RC.best(RC.ladder(TAUGHT, cell_rows(rows, w, 'W', 'OLD', 'old_end'), 'OLD'))[1]
        out.append({'world': w, 'cand': v, 'npt': npt, 'best_names': names, 'best': best,
                    'comp': max(chance, npt, ctrl, best), 'd': v - max(chance, npt, ctrl, best),
                    'g': v - max(chance, npt, best), 'ctrl': ctrl,
                    'old_end': old_end, 'old_final': old_fin, 'old_drop': old_end - old_fin,
                    'old_acq': old_end - old_lad, 'ladder': lad})
    return out


def verdict(rows, margin=0.05, absolute_floor=0.25, per_world_floor=-0.05, max_drop=0.1, acq=0.05):
    pw = per_world(rows)
    d = [x['d'] for x in pw]
    mean_d, sd, lcb = RC.paired_lcb(d)
    _, _, drop_ucb = RC.paired_ucb([x['old_drop'] for x in pw])
    mean_cand = sum(x['cand'] for x in pw) / len(pw)
    mean_acq = sum(x['old_acq'] for x in pw) / len(pw)
    ok = (lcb > margin and mean_cand >= absolute_floor and min(d) >= per_world_floor
          and drop_ucb <= max_drop and mean_acq >= acq and len(pw) >= 8)
    return {'verdict': 'PASS' if ok else 'FAIL', 'mean_cand': mean_cand, 'mean_d': mean_d, 'sd': sd,
            'mean_g': sum(x['g'] for x in pw) / len(pw),
            'lcb': lcb, 'drop_ucb': drop_ucb, 'mean_acq': mean_acq, 'per_world': pw}


def headline(rows, bh=None, k=1, role='MAIN'):
    v = verdict(rows)
    pw = v['per_world']
    names = set(n for x in pw for n in x['best_names'])
    name = sorted(names)[0] if len(names) == 1 else '/'.join(sorted(names))
    best = sum(x['best'] for x in pw) / len(pw)
    g = v['mean_g']
    old_fin = sum(x['old_final'] for x in pw) / len(pw)
    drop = sum(x['old_drop'] for x in pw) / len(pw)
    n = sum(1 for r in rows if r['arm'] == 'W' and r['pool'] == 'HELD' and r['checkpoint'] == 'final'
            and stratum(*_ab(r)) in ('S2', 'S3', 'S4') and determined(*_ab(r)))
    earned = 'E3' if v['verdict'] == 'PASS' else 'NONE'
    return ('%s | earned %s (attempted E3) | held exact own-EOS %s%% [S2+ determined, n=%d] vs best '
            'shortcut %s %s%% | G %spp | OLD final %s%% (drop %spp) | %d %s worlds | best of k=%d'
            % (v['verdict'], earned, RC.pct(v['mean_cand']), n, name, RC.pct(best), RC.pct(g),
               RC.pct(old_fin), RC.pct(drop), len(pw), role, k))


def scores(rows, role, bh=None, cpu_h=0.2):
    cells = {}
    for r in rows:
        a, b = _ab(r)
        key = (r['world'], r['world_role'], r['arm'], r['pool'], r['checkpoint'], stratum(a, b),
               determined(a, b))
        n, ne = cells.get(key, (0, 0))
        cells[key] = (n + 1, ne + (1 if RC.is_exact(r) else 0))
    cell_list = [{'world': k[0], 'world_role': k[1], 'arm': k[2], 'pool': k[3], 'checkpoint': k[4],
                  'stratum': k[5], 'determined': k[6], 'n': v[0], 'n_exact': v[1]}
                 for k, v in sorted(cells.items(), key=lambda kv: tuple(str(x) for x in kv[0]))]
    pw = per_world(rows)
    lad = {}
    for x in pw:
        lad['%s|HELD|final' % x['world']] = {m: round(val, 6) for m, val in sorted(x['ladder'].items())}
    names = sorted(set(n for x in pw for n in x['best_names']))
    g = sum(x['g'] for x in pw) / len(pw)
    held = [r for r in rows if r['arm'] == 'W' and r['pool'] == 'HELD']
    em = Counter(RC.emitted_from_stream(RC.dec(r['stream_hex']))[0] for r in held)
    tot = float(sum(em.values()))
    ent = -sum((c / tot) * math.log(c / tot, 2) for c in em.values())
    first = [r for r in held if r['world'] == pw[0]['world']]
    strata = Counter(stratum(*_ab(r)) for r in first)
    det = sum(1 for r in first if determined(*_ab(r)))
    end = sum(x['old_end'] for x in pw) / len(pw)
    fin = sum(x['old_final'] for x in pw) / len(pw)
    return {'schema': 'minifly.scores.v1', 'role': role, 'cells': cell_list, 'ladder': lad,
            'best_shortcut': {'name': names[0] if len(names) == 1 else '/'.join(names),
                              'value': round(sum(x['best'] for x in pw) / len(pw), 6)},
            'strata': dict(sorted(strata.items())), 'determinacy': {'determined': det, 'total': len(first)},
            'modal_fraction': round(max(em.values()) / tot, 6), 'entropy_bits': round(ent, 6),
            'G': round(g, 6), 'G_per_sample': round(g / 0.15, 6), 'G_per_cpu_h': round(g / cpu_h, 6),
            'old': {'end': round(end, 6), 'final': round(fin, 6), 'drop': round(end - fin, 6),
                    'acq': round(sum(x['old_acq'] for x in pw) / len(pw), 6)},
            'udi': None, 'designs_tried_k': 1, 'inert_alarms': [], 'met_mismatches': [],
            'note': 'executor copy from corpus/_build/scores_doc.py; MFC score rewrites it'}
