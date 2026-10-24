"""MRK-1 rule qualification (WP3): RED-01..11 of SPEC 7.4 and the DEC-01 property fixtures.

Python 3.9+, standard library only. Synthetic cells are built from the lock's own arms, worlds, pools,
strata and n_items_per_pool and evaluated with the program decide (decide.evaluate). Nothing here reads
delivery outputs: the checks qualify the declared rule before any learner outcome.

Synthetic data conventions:
- worlds: lock.worlds.official.ids when listed, else RED_W01..RED_Wn with n = official.n (or min_worlds);
- "at chance": floor(chance * n) exact items (so the comparator is chance itself);
- a "clean" level r: per-world exact counts at the binomial quantiles (i + 0.5) / N of Binomial(n, r), so the
  worlds vary like real ones (identical worlds would trigger the Hoeffding fallback of SPEC 9.5);
- counted items sit in the first counted stratum among S2..S4 (else the first counted one) and are determined
  when a witness is declared.
"""
from __future__ import annotations

import copy
import hashlib
import math
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import common
import decide

ALL_PASS = {'inert_alarms': [], 'adm_tb': 'PASS', 'adm_ctrl': 'PASS'}
BIG_N = 10 ** 9


def binom_quantiles(n: int, p: float, k_worlds: int) -> List[int]:
    """Counts at the (i + 0.5) / k quantiles of Binomial(n, p), i = 0..k-1."""
    p = min(1.0, max(0.0, p))
    if p <= 0.0:
        return [0] * k_worlds
    if p >= 1.0:
        return [n] * k_worlds
    lp, lq = math.log(p), math.log1p(-p)
    cdf = []
    acc = 0.0
    for k in range(n + 1):
        acc += math.exp(math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * lp + (n - k) * lq)
        cdf.append(acc)
    out = []
    for i in range(k_worlds):
        q = (i + 0.5) / k_worlds
        k = next((j for j, c in enumerate(cdf) if c >= q - 1e-12), n)
        out.append(k)
    return out


class Synth:
    """Synthetic cells for one lock."""

    def __init__(self, lock: dict, reducer_path: Optional[str] = None):
        p, problems, nonfinite = decide.parse_params(lock)
        if problems or nonfinite:
            raise common.InvalidInput('lock does not define a decidable rule: %s' % '; '.join(problems + nonfinite))
        self.lock = lock
        self.p = p
        nip = p['n_items_per_pool']
        n = nip.get(p['pool'])
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise common.InvalidInput('primary.n_items_per_pool.%s must be a positive integer, got %r' % (p['pool'], n))
        self.n = n
        n_old = nip.get('OLD', n)
        self.n_old = n_old if isinstance(n_old, int) and not isinstance(n_old, bool) and n_old >= 1 else n
        worlds = lock.get('worlds') if isinstance(lock.get('worlds'), dict) else {}
        off = worlds.get('official') if isinstance(worlds.get('official'), dict) else {}
        ids = [w for w in (off.get('ids') or []) if isinstance(w, str)]
        k = len(ids) or (off.get('n') if isinstance(off.get('n'), int) and not isinstance(off.get('n'), bool) and off.get('n') > 0
                         else p['min_worlds'])
        self.worlds = ids or ['RED_W%02d' % (i + 1) for i in range(k)]
        if decide.strata_applicable(lock):
            sc = p['strata_counted']
            self.stratum = next((x for x in sc if x in ('S2', 'S3', 'S4')), sc[0])
        else:
            self.stratum = None
        self.det = True if decide.determinacy_applicable(lock) else None
        self.balanced = p['metric'] == 'balanced_exact_own_eos'
        self.reducer_path = reducer_path
        self.has_old = p['old_pool_exists'] or bool(p['guards'])

    # counts -----------------------------------------------------------------------
    def chance_count(self, n: int) -> int:
        return int(math.floor(self.p['chance'] * n + 1e-12))

    def level(self, r: float, n: Optional[int] = None, worlds: Optional[Sequence[str]] = None) -> Dict[str, int]:
        ws = list(worlds if worlds is not None else self.worlds)
        n = self.n if n is None else n
        return dict(zip(ws, binom_quantiles(n, r, len(ws))))

    def flat(self, k: int, worlds: Optional[Sequence[str]] = None) -> Dict[str, int]:
        return {w: k for w in (worlds if worlds is not None else self.worlds)}

    # cells -------------------------------------------------------------------------
    def _group(self, w: str, role: str, arm: str, pool: str, ck: str, stratum: Any, det: Any, n: int, k: Any) -> dict:
        g = {'world': w, 'world_role': role, 'arm': arm, 'pool': pool, 'checkpoint': ck, 'stratum': stratum,
             'determined': det, 'n': n, 'n_rows': n, 'n_exact': k}
        if self.balanced:
            g['answer'] = '61'
        return g

    def cells(self, cand: Dict[str, int], worlds: Optional[Sequence[str]] = None, role: str = 'official',
              comps: Optional[Dict[str, Dict[str, int]]] = None, ladder: Optional[Dict[str, Dict[str, int]]] = None,
              old_end: Optional[Dict[str, int]] = None, old_final: Optional[Dict[str, int]] = None,
              old_ladder: Optional[Dict[str, Dict[str, int]]] = None, n_primary: Optional[int] = None,
              extra_groups: Optional[List[Tuple[str, Dict[str, Dict[str, int]], int]]] = None) -> dict:
        """Cells dict as decide.evaluate consumes it. Counts are exact items per world on the counted items;
        unspecified comparators and ladder members sit at chance; OLD defaults to the clean level 0.9 at both
        checkpoints with the OLD ladder at chance."""
        p = self.p
        ws = list(worlds if worlds is not None else self.worlds)
        n = self.n if n_primary is None else n_primary
        comps = comps or {}
        ladder = ladder or {}
        old_ladder = old_ladder or {}
        clean_old = self.level(0.9, self.n_old, ws)
        old_end = old_end if old_end is not None else clean_old
        old_final = old_final if old_final is not None else old_end
        out: List[dict] = []
        lad: Dict[str, Dict[str, float]] = {}
        lcells: Dict[str, Dict[str, List[dict]]] = {}
        status: Dict[str, str] = {}
        cc = self.chance_count(n)
        cc_old = self.chance_count(self.n_old)
        arms = [p['candidate_arm']] + [a for a in p['comparator_arms'] if a != p['candidate_arm']]
        for w in ws:
            for a in arms:
                k = cand[w] if a == p['candidate_arm'] else comps.get(a, {}).get(w, cc)
                out.append(self._group(w, role, a, p['pool'], p['checkpoint'], self.stratum, self.det, n, k))
                for st, by_arm, n_x in extra_groups or []:
                    kx = by_arm.get(a, {}).get(w, self.chance_count(n_x))
                    out.append(self._group(w, role, a, p['pool'], p['checkpoint'], st, self.det, n_x, kx))
            key = decide.lkey(w, p['pool'], p['checkpoint'])
            lad[key] = {}
            lcells[key] = {}
            for m in decide.pool_members(p['pool']):
                k = ladder.get(m, {}).get(w, cc)
                lad[key][m] = k / float(n)
                lcells[key][m] = [{'stratum': self.stratum, 'determined': self.det, 'n': n, 'n_rows': n, 'n_exact': k}]
                if self.balanced:
                    lcells[key][m][0]['answer'] = '61'
            status[key] = 'ok'
            if self.has_old:
                out.append(self._group(w, role, p['candidate_arm'], 'OLD', p['old_end'], 'S1', self.det, self.n_old,
                                       old_end[w]))
                out.append(self._group(w, role, p['candidate_arm'], 'OLD', p['final'], 'S1', self.det, self.n_old,
                                       old_final[w]))
                okey = decide.lkey(w, 'OLD', p['old_end'])
                lad[okey] = {m: old_ladder.get(m, {}).get(w, cc_old) / float(self.n_old)
                             for m in decide.pool_members('OLD')}
                status[okey] = 'ok'
        return {'schema': 'minifly.cells.v1', 'synthetic': True, 'metric': p['metric'],
                'worlds': {w: role for w in ws}, 'cells': out, 'ladder': lad, 'ladder_status': status,
                'ladder_cells': lcells}

    def pre(self, worlds: Optional[Sequence[str]] = None, n_primary: Optional[int] = None) -> dict:
        nip = dict(self.p['n_items_per_pool'])
        nip[self.p['pool']] = self.n if n_primary is None else n_primary
        if self.has_old:
            nip['OLD'] = self.n_old
        pre = dict(ALL_PASS)
        pre.update({'gates': {g: 'PASS' for g in self.p['gates']}, 'n_items_per_pool': nip,
                    'expected_worlds': list(worlds if worlds is not None else self.worlds)})
        if self.reducer_path:
            pre['reducer_path'] = self.reducer_path
        return pre

    def clean(self, worlds: Optional[Sequence[str]] = None, role: str = 'official') -> dict:
        ws = list(worlds if worlds is not None else self.worlds)
        return self.cells(self.level(0.9, worlds=ws), worlds=ws, role=role)

    def evaluate(self, cells: dict, worlds: Optional[Sequence[str]] = None, n_primary: Optional[int] = None) -> dict:
        return decide.evaluate(cells, self.lock, self.pre(worlds, n_primary))


def _synth(ctx: common.Ctx) -> Synth:
    lock = ctx.lock
    pr = lock.get('primary') if isinstance(lock.get('primary'), dict) else {}
    rp = None
    cr = pr.get('custom_reducer') if isinstance(pr.get('custom_reducer'), dict) else None
    if pr.get('rule_type') == 'custom' and cr and isinstance(cr.get('path'), str):
        rp = str(ctx.path(cr['path']))
    return Synth(lock, rp)


def _brief(v: dict) -> str:
    s = v.get('verdict')
    if v.get('shadow_verdict'):
        s += '(shadow %s)' % v['shadow_verdict']
    if v.get('invalid_reason'):
        s += '[%s]' % v['invalid_reason']
    return s


def _must_not_pass(cid: str, cases: List[Tuple[str, dict]], what: str) -> common.Result:
    bad = [(name, v) for name, v in cases if decide.passes(v)]
    detail = '; '.join('%s -> %s' % (name, _brief(v)) for name, v in cases)
    if bad:
        return common.Result(cid, common.FAIL, '%s PASSes: %s' % (what, detail[:3000]))
    return common.Result(cid, common.PASS, '%s never PASSes: %s' % (what, detail[:3000]))


def _guard_must_reject(cid: str, s: 'Synth', kind: str, cases: List[Tuple[str, dict]], what: str) -> common.Result:
    """RED-09/RED-11: the case must not PASS *and* the lock's own guard of `kind` must report FAIL on it. A
    rule that rejects the case only because it rejects everything (no such guard, an unreachable rule) does
    not show that the OLD guard works, so it FAILs here too (LOCK-04 / RED-01 name the cause)."""
    gids = [g['id'] for g in s.p['guards'] if g['kind'] == kind]
    if not gids:
        return common.Result(cid, common.FAIL, '%s is not rejected by any guard: the lock declares no %s guard although '
                             'an OLD pool exists' % (what, kind))
    bad = []
    for name, v in cases:
        gs = v.get('guards') or {}
        if decide.passes(v):
            bad.append('%s PASSes (%s)' % (name, _brief(v)))
        elif not any(gs.get(g) == 'FAIL' for g in gids):
            bad.append('%s is rejected, but not by guard %s (guards %s; %s)'
                       % (name, ','.join(gids), {g: gs.get(g) for g in gids}, _brief(v)))
    detail = '; '.join('%s -> %s, guard %s' % (name, _brief(v), {g: (v.get('guards') or {}).get(g) for g in gids})
                       for name, v in cases)
    if bad:
        return common.Result(cid, common.FAIL, '%s: %s' % (what, '; '.join(bad)[:3000]))
    return common.Result(cid, common.PASS, '%s never PASSes and guard %s rejects it: %s' % (what, ','.join(gids), detail[:3000]))


def _not_applicable_old(s: Synth, cid: str) -> Optional[common.Result]:
    if not s.p['old_pool_exists']:
        return common.Result(cid, common.NOT_APPLICABLE, 'no OLD pool in the lock (n_items_per_pool, split.old_manifest)')
    return None


@common.check('RED-01')
def check_red_01(ctx: common.Ctx) -> common.Result:
    s = _synth(ctx)
    v = s.evaluate(s.clean())
    if v.get('verdict') != 'PASS':
        return common.Result('RED-01', common.FAIL, 'the rule is unreachable: a clean positive (candidate 0.9, all else at '
                             'chance %.3f, %d worlds x %d items) gives %s: %s'
                             % (s.p['chance'], len(s.worlds), s.n, _brief(v), '; '.join(v.get('reasons', [])[:4])))
    prim = v.get('primary') or {}
    return common.Result('RED-01', common.PASS, 'clean positive PASSes (%d worlds x %d items; mean d %.3f, %s %s)'
                         % (len(s.worlds), s.n, prim.get('mean_d') or 0.0, prim.get('method'),
                            ('LCB %.3f' % prim['lcb']) if prim.get('lcb') is not None else ''))


@common.check('RED-02')
def check_red_02(ctx: common.Ctx) -> common.Result:
    s = _synth(ctx)
    strong = s.level(0.6)
    cases = []
    for a in s.p['comparator_arms']:
        cases.append(('candidate == arm %s' % a, s.evaluate(s.cells(dict(strong), comps={a: strong}))))
    for m in ('CONST', 'FIXED_1', 'MODE_REPEAT'):
        if m in decide.pool_members(s.p['pool']):
            cases.append(('candidate == shortcut %s' % m, s.evaluate(s.cells(dict(strong), ladder={m: strong}))))
    cc = s.chance_count(s.n)
    cases.append(('candidate == chance', s.evaluate(s.cells(s.flat(cc)))))
    return _must_not_pass('RED-02', cases, 'candidate equal to the strongest comparator')


@common.check('RED-03')
def check_red_03(ctx: common.Ctx) -> common.Result:
    s = _synth(ctx)
    cases = []
    w0 = s.worlds[0]
    for a in s.p['comparator_arms']:
        c = s.clean()
        c['cells'] = [g for g in c['cells'] if not (g['arm'] == a and g['world'] == w0)]
        cases.append(('comparator %s missing in %s' % (a, w0), s.evaluate(c)))
    c = s.clean()
    key = decide.lkey(w0, s.p['pool'], s.p['checkpoint'])
    c['ladder'].pop(key, None)
    c['ladder_status'].pop(key, None)
    cases.append(('ladder missing in %s' % w0, s.evaluate(c)))
    if not s.p['comparator_arms']:
        c = s.clean()
        c['cells'] = [g for g in c['cells'] if not (g['arm'] == s.p['candidate_arm'] and g['world'] == w0)]
        cases.append(('candidate missing in %s (no comparator arms declared)' % w0, s.evaluate(c)))
    bad = [(n, v) for n, v in cases if v.get('verdict') != 'INCOMPLETE']
    detail = '; '.join('%s -> %s' % (n, _brief(v)) for n, v in cases)
    if bad:
        return common.Result('RED-03', common.FAIL, 'missing inputs do not give INCOMPLETE: %s' % detail)
    return common.Result('RED-03', common.PASS, 'missing inputs give INCOMPLETE: %s' % detail)


@common.check('RED-04')
def check_red_04(ctx: common.Ctx) -> common.Result:
    s = _synth(ctx)
    mw = s.p['min_worlds']
    k = max(0, mw - 1)
    ws = ['RED_FEW%02d' % (i + 1) for i in range(k)]
    c = s.cells(s.level(0.9, worlds=ws), worlds=ws) if ws else s.cells({}, worlds=[])
    v = s.evaluate(c, worlds=ws)
    return _must_not_pass('RED-04', [('%d worlds (min_worlds %d)' % (k, mw), v)], 'fewer than min_worlds')


@common.check('RED-05')
def check_red_05(ctx: common.Ctx) -> common.Result:
    s = _synth(ctx)
    cases = []
    for label, val in (('None', None), ('NaN', float('nan'))):
        c = s.clean()
        for g in c['cells']:
            if g['arm'] == s.p['candidate_arm'] and g['world'] == s.worlds[0] and g['pool'] == s.p['pool']:
                g['n_exact'] = val
                break
        cases.append(('candidate n_exact %s' % label, s.evaluate(c)))
    c = s.clean()
    key = decide.lkey(s.worlds[0], s.p['pool'], s.p['checkpoint'])
    c['ladder'][key]['CONST'] = float('nan')
    cases.append(('ladder CONST NaN', s.evaluate(c)))
    bad = [(n, v) for n, v in cases if v.get('verdict') != 'INVALID']
    detail = '; '.join('%s -> %s' % (n, _brief(v)) for n, v in cases)
    if bad:
        return common.Result('RED-05', common.FAIL, 'None/NaN do not give INVALID: %s' % detail)
    return common.Result('RED-05', common.PASS, 'None/NaN give INVALID: %s' % detail)


@common.check('RED-06')
def check_red_06(ctx: common.Ctx) -> common.Result:
    s = _synth(ctx)
    v = s.evaluate(s.clean(role='dev'))
    if v.get('verdict') != 'DEV_NOT_EVIDENCE':
        return common.Result('RED-06', common.FAIL, 'all-dev clean positive gives %s, not DEV_NOT_EVIDENCE' % _brief(v))
    return common.Result('RED-06', common.PASS, 'all-dev worlds give %s' % _brief(v))


@common.check('RED-07')
def check_red_07(ctx: common.Ctx) -> common.Result:
    s = _synth(ctx)
    n = BIG_N
    comp = s.p['chance']
    target = comp + s.p['margin_abs'] - 1e-9
    k = int(math.floor(target * n))
    cand = {}
    spread = (s.p['aggregator'] or {}).get('type') == 'paired_lcb'
    for i, w in enumerate(s.worlds):
        off = 0
        if spread and len(s.worlds) >= 2:
            off = (100 if i % 2 == 0 else -100) if (i < len(s.worlds) - (len(s.worlds) % 2)) else 0
        cand[w] = max(0, min(n, k + off))
    c = s.cells(cand, n_primary=n)
    v = s.evaluate(c, n_primary=n)
    return _must_not_pass('RED-07', [('candidate = comparator %.4f + margin %.4f - 1e-9 (%s)'
                                      % (comp, s.p['margin_abs'], 'spread +-1e-7' if spread else 'flat'), v)],
                          'candidate just below comparator + margin')


@common.check('RED-08')
def check_red_08(ctx: common.Ctx) -> common.Result:
    lock = ctx.lock
    if not decide.strata_applicable(lock):
        return common.Result('RED-08', common.NOT_APPLICABLE, 'strata are not available for a non-addition task')
    s = _synth(ctx)
    m = s.n
    cc = s.chance_count(s.n)
    gain = {s.p['candidate_arm']: s.level(0.9, m)}
    extra = [('S0', gain, m), ('S1', gain, m)]
    c = s.cells(s.flat(cc), extra_groups=extra)
    nip = s.n + 2 * m
    v = decide.evaluate(c, lock, s.pre(n_primary=nip))
    note = ''
    if set(s.p['strata_counted']) & {'S0', 'S1'}:
        note = ' (the lock counts %s)' % sorted(set(s.p['strata_counted']) & {'S0', 'S1'})
    return _must_not_pass('RED-08', [('0.9 on S0/S1, chance on counted strata%s' % note, v)], 'a gain only on S0/S1')


@common.check('RED-09')
def check_red_09(ctx: common.Ctx) -> common.Result:
    s = _synth(ctx)
    na = _not_applicable_old(s, 'RED-09')
    if na:
        return na
    drops = [g for g in s.p['guards'] if g['kind'] == 'old_drop']
    max_drop = drops[0]['max_drop'] if drops else 0.10
    end = s.level(0.9, s.n_old)
    d = int(math.ceil((max_drop + 0.2) * s.n_old))
    final = {w: max(0, k - d) for w, k in end.items()}
    v = s.evaluate(s.cells(s.level(0.9), old_end=end, old_final=final))
    return _guard_must_reject('RED-09', s, 'old_drop',
                              [('OLD drop %d of %d items (max_drop %s)' % (d, s.n_old, max_drop), v)],
                              'an OLD guard failure')


@common.check('RED-10')
def check_red_10(ctx: common.Ctx) -> common.Result:
    import scorer
    s = _synth(ctx)
    strong = s.level(0.9)
    cases = []
    for m in decide.pool_members(s.p['pool']):
        c = s.cells(s.flat(s.chance_count(s.n)), ladder={m: strong})
        sub = scorer.substitute_shortcut(c, s.lock)
        if set(sub.get('substituted', {}).values()) != {m}:
            return common.Result('RED-10', common.FAIL, 'substitution picked %s instead of %s' % (sub.get('substituted'), m))
        cases.append((m, s.evaluate(sub)))
    return _must_not_pass('RED-10', cases, 'the best shortcut substituted for the candidate')


@common.check('RED-11')
def check_red_11(ctx: common.Ctx) -> common.Result:
    s = _synth(ctx)
    na = _not_applicable_old(s, 'RED-11')
    if na:
        return na
    lvl = s.level(0.5, s.n_old)
    cases = []
    for m in ('CONST', 'FIXED_1', 'MODE_REPEAT'):
        c = s.cells(s.level(0.9), old_end=lvl, old_final=dict(lvl), old_ladder={m: lvl})
        cases.append(('OLD_END = %s, drop 0' % m, s.evaluate(c)))
    return _guard_must_reject('RED-11', s, 'old_acq', cases, 'OLD_END at shortcut level with zero drop')


# --------------------------------------------------------------------------------------
# Property fixtures for DEC-01 (custom rules)
# --------------------------------------------------------------------------------------

def _seed_int(seed: Any) -> int:
    if isinstance(seed, int) and not isinstance(seed, bool):
        return seed
    s = str(seed).strip().lower()
    if s.startswith('0x'):
        s = s[2:]
    try:
        return int(s, 16)
    except ValueError:
        return int(hashlib.sha256(s.encode('utf-8')).hexdigest()[:16], 16)


def property_fixtures(lock: dict, seed: Any, n: int = 50) -> List[dict]:
    """n reviewer-seeded fixtures {id, low, high, expected_worlds, n_items_per_pool}: random effect sizes near
    margin_abs; 'high' raises the candidate by the same number of items in every world. The property is
    monotonicity: if 'low' PASSes then 'high' PASSes. Outputs must also be verdict-enum values."""
    s = Synth(lock)
    rng = random.Random(_seed_int(seed))
    out = []
    p = s.p
    for i in range(int(n)):
        comp_lvl = rng.uniform(p['chance'], max(p['chance'], min(0.8, p['chance'] + 0.5)))
        eff = p['margin_abs'] + rng.uniform(-0.1, 0.1)
        cand_lvl = min(1.0, max(0.0, comp_lvl + eff))
        cand = {w: decide._binom(rng, s.n, cand_lvl) for w in s.worlds}
        comps = {a: {w: decide._binom(rng, s.n, comp_lvl) for w in s.worlds} for a in p['comparator_arms']}
        room = min(s.n - k for k in cand.values())
        delta = rng.randint(1, max(1, s.n // 10)) if room > 0 else 0
        delta = min(delta, room)
        low = s.cells(cand, comps=comps)
        high = s.cells({w: k + delta for w, k in cand.items()}, comps=comps)
        out.append({'id': 'PF%02d' % (i + 1), 'low': low, 'high': high, 'delta_items': delta,
                    'expected_worlds': list(s.worlds), 'n_items_per_pool': s.pre()['n_items_per_pool']})
    return out
