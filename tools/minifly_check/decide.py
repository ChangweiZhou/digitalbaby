"""MRK-1 program-owned decision rule (WP3): SPEC Sections 9.5 (declarative rule), 9.8 (OC), 7.4 OC-01/DEC-01.

Python 3.9+, standard library only. A delivery supplies declarations (ENDPOINT_LOCK.primary, gates,
guards) and saved outputs; this module alone turns scored cells into a verdict.

Cells (the input of evaluate) are the dict returned by scorer.score_rows:
  cells:   [{world, world_role, arm, pool, checkpoint, stratum, determined, n, n_rows, n_exact[, answer]}]
           n = items of the (world, pool) item universe in that stratum/determined group; n_rows = rows of
           this arm present among them; n_exact = exact own-EOS native rows. 'answer' (expected hex) is
           present when the metric is balanced_exact_own_eos.
  worlds:  {world: role}
  ladder:  {'<world>|<pool>|<checkpoint>': {member: value}} computed on the same items as the arm rates
           (the counted items for the primary pool/checkpoint and for HELD/final; all items otherwise).
  ladder_status: {'<world>|<pool>|<checkpoint>': 'ok' | reason}
evaluate(cells, lock, preconditions) never trusts a verdict from anywhere else.
"""
from __future__ import annotations

import copy
import json
import math
import os
import random
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import common

DECIDE_VERSION = '1.0.0'
SCHEMA_VERDICT = 'minifly.verdict.v1'
VERDICTS = ('PASS', 'FAIL', 'INVALID', 'INCOMPLETE', 'DEV_NOT_EVIDENCE', 'CANNOT_TEST_CLAIM_AS_SPECIFIED')
INVALID_REASONS = ('INTEGRITY_ALARM', 'TREATMENT_NOT_DELIVERED', 'NONFINITE')
METRICS = ('exact_own_eos', 'balanced_exact_own_eos')
AGGREGATORS = ('paired_lcb', 'k_of_n', 'all_worlds')
RULE_TYPES = ('declarative', 'custom')
PRE_RUN_STAGES = ('lock', 'technical', 'roster', 'official', 'testbed', 'dev', 'rehearsal')  # DEC-01: VERDICT not yet due
POOLS = ('TAUGHT', 'OLD', 'NEW', 'HELD', 'PRESENTATION')
STRATA = ('S0', 'S1', 'S2', 'S3', 'S4')
GUARD_KINDS = ('old_drop', 'old_acq')
# SPEC 9.5 as amended (SPEC 20.6): the decision comparator holds the declared comparator_arms plus every declared
# arm of these roles, so leaving a control out of comparator_arms never weakens the rule (declaration
# minimization). oracle_plumbing and positive_control arms (expected to be strong) and evaluator-only CEILING_*/
# FLOOR_* arms are never added; the testbed view (mode 'testbed') adds none.
IMPLIED_COMPARATOR_ROLES = ('no_prior_teaching', 'shuf', 'frozen', 'lesion')
OFFICIAL_ROLE = 'official'
NON_EVIDENCE_ROLES = ('technical', 'dev', 'rehearsal')
CUSTOM_TIMEOUT_S = 20.0

# Shortcut ladder (SPEC 9.2), in tie-break order.
FIXED_MEMBERS = tuple('FIXED_%d' % k for k in range(9))
EXPO_INDEP_MEMBERS = ('CONST', 'LENFREQ', 'SUFFIX1', 'SUFFIX2', 'SUFFIX3', 'COPY_A', 'COPY_B', 'ID_ZERO',
                      'DIGIT_NO_CARRY') + FIXED_MEMBERS + ('MODE_REPEAT',)
EXPO_DEP_MEMBERS = ('EXACT', 'CANON', 'NEAREST')
LADDER_MEMBERS = EXPO_INDEP_MEMBERS + EXPO_DEP_MEMBERS
OPERAND_MEMBERS = ('COPY_A', 'COPY_B', 'ID_ZERO', 'DIGIT_NO_CARRY', 'CANON')


def pool_members(pool: str) -> Tuple[str, ...]:
    """Ladder subset per pool: TAUGHT/OLD/NEW exposure-independent; PRESENTATION full minus CANON; HELD full."""
    if pool in ('TAUGHT', 'OLD', 'NEW'):
        return EXPO_INDEP_MEMBERS
    if pool == 'PRESENTATION':
        return tuple(m for m in LADDER_MEMBERS if m != 'CANON')
    if pool == 'HELD':
        return LADDER_MEMBERS
    return ()


def lkey(world: str, pool: str, checkpoint: str) -> str:
    return '%s|%s|%s' % (world, pool, checkpoint)


# --------------------------------------------------------------------------------------
# Statistics: Student-t via the regularized incomplete beta function (Lentz continued fraction)
# --------------------------------------------------------------------------------------

def _betacf(a: float, b: float, x: float) -> float:
    fpmin = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, 10001):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < 1e-16:
            return h
    raise common.CheckerError('incomplete beta continued fraction did not converge (a=%r b=%r x=%r)' % (a, b, x))


def betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if a <= 0 or b <= 0:
        raise ValueError('betainc needs a, b > 0')
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    bt = math.exp(lbt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: float) -> float:
    if df <= 0:
        raise ValueError('df must be > 0')
    x = df / (df + t * t)
    ib = betainc(df / 2.0, 0.5, x)
    return 1.0 - 0.5 * ib if t >= 0 else 0.5 * ib


def t_ppf(p: float, df: float, tol: float = 1e-10) -> float:
    """Student-t quantile by bisection on t_cdf, to `tol` absolute."""
    if not (0.0 < p < 1.0):
        raise ValueError('p must be in (0, 1)')
    if p == 0.5:
        return 0.0
    if p < 0.5:
        return -t_ppf(1.0 - p, df, tol)
    lo, hi = 0.0, 1.0
    while t_cdf(hi, df) < p:
        lo, hi = hi, hi * 2.0
        if hi > 1e15:
            raise common.CheckerError('t quantile out of range (p=%r df=%r)' % (p, df))
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, df) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def hoeffding_halfwidth(alpha: float, n: int) -> float:
    """2 * sqrt(ln(1/alpha) / (2n)): the one-sided Hoeffding half-width for differences in [-1, 1]."""
    return 2.0 * math.sqrt(math.log(1.0 / alpha) / (2.0 * n))


def lower_bound(values: Sequence[float], alpha: float) -> Tuple[float, str, Optional[float], float]:
    """(LCB, method, t_quantile, sd): paired t when n >= 2 and sd > 0, else the Hoeffding bound (SPEC 9.5)."""
    n = len(values)
    if n == 0:
        return float('-inf'), 'none', None, 0.0
    mean = math.fsum(values) / n
    sd = statistics.stdev(values) if n >= 2 else 0.0
    if n >= 2 and sd > 0:
        q = t_ppf(1.0 - alpha, n - 1)
        return mean - q * sd / math.sqrt(n), 't', q, sd
    return mean - hoeffding_halfwidth(alpha, n), 'hoeffding', None, sd


def upper_bound(values: Sequence[float], alpha: float) -> Tuple[float, str, Optional[float], float]:
    """(UCB, method, t_quantile, sd) for a guard under paired_lcb: mean + t * sd / sqrt(n).

    sd == 0 with n >= 2 gives the mean (the paired t bound; the Hoeffding fallback of SPEC 9.5 is stated for
    the primary LCB only). n == 1 has no variance estimate and uses the Hoeffding upper bound."""
    n = len(values)
    if n == 0:
        return float('inf'), 'none', None, 0.0
    mean = math.fsum(values) / n
    if n == 1:
        return mean + hoeffding_halfwidth(alpha, n), 'hoeffding', None, 0.0
    sd = statistics.stdev(values)
    if sd > 0:
        q = t_ppf(1.0 - alpha, n - 1)
        return mean + q * sd / math.sqrt(n), 't', q, sd
    return mean, 't_sd0', None, 0.0


# --------------------------------------------------------------------------------------
# Cells helpers (shared with scorer and reducercheck)
# --------------------------------------------------------------------------------------

def is_finite_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def sanitize(obj: Any) -> Any:
    """JSON-safe deterministic copy: non-finite floats become strings, tuples/sets become lists."""
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, (set, frozenset)):
        return sorted((sanitize(v) for v in obj), key=lambda x: json.dumps(x, sort_keys=True))
    if isinstance(obj, float) and not math.isfinite(obj):
        return repr(obj)
    if isinstance(obj, bytes):
        return obj.hex()
    return obj


def cells_sha256(cells: Any) -> str:
    return common.sha256_bytes(common.canonical_json(sanitize(cells)))


def index_cells(cell_list: Sequence[dict]) -> Dict[Tuple[str, str, str, str], List[dict]]:
    idx: Dict[Tuple[str, str, str, str], List[dict]] = {}
    for c in cell_list or []:
        if not isinstance(c, dict):
            continue
        key = (c.get('world'), c.get('arm'), c.get('pool'), c.get('checkpoint'))
        idx.setdefault(key, []).append(c)
    return idx


def strata_applicable(lock: dict) -> bool:
    task = lock.get('task') if isinstance(lock.get('task'), dict) else {}
    fmt = task.get('format') if isinstance(task.get('format'), dict) else {}
    return fmt.get('kind') == 'addition'


def determinacy_applicable(lock: dict) -> bool:
    u = lock.get('units')
    return isinstance(u, dict) and bool(u.get('witness'))


def counted_filter(lock: dict) -> Callable[[dict], bool]:
    """Counted items of the primary: stratum in strata_counted (addition tasks) and determined (when a
    units witness is declared). Items whose stratum or determinacy could not be computed never count."""
    primary = lock.get('primary') if isinstance(lock.get('primary'), dict) else {}
    sc = primary.get('strata_counted')
    counted = set(sc) if isinstance(sc, list) else set()
    use_strata = strata_applicable(lock)
    use_det = determinacy_applicable(lock)

    def f(g: dict) -> bool:
        if use_strata and g.get('stratum') not in counted:
            return False
        if use_det and g.get('determined') is not True:
            return False
        return True
    return f


def all_items(_g: dict) -> bool:
    return True


def group_rate(groups: Sequence[dict], flt: Callable[[dict], bool], metric: str) -> Tuple[Optional[float], int, int]:
    """(rate, n, n_rows) over the groups passing flt. exact_own_eos: sum(n_exact)/sum(n). balanced: mean over
    expected answers of per-answer rates. rate is None when n == 0."""
    sel = [g for g in groups if flt(g)]
    n = sum(int(g.get('n') or 0) for g in sel)
    n_rows = sum(int(g.get('n_rows') or 0) for g in sel)
    if n == 0:
        return None, 0, n_rows
    if metric == 'balanced_exact_own_eos':
        by: Dict[Any, List[int]] = {}
        for g in sel:
            if 'answer' not in g:
                raise common.InvalidInput('balanced_exact_own_eos needs cells split by expected answer')
            acc = by.setdefault(g.get('answer'), [0, 0])
            acc[0] += int(g.get('n') or 0)
            acc[1] += int(g.get('n_exact') or 0)
        rates = [e / float(t) for t, e in by.values() if t > 0]
        return (math.fsum(rates) / len(rates)) if rates else None, n, n_rows
    return sum(int(g.get('n_exact') or 0) for g in sel) / float(n), n, n_rows


def ladder_best(cells: dict, world: str, pool: str, checkpoint: str) -> Optional[Tuple[str, float]]:
    """Best allowed ladder member (name, value) for the pool, or None when unavailable."""
    key = lkey(world, pool, checkpoint)
    st = (cells.get('ladder_status') or {}).get(key, 'ok')
    if st != 'ok':
        return None
    vals = (cells.get('ladder') or {}).get(key)
    if not isinstance(vals, dict):
        return None
    best = None
    for m in pool_members(pool):
        if m in vals and is_finite_number(vals[m]):
            if best is None or vals[m] > best[1]:
                best = (m, float(vals[m]))
    return best


# --------------------------------------------------------------------------------------
# Lock parameters
# --------------------------------------------------------------------------------------

def _num_param(d: dict, key: str, label: str, problems: List[str], nonfinite: List[str],
               lo: Optional[float] = None, hi: Optional[float] = None) -> Optional[float]:
    v = d.get(key)
    if isinstance(v, float) and not math.isfinite(v):
        nonfinite.append('%s.%s is %r' % (label, key, v))
        return None
    if not is_finite_number(v):
        problems.append('%s.%s must be a finite number, got %r' % (label, key, v))
        return None
    v = float(v)
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        problems.append('%s.%s=%r outside [%s, %s]' % (label, key, v, lo, hi))
        return None
    return v


def implied_comparator_arms(lock: Any, declared: Sequence[str] = (), candidate: Any = None) -> List[str]:
    """Declared arms whose role is in IMPLIED_COMPARATOR_ROLES (lock order), minus the candidate, the declared
    comparators and evaluator-only CEILING_*/FLOOR_* ids (SPEC 20.6)."""
    out: List[str] = []
    arms = lock.get('arms') if isinstance(lock, dict) else None
    for a in arms if isinstance(arms, list) else []:
        if not isinstance(a, dict):
            continue
        aid = a.get('id')
        if not isinstance(aid, str) or not aid or aid == candidate or aid in declared or aid in out:
            continue
        if common.is_evaluator_arm(aid) or a.get('role') not in IMPLIED_COMPARATOR_ROLES:
            continue
        out.append(aid)
    return out


def decision_comparator_arms(lock: Any) -> List[str]:
    """The arms of the decision comparator (SPEC 9.5 as amended in 20.6): declared comparator_arms, then the
    implied control arms."""
    return list(parse_params(lock)[0].get('comparator_arms') or [])


def parse_params(lock: Any, implied_comparators: bool = True) -> Tuple[dict, List[str], List[str]]:
    """(params, problems, nonfinite) from ENDPOINT_LOCK. Problems make the verdict FAIL; non-finite
    values make it INVALID NONFINITE. p['comparator_arms'] is the decision comparator (declared comparator_arms
    plus the implied control arms, SPEC 20.6); p['declared_comparator_arms'] is the declared list alone (LAD-03,
    CLM-03 and canary M06 read that). implied_comparators=False is the cycle-0 testbed view only."""
    problems: List[str] = []
    nonfinite: List[str] = []
    if not isinstance(lock, dict):
        return {}, ['ENDPOINT_LOCK is not an object'], []
    pr = lock.get('primary')
    if not isinstance(pr, dict):
        return {}, ['lock.primary missing'], []
    p: Dict[str, Any] = {}
    p['candidate_arm'] = pr.get('candidate_arm')
    if not isinstance(p['candidate_arm'], str) or not p['candidate_arm']:
        problems.append('primary.candidate_arm must be an arm id')
    comps = pr.get('comparator_arms')
    if not isinstance(comps, list) or not all(isinstance(c, str) and c for c in comps):
        problems.append('primary.comparator_arms must be a list of arm ids')
        comps = []
    p['declared_comparator_arms'] = list(comps)
    implied = implied_comparator_arms(lock, comps, p['candidate_arm']) if implied_comparators else []
    p['implied_comparator_arms'] = implied
    p['comparator_arms'] = list(comps) + implied
    p['pool'] = pr.get('pool')
    if p['pool'] not in POOLS:
        problems.append('primary.pool %r not in %s' % (p['pool'], '|'.join(POOLS)))
    p['checkpoint'] = pr.get('checkpoint')
    if not isinstance(p['checkpoint'], str) or not p['checkpoint']:
        problems.append('primary.checkpoint must be a checkpoint name')
    p['metric'] = pr.get('metric')
    if p['metric'] not in METRICS:
        problems.append('primary.metric %r is not computable by the program decide (%s)' % (p['metric'], '|'.join(METRICS)))
    p['chance'] = _num_param(pr, 'chance', 'primary', problems, nonfinite, 0.0, 1.0)
    p['margin_abs'] = _num_param(pr, 'margin_abs', 'primary', problems, nonfinite)
    p['absolute_floor'] = _num_param(pr, 'absolute_floor', 'primary', problems, nonfinite)
    p['per_world_floor'] = _num_param(pr, 'per_world_floor', 'primary', problems, nonfinite)
    mw = pr.get('min_worlds')
    if isinstance(mw, bool) or not isinstance(mw, int) or mw < 1:
        problems.append('primary.min_worlds must be an integer >= 1, got %r' % (mw,))
        mw = None
    p['min_worlds'] = mw
    agg = pr.get('aggregator')
    p['aggregator'] = None
    if not isinstance(agg, dict) or agg.get('type') not in AGGREGATORS:
        problems.append('primary.aggregator.type must be one of %s, got %r'
                        % ('|'.join(AGGREGATORS), agg.get('type') if isinstance(agg, dict) else agg))
    else:
        a = {'type': agg['type']}
        if agg['type'] == 'paired_lcb':
            a['alpha'] = _num_param(agg, 'alpha', 'primary.aggregator', problems, nonfinite, 1e-12, 0.5)
            if a['alpha'] == 0.5 or (a['alpha'] is not None and a['alpha'] <= 0):
                problems.append('primary.aggregator.alpha must be in (0, 0.5)')
        elif agg['type'] == 'k_of_n':
            k = agg.get('k')
            if isinstance(k, bool) or not isinstance(k, int) or k < 1:
                problems.append('primary.aggregator.k must be an integer >= 1, got %r' % (k,))
                k = None
            a['k'] = k
            # alpha is still used for guard bounds only under paired_lcb; none here
        p['aggregator'] = a
    rt = pr.get('rule_type', 'declarative')
    if rt not in RULE_TYPES:
        problems.append('primary.rule_type %r not in %s' % (rt, '|'.join(RULE_TYPES)))
    p['rule_type'] = rt
    sc = pr.get('strata_counted')
    if strata_applicable(lock):
        if not isinstance(sc, list) or not sc or not all(s in STRATA for s in sc):
            problems.append('primary.strata_counted must be a non-empty subset of %s for addition tasks, got %r'
                            % ('|'.join(STRATA), sc))
    p['strata_counted'] = list(sc) if isinstance(sc, list) else []
    nip = pr.get('n_items_per_pool')
    p['n_items_per_pool'] = dict(nip) if isinstance(nip, dict) else {}
    # gates and guards
    gates = lock.get('gates') or []
    if not isinstance(gates, list):
        problems.append('lock.gates must be a list')
        gates = []
    p['gates'] = []
    for g in gates:
        if not isinstance(g, dict) or not isinstance(g.get('id'), str) or not g.get('id'):
            problems.append('gate entry without id: %r' % (g,))
            continue
        p['gates'].append(g['id'])
    guards = lock.get('guards') or []
    if not isinstance(guards, list):
        problems.append('lock.guards must be a list')
        guards = []
    p['guards'] = []
    for g in guards:
        if not isinstance(g, dict) or g.get('kind') not in GUARD_KINDS:
            problems.append('guard %r: kind must be one of %s' % (g, '|'.join(GUARD_KINDS)))
            continue
        gid = g.get('id') or g['kind']
        spec = {'id': gid, 'kind': g['kind'], 'pool': g.get('pool') or 'OLD'}
        if spec['pool'] not in POOLS:
            problems.append('guard %s: pool %r invalid' % (gid, spec['pool']))
            continue
        if g['kind'] == 'old_drop':
            spec['max_drop'] = _num_param(g, 'max_drop', 'guard %s' % gid, problems, nonfinite)
        else:
            spec['margin'] = _num_param(g, 'margin', 'guard %s' % gid, problems, nonfinite)
        p['guards'].append(spec)
    split = lock.get('split') if isinstance(lock.get('split'), dict) else {}
    p['old_pool_exists'] = ('OLD' in p['n_items_per_pool']) or bool(split.get('old_manifest'))
    p['old_end'] = 'old_end'
    p['final'] = 'final'
    return p, problems, nonfinite


def testbed_lock(lock: dict, arm: str) -> dict:
    """Lock view for the cycle-0 testbed (SPEC 3.1 C0, 6.2 item 6): the evaluator-only arm takes the
    candidate's place, there are no learner comparator arms yet, and each technical world must clear the
    margin on its own (all_worlds, min_worlds 1). Floors, chance, ladder and guards stay the delivery's."""
    lv = copy.deepcopy(lock)
    pr = lv.setdefault('primary', {})
    pr['candidate_arm'] = arm
    pr['comparator_arms'] = []
    pr['min_worlds'] = 1
    pr['aggregator'] = {'type': 'all_worlds'}
    pr['rule_type'] = 'declarative'
    lv['gates'] = []
    return lv


# --------------------------------------------------------------------------------------
# The rule on per-world vectors (shared by evaluate and oc)
# --------------------------------------------------------------------------------------

def apply_rule(d: Sequence[float], v_cand: Sequence[float], p: dict) -> Tuple[bool, dict]:
    """Primary aggregator plus min_worlds and the absolute and per-world floors."""
    n = len(d)
    agg = p['aggregator'] or {}
    margin = p['margin_abs']
    info: Dict[str, Any] = {'method': agg.get('type'), 'n_worlds': n, 'margin_abs': margin}
    if n == 0:
        info.update({'pass': False, 'reason': 'no worlds'})
        return False, info
    mean_d = math.fsum(d) / n
    info['mean_d'] = mean_d
    if agg.get('type') == 'paired_lcb':
        lcb, method, q, sd = lower_bound(d, agg['alpha'])
        info.update({'lcb': lcb, 'lcb_method': method, 't_quantile': q, 'sd': sd, 'alpha': agg['alpha']})
        prim = lcb > margin
    elif agg.get('type') == 'k_of_n':
        cnt = sum(1 for x in d if x > margin)
        info.update({'count_above': cnt, 'k': agg.get('k'), 'lcb': None})
        prim = agg.get('k') is not None and cnt >= agg['k']
    else:
        cnt = sum(1 for x in d if x > margin)
        info.update({'count_above': cnt, 'lcb': None})
        prim = cnt == n
    mean_cand = math.fsum(v_cand) / n
    mw_ok = p['min_worlds'] is not None and n >= p['min_worlds']
    abs_ok = mean_cand >= p['absolute_floor']
    pw_ok = min(d) >= p['per_world_floor']
    info.update({'primary_pass': prim, 'mean_cand': mean_cand,
                 'min_worlds': {'required': p['min_worlds'], 'n_worlds': n, 'pass': mw_ok},
                 'absolute_floor': {'floor': p['absolute_floor'], 'mean_cand': mean_cand, 'pass': abs_ok},
                 'per_world_floor': {'floor': p['per_world_floor'], 'min_d': min(d), 'pass': pw_ok}})
    ok = prim and mw_ok and abs_ok and pw_ok
    info['pass'] = ok
    return ok, info


# --------------------------------------------------------------------------------------
# Custom reducers (rule_type custom): subprocess only
# --------------------------------------------------------------------------------------

def run_custom_reducer(path: Any, cells: Any, timeout: float = CUSTOM_TIMEOUT_S) -> Tuple[Optional[str], str]:
    """Run a custom reducer as `python3 -I <path>` with canonical cells on stdin and a 20 s timeout.
    Returns (verdict, detail); verdict is None for a non-enum, failed or timed-out run (FAIL)."""
    p = Path(str(path)) if path else None
    if p is None or not p.is_file():
        return None, 'custom reducer %s not found' % path
    data = common.canonical_json(sanitize(cells))
    tmp = tempfile.mkdtemp(prefix='mfc_reducer_')
    env = {'PATH': os.environ.get('PATH', '/usr/bin:/bin'), 'LC_ALL': 'C', 'PYTHONDONTWRITEBYTECODE': '1'}
    try:
        proc = subprocess.run([sys.executable or 'python3', '-I', str(p.resolve())], input=data,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, cwd=tmp, env=env)
    except subprocess.TimeoutExpired:
        return None, 'custom reducer timed out after %ss' % timeout
    except OSError as e:
        return None, 'custom reducer could not start: %s' % e
    finally:
        try:
            for f in os.listdir(tmp):
                fp = os.path.join(tmp, f)
                if os.path.isfile(fp):
                    os.remove(fp)
            os.rmdir(tmp)
        except OSError:
            pass
    if proc.returncode != 0:
        return None, 'custom reducer exited %d: %s' % (proc.returncode,
                                                     proc.stderr.decode('utf-8', 'replace').strip()[-300:])
    out = proc.stdout.decode('utf-8', 'replace').strip()
    v: Any = out
    if out.startswith('{'):
        try:
            obj = json.loads(out)
            v = obj.get('verdict') if isinstance(obj, dict) else None
        except ValueError:
            v = None
    if v not in VERDICTS:
        return None, 'custom reducer returned a non-enum value %r' % (out[:120],)
    return v, 'custom reducer returned %s' % v


# --------------------------------------------------------------------------------------
# evaluate (SPEC 9.5)
# --------------------------------------------------------------------------------------

def _scan_nonfinite(cells: dict) -> Tuple[List[str], List[str]]:
    nonfinite: List[str] = []
    malformed: List[str] = []
    for c in cells.get('cells') or []:
        if not isinstance(c, dict):
            malformed.append('cell is not an object: %r' % (c,))
            continue
        vals = {}
        for k in ('n', 'n_rows', 'n_exact'):
            v = c.get(k)
            if v is None or (isinstance(v, float) and not math.isfinite(v)):
                nonfinite.append('%s/%s/%s/%s %s=%r' % (c.get('world'), c.get('arm'), c.get('pool'),
                                                        c.get('checkpoint'), k, v))
            elif isinstance(v, bool) or not isinstance(v, int) or v < 0:
                malformed.append('%s/%s/%s/%s %s=%r is not a count' % (c.get('world'), c.get('arm'), c.get('pool'),
                                                                     c.get('checkpoint'), k, v))
            else:
                vals[k] = v
        if len(vals) == 3 and not (vals['n_exact'] <= vals['n_rows'] <= vals['n']):
            malformed.append('%s/%s/%s/%s counts inconsistent (n=%d n_rows=%d n_exact=%d)'
                             % (c.get('world'), c.get('arm'), c.get('pool'), c.get('checkpoint'),
                                vals['n'], vals['n_rows'], vals['n_exact']))
    for key, vals in (cells.get('ladder') or {}).items():
        if not isinstance(vals, dict):
            malformed.append('ladder %s is not an object' % key)
            continue
        for m, v in vals.items():
            if v is None or (isinstance(v, float) and not math.isfinite(v)):
                nonfinite.append('ladder %s %s=%r' % (key, m, v))
            elif not is_finite_number(v) or not (0.0 <= v <= 1.0):
                malformed.append('ladder %s %s=%r outside [0, 1]' % (key, m, v))
    return nonfinite, malformed


def _pre_status(v: Any) -> str:
    if v is True or v == 'PASS':
        return 'PASS'
    if v is False or v == 'FAIL':
        return 'FAIL'
    return 'MISSING'


def _required_cells(p: dict, world: str, mode: str) -> List[Tuple[str, str, str, str, str]]:
    """[(kind, arm, pool, checkpoint, why)] required for one world."""
    req = [('primary', p['candidate_arm'], p['pool'], p['checkpoint'], 'candidate')]
    for a in p['comparator_arms']:
        req.append(('primary', a, p['pool'], p['checkpoint'], 'comparator'))
    for g in p['guards']:
        if g['kind'] == 'old_drop':
            req.append(('guard', p['candidate_arm'], g['pool'], p['old_end'], g['id']))
            req.append(('guard', p['candidate_arm'], g['pool'], p['final'], g['id']))
        elif g['kind'] == 'old_acq':
            req.append(('guard', p['candidate_arm'], g['pool'], p['old_end'], g['id']))
    return req


def _world_problems(cells: dict, idx: dict, p: dict, world: str, mode: str, expected_n: dict,
                    flt: Callable[[dict], bool]) -> Tuple[List[str], set]:
    """(missing reasons, guard ids not measured) for one world."""
    missing: List[str] = []
    unmeasured = set()
    for kind, arm, pool, ck, why in _required_cells(p, world, mode):
        groups = idx.get((world, arm, pool, ck))
        reason = None
        if not groups:
            reason = 'no cells for %s/%s/%s/%s (%s)' % (world, arm, pool, ck, why)
        else:
            n = sum(int(g.get('n') or 0) for g in groups if isinstance(g.get('n'), int))
            n_rows = sum(int(g.get('n_rows') or 0) for g in groups if isinstance(g.get('n_rows'), int))
            en = expected_n.get(pool)
            if n_rows < n:
                reason = '%s/%s/%s/%s has rows for %d of %d items' % (world, arm, pool, ck, n_rows, n)
            elif isinstance(en, int) and not isinstance(en, bool) and n != en:
                reason = '%s/%s/%s/%s has %d items, lock n_items_per_pool.%s=%d' % (world, arm, pool, ck, n, pool, en)
            elif kind == 'primary' and sum(int(g.get('n') or 0) for g in groups
                                           if flt(g) and isinstance(g.get('n'), int)) == 0:
                reason = '%s/%s/%s/%s has no counted items' % (world, arm, pool, ck)
        if reason:
            if kind == 'guard' and mode == 'testbed':
                unmeasured.add(why)
            else:
                missing.append(reason)
    if ladder_best(cells, world, p['pool'], p['checkpoint']) is None:
        missing.append('no ladder for %s (%s)' % (lkey(world, p['pool'], p['checkpoint']),
                                                  (cells.get('ladder_status') or {}).get(
                                                      lkey(world, p['pool'], p['checkpoint']), 'absent')))
    for g in p['guards']:
        if g['kind'] == 'old_acq' and ladder_best(cells, world, g['pool'], p['old_end']) is None:
            if mode == 'testbed':
                unmeasured.add(g['id'])
            else:
                missing.append('no ladder for %s (guard %s)' % (lkey(world, g['pool'], p['old_end']), g['id']))
    return missing, unmeasured


def _core(cells: dict, idx: dict, lock: dict, p: dict, worlds: List[str], pre: dict, mode: str,
          unmeasured: set) -> dict:
    """Evaluate the rule on complete worlds. Returns {'verdict': PASS|FAIL, primary, gates, guards, ...}."""
    flt = counted_filter(lock)
    metric = p['metric']
    per_world = []
    d_vals: List[float] = []
    cand_vals: List[float] = []
    comp_vals: List[float] = []
    for w in worlds:
        v_cand, n_c, _ = group_rate(idx.get((w, p['candidate_arm'], p['pool'], p['checkpoint']), []), flt, metric)
        comps = [('chance', p['chance'])]
        for a in p['comparator_arms']:
            v, _, _ = group_rate(idx.get((w, a, p['pool'], p['checkpoint']), []), flt, metric)
            comps.append((a, v if v is not None else 0.0))
        best = ladder_best(cells, w, p['pool'], p['checkpoint'])
        if best is not None:
            comps.append(('shortcut:' + best[0], best[1]))
        src, v_comp = comps[0]
        for name, v in comps[1:]:
            if v > v_comp:
                src, v_comp = name, v
        v_cand = v_cand if v_cand is not None else 0.0
        d = v_cand - v_comp
        per_world.append({'world': w, 'role': (cells.get('worlds') or {}).get(w), 'v_cand': v_cand, 'n_counted': n_c,
                          'v_comp': v_comp, 'comparator': src, 'best_shortcut': ({'name': best[0], 'value': best[1]}
                                                                                if best else None),
                          'comparator_arms': {a: v for a, v in comps if a not in ('chance',) and
                                              not a.startswith('shortcut:')}, 'd': d})
        d_vals.append(d)
        cand_vals.append(v_cand)
        comp_vals.append(v_comp)
    ok, info = apply_rule(d_vals, cand_vals, p)
    reasons: List[str] = []
    if not info.get('primary_pass'):
        reasons.append('primary aggregator %s did not clear margin_abs %s' % (info.get('method'), p['margin_abs']))
    for k in ('min_worlds', 'absolute_floor', 'per_world_floor'):
        if isinstance(info.get(k), dict) and not info[k]['pass']:
            reasons.append('%s not met: %s' % (k, info[k]))
    primary = {'mean_cand': info.get('mean_cand'), 'mean_comp': (math.fsum(comp_vals) / len(comp_vals)) if comp_vals else None,
               'per_world': d_vals, 'per_world_detail': per_world, 'lcb': info.get('lcb'), 'method': info.get('method'),
               'lcb_method': info.get('lcb_method'), 't_quantile': info.get('t_quantile'), 'sd': info.get('sd'),
               'mean_d': info.get('mean_d'), 'count_above': info.get('count_above'), 'k': info.get('k'),
               'margin_abs': p['margin_abs'], 'n_worlds': len(worlds), 'pass': info.get('primary_pass'),
               'comparator_arms': list(p['comparator_arms']),
               'implied_comparator_arms': list(p.get('implied_comparator_arms') or [])}
    floors = {k: info.get(k) for k in ('min_worlds', 'absolute_floor', 'per_world_floor')}
    # plumbing gates
    gates: Dict[str, str] = {}
    gates_ok = True
    if mode != 'testbed':
        given = pre.get('gates') if isinstance(pre.get('gates'), dict) else {}
        for gid in p['gates']:
            st = _pre_status(given.get(gid))
            gates[gid] = st
            if st != 'PASS':
                gates_ok = False
                reasons.append('plumbing gate %s %s' % (gid, st))
    # guards
    guards: Dict[str, str] = {}
    gdetail: Dict[str, dict] = {}
    guards_ok = True
    kinds = set(g['kind'] for g in p['guards'])
    if p['old_pool_exists']:
        for need in GUARD_KINDS:
            if need not in kinds:
                guards[need] = 'MISSING_IN_LOCK'
                guards_ok = False
                reasons.append('an OLD pool exists but the lock has no %s guard' % need)
    alpha = (p['aggregator'] or {}).get('alpha')
    for g in p['guards']:
        if g['id'] in unmeasured:
            guards[g['id']] = 'NOT_MEASURED'
            gdetail[g['id']] = {'kind': g['kind'], 'note': 'testbed rows do not measure this guard'}
            continue
        vals = []
        for w in worlds:
            end, _, _ = group_rate(idx.get((w, p['candidate_arm'], g['pool'], p['old_end']), []), all_items, metric)
            end = end if end is not None else 0.0
            if g['kind'] == 'old_drop':
                fin, _, _ = group_rate(idx.get((w, p['candidate_arm'], g['pool'], p['final']), []), all_items, metric)
                vals.append(end - (fin if fin is not None else 0.0))
            else:
                best = ladder_best(cells, w, g['pool'], p['old_end'])
                vals.append(end - (best[1] if best else 1.0))
        mean = math.fsum(vals) / len(vals) if vals else None
        if g['kind'] == 'old_drop':
            if alpha is not None and (p['aggregator'] or {}).get('type') == 'paired_lcb':
                bound, method, _q, _sd = upper_bound(vals, alpha)
            else:
                bound, method = mean, 'mean'
            okg = bound is not None and g['max_drop'] is not None and bound <= g['max_drop']
            gdetail[g['id']] = {'kind': 'old_drop', 'mean_drop': mean, 'bound': bound, 'bound_method': method,
                                'max_drop': g['max_drop'], 'per_world': vals}
        else:
            okg = mean is not None and g['margin'] is not None and mean >= g['margin']
            gdetail[g['id']] = {'kind': 'old_acq', 'mean_acq': mean, 'margin': g['margin'], 'per_world': vals}
        guards[g['id']] = 'PASS' if okg else 'FAIL'
        if not okg:
            guards_ok = False
            reasons.append('guard %s FAIL: %s' % (g['id'], gdetail[g['id']]))
    verdict = 'PASS' if (ok and gates_ok and guards_ok) else 'FAIL'
    custom = None
    if p['rule_type'] == 'custom' and mode != 'testbed':
        rv, why = run_custom_reducer(pre.get('reducer_path'), cells)
        custom = {'verdict': rv, 'detail': why}
        mean_d = info.get('mean_d')
        program_ok = (mean_d is not None and mean_d > p['margin_abs'] and info['min_worlds']['pass']
                      and info['absolute_floor']['pass'] and info['per_world_floor']['pass'] and gates_ok and guards_ok)
        if rv is None:
            verdict = 'FAIL'
            reasons.append(why)
        elif rv == 'PASS':
            verdict = 'PASS' if program_ok else 'FAIL'
            if not program_ok:
                reasons.append('custom reducer PASS overruled: mean d must exceed margin_abs and floors, gates and '
                               'guards must pass')
        else:
            verdict = 'FAIL' if rv in ('DEV_NOT_EVIDENCE', 'PASS') else rv
            reasons.append(why)
    return {'verdict': verdict, 'primary': primary, 'floors': floors, 'gates': gates, 'guards': guards,
            'guard_detail': gdetail, 'reasons': reasons, 'custom_reducer': custom}


def evaluate(cells: Any, lock: Any, preconditions: Optional[dict] = None) -> dict:
    """The program-owned decision rule (SPEC 9.5). Returns the VERDICT.json body.

    preconditions keys: inert_alarms (list; non-empty -> INVALID INTEGRITY_ALARM), adm_tb and adm_ctrl
    ('PASS' | 'FAIL' | other = not established), gates ({gate_id: 'PASS'|True|...}), expected_worlds (list),
    expected_n_worlds (int), n_items_per_pool (override for synthetic cells), endpoint_lock_sha256,
    source_lock_sha256, reducer_path (custom rules), mode ('standard' | 'testbed')."""
    pre = dict(preconditions or {})
    mode = pre.get('mode') or 'standard'
    if not isinstance(cells, dict):
        cells = {'cells': list(cells or [])}
    out: Dict[str, Any] = {
        'schema': SCHEMA_VERDICT, 'verdict': None, 'invalid_reason': None, 'shadow_verdict': None,
        'primary': {}, 'gates': {}, 'guards': {}, 'guard_detail': {}, 'floors': {}, 'reasons': [],
        'preconditions': {}, 'world_roles': {}, 'mode': mode, 'decide_version': DECIDE_VERSION,
        'endpoint_lock_sha256': pre.get('endpoint_lock_sha256'), 'source_lock_sha256': pre.get('source_lock_sha256'),
        'cells_sha256': cells_sha256(cells),
    }

    def done(verdict, reason=None, shadow=None, invalid=None):
        out['verdict'] = verdict
        out['shadow_verdict'] = shadow
        out['invalid_reason'] = invalid
        if reason:
            out['reasons'].insert(0, reason)
        return out

    # world roles first: a precondition verdict (INVALID, CANNOT_TEST) on official worlds still names the worlds
    # it was computed on, so the claim reports '<n> <ROLE> worlds' (SPEC 9.9) and never mistakes an official
    # INVALID for a dev-only result (CLM-09)
    roles = dict(cells.get('worlds') or {})
    expected = pre.get('expected_worlds')
    if expected is None:
        expected = sorted(roles)
    expected = [str(w) for w in expected]
    out['world_roles'] = {w: roles.get(w) for w in expected}
    inert = pre.get('inert_alarms')
    pstat = {'inert': 'FAIL' if inert else ('PASS' if isinstance(inert, list) else 'MISSING'),
             'adm_tb': _pre_status(pre.get('adm_tb')), 'adm_ctrl': _pre_status(pre.get('adm_ctrl'))}
    if mode == 'testbed':
        pstat = {'inert': 'PASS', 'adm_tb': 'PASS', 'adm_ctrl': 'PASS'}
    out['preconditions'] = dict(pstat)
    # 1-3: preconditions in order
    if pstat['inert'] == 'FAIL':
        return done('INVALID', 'INERT alarm: %s' % ('; '.join(str(a) for a in inert)[:500]), invalid='INTEGRITY_ALARM')
    if pstat['adm_tb'] == 'FAIL':
        return done('CANNOT_TEST_CLAIM_AS_SPECIFIED', 'ADM-TB failure: the testbed cannot test the claim')
    if pstat['adm_ctrl'] == 'FAIL':
        return done('INVALID', 'ADM-CTRL failure', invalid='TREATMENT_NOT_DELIVERED')
    p, problems, lock_nonfinite = parse_params(lock, implied_comparators=(mode != 'testbed'))
    if lock_nonfinite:
        return done('INVALID', 'non-finite lock values: %s' % '; '.join(lock_nonfinite), invalid='NONFINITE')
    if problems:
        out['reasons'].extend(problems)
        return done('FAIL', 'lock does not define a decidable rule')
    idx = index_cells(cells.get('cells') or [])
    expected_n = pre.get('n_items_per_pool')
    if not isinstance(expected_n, dict):
        expected_n = p['n_items_per_pool']
    flt = counted_filter(lock)
    missing: List[str] = []
    complete: List[str] = []
    unmeasured_all = set()
    enw = pre.get('expected_n_worlds')
    if isinstance(enw, int) and not isinstance(enw, bool) and len(expected) < enw:
        missing.append('%d worlds present, %d expected' % (len(expected), enw))
    if not expected:
        missing.append('no worlds to evaluate')
    for w in expected:
        miss, unm = _world_problems(cells, idx, p, w, mode, expected_n, flt)
        unmeasured_all |= unm
        if miss:
            missing.extend(miss)
        else:
            complete.append(w)
    unknown = [k for k, v in pstat.items() if v != 'PASS']
    nonfinite, malformed = _scan_nonfinite(cells)
    if malformed:
        out['reasons'].extend(malformed[:50])
        return done('FAIL', 'malformed cells')
    # 4: missing cells (or preconditions not established) -> INCOMPLETE with a shadow on complete worlds
    if missing or unknown:
        out['reasons'].extend(missing[:100])
        if unknown:
            out['reasons'].append('preconditions not established: %s' % ', '.join(unknown))
        shadow = None
        if complete and not nonfinite:
            core = _core(cells, idx, lock, p, complete, pre, mode, unmeasured_all)
            shadow = core['verdict']
            for k in ('primary', 'floors', 'gates', 'guards', 'guard_detail'):
                out[k] = core[k]
            out['shadow_worlds'] = complete
        return done('INCOMPLETE', 'missing cells or preconditions', shadow=shadow)
    # 5: non-finite values
    if nonfinite:
        out['reasons'].extend(nonfinite[:50])
        return done('INVALID', 'non-finite values in cells', invalid='NONFINITE')
    core = _core(cells, idx, lock, p, expected, pre, mode, unmeasured_all)
    for k in ('primary', 'floors', 'gates', 'guards', 'guard_detail'):
        out[k] = core[k]
    out['reasons'].extend(core['reasons'])
    if core.get('custom_reducer') is not None:
        out['custom_reducer'] = core['custom_reducer']
    non_evidence = [w for w in expected if roles.get(w) != OFFICIAL_ROLE]
    if non_evidence or mode == 'testbed':
        return done('DEV_NOT_EVIDENCE', 'worlds %s are not official evidence' % ','.join(non_evidence[:10]),
                    shadow=core['verdict'])
    return done(core['verdict'])


def verdict_or_shadow(v: dict) -> Optional[str]:
    """The decisive value for screens (testbed, substitution): the shadow verdict under DEV_NOT_EVIDENCE."""
    if v.get('verdict') == 'DEV_NOT_EVIDENCE':
        return v.get('shadow_verdict')
    return v.get('verdict')


def passes(v: dict) -> bool:
    """True when either the verdict or its shadow is PASS (used by must-not-PASS qualifications)."""
    return v.get('verdict') == 'PASS' or v.get('shadow_verdict') == 'PASS'


# --------------------------------------------------------------------------------------
# OC (SPEC 9.8)
# --------------------------------------------------------------------------------------

def _binom(rng: random.Random, n: int, p: float) -> int:
    """Binomial(n, p) by CDF inversion in chunks of at most 256 trials (no underflow), stdlib only."""
    if n <= 0 or p <= 0.0:
        return 0
    if p >= 1.0:
        return n
    flip = p > 0.5
    q = 1.0 - p if flip else p
    total = 0
    left = n
    r = q / (1.0 - q)
    while left > 0:
        m = min(left, 256)
        left -= m
        u = rng.random()
        pk = (1.0 - q) ** m
        cdf = pk
        k = 0
        while u > cdf and k < m:
            pk *= r * (m - k) / (k + 1.0)
            k += 1
            cdf += pk
        total += k
    return n - total if flip else total


def oc(lock: dict, dev_summary: dict, sims: int = 2000, seed: int = 0) -> dict:
    """Monte Carlo operating characteristic of the locked rule (random.Random(seed)).

    Per world: comparator ~ Binomial(n, p_comp)/n (floored at chance); candidate ~ Binomial(n, p_comp + delta_w)/n
    with delta_w ~ Normal(target_delta, between_world_sd) (delta_w = 0 mean under the null). dev_summary keys:
    per_world_d [], mean_comp, n_items (counted items per world), optional target_delta, between_world_sd,
    n_worlds. Lock overrides (stricter of the two is used): primary.oc_target_delta, primary.oc_between_world_sd.
    Guards and plumbing gates are not simulated (assumed to pass)."""
    p, problems, nonfinite = parse_params(lock)
    if problems or nonfinite:
        raise common.InvalidInput('lock does not define a decidable rule: %s' % '; '.join(problems + nonfinite))
    ds = dict(dev_summary or {})
    pw = [float(x) for x in (ds.get('per_world_d') or []) if is_finite_number(x)]
    pr = lock.get('primary') or {}
    targets = []
    if is_finite_number(pr.get('oc_target_delta')):
        targets.append(float(pr['oc_target_delta']))
    if is_finite_number(ds.get('target_delta')):
        targets.append(float(ds['target_delta']))
    if pw:
        targets.append(math.fsum(pw) / len(pw))
    if not targets:
        raise common.MissingInput('OC needs a target effect (dev per-world d or primary.oc_target_delta)')
    target = min(targets)
    sds = []
    if is_finite_number(pr.get('oc_between_world_sd')):
        sds.append(float(pr['oc_between_world_sd']))
    if is_finite_number(ds.get('between_world_sd')):
        sds.append(float(ds['between_world_sd']))
    if len(pw) >= 2:
        sds.append(statistics.stdev(pw))
    if not sds:
        raise common.MissingInput('OC needs a between-world SD (>= 2 dev worlds or primary.oc_between_world_sd)')
    sd = max(sds)
    if not is_finite_number(ds.get('mean_comp')):
        raise common.MissingInput('OC needs the dev mean comparator rate')
    p_comp = min(1.0, max(0.0, float(ds['mean_comp'])))
    n_items = ds.get('n_items')
    if isinstance(n_items, bool) or not isinstance(n_items, int) or n_items < 1:
        raise common.MissingInput('OC needs the counted items per world (n_items >= 1), got %r' % (n_items,))
    worlds = lock.get('worlds') if isinstance(lock.get('worlds'), dict) else {}
    off = worlds.get('official') if isinstance(worlds.get('official'), dict) else {}
    n_w = len(off.get('ids') or []) or (off.get('n') if isinstance(off.get('n'), int) and not isinstance(off.get('n'), bool) else 0)
    if not n_w and isinstance(ds.get('n_worlds'), int):
        n_w = ds['n_worlds']
    if not n_w:
        n_w = p['min_worlds'] or 0
    if n_w < 1:
        raise common.MissingInput('OC needs the number of official worlds')
    chance = p['chance']
    rng = random.Random(seed)

    def run(mean_delta: float) -> float:
        passes_ = 0
        for _ in range(int(sims)):
            d, vc = [], []
            for _w in range(n_w):
                delta = mean_delta + (rng.gauss(0.0, sd) if sd > 0 else 0.0)
                pc = min(1.0, max(0.0, p_comp + delta))
                x_comp = _binom(rng, n_items, p_comp) / float(n_items)
                x_cand = _binom(rng, n_items, pc) / float(n_items)
                comp = max(chance, x_comp)
                d.append(x_cand - comp)
                vc.append(x_cand)
            ok, _info = apply_rule(d, vc, p)
            passes_ += 1 if ok else 0
        return passes_ / float(sims)

    p_target = run(target)
    p_null = run(0.0)
    return {'p_pass_target': p_target, 'p_pass_null': p_null, 'sims': int(sims), 'seed': seed, 'n_worlds': n_w,
            'n_items': n_items, 'p_comp': p_comp, 'chance': chance, 'target_delta': target,
            'between_world_sd': sd, 'aggregator': p['aggregator'], 'margin_abs': p['margin_abs'],
            'note': 'guards and plumbing gates are not simulated; target = min(declared, dev mean d), sd = max(declared, dev sd)'}


def _oc_min_power(ctx: common.Ctx) -> float:
    tf = ctx.kb.constraints().get('threshold_floors')
    v = tf.get('oc_min_power') if isinstance(tf, dict) else None
    if not is_finite_number(v):
        raise common.MissingInput('CONSTRAINTS.threshold_floors.oc_min_power missing')
    return float(v)


@common.check('OC-01')
def check_oc_01(ctx: common.Ctx) -> common.Result:
    import scorer
    summary = scorer.dev_summary(ctx)
    res = oc(ctx.lock, summary)
    need = _oc_min_power(ctx)
    ok = res['p_pass_target'] >= need
    detail = 'P(pass | target)=%.3f %s oc_min_power %.2f (P(pass | null)=%.3f; %d sims, %d worlds x %d items, ' \
             'p_comp %.3f, target %.3f, sd %.3f)' % (res['p_pass_target'], '>=' if ok else '<', need,
                                                     res['p_pass_null'], res['sims'], res['n_worlds'], res['n_items'],
                                                     res['p_comp'], res['target_delta'], res['between_world_sd'])
    if not ok:
        detail += '; worlds_role is capped at MAIN'
    return common.Result('OC-01', common.PASS if ok else common.FAIL, detail, data={'oc': res})


def cmd_oc(args: Any) -> int:
    """MFC oc <dir>: operating characteristic of the locked rule from dev rows. Exit 0 when
    P(pass | target) >= oc_min_power, 1 when below, 2 when inputs are missing."""
    import scorer
    ctx = common.make_ctx(args, delivery=args.dir, stage='oc', options={'invoked_by': 'oc'})
    ok, why = common.v1_status(ctx.delivery)
    if not ok:
        sys.stderr.write('legacy delivery: use retro (%s)\n' % why)
        return common.EXIT_MISSING
    try:
        summary = scorer.dev_summary(ctx)
        res = oc(ctx.lock, summary)
        need = _oc_min_power(ctx)
    except common.MissingInput as e:
        sys.stderr.write('MISSING: %s\n' % e)
        return common.EXIT_MISSING
    except common.InvalidInput as e:
        sys.stderr.write('FAIL (invalid input): %s\n' % e)
        return common.EXIT_FAIL
    res['oc_min_power'] = need
    res['pass'] = res['p_pass_target'] >= need
    if getattr(args, 'json', False):
        print(json.dumps(sanitize(res), indent=2, sort_keys=True))
    else:
        print('oc %s: P(pass | target)=%.3f P(pass | null)=%.3f (min power %.2f) -> %s'
              % (ctx.delivery_rel, res['p_pass_target'], res['p_pass_null'], need, 'PASS' if res['pass'] else 'FAIL'))
    return common.EXIT_PASS if res['pass'] else common.EXIT_FAIL


# --------------------------------------------------------------------------------------
# DEC-01
# --------------------------------------------------------------------------------------

VERDICT_COMPARED_FIELDS = ('verdict', 'invalid_reason', 'shadow_verdict', 'primary', 'gates', 'guards',
                           'decide_version', 'endpoint_lock_sha256', 'source_lock_sha256', 'cells_sha256',
                           'world_roles')
CUSTOM_RULE_SCOPE = 'custom_rule:%s'
N_PROPERTY_FIXTURES = 50


def _family(agent: Any) -> Optional[str]:
    if not isinstance(agent, str) or not agent.strip():
        return None
    return agent.strip().split(':', 1)[0].strip().lower()


def custom_rule_problems(ctx: common.Ctx) -> Tuple[List[str], List[str]]:
    """(problems, missing) for rule_type custom: UD, non-executor reducer author and reviewer, endpoint-locked
    reducer, and 50 reviewer-seeded property fixtures (monotone, enum outputs)."""
    import reducercheck
    lock = ctx.lock
    pr = lock.get('primary') or {}
    problems: List[str] = []
    missing: List[str] = []
    name = ctx.delivery_name
    scope = CUSTOM_RULE_SCOPE % name
    uds = lock.get('user_decisions') or []
    found_ud = False
    for ud_id in uds if isinstance(uds, list) else []:
        row = ctx.kb.user_decision(ud_id)
        if row and scope in (row.get('scope') or []):
            found_ud = True
            break
    if not found_ud:
        problems.append('no USER_DECISIONS row cited in lock.user_decisions with scope %s' % scope)
    author = _family(lock.get('author_agent'))
    cr = pr.get('custom_reducer') if isinstance(pr.get('custom_reducer'), dict) else None
    rpath = None
    if cr is None or not isinstance(cr.get('path'), str):
        problems.append('primary.custom_reducer {path, author_agent} missing')
    else:
        rel = common.norm_rel(cr['path'])
        rpath = ctx.path(rel)
        if not rpath.is_file():
            missing.append('custom reducer %s absent' % rel)
            rpath = None
        else:
            locked = (lock.get('endpoint_files_sha256') or {}).get(rel)
            if locked != common.sha256_file(rpath):
                problems.append('custom reducer %s is not endpoint-locked with a matching sha256' % rel)
        if _family(cr.get('author_agent')) in (None, author):
            problems.append('custom reducer author %r must differ from the executor %r'
                            % (cr.get('author_agent'), lock.get('author_agent')))
    rv_path = ctx.path('review/REVIEW.json')
    seed = None
    if not rv_path.is_file():
        missing.append('review/REVIEW.json absent (custom rules need a REVIEW by a non-executor)')
    else:
        rv = ctx.read_json('review/REVIEW.json')
        rfam = _family(rv.get('reviewer_agent') if isinstance(rv, dict) else None)
        if rfam is None or rfam == author:
            problems.append('REVIEW reviewer_agent %r is the executor family %r'
                            % (rv.get('reviewer_agent') if isinstance(rv, dict) else None, author))
        seed = rv.get('property_fixture_seed') if isinstance(rv, dict) else None
        if not isinstance(seed, str) or not seed:
            missing.append('REVIEW.property_fixture_seed (reviewer-chosen hex seed) absent')
            seed = None
    if rpath is not None and seed is not None and not problems:
        fixtures = reducercheck.property_fixtures(lock, seed, n=N_PROPERTY_FIXTURES)
        bad = []
        for fx in fixtures:
            lo, why_lo = run_custom_reducer(rpath, fx['low'])
            hi, why_hi = run_custom_reducer(rpath, fx['high'])
            if lo is None or hi is None:
                bad.append('%s: %s / %s' % (fx['id'], why_lo, why_hi))
            elif lo == 'PASS' and hi != 'PASS':
                bad.append('%s: raising the candidate turned PASS into %s' % (fx['id'], hi))
        if bad:
            problems.append('%d of %d property fixtures failed: %s' % (len(bad), len(fixtures), '; '.join(bad[:5])))
    return problems, missing


@common.check('DEC-01')
def check_dec_01(ctx: common.Ctx) -> common.Result:
    import scorer
    lock = ctx.lock
    pr = lock.get('primary') if isinstance(lock.get('primary'), dict) else {}
    rt = pr.get('rule_type')
    notes: List[str] = []
    problems: List[str] = []
    missing: List[str] = []
    if rt not in RULE_TYPES:
        problems.append('primary.rule_type %r not in %s' % (rt, '|'.join(RULE_TYPES)))
    elif rt == 'custom':
        pb, ms = custom_rule_problems(ctx)
        problems.extend(pb)
        missing.extend(ms)
        notes.append('custom rule checked (UD, authors, review, property fixtures)')
    else:
        notes.append('declarative rule evaluated only by program decide')
    vrel = 'results/official/VERDICT.json'
    has_official = scorer.role_has_rows(ctx, 'official')
    if ctx.path(vrel).is_file():
        stored = ctx.read_json(vrel)
        if not isinstance(stored, dict):
            problems.append('%s is not an object' % vrel)
        elif not has_official:
            problems.append('%s exists but there are no official OUTPUTS rows' % vrel)
        else:
            for k in ('decide_version', 'endpoint_lock_sha256', 'source_lock_sha256', 'cells_sha256'):
                if not isinstance(stored.get(k), str) or not stored.get(k):
                    problems.append('%s lacks %s' % (vrel, k))
            fresh = scorer.build_verdict(ctx)
            diffs = []
            for k in VERDICT_COMPARED_FIELDS:
                if common.canonical_json(sanitize(stored.get(k))) != common.canonical_json(sanitize(fresh.get(k))):
                    diffs.append(k)
            if diffs:
                problems.append('VERDICT.json differs from a fresh program decide on: %s (stored verdict %r, fresh %r)'
                                % (', '.join(diffs), stored.get('verdict'), fresh.get('verdict')))
            else:
                notes.append('VERDICT.json equals a fresh decide (%s, cells %s)'
                             % (fresh.get('verdict'), str(fresh.get('cells_sha256'))[:12]))
    elif has_official and ctx.stage in PRE_RUN_STAGES:
        # pre-run stage bundles (lock/roster/official, launch stages) re-run on a finished delivery: VERDICT.json
        # is written by `score --role official` after the run and is required from the publish stage on
        notes.append('official OUTPUTS rows exist and %s is not yet written; it is required at publish (stage %s)'
                     % (vrel, ctx.stage))
    elif has_official:
        missing.append('official OUTPUTS rows exist but %s does not: run MFC score --role official' % vrel)
    else:
        notes.append('no official VERDICT yet')
    if problems:
        return common.Result('DEC-01', common.FAIL, '; '.join(problems + missing))
    if missing:
        return common.Result('DEC-01', common.MISSING, '; '.join(missing))
    return common.Result('DEC-01', common.PASS, '; '.join(notes))
