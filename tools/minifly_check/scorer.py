"""MRK-1 canonical scorer (WP3): SPEC Sections 9.1-9.4, 9.6, 9.7 and checks MET-01/02, LAD-01..04,
STR-01, DET-01, INERT-01, UDI-01. Python 3.9+, standard library only.

The scorer recomputes everything from raw rows: emitted bytes and own-EOS from stream_hex, the expected
answer from prompt_hex (addition formats), strata and determinacy from item content and the candidate's
TAUGHT manifest, and the shortcut ladder from the candidate's TAUGHT manifest and its own streams. The
runner's emitted_hex/own_eos/expected_hex are only compared (two-source agreement), never trusted.

score(ctx, role) writes results/<role>/SCORES.json, results/official/VERDICT.json (role official) and
results/admission/TESTBED.json (role admission). Those three files are checker-written (RUN-01 exempt).
"""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import addition
import common
import decide

SCHEMA_SCORES = 'minifly.scores.v1'
SCHEMA_TESTBED = 'minifly.testbed.v1'
SCHEMA_CELLS = 'minifly.cells.v1'
ROLES = ('admission', 'dev', 'rehearsal', 'official')
WORLD_ROLE_OF = {'admission': 'technical', 'dev': 'dev', 'rehearsal': 'rehearsal', 'official': 'official'}
ANSWER_PATHS = ('native', 'head', 'evaluator')
LEARNER_ROLES = ('dev', 'rehearsal', 'official')
REQUIRED_ROW_FIELDS = ('world', 'arm', 'pool', 'checkpoint', 'item_id', 'prompt_hex')
MAX_LISTED = 100
LAD04_MAX_MODAL = 0.9
LAD04_MIN_ENTROPY = 0.5
COUNTED_KEYS_EXTRA = (('HELD', 'final'),)


# --------------------------------------------------------------------------------------
# Lock helpers
# --------------------------------------------------------------------------------------

def _fmt(lock: dict) -> dict:
    task = lock.get('task') if isinstance(lock.get('task'), dict) else {}
    fmt = task.get('format') if isinstance(task.get('format'), dict) else None
    if fmt is None:
        raise common.InvalidInput('lock.task.format missing')
    return fmt


def _gen_cap(fmt: dict) -> int:
    g = fmt.get('gen_cap_bytes')
    if isinstance(g, bool) or not isinstance(g, int) or g < 1:
        raise common.InvalidInput('task.format.gen_cap_bytes must be an integer >= 1, got %r' % (g,))
    return g


def _eos(fmt: dict) -> bytes:
    v = fmt.get('eos_hex')
    try:
        b = bytes.fromhex(v) if isinstance(v, str) else b''
    except ValueError:
        b = b''
    if not b:
        raise common.InvalidInput('task.format.eos_hex must be non-empty hex, got %r' % (v,))
    return b


def lock_arms(lock: dict) -> List[str]:
    out = []
    for a in lock.get('arms') or []:
        if isinstance(a, dict) and isinstance(a.get('id'), str):
            out.append(a['id'])
    return out


def arms_with_role(lock: dict, role: str) -> List[str]:
    return [a['id'] for a in (lock.get('arms') or []) if isinstance(a, dict) and a.get('role') == role
            and isinstance(a.get('id'), str)]


def world_roles_from_lock(lock: dict) -> Dict[str, str]:
    """{world: role} from lock.worlds; a world listed under two roles raises InvalidInput."""
    W = lock.get('worlds') if isinstance(lock.get('worlds'), dict) else {}
    out: Dict[str, str] = {}
    lists = [('technical', W.get('technical')), ('dev', W.get('dev')), ('rehearsal', W.get('rehearsal'))]
    off = W.get('official') if isinstance(W.get('official'), dict) else {}
    lists.append(('official', off.get('ids')))
    for role, lst in lists:
        for w in lst or []:
            if not isinstance(w, str):
                continue
            if w in out and out[w] != role:
                raise common.InvalidInput('world %s is listed as both %s and %s in lock.worlds' % (w, out[w], role))
            out[w] = role
    return out


def lock_worlds(lock: dict, role: str) -> List[str]:
    W = lock.get('worlds') if isinstance(lock.get('worlds'), dict) else {}
    if role in ('admission', 'technical'):
        return [w for w in (W.get('technical') or []) if isinstance(w, str)]
    if role in ('dev', 'rehearsal'):
        return [w for w in (W.get(role) or []) if isinstance(w, str)]
    off = W.get('official') if isinstance(W.get('official'), dict) else {}
    return [w for w in (off.get('ids') or []) if isinstance(w, str)]


def official_n(lock: dict) -> Optional[int]:
    W = lock.get('worlds') if isinstance(lock.get('worlds'), dict) else {}
    off = W.get('official') if isinstance(W.get('official'), dict) else {}
    n = off.get('n')
    return n if isinstance(n, int) and not isinstance(n, bool) else None


# --------------------------------------------------------------------------------------
# Taught-set index for the ladder
# --------------------------------------------------------------------------------------

def _modal(counter: Dict[bytes, int]) -> Optional[bytes]:
    if not counter:
        return None
    return min(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0]


class TaughtIndex:
    """Ladder predictions from one arm's TAUGHT rows (SPEC 9.2), plus the addition.Taught for strata."""

    def __init__(self, rows: Sequence[dict], fmt: dict, label: str):
        self.fmt = fmt
        self.is_add = addition.is_addition(fmt)
        self.prompts: List[bytes] = []
        self.answers: List[Optional[bytes]] = []
        const: Dict[bytes, int] = {}
        by_len: Dict[int, Dict[bytes, int]] = {}
        by_suf: Dict[int, Dict[bytes, Dict[bytes, int]]] = {1: {}, 2: {}, 3: {}}
        exact: Dict[bytes, Dict[bytes, int]] = {}
        canon: Dict[Tuple[str, str], Dict[bytes, int]] = {}
        for i, r in enumerate(rows):
            ph = r.get('prompt_hex')
            try:
                pb = bytes.fromhex(ph)
            except (TypeError, ValueError):
                raise common.InvalidInput('%s row %d: prompt_hex %r is not hex' % (label, i + 1, ph))
            ah = r.get('answer_hex')
            ab = None
            if ah is not None:
                try:
                    ab = bytes.fromhex(ah)
                except (TypeError, ValueError):
                    raise common.InvalidInput('%s row %d: answer_hex %r is not hex' % (label, i + 1, ah))
            self.prompts.append(pb)
            self.answers.append(ab)
            if ab is None:
                continue
            const[ab] = const.get(ab, 0) + 1
            d = by_len.setdefault(len(pb), {})
            d[ab] = d.get(ab, 0) + 1
            for k in (1, 2, 3):
                dk = by_suf[k].setdefault(pb[-k:], {})
                dk[ab] = dk.get(ab, 0) + 1
            de = exact.setdefault(pb, {})
            de[ab] = de.get(ab, 0) + 1
            if self.is_add:
                pr = addition.parse(ph, fmt)
                if pr is not None:
                    dc = canon.setdefault(addition.canonical(*pr), {})
                    dc[ab] = dc.get(ab, 0) + 1
        self.const = _modal(const)
        self.by_len = {k: _modal(v) for k, v in by_len.items()}
        self.by_suf = {k: {s: _modal(v) for s, v in d.items()} for k, d in by_suf.items()}
        self.exact = {k: _modal(v) for k, v in exact.items()}
        self.canon = {k: _modal(v) for k, v in canon.items()}
        # NEAREST: taught prompts with an answer, sorted (ties go to the smallest prompt). Positional byte
        # matches are counted with bit-sliced counters over bitmasks of taught prompts per (position, byte).
        nearest = sorted(set(p for p, a in zip(self.prompts, self.answers) if a is not None))
        self._near_prompts = nearest
        self._near_all = (1 << len(nearest)) - 1
        masks: Dict[Tuple[int, int], int] = {}
        for j, p in enumerate(nearest):
            bit = 1 << j
            for pos, byte in enumerate(p):
                masks[(pos, byte)] = masks.get((pos, byte), 0) | bit
        # a (position, byte) shared by every taught prompt adds 1 to all of them and cannot change the argmax
        self._near_masks = {k: m for k, m in masks.items() if m != self._near_all}
        self.taught = addition.Taught.from_rows(rows, fmt, label) if self.is_add else None
        self._cache: Dict[Tuple[bytes, bool], Dict[str, bytes]] = {}

    def nearest(self, pb: bytes) -> Optional[bytes]:
        """Answer of the taught prompt with the most positional byte matches (ties: smallest prompt)."""
        if not self._near_prompts:
            return None
        planes: List[int] = []
        masks = self._near_masks
        for pos, byte in enumerate(pb):
            carry = masks.get((pos, byte), 0)
            i = 0
            while carry:
                if i == len(planes):
                    planes.append(carry)
                    break
                nxt = planes[i] & carry
                planes[i] ^= carry
                carry = nxt
                i += 1
        cand = self._near_all
        for plane in reversed(planes):
            t = cand & plane
            if t:
                cand = t
        j = (cand & -cand).bit_length() - 1
        return self.exact.get(self._near_prompts[j])

    def predictions(self, ph: str, pb: bytes, with_dep: bool) -> Dict[str, bytes]:
        """Exposure-independent (and, when with_dep, exposure-dependent) taught/operand member predictions."""
        key = (pb, with_dep)
        got = self._cache.get(key)
        if got is not None:
            return got
        c = self.const
        out: Dict[str, bytes] = {'CONST': c}
        out['LENFREQ'] = self.by_len.get(len(pb)) or c
        for k in (1, 2, 3):
            out['SUFFIX%d' % k] = self.by_suf[k].get(pb[-k:]) or c
        if self.is_add:
            pr = addition.parse(ph, self.fmt)
            if pr is None:
                for m in ('COPY_A', 'COPY_B', 'ID_ZERO', 'DIGIT_NO_CARRY'):
                    out[m] = c
            else:
                a, b = addition.norm(pr[0]), addition.norm(pr[1])
                out['COPY_A'] = a.encode('ascii')
                out['COPY_B'] = b.encode('ascii')
                out['ID_ZERO'] = b.encode('ascii') if a == '0' else (a.encode('ascii') if b == '0' else c)
                out['DIGIT_NO_CARRY'] = _digit_no_carry(a, b).encode('ascii')
        if with_dep:
            out['EXACT'] = self.exact.get(pb) or c
            if self.is_add:
                pr = addition.parse(ph, self.fmt)
                out['CANON'] = (self.canon.get(addition.canonical(*pr)) if pr is not None else None) or c
            out['NEAREST'] = self.nearest(pb) or c
        self._cache[key] = out
        return out


def _digit_no_carry(a: str, b: str) -> str:
    n = max(len(a), len(b))
    a, b = a.rjust(n, '0'), b.rjust(n, '0')
    return addition.norm(''.join(str((int(x) + int(y)) % 10) for x, y in zip(a, b)))


def _normalize_taught_map(taught: Any) -> Dict[Tuple[str, str], List[dict]]:
    out: Dict[Tuple[str, str], List[dict]] = {}
    if not taught:
        return out
    for k, v in dict(taught).items():
        if isinstance(k, tuple) and len(k) == 2:
            out[(str(k[0]), str(k[1]))] = list(v or [])
        elif isinstance(k, str) and isinstance(v, dict):
            for arm, rows in v.items():
                out[(k, str(arm))] = list(rows or [])
        elif isinstance(k, str) and '|' in k:
            w, a = k.split('|', 1)
            out[(w, a)] = list(v or [])
        else:
            raise common.InvalidInput('taught_by_world_arm keys must be (world, arm) tuples, got %r' % (k,))
    return out


# --------------------------------------------------------------------------------------
# score_rows (SPEC 7.3)
# --------------------------------------------------------------------------------------

def _entropy_bits(counter: Dict[bytes, int]) -> float:
    n = sum(counter.values())
    if n <= 0:
        return 0.0
    h = 0.0
    for c in counter.values():
        if c > 0:
            q = c / float(n)
            h -= q * math.log(q, 2)
    return h


def score_rows(rows: Iterable[dict], lock: dict, taught_by_world_arm: Any = None, allow_evaluator: bool = False,
               item_detail: Optional[Tuple[str, str]] = None,
               exposed_orbits: Optional[Iterable[Tuple[str, str]]] = None) -> dict:
    """Recompute and aggregate OUTPUTS rows into cells (SPEC 8 SCORES.cells) plus the shortcut ladder.

    rows: iterable of minifly.outputs.v1 rows (consumed once). taught_by_world_arm: {(world, arm): [TAUGHT rows]}
    (also {world: {arm: rows}}). allow_evaluator: count answer_path 'evaluator' rows of CEILING_*/FLOOR_* arms
    (cycle-0 testbed only); learner arms count only answer_path native. item_detail=(pool, checkpoint): also
    return the candidate's per-item exactness there (for UDI) under 'item_exact'. exposed_orbits: canonical orbits
    whose expected answer was fed with the orbit in some exposure stream (admitcheck.exposed_held_orbits); such an
    item is S1 (taught) in every world and never counted (SPEC 9.3 as amended in 20.11). S0 stays S0.

    A conflicting duplicate key (world, arm, pool, checkpoint, item_id), an unknown pool or checkpoint, a
    missing required field, or the same item_id with two prompts raises InvalidInput (fail closed)."""
    fmt = _fmt(lock)
    is_add = addition.is_addition(fmt)
    gen_cap = _gen_cap(fmt)
    eos = _eos(fmt)
    if is_add:
        addition._require_normalized(fmt)
    primary = lock.get('primary') if isinstance(lock.get('primary'), dict) else {}
    cand = primary.get('candidate_arm')
    metric = primary.get('metric')
    balanced = metric == 'balanced_exact_own_eos'
    ppool, pck = primary.get('pool'), primary.get('checkpoint')
    cks = lock.get('checkpoints')
    if not isinstance(cks, dict) or not cks:
        raise common.InvalidInput('lock.checkpoints must be a non-empty object')
    ck_names = set(cks)
    units = lock.get('units') if isinstance(lock.get('units'), dict) else None
    witness = units.get('witness') if units else None
    commutative = bool(units.get('commutative')) if units else False
    if witness is not None and witness not in addition.WITNESSES:
        raise common.InvalidInput('units.witness must be one of %s, got %r' % ('|'.join(addition.WITNESSES), witness))
    if witness is not None and not is_add:
        raise common.InvalidInput('units.witness %s needs an addition task format' % witness)
    lock_roles = world_roles_from_lock(lock)
    off = (lock.get('worlds') or {}).get('official') if isinstance(lock.get('worlds'), dict) else None
    official_listed = bool(isinstance(off, dict) and off.get('ids'))
    tmap = _normalize_taught_map(taught_by_world_arm)
    exposed = set(addition.canonical(a, b) for a, b in (exposed_orbits or ())) if is_add else set()
    counted_keys = set([(ppool, pck)] + list(COUNTED_KEYS_EXTRA))

    tindex: Dict[str, Optional[TaughtIndex]] = {}

    def tidx(world: str) -> Optional[TaughtIndex]:
        if world not in tindex:
            trs = tmap.get((world, cand))
            tindex[world] = TaughtIndex(trs, fmt, 'TAUGHT/%s/%s' % (world, cand)) if trs is not None else None
        return tindex[world]

    item_cache: Dict[Tuple[str, str], tuple] = {}
    universe: Dict[Tuple[str, str], Dict[str, tuple]] = {}
    results: Dict[Tuple[str, str, str, str], Dict[str, bool]] = {}
    cand_stream: Dict[Tuple[str, str, str], Dict[str, bytes]] = {}
    cand_out: Dict[Tuple[str, str, str], Dict[bytes, int]] = {}
    digests: Dict[str, List[int]] = {}
    row_roles: Dict[str, Set[str]] = {}
    mism: List[dict] = []
    mism_count: Dict[str, int] = {}
    met02: List[dict] = []
    met02_count = 0
    n_rows = 0
    nonnative = 0
    item_exact: Dict[str, Dict[str, tuple]] = {}

    def note(kind: str, row: dict, field: str, got: Any, want: Any):
        mism_count[field] = mism_count.get(field, 0) + 1
        if len(mism) < MAX_LISTED:
            mism.append({'world': row.get('world'), 'arm': row.get('arm'), 'pool': row.get('pool'),
                         'checkpoint': row.get('checkpoint'), 'item_id': row.get('item_id'), 'field': field,
                         'row_value': got, 'recomputed': want})

    def make_info(world: str, ph: str, row: dict) -> tuple:
        if is_add:
            c = item_cache.get((world, ph))
            if c is None:
                exp = addition.expected_bytes(ph, fmt)
                ti = tidx(world)
                st = det = None
                if exp is not None and ti is not None:
                    st = addition.stratum({'prompt_hex': ph}, ti.taught)
                    if exposed and st not in (None, 'S0'):
                        pr = addition.parse(ph, fmt)
                        if pr is not None and addition.canonical(*pr) in exposed:
                            st = 'S1'
                    if witness is not None:
                        det = addition.determined({'prompt_hex': ph}, ti.taught, witness, commutative)
                c = (exp, st, det)
                item_cache[(world, ph)] = c
            exp, st, det = c
        else:
            eh = row.get('expected_hex')
            try:
                exp = bytes.fromhex(eh) if isinstance(eh, str) else None
            except ValueError:
                exp = None
            if exp is None:
                raise common.InvalidInput('row %s/%s/%s: expected_hex %r is not hex (non-addition task)'
                                          % (world, row.get('arm'), row.get('item_id'), eh))
            st = det = None
        ans = exp.hex() if (balanced and exp is not None) else None
        return (ph, exp, st, det, ans, exp.hex() if exp is not None else None)

    pools_ok = frozenset(decide.POOLS)
    arm_flags: Dict[str, Tuple[bool, bool]] = {}  # arm -> (is_candidate, is_evaluator)
    fromhex = bytes.fromhex
    want_detail = item_detail
    for row in rows:
        n_rows += 1
        if type(row) is not dict:
            raise common.InvalidInput('OUTPUTS row %d is not an object' % n_rows)
        try:
            world = row['world']
            arm = row['arm']
            pool = row['pool']
            ck = row['checkpoint']
            iid = row['item_id']
            ph = row['prompt_hex']
        except KeyError as e:
            raise common.InvalidInput('OUTPUTS row %d lacks %s' % (n_rows, e))
        if not (type(world) is str and type(arm) is str and type(iid) is str and type(ph) is str
                and world and arm and iid and ph):
            raise common.InvalidInput('OUTPUTS row %d: world/arm/item_id/prompt_hex must be non-empty strings' % n_rows)
        if pool not in pools_ok:
            raise common.InvalidInput('OUTPUTS row %d: pool %r not in %s' % (n_rows, pool, '|'.join(decide.POOLS)))
        if ck not in ck_names:
            raise common.InvalidInput('OUTPUTS row %d: checkpoint %r is not a key of lock.checkpoints' % (n_rows, ck))
        fl = arm_flags.get(arm)
        if fl is None:
            fl = arm_flags[arm] = (arm == cand, common.is_evaluator_arm(arm))
        k4 = (world, arm, pool, ck)
        res = results.get(k4)
        if res is None:
            res = results[k4] = {}
        if iid in res:
            raise common.InvalidInput('duplicate OUTPUTS rows for %s/%s/%s/%s item %s' % (world, arm, pool, ck, iid))
        u = universe.get((world, pool))
        if u is None:
            u = universe[(world, pool)] = {}
        info = u.get(iid)
        if info is None:
            info = make_info(world, ph, row)
            u[iid] = info
        elif info[0] != ph:
            raise common.InvalidInput('item %s/%s/%s has two prompts (%s, %s)' % (world, pool, iid, info[0], ph))
        elif not is_add:
            eh = row.get('expected_hex')
            if not isinstance(eh, str) or eh.lower() != info[1].hex():
                raise common.InvalidInput('item %s/%s/%s has two expected answers' % (world, pool, iid))
        exp = info[1]
        rr = row.get('world_role')
        rs = row_roles.get(world)
        if rs is None:
            rs = row_roles[world] = set()
        rs.add(rr if type(rr) is str else repr(rr))
        if is_add:
            want = info[5]
            if row.get('expected_hex') != want:
                met02_count += 1
                if len(met02) < MAX_LISTED:
                    met02.append({'world': world, 'arm': arm, 'pool': pool, 'checkpoint': ck, 'item_id': iid,
                                  'prompt_hex': ph, 'row_expected_hex': row.get('expected_hex'),
                                  'recomputed': want if want is not None else 'unparseable prompt'})
        sh = row.get('stream_hex')
        s = None
        if type(sh) is str:
            try:
                s = fromhex(sh)
            except ValueError:
                s = None
        em = b''
        own = False
        if s is None:
            ok_stream = False
            note('MET-01', row, 'stream_hex', sh if type(sh) is not str else sh[:64], 'hex of gen_cap bytes')
        else:
            ok_stream = len(s) == gen_cap
            if not ok_stream:
                note('MET-01', row, 'stream_len', len(s), gen_cap)
                s = s[:gen_cap]
            pos = s.find(eos)
            if pos >= 0:
                em = s[:pos]
                own = True
            else:
                em = s
            emh = em.hex()
            if row.get('emitted_hex') != emh:
                note('MET-01', row, 'emitted_hex', row.get('emitted_hex'), emh)
            if row.get('own_eos') is not own:
                note('MET-01', row, 'own_eos', row.get('own_eos'), own)
        ap = row.get('answer_path')
        if ap == 'native':
            path_ok = True
        else:
            if ap not in ANSWER_PATHS:
                note('MET-01', row, 'answer_path', ap, '|'.join(ANSWER_PATHS))
            path_ok = allow_evaluator and ap == 'evaluator' and fl[1]
            if not path_ok:
                nonnative += 1
        exact = ok_stream and own and path_ok and exp is not None and em == exp
        res[iid] = exact
        if fl[0]:
            if s is not None:
                g3 = (world, pool, ck)
                cs = cand_stream.get(g3)
                if cs is None:
                    cs = cand_stream[g3] = {}
                    cand_out[g3] = {}
                cs[iid] = s
                co = cand_out[g3]
                co[em] = co.get(em, 0) + 1
            if want_detail is not None and (pool, ck) == want_detail:
                item_exact.setdefault(world, {})[iid] = (ph, exact)
        if not fl[1]:
            dg = digests.get(arm)
            if dg is None:
                dg = digests[arm] = [0, 0, 0]
            dg[0] += 1
            dg[1] += hash((world, pool, ck, iid))
            dg[2] += hash((world, pool, ck, iid, sh if type(sh) is str else None))

    # arms with identical stream_hex on every item (multiset digests; in-process hashing, not stored)
    identical_pairs: List[list] = []
    darms = sorted(a for a, d in digests.items() if d[0] > 0)
    for i, a in enumerate(darms):
        for b in darms[i + 1:]:
            if digests[a] == digests[b]:
                identical_pairs.append([a, b, digests[a][0]])

    # world roles: the lock decides; a row claiming a non-official role always wins (stricter)
    worlds: Dict[str, str] = {}
    conflicts: List[dict] = []
    for w in sorted(row_roles):
        lr = lock_roles.get(w)
        rr = row_roles[w]
        cands = set(rr) | ({lr} if lr else set())
        if lr and rr == {lr}:
            worlds[w] = lr
            continue
        if lr is None and rr == {'official'}:
            if official_listed:
                worlds[w] = 'unlisted'
                conflicts.append({'world': w, 'lock_role': None, 'row_roles': sorted(rr),
                                  'note': 'official rows for a world not in lock.worlds.official.ids'})
            else:
                worlds[w] = 'official'
            continue
        if lr is None and len(rr) == 1 and next(iter(rr)) in ('technical', 'dev', 'rehearsal'):
            worlds[w] = next(iter(rr))
            conflicts.append({'world': w, 'lock_role': None, 'row_roles': sorted(rr),
                              'note': 'world not listed in lock.worlds'})
            continue
        non_off = sorted(r for r in cands if r in ('technical', 'dev', 'rehearsal'))
        worlds[w] = non_off[0] if non_off else 'unknown'
        conflicts.append({'world': w, 'lock_role': lr, 'row_roles': sorted(rr)})

    # cells
    cells: List[dict] = []
    for (w, a, pool, ck) in sorted(results):
        res = results[(w, a, pool, ck)]
        groups: Dict[tuple, List[int]] = {}
        for iid, info in universe[(w, pool)].items():
            g = (info[2], info[3], info[4])
            acc = groups.get(g)
            if acc is None:
                acc = groups[g] = [0, 0, 0]
            acc[0] += 1
            ex = res.get(iid)
            if ex is not None:
                acc[1] += 1
                if ex:
                    acc[2] += 1
        for g in sorted(groups, key=lambda t: (str(t[0]), str(t[1]), str(t[2]))):
            n, nr, ne = groups[g]
            c = {'world': w, 'world_role': worlds.get(w), 'arm': a, 'pool': pool, 'checkpoint': ck,
                 'stratum': g[0], 'determined': g[1], 'n': n, 'n_rows': nr, 'n_exact': ne}
            if balanced:
                c['answer'] = g[2]
            cells.append(c)

    # shortcut ladder per (world, pool, checkpoint) from the candidate's TAUGHT and its own streams
    cfilter = decide.counted_filter(lock)
    ladder: Dict[str, Dict[str, float]] = {}
    ladder_status: Dict[str, str] = {}
    ladder_cells: Dict[str, Dict[str, List[dict]]] = {}
    output_stats: Dict[str, dict] = {}
    skips = sorted(decide.OPERAND_MEMBERS) if not is_add else []
    g3s = sorted(set((w, pool, ck) for (w, _a, pool, ck) in results))
    static_groups: Dict[Tuple[str, str], Dict[str, Dict[tuple, List[int]]]] = {}

    def static_for(w: str, pool: str, ti: TaughtIndex, members: List[str]) -> Dict[str, Dict[tuple, List[int]]]:
        """Group counts of the members that do not depend on the checkpoint (taught and operand members)."""
        got = static_groups.get((w, pool))
        if got is not None:
            return got
        smembers = [m for m in members if not m.startswith('FIXED_') and m != 'MODE_REPEAT']
        with_dep = any(m in decide.EXPO_DEP_MEMBERS for m in smembers)
        mg: Dict[str, Dict[tuple, List[int]]] = {m: {} for m in smembers}
        for iid, info in universe[(w, pool)].items():
            ph, exp, st, det, ans, _eh = info
            g = (st, det, ans)
            try:
                pb = bytes.fromhex(ph)
            except ValueError:
                pb = b''
            preds = ti.predictions(ph, pb, with_dep)
            for m in smembers:
                acc = mg[m].get(g)
                if acc is None:
                    acc = mg[m][g] = [0, 0]
                acc[0] += 1
                pr = preds.get(m)
                if exp is not None and pr is not None and pr == exp:
                    acc[1] += 1
        static_groups[(w, pool)] = mg
        return mg

    for (w, pool, ck) in g3s:
        key = decide.lkey(w, pool, ck)
        co = cand_out.get((w, pool, ck)) or {}
        if co:
            n_out = sum(co.values())
            mo = _modal(co)
            output_stats[key] = {'n': n_out, 'modal_output_hex': mo.hex() if mo is not None else None,
                                 'modal_fraction': co[mo] / float(n_out) if mo is not None else None,
                                 'entropy_bits': _entropy_bits(co)}
        ti = tidx(w)
        if ti is None or ti.const is None:
            ladder_status[key] = 'taught_missing: no TAUGHT rows with answers for candidate %s in world %s' % (cand, w)
            continue
        members = [m for m in decide.pool_members(pool) if not (m in decide.OPERAND_MEMBERS and not is_add)]
        mgroups: Dict[str, Dict[tuple, List[int]]] = dict(static_for(w, pool, ti, members))
        # MODE_REPEAT: candidate's modal output byte repeated to its modal emitted length
        bytec: Dict[int, int] = {}
        lenc: Dict[int, int] = {}
        for emb, cnt in co.items():
            lenc[len(emb)] = lenc.get(len(emb), 0) + cnt
            for byte in emb:
                bytec[byte] = bytec.get(byte, 0) + cnt
        mlen = min(lenc.items(), key=lambda kv: (-kv[1], kv[0]))[0] if lenc else 0
        mbyte = min(bytec.items(), key=lambda kv: (-kv[1], kv[0]))[0] if bytec else None
        mode_pred = (bytes([mbyte]) * mlen) if mbyte is not None else b''
        streams = cand_stream.get((w, pool, ck)) or {}
        fixed = [m for m in members if m.startswith('FIXED_')]
        fixed_k = [int(m[6:]) for m in fixed]
        fg: Dict[int, Dict[tuple, List[int]]] = {k: {} for k in fixed_k}
        mode_g: Dict[tuple, List[int]] = {}
        use_mode = 'MODE_REPEAT' in members
        for iid, info in universe[(w, pool)].items():
            exp = info[1]
            g = (info[2], info[3], info[4])
            s_ = streams.get(iid)
            hits: Tuple[int, ...] = ()
            if s_ is not None and exp is not None:
                if len(s_) > 8:
                    le = len(exp)
                    hits = (le,) if (le <= 8 and s_[:le] == exp) else ()
                else:
                    hits = tuple(k for k in fixed_k if s_[:k] == exp)
            for k in fixed_k:
                d_ = fg[k]
                acc = d_.get(g)
                if acc is None:
                    acc = d_[g] = [0, 0]
                acc[0] += 1
                if k in hits:
                    acc[1] += 1
            if use_mode:
                acc = mode_g.get(g)
                if acc is None:
                    acc = mode_g[g] = [0, 0]
                acc[0] += 1
                if exp is not None and mode_pred == exp:
                    acc[1] += 1
        for k in fixed_k:
            mgroups['FIXED_%d' % k] = fg[k]
        if use_mode:
            mgroups['MODE_REPEAT'] = mode_g
        flt = cfilter if (pool, ck) in counted_keys else decide.all_items
        vals: Dict[str, float] = {}
        lc: Dict[str, List[dict]] = {}
        for m in members:
            gl = []
            for g, (n, ne) in sorted(mgroups[m].items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]), str(kv[0][2]))):
                e = {'stratum': g[0], 'determined': g[1], 'n': n, 'n_rows': n, 'n_exact': ne}
                if balanced:
                    e['answer'] = g[2]
                gl.append(e)
            v, n_sel, _ = decide.group_rate(gl, flt, metric if metric in decide.METRICS else 'exact_own_eos')
            if v is not None:
                vals[m] = v
            if (pool, ck) in counted_keys:
                lc[m] = gl
        ladder[key] = vals
        ladder_status[key] = 'ok' if vals else 'no items in the evaluated subset'
        if lc:
            ladder_cells[key] = lc

    # best shortcut on the primary (pool, checkpoint)
    per_world_best: Dict[str, dict] = {}
    member_sums: Dict[str, List[float]] = {}
    for w in sorted(worlds):
        key = decide.lkey(w, ppool, pck)
        b = decide.ladder_best({'ladder': ladder, 'ladder_status': ladder_status}, w, ppool, pck)
        if b is None:
            continue
        per_world_best[w] = {'name': b[0], 'value': b[1]}
        for m, v in ladder.get(key, {}).items():
            if m in decide.pool_members(ppool):
                member_sums.setdefault(m, []).append(v)
    best_shortcut = None
    if per_world_best:
        means = {m: math.fsum(v) / len(v) for m, v in member_sums.items() if len(v) == len(per_world_best)}
        order = {m: i for i, m in enumerate(decide.LADDER_MEMBERS)}
        name = min(means, key=lambda m: (-means[m], order.get(m, 99))) if means else None
        best_shortcut = {'name': name, 'member_mean': means.get(name) if name else None,
                         'value': math.fsum(b['value'] for b in per_world_best.values()) / len(per_world_best),
                         'per_world': per_world_best, 'pool': ppool, 'checkpoint': pck}

    out = {
        'schema': SCHEMA_CELLS,
        'metric': metric,
        'primary': {'pool': ppool, 'checkpoint': pck, 'candidate_arm': cand,
                    'strata_counted': list(primary.get('strata_counted') or [])},
        'strata_applicable': is_add,
        'determinacy_applicable': witness is not None,
        'witness': witness,
        'commutative': commutative,
        'n_rows': n_rows,
        'worlds': worlds,
        'role_conflicts': conflicts,
        'cells': cells,
        'ladder': ladder,
        'ladder_status': ladder_status,
        'ladder_cells': ladder_cells,
        'ladder_skipped_members': skips,
        'best_shortcut': best_shortcut,
        'output_stats': output_stats,
        'met_mismatches': mism,
        'met_mismatch_counts': mism_count,
        'met02_mismatches': met02,
        'met02_mismatch_count': met02_count,
        'non_native_rows': nonnative,
        'identical_arm_pairs': identical_pairs,
    }
    if item_detail is not None:
        out['item_exact'] = {w: {i: [v[0], v[1]] for i, v in sorted(d.items())} for w, d in sorted(item_exact.items())}
    return out


# --------------------------------------------------------------------------------------
# Delivery-level reading and caching
# --------------------------------------------------------------------------------------

def outputs_pattern(role: str) -> str:
    return 'results/%s/OUTPUTS.jsonl' % role


def role_has_rows(ctx: common.Ctx, role: str) -> bool:
    key = ('wp3_has_rows', role)
    if key not in ctx._cache:
        found = False
        for _row in ctx.iter_rows(outputs_pattern(role)):
            found = True
            break
        ctx._cache[key] = found
    return ctx._cache[key]


def load_taught(ctx: common.Ctx, lock: Optional[dict] = None) -> Dict[Tuple[str, str], List[dict]]:
    """{(world, arm): rows} from <split.taught_manifest>/<world>/<arm>.jsonl (gz and bundles allowed)."""
    if 'wp3_taught' in ctx._cache:
        return ctx._cache['wp3_taught']
    lock = lock or ctx.lock
    split = lock.get('split') if isinstance(lock.get('split'), dict) else {}
    tdir = common.norm_rel(split.get('taught_manifest') or 'manifests/TAUGHT')
    out: Dict[Tuple[str, str], List[dict]] = {}
    for logical, _kind, _ref, _gz in ctx.row_sources(tdir + '/*/*.jsonl'):
        rest = logical[len(tdir) + 1:]
        parts = rest.split('/')
        if len(parts) != 2 or not parts[1].endswith('.jsonl'):
            continue
        world, arm = parts[0], parts[1][:-len('.jsonl')]
        rows = []
        for label, lineno, r in ctx.iter_rows_with_source(logical):
            for f, want in (('world', world), ('arm', arm)):
                if r.get(f) is not None and r.get(f) != want:
                    raise common.InvalidInput('%s:%d has %s %r, path says %r' % (label, lineno, f, r.get(f), want))
            rows.append(r)
        out[(world, arm)] = rows
    ctx._cache['wp3_taught'] = out
    return out


def udi_variant_worlds(ctx: common.Ctx, lock: dict) -> Tuple[Set[str], Optional[dict], Optional[str]]:
    """(variant world ids, manifest, error) when lock.udi is declared."""
    udi = lock.get('udi')
    if not isinstance(udi, dict):
        return set(), None, None
    rel = udi.get('variants_manifest')
    if not isinstance(rel, str) or not rel:
        return set(), None, 'lock.udi.variants_manifest missing'
    try:
        man = ctx.read_json(rel)
    except common.MissingInput as e:
        return set(), None, str(e)
    if not isinstance(man, dict) or not isinstance(man.get('variants'), list):
        raise common.InvalidInput('%s must be {"variants": [...]}' % rel)
    vw = set()
    for v in man['variants']:
        if isinstance(v, dict) and isinstance(v.get('variant_world'), str):
            vw.add(v['variant_world'])
    return vw, man, None


def taught_with_aliases(taught: Dict[Tuple[str, str], List[dict]], cand: str, arms: Iterable[str]) -> Dict[Tuple[str, str], List[dict]]:
    """Testbed arms use their own TAUGHT manifest when present, else the candidate's (the exact split)."""
    out = dict(taught)
    for (w, a), rows in list(taught.items()):
        if a == cand:
            for arm in arms:
                out.setdefault((w, arm), rows)
    return out


def split_manifest_items(ctx: common.Ctx, lock: Optional[dict] = None) -> Dict[str, Dict[str, str]]:
    """{pool: {item_id: prompt_hex}} of the HELD/OLD/NEW manifests the lock names (split.*_manifest). These are
    the items the lock fixes (SPEC 6.2; AMD-05 verifies their bytes against split.held_set_sha256 and
    split_sha256), so they are the only items a HELD/OLD/NEW cell may hold. A named manifest that is missing
    raises MissingInput; a malformed one (no item_id/prompt_hex, a repeated item_id) raises InvalidInput."""
    key = 'wp3_split_items'
    if key in ctx._cache:
        return ctx._cache[key]
    lock = lock or ctx.lock
    split = lock.get('split') if isinstance(lock.get('split'), dict) else {}
    out: Dict[str, Dict[str, str]] = {}
    for skey, pool in common.SPLIT_MANIFESTS:
        rel = split.get(skey)
        if not isinstance(rel, str) or not rel:
            continue  # LOCK-01 requires the key
        items: Dict[str, str] = {}
        for i, r in enumerate(ctx.read_jsonl(rel), 1):
            iid, ph = r.get('item_id') if isinstance(r, dict) else None, r.get('prompt_hex') if isinstance(r, dict) else None
            if not isinstance(iid, str) or not isinstance(ph, str):
                raise common.InvalidInput('%s:%d needs string item_id and prompt_hex' % (rel, i))
            if iid in items:
                raise common.InvalidInput('%s:%d repeats item_id %s' % (rel, i, iid))
            items[iid] = ph.lower()
        out[pool] = items
    ctx._cache[key] = out
    return out


def exposed_orbits_of(ctx: common.Ctx) -> Set[Tuple[str, str]]:
    """Held orbits taught with their answers in any exposure stream (SPEC 9.3 / 20.11), from ADM-EXPO's content scan.
    Only orbit-keyed (E3 addition) locks have any; the scan's own errors propagate (fail closed)."""
    lock = ctx.lock
    fmt = (lock.get('task') or {}).get('format') if isinstance(lock.get('task'), dict) else None
    split = lock.get('split') if isinstance(lock.get('split'), dict) else {}
    if not addition.is_addition(fmt) or lock.get('claim_level_target') == 'E2' or not split.get('held_manifest'):
        return set()  # exact-prompt locks have no orbit strata; a lock without a held manifest fails LOCK-01
    import admitcheck  # noqa: E402 - admitcheck imports the scorer lazily, never at import time
    return admitcheck.exposed_held_orbits(ctx)


def score_role(ctx: common.Ctx, role: str) -> dict:
    """Cached scoring of results/<role>/OUTPUTS rows: {'role', 'cells', 'testbed_rows', 'variant_rows', ...}.

    HELD/OLD/NEW rows must carry an item of the locked split manifest with its manifest prompt; any other row
    raises InvalidInput (the canonical scorer never counts an item the lock did not fix). The item set of every
    (world, arm, pool, checkpoint) cell of those pools is kept under 'split_cells' for LAD-01."""
    if role not in ROLES:
        raise common.InvalidInput('role must be one of %s' % '|'.join(ROLES))
    key = ('wp3_scored', role)
    if key in ctx._cache:
        return ctx._cache[key]
    lock = ctx.lock
    taught = load_taught(ctx, lock)
    vworlds, _man, _err = udi_variant_worlds(ctx, lock)
    split_items = split_manifest_items(ctx, lock)
    split_cells: Dict[Tuple[Any, Any, Any, Any], Set[str]] = {}
    foreign: List[str] = []
    n_foreign = [0]
    testbed_rows: List[dict] = []
    variant_rows: List[dict] = []
    count = [0]

    def gen() -> Iterator[dict]:
        for r in ctx.iter_rows(outputs_pattern(role)):
            count[0] += 1
            man = split_items.get(r.get('pool')) if isinstance(r, dict) else None
            if man is not None:
                iid, ph = r.get('item_id'), r.get('prompt_hex')
                want = man.get(iid) if isinstance(iid, str) else None
                if want is None or not isinstance(ph, str) or ph.lower() != want:
                    n_foreign[0] += 1
                    if len(foreign) < 5:
                        foreign.append('%s/%s/%s/%s item %r prompt %r' % (r.get('world'), r.get('arm'), r.get('pool'),
                                                                        r.get('checkpoint'), iid, ph))
                else:
                    split_cells.setdefault((r.get('world'), r.get('arm'), r.get('pool'), r.get('checkpoint')),
                                           set()).add(iid)
            if isinstance(r, dict) and r.get('world') in vworlds:
                variant_rows.append(r)
                continue
            if role == 'admission' and isinstance(r, dict) and common.is_evaluator_arm(r.get('arm')):
                testbed_rows.append(r)
            yield r

    item_detail = ('HELD', 'final') if vworlds else None
    exposed = exposed_orbits_of(ctx)
    cells = score_rows(gen(), lock, taught, allow_evaluator=(role == 'admission'), item_detail=item_detail,
                       exposed_orbits=exposed)
    if count[0] == 0:
        raise common.MissingInput('no rows in %s (or .gz / RESULT_BUNDLE*.zip members)' % outputs_pattern(role))
    if n_foreign[0]:
        raise common.InvalidInput('%s: %d HELD/OLD/NEW row(s) carry an item or prompt that is not in the locked split '
                                  'manifest (the held set is fixed in the lock), e.g. %s'
                                  % (outputs_pattern(role), n_foreign[0], '; '.join(foreign)))
    item_exact = cells.pop('item_exact', None)
    out = {'role': role, 'cells': cells, 'n_rows': count[0], 'testbed_rows': testbed_rows,
           'variant_rows': variant_rows, 'variant_worlds': sorted(vworlds), 'item_exact': item_exact,
           'taught': taught, 'split_cells': split_cells, 'exposed_orbits': sorted(exposed)}
    ctx._cache[key] = out
    return out


def expected_worlds(lock: dict, role: str, cells: dict) -> Tuple[List[str], Optional[int]]:
    listed = lock_worlds(lock, role)
    if listed:
        return listed, None
    want = WORLD_ROLE_OF[role]
    present = sorted(w for w, r in (cells.get('worlds') or {}).items() if r == want)
    return present, (official_n(lock) if role == 'official' else None)


# --------------------------------------------------------------------------------------
# Preconditions (SPEC 9.5) and verdicts
# --------------------------------------------------------------------------------------

def inert_alarms(cells: dict, lock: dict, worlds: Optional[Sequence[str]] = None) -> List[dict]:
    """INERT-01: (1) two declared arms with identical stream_hex on every item in every world; (2) a primary
    contrast (candidate minus a comparator arm) that is exactly zero in every world (the [0,0] interval)."""
    alarms: List[dict] = []
    declared = set(lock_arms(lock))
    for a, b, n in cells.get('identical_arm_pairs') or []:
        if a in declared and b in declared:
            alarms.append({'kind': 'identical_streams', 'arms': [a, b], 'rows': n,
                           'detail': 'arms %s and %s have identical stream_hex on every item in every world' % (a, b)})
    p, problems, nonfinite = decide.parse_params(lock)
    if problems or nonfinite:
        return alarms
    ws = list(worlds) if worlds is not None else sorted(cells.get('worlds') or {})
    if len(ws) < 2:
        return alarms
    idx = decide.index_cells(cells.get('cells') or [])
    flt = decide.counted_filter(lock)
    # the declared contrast arms only: a failing candidate at the level of an implied control (a FROZEN or SHUF arm
    # at 0 on the counted items) is an honest FAIL through the decision comparator (SPEC 20.6), not an integrity
    # alarm; identical streams of any two declared arms stay an alarm (above)
    for comp in p['declared_comparator_arms']:
        diffs = []
        for w in ws:
            vc, _, _ = decide.group_rate(idx.get((w, p['candidate_arm'], p['pool'], p['checkpoint']), []), flt, p['metric'])
            va, _, _ = decide.group_rate(idx.get((w, comp, p['pool'], p['checkpoint']), []), flt, p['metric'])
            if vc is None or va is None:
                diffs = None
                break
            diffs.append(vc - va)
        if diffs and all(abs(d) < 1e-12 for d in diffs):
            alarms.append({'kind': 'zero_variance_contrast', 'arms': [p['candidate_arm'], comp], 'interval': [0.0, 0.0],
                           'detail': 'candidate minus %s is exactly 0 in all %d worlds' % (comp, len(ws))})
    return alarms


def gate_status(ctx: common.Ctx, lock: dict) -> Dict[str, str]:
    """Plumbing gate results from results/admission/FIXTURES.jsonl: PASS when the gate's negative fixtures all
    returned false and its positive fixtures all returned true (each kind present); MISSING when a kind is absent."""
    gates = [g.get('id') for g in (lock.get('gates') or []) if isinstance(g, dict)]
    if not gates:
        return {}
    try:
        rows = list(ctx.iter_rows('results/admission/FIXTURES.jsonl'))
    except common.MissingInput:
        rows = []
    out = {}
    for gid in gates:
        neg = [r for r in rows if r.get('gate_id') == gid and r.get('kind') == 'negative']
        pos = [r for r in rows if r.get('gate_id') == gid and r.get('kind') == 'positive']
        if not neg or not pos:
            out[gid] = 'MISSING'
        elif all(r.get('result') is False for r in neg) and all(r.get('result') is True for r in pos):
            out[gid] = 'PASS'
        else:
            out[gid] = 'FAIL'
    return out


ADM_CTRL_IDS = ('ADM-CTRL-ZERO', 'ADM-CTRL-ISO', 'ADM-CTRL-DOSE')


def _adm_ctrl_status(ctx: common.Ctx) -> Tuple[str, str]:
    """ADM-CTRL precondition from the registered admission checks (WP4); MISSING when unavailable."""
    try:
        idx = common.rule_index(ctx.rules or common.load_rules())
    except common.CheckerError as e:
        return 'MISSING', str(e)
    statuses = []
    for rid in ADM_CTRL_IDS:
        rule = idx.get(rid)
        if rule is None:
            return 'MISSING', '%s not in RULES' % rid
        for r in common.run_check(ctx, rule):
            statuses.append((rid, r.status, r.detail))
    if any(s == common.FAIL for _i, s, _d in statuses):
        return 'FAIL', '; '.join('%s %s' % (i, s) for i, s, _d in statuses)
    if all(s in (common.PASS, common.WARN, common.NOT_APPLICABLE) for _i, s, _d in statuses):
        return 'PASS', 'ADM-CTRL checks pass'
    return 'MISSING', '; '.join('%s %s: %s' % (i, s, d[:120]) for i, s, d in statuses if s != common.PASS)


def _testbed_status(ctx: common.Ctx) -> Tuple[str, str]:
    try:
        tb = testbed(ctx)
    except common.MissingInput as e:
        return 'MISSING', str(e)
    r = tb.get('result')
    if r == 'PASS':
        return 'PASS', 'testbed: a ceiling passes and every floor fails'
    if r == 'CANNOT_TEST_CLAIM_AS_SPECIFIED':
        return 'FAIL', 'testbed: %s' % tb.get('reason')
    return 'MISSING', 'testbed: %s' % tb.get('reason')


def preconditions(ctx: common.Ctx, cells: dict, role: str) -> dict:
    lock = ctx.lock
    worlds, n = expected_worlds(lock, role, cells)
    tb, tb_why = _testbed_status(ctx)
    ctrl, ctrl_why = _adm_ctrl_status(ctx)
    pre = {'inert_alarms': inert_alarms(cells, lock, worlds), 'adm_tb': tb, 'adm_ctrl': ctrl,
           'gates': gate_status(ctx, lock), 'expected_worlds': worlds, 'expected_n_worlds': n,
           'endpoint_lock_sha256': ctx.endpoint_lock_sha256, 'source_lock_sha256': ctx.source_lock_sha256,
           'detail': {'adm_tb': tb_why, 'adm_ctrl': ctrl_why}}
    pr = lock.get('primary') if isinstance(lock.get('primary'), dict) else {}
    cr = pr.get('custom_reducer') if isinstance(pr.get('custom_reducer'), dict) else None
    if pr.get('rule_type') == 'custom' and cr and isinstance(cr.get('path'), str):
        pre['reducer_path'] = str(ctx.path(cr['path']))
    return pre


def evaluate_role(ctx: common.Ctx, role: str) -> dict:
    """decide.evaluate on the role's cells with the delivery's real preconditions (cached)."""
    key = ('wp3_verdict', role)
    if key in ctx._cache:
        return ctx._cache[key]
    sr = score_role(ctx, role)
    pre = preconditions(ctx, sr['cells'], role)
    detail = pre.pop('detail')
    v = decide.evaluate(sr['cells'], ctx.lock, pre)
    v['preconditions_detail'] = detail
    v['role'] = role
    ctx._cache[key] = v
    return v


def build_verdict(ctx: common.Ctx) -> dict:
    """The official VERDICT.json body (not written)."""
    v = dict(evaluate_role(ctx, 'official'))
    v['delivery'] = ctx.delivery_rel
    v['checker_version'] = common.CHECKER_VERSION
    return v


def dev_summary(ctx: common.Ctx) -> dict:
    """Per-world d_w, mean comparator and counted items per world from dev rows (OC input)."""
    if not role_has_rows(ctx, 'dev'):
        raise common.MissingInput('no dev rows (results/dev/OUTPUTS.jsonl): OC needs dev d_w')
    sr = score_role(ctx, 'dev')
    worlds, _n = expected_worlds(ctx.lock, 'dev', sr['cells'])
    pre = {'inert_alarms': [], 'adm_tb': 'PASS', 'adm_ctrl': 'PASS', 'gates': {}, 'expected_worlds': worlds}
    v = decide.evaluate(sr['cells'], ctx.lock, pre)
    pw = (v.get('primary') or {}).get('per_world_detail') or []
    if not pw:
        raise common.MissingInput('dev rows give no per-world contrast (%s)' % '; '.join(v.get('reasons', [])[:3]))
    ns = [x['n_counted'] for x in pw if isinstance(x.get('n_counted'), int)]
    return {'per_world_d': [x['d'] for x in pw], 'mean_comp': (v['primary'] or {}).get('mean_comp'),
            'n_items': int(min(ns)) if ns else None, 'n_worlds_dev': len(pw), 'dev_verdict': v.get('verdict'),
            'dev_shadow_verdict': v.get('shadow_verdict')}


# --------------------------------------------------------------------------------------
# Testbed (cycle 0), acquisition and SHUF summaries
# --------------------------------------------------------------------------------------

def testbed(ctx: common.Ctx) -> dict:
    """Score every CEILING_*/FLOOR_* config with the delivery's rule (decide.testbed_lock view). At least one
    declared ceiling must PASS and every floor (so also the best floor) must FAIL."""
    if 'wp3_testbed' in ctx._cache:
        return ctx._cache['wp3_testbed']
    lock = ctx.lock
    if not role_has_rows(ctx, 'admission'):
        raise common.MissingInput('no admission rows: run MFC launch --stage testbed <dir> -- python3 -I admit.py --testbed')
    sr = score_role(ctx, 'admission')
    rows = sr['testbed_rows']
    tb = lock.get('testbed') if isinstance(lock.get('testbed'), dict) else {}
    configs: List[Tuple[str, str, Optional[str], bool]] = []
    for kind, prefix in (('ceiling_grid', 'CEILING_'), ('floor_grid', 'FLOOR_')):
        for c in tb.get(kind) or []:
            if isinstance(c, dict) and isinstance(c.get('id'), str):
                configs.append((prefix + c['id'], 'ceiling' if prefix == 'CEILING_' else 'floor', c['id'], True))
    declared = set(a for a, _k, _i, _d in configs)
    by_arm: Dict[str, List[dict]] = {}
    for r in rows:
        by_arm.setdefault(r.get('arm'), []).append(r)
    for arm in sorted(a for a in by_arm if a not in declared):
        configs.append((arm, 'ceiling' if arm.startswith('CEILING_') else 'floor', None, False))
    cand = (lock.get('primary') or {}).get('candidate_arm')
    tworlds = lock_worlds(lock, 'technical')
    nip = ((lock.get('primary') or {}).get('n_items_per_pool')) or {}
    results = []
    for arm, kind, cid, decl in configs:
        arows = by_arm.get(arm) or []
        ent: Dict[str, Any] = {'arm': arm, 'kind': kind, 'config_id': cid, 'declared': decl}
        if not arows:
            ent.update({'status': 'MISSING', 'reason': 'no rows for %s' % arm})
            results.append(ent)
            continue
        lv = decide.testbed_lock(lock, arm)
        tmap = taught_with_aliases(sr['taught'], cand, [arm])
        cells = score_rows(arows, lv, tmap, allow_evaluator=True, exposed_orbits=sr.get('exposed_orbits'))
        worlds = tworlds or sorted(cells['worlds'])
        v = decide.evaluate(cells, lv, {'mode': 'testbed', 'expected_worlds': worlds, 'n_items_per_pool': nip})
        res = decide.verdict_or_shadow(v)
        prim = v.get('primary') or {}
        ent.update({'status': 'PASS' if res == 'PASS' else ('FAIL' if res == 'FAIL' else 'MISSING'),
                    'verdict': v.get('verdict'), 'shadow_verdict': v.get('shadow_verdict'),
                    'mean_cand': prim.get('mean_cand'), 'mean_comp': prim.get('mean_comp'),
                    'mean_d': prim.get('mean_d'), 'per_world': prim.get('per_world'),
                    'guards': v.get('guards'), 'reasons': (v.get('reasons') or [])[:10],
                    'best_shortcut': (cells.get('best_shortcut') or {}).get('name')})
        results.append(ent)
    ceil = [e for e in results if e['kind'] == 'ceiling' and e['declared']]
    floors = [e for e in results if e['kind'] == 'floor']
    if not ceil:
        ceiling_any = False
        c_reason = 'lock.testbed.ceiling_grid is empty'
    elif any(e['status'] == 'PASS' for e in ceil):
        ceiling_any = True
        c_reason = 'ceiling %s passes' % ','.join(e['arm'] for e in ceil if e['status'] == 'PASS')
    elif all(e['status'] == 'FAIL' for e in ceil):
        ceiling_any = False
        c_reason = 'every ceiling fails the rule'
    else:
        ceiling_any = None
        c_reason = 'no ceiling passes and ceilings %s are incomplete' % ','.join(e['arm'] for e in ceil if e['status'] == 'MISSING')
    scored_f = [e for e in floors if e['status'] in ('PASS', 'FAIL')]
    best_floor = None
    if scored_f:
        bf = max(scored_f, key=lambda e: ((e.get('mean_d') if e.get('mean_d') is not None else -9.0),
                                          (e.get('mean_cand') or 0.0), e['arm']))
        best_floor = {'arm': bf['arm'], 'mean_cand': bf.get('mean_cand'), 'mean_d': bf.get('mean_d'), 'status': bf['status']}
    if not floors:
        floors_fail = None
        f_reason = 'lock.testbed.floor_grid is empty'
    elif any(e['status'] == 'PASS' for e in floors):
        floors_fail = False
        f_reason = 'floor %s passes the rule' % ','.join(e['arm'] for e in floors if e['status'] == 'PASS')
    elif all(e['status'] == 'FAIL' for e in floors):
        floors_fail = True
        f_reason = 'every floor fails the rule (best %s)' % (best_floor['arm'] if best_floor else '-')
    else:
        floors_fail = None
        f_reason = 'floors %s are incomplete' % ','.join(e['arm'] for e in floors if e['status'] == 'MISSING')
    if ceiling_any is False or floors_fail is False:
        result = 'CANNOT_TEST_CLAIM_AS_SPECIFIED'
    elif ceiling_any and floors_fail:
        result = 'PASS'
    else:
        result = 'MISSING'
    out = {'schema': SCHEMA_TESTBED, 'delivery': ctx.delivery_rel, 'endpoint_lock_sha256': ctx.endpoint_lock_sha256,
           'source_lock_sha256': ctx.source_lock_sha256, 'checker_version': common.CHECKER_VERSION,
           'decide_version': decide.DECIDE_VERSION, 'rule_view': 'decide.testbed_lock: all_worlds per technical world, '
           'min_worlds 1, no learner comparators, guards only where measured', 'configs': results,
           'ceiling_any_pass': ceiling_any, 'floors_all_fail': floors_fail, 'best_floor': best_floor,
           'result': result, 'reason': '%s; %s' % (c_reason, f_reason)}
    ctx._cache['wp3_testbed'] = out
    return out


def taught_acquisition(ctx: common.Ctx) -> dict:
    """ADM-ACQ input: candidate exact own-EOS on TAUGHT at the prefix checkpoint vs the best
    exposure-independent shortcut there plus admission_thresholds.taught_acq_margin_min, per technical world."""
    lock = ctx.lock
    sr = score_role(ctx, 'admission')
    cells = sr['cells']
    p, problems, _nf = decide.parse_params(lock)
    metric = p.get('metric') if p.get('metric') in decide.METRICS else 'exact_own_eos'
    cand = (lock.get('primary') or {}).get('candidate_arm')
    at = lock.get('admission_thresholds') if isinstance(lock.get('admission_thresholds'), dict) else {}
    margin = at.get('taught_acq_margin_min')
    idx = decide.index_cells(cells['cells'])
    # admission worlds only (common.admission_worlds): testbed-only technical worlds carry no learner run
    worlds = common.admission_worlds(ctx)[0] or sorted(w for w, r in cells['worlds'].items() if r == 'technical')
    per = []
    for w in worlds:
        groups = idx.get((w, cand, 'TAUGHT', 'prefix'))
        v = decide.group_rate(groups, decide.all_items, metric)[0] if groups else None
        best = decide.ladder_best(cells, w, 'TAUGHT', 'prefix')
        ok = (v is not None and best is not None and decide.is_finite_number(margin) and v >= best[1] + margin)
        per.append({'world': w, 'cand': v, 'best_shortcut': {'name': best[0], 'value': best[1]} if best else None,
                    'pass': ok})
    status = 'MISSING' if (not per or any(x['cand'] is None or x['best_shortcut'] is None for x in per)
                           or not decide.is_finite_number(margin)) else ('PASS' if all(x['pass'] for x in per) else 'FAIL')
    return {'checkpoint': 'prefix', 'pool': 'TAUGHT', 'margin': margin, 'per_world': per, 'status': status}


def shuf_summary(ctx: common.Ctx) -> dict:
    """ADM-SHUF input: agreement of each SHUF arm's TAUGHT labels with the true answers and orbit consistency."""
    lock = ctx.lock
    fmt = _fmt(lock)
    if not addition.is_addition(fmt):
        return {'status': 'NOT_APPLICABLE', 'reason': 'non-addition task', 'arms': []}
    taught = load_taught(ctx, lock)
    shufs = arms_with_role(lock, 'shuf')
    out = []
    for (w, a), rows in sorted(taught.items()):
        if a not in shufs:
            continue
        agree = 0
        n = 0
        orbits: Dict[Tuple[str, str], Set[str]] = {}
        truths: Set[str] = set()
        for r in rows:
            pr = addition.parse(r.get('prompt_hex'), fmt)
            if pr is None:
                raise common.InvalidInput('TAUGHT/%s/%s: prompt %r does not parse' % (w, a, r.get('prompt_hex')))
            truth = addition.expected(r.get('prompt_hex'), fmt)
            truths.add(truth)
            n += 1
            agree += 1 if r.get('answer_hex') == truth else 0
            orbits.setdefault(addition.canonical(*pr), set()).add(str(r.get('answer_hex')))
        bad = sorted('+'.join(o) for o, labs in orbits.items() if len(labs) > 1)
        out.append({'world': w, 'arm': a, 'n': n, 'agreement': (agree / float(n)) if n else None,
                    'orbit_consistent': not bad, 'inconsistent_orbits': bad[:20], 'distinct_true_answers': len(truths),
                    'binary_derangement': len(truths) == 2 and n > 0 and agree == 0})
    return {'status': 'OK' if out else 'MISSING', 'arms': out, 'shuf_arms': shufs}


# --------------------------------------------------------------------------------------
# UDI (SPEC 9.7)
# --------------------------------------------------------------------------------------

def _taught_multiset(rows: Sequence[dict]) -> Dict[tuple, int]:
    out: Dict[tuple, int] = {}
    for r in rows:
        k = (r.get('prompt_hex'), r.get('answer_hex'), r.get('phase'), r.get('feedback_bearing'), r.get('presentations'))
        out[k] = out.get(k, 0) + 1
    return out


def udi(ctx: common.Ctx, role: str = 'official') -> dict:
    """Leave-units-out UDI with a world-level LCB, and the validity problems of the variants."""
    lock = ctx.lock
    fmt = _fmt(lock)
    vworlds, man, err = udi_variant_worlds(ctx, lock)
    if man is None:
        raise common.MissingInput('UDI variants manifest unavailable: %s' % (err or 'lock.udi is null'))
    units = lock.get('units') if isinstance(lock.get('units'), dict) else {}
    witness = units.get('witness')
    commutative = bool(units.get('commutative'))
    if witness not in addition.WITNESSES or not addition.is_addition(fmt):
        raise common.InvalidInput('UDI needs an addition task with units.witness pair_rectangle|column_closure')
    sr = score_role(ctx, role)
    base_items = sr['item_exact'] or {}
    cand = (lock.get('primary') or {}).get('candidate_arm')
    taught = sr['taught']
    var_cells = (score_rows(sr['variant_rows'], lock, taught, item_detail=('HELD', 'final'),
                            exposed_orbits=sr.get('exposed_orbits')) if sr['variant_rows'] else None)
    var_items = (var_cells or {}).get('item_exact') or {}
    counted = set(((lock.get('primary') or {}).get('strata_counted')) or [])
    problems: List[str] = []
    values: List[float] = []
    per: List[dict] = []
    for i, v in enumerate(man.get('variants') or []):
        lab = 'variant %d' % (i + 1)
        if not isinstance(v, dict):
            problems.append('%s is not an object' % lab)
            continue
        bw, vw = v.get('base_world'), v.get('variant_world')
        removed = v.get('removed_taught_prompt_hex') or []
        D = list(v.get('dependent_item_ids') or [])
        I = list(v.get('independent_item_ids') or [])
        lab = '%s (%s -> %s)' % (lab, bw, vw)
        pb = []
        if bw not in (sr['cells'].get('worlds') or {}):
            pb.append('base world has no %s rows' % role)
        if vw not in var_items:
            pb.append('variant world has no candidate HELD/final rows')
        if not removed:
            pb.append('removed set is empty')
        if not D or not I:
            pb.append('dependent and independent sets must both be non-empty')
        if set(D) & set(I):
            pb.append('dependent and independent sets overlap')
        tb = taught.get((bw, cand))
        tv = taught.get((vw, cand))
        if tb is None or tv is None:
            pb.append('TAUGHT manifests for %s in both worlds are required' % cand)
        else:
            rem = set(removed)
            base_prompts = set(r.get('prompt_hex') for r in tb)
            if not rem <= base_prompts:
                pb.append('removed prompts not in TAUGHT(base): %s' % sorted(rem - base_prompts)[:5])
            expect = _taught_multiset([r for r in tb if r.get('prompt_hex') not in rem])
            if _taught_multiset(tv) != expect:
                pb.append('TAUGHT(variant) != TAUGHT(base) minus the removed set')
            for (w2, a2), rows2 in taught.items():
                if w2 == bw and a2 != cand and (vw, a2) in taught:
                    if _taught_multiset(taught[(vw, a2)]) != _taught_multiset(
                            [r for r in rows2 if r.get('prompt_hex') not in rem]):
                        pb.append('TAUGHT(variant)/%s != TAUGHT(base)/%s minus the removed set' % (a2, a2))
        bitems = base_items.get(bw) or {}
        vitems = var_items.get(vw) or {}
        if not pb:
            T_b = addition.Taught.from_rows(tb, fmt, 'TAUGHT/%s/%s' % (bw, cand))
            T_v = addition.Taught.from_rows(tv, fmt, 'TAUGHT/%s/%s' % (vw, cand))
            for iid in D + I:
                if iid not in bitems or iid not in vitems:
                    pb.append('item %s lacks candidate HELD/final rows in base or variant' % iid)
                    continue
                ph = bitems[iid][0]
                if vitems[iid][0] != ph:
                    pb.append('item %s has different prompts in base and variant' % iid)
                    continue
                st = addition.stratum({'prompt_hex': ph}, T_b)
                if st not in counted or addition.determined({'prompt_hex': ph}, T_b, witness, commutative) is not True:
                    pb.append('item %s is not a counted determined item in the base world' % iid)
                    continue
                dv = addition.determined({'prompt_hex': ph}, T_v, witness, commutative)
                if iid in D and dv is not False:
                    pb.append('dependent item %s stays determined after removal' % iid)
                if iid in I and dv is not True:
                    pb.append('independent item %s becomes undetermined after removal' % iid)
        if pb:
            problems.extend('%s: %s' % (lab, x) for x in pb)
            per.append({'base_world': bw, 'variant_world': vw, 'valid': False, 'problems': pb})
            continue

        def acc(items: dict, ids: List[str]) -> float:
            return sum(1 for x in ids if items[x][1]) / float(len(ids))
        val = (acc(bitems, D) - acc(vitems, D)) - (acc(bitems, I) - acc(vitems, I))
        values.append(val)
        per.append({'base_world': bw, 'variant_world': vw, 'valid': True, 'acc_base_D': acc(bitems, D),
                    'acc_var_D': acc(vitems, D), 'acc_base_I': acc(bitems, I), 'acc_var_I': acc(vitems, I), 'udi': val})
    udi_cfg = lock.get('udi') if isinstance(lock.get('udi'), dict) else {}
    alpha = udi_cfg.get('alpha')
    if not decide.is_finite_number(alpha) or not (0 < alpha < 0.5):
        agg = ((lock.get('primary') or {}).get('aggregator') or {})
        alpha = agg.get('alpha') if decide.is_finite_number(agg.get('alpha')) else 0.05
    out = {'valid': not problems and bool(values), 'problems': problems[:50], 'per_variant': per, 'alpha': alpha,
           'value': (math.fsum(values) / len(values)) if values else None, 'lcb': None, 'lcb_method': None}
    if values:
        lcb, method, _q, _sd = decide.lower_bound(values, alpha)
        out['lcb'] = lcb
        out['lcb_method'] = method
    if not man.get('variants'):
        out['valid'] = False
        out['problems'].append('no variants declared')
    return out


def udi_applicable(ctx: common.Ctx) -> bool:
    return isinstance(ctx.lock.get('udi'), dict) or common.adoption_intended(ctx)


# --------------------------------------------------------------------------------------
# SCORES summary and score()
# --------------------------------------------------------------------------------------

def _mean(xs: Sequence[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return math.fsum(xs) / len(xs) if xs else None


def _rate(idx: dict, w: str, arm: str, pool: str, ck: str, flt: Callable, metric: str) -> Optional[float]:
    groups = idx.get((w, arm, pool, ck))
    if not groups:
        return None
    return decide.group_rate(groups, flt, metric)[0]


def strata_counts(lock: dict, cells: dict, worlds: Sequence[str]) -> Tuple[Dict[str, int], int, int]:
    """(stratum -> items, determined counted items, counted-strata items) on the primary pool and checkpoint,
    pooled over (world, item) instances. Items without a computable stratum are counted as UNCOMPUTED."""
    p = decide.parse_params(lock)[0]
    cand = p.get('candidate_arm')
    idx = decide.index_cells(cells['cells'])
    strata: Dict[str, int] = {}
    det_total = det_yes = 0
    counted_set = set(p.get('strata_counted') or [])
    for w in worlds:
        for g in idx.get((w, cand, p.get('pool'), p.get('checkpoint')), []):
            st = g.get('stratum') if g.get('stratum') is not None else 'UNCOMPUTED'
            strata[st] = strata.get(st, 0) + g['n']
            if g.get('stratum') in counted_set:
                det_total += g['n']
                if g.get('determined') is True:
                    det_yes += g['n']
    return strata, det_yes, det_total


def comparator_strength(lock: dict, cells: dict, worlds: Sequence[str],
                        decision: bool = False) -> Tuple[Dict[str, Optional[float]], List[str], Optional[dict]]:
    """(comparator arm -> mean counted rate, arms below the best shortcut (WEAK_COMPARATOR), best_shortcut).
    The declared comparator_arms (LAD-03 reads these); decision=True: the decision comparator of SPEC 20.6
    (declared plus implied control arms)."""
    p = decide.parse_params(lock)[0]
    metric = p.get('metric') if p.get('metric') in decide.METRICS else 'exact_own_eos'
    idx = decide.index_cells(cells['cells'])
    flt = decide.counted_filter(lock)
    comparators: Dict[str, Optional[float]] = {}
    for a in (p.get('comparator_arms') if decision else p.get('declared_comparator_arms')) or []:
        vals = [_rate(idx, w, a, p.get('pool'), p.get('checkpoint'), flt, metric) for w in worlds]
        comparators[a] = _mean(vals) if vals and all(v is not None for v in vals) else None
    best = cells.get('best_shortcut') or None
    weak = sorted(a for a, v in comparators.items() if v is not None and best and best.get('value') is not None
                  and v < best['value'])
    return comparators, weak, best


def summarize(ctx: common.Ctx, role: str, sr: dict) -> dict:
    lock = ctx.lock
    cells = sr['cells']
    p, problems, _nf = decide.parse_params(lock)
    metric = p.get('metric') if p.get('metric') in decide.METRICS else 'exact_own_eos'
    cand = (lock.get('primary') or {}).get('candidate_arm')
    ppool, pck = p.get('pool'), p.get('checkpoint')
    idx = decide.index_cells(cells['cells'])
    flt = decide.counted_filter(lock)
    worlds, _n = expected_worlds(lock, role, cells)
    worlds = [w for w in worlds if w in cells['worlds']] or sorted(cells['worlds'])
    chance = p.get('chance') if p.get('chance') is not None else 0.0
    npt = arms_with_role(lock, 'no_prior_teaching')
    # G (SPEC 9.6): HELD, final, counted
    g_parts = []
    held_vals = []
    for w in worlds:
        v = _rate(idx, w, cand, 'HELD', 'final', flt, metric)
        held_vals.append(v)
        best = decide.ladder_best(cells, w, 'HELD', 'final')
        comps = [chance] + [x for x in (_rate(idx, w, a, 'HELD', 'final', flt, metric) for a in npt) if x is not None]
        if best is not None:
            comps.append(best[1])
        g_parts.append(None if (v is None or best is None) else v - max(comps))
    G = _mean(g_parts) if g_parts and all(x is not None for x in g_parts) else None
    held = _mean(held_vals) if held_vals and all(x is not None for x in held_vals) else None
    # receipts for G_per_sample
    pres = []
    for w in worlds:
        rp = ctx.path('results/receipts/%s/%s.json' % (w, cand))
        try:
            rc = common.load_json_file(rp)
            tp = rc.get('taught_presentations') if isinstance(rc, dict) else None
            pres.append(float(tp) if decide.is_finite_number(tp) else None)
        except (common.MissingInput, common.InvalidInput):
            pres.append(None)
    mean_pres = _mean(pres) if pres and all(x is not None for x in pres) else None
    G_per_sample = (G / (mean_pres / 1000.0)) if (G is not None and mean_pres) else None
    cpu_basis = None
    cpu_h = None
    try:
        fam_id, members = common.family_of(ctx)
        cpu_h = common.family_cpu_h(ctx, members)
        cpu_basis = 'family %s (%d members), all attempts incl. failed' % (fam_id, len(members))
    except (common.MissingInput, common.InvalidInput, common.CheckerError) as e:
        try:
            cpu_h = common.attempts_cpu_s(common.read_attempts(ctx)) / 3600.0
            cpu_basis = 'this delivery runlog/ATTEMPTS.jsonl only (family unavailable: %s)' % e
        except (common.MissingInput, common.InvalidInput) as e2:
            cpu_basis = 'unavailable: %s' % e2
    G_per_cpu_h = (G / cpu_h) if (G is not None and cpu_h) else None
    # OLD
    ends, finals, acqs = [], [], []
    for w in worlds:
        e = _rate(idx, w, cand, 'OLD', 'old_end', decide.all_items, metric)
        f = _rate(idx, w, cand, 'OLD', 'final', decide.all_items, metric)
        b = decide.ladder_best(cells, w, 'OLD', 'old_end')
        ends.append(e)
        finals.append(f)
        acqs.append(None if (e is None or b is None) else e - b[1])
    old_end = _mean(ends) if ends and all(x is not None for x in ends) else None
    old_final = _mean(finals) if finals and all(x is not None for x in finals) else None
    old = {'end': old_end, 'final': old_final,
           'drop': (old_end - old_final) if (old_end is not None and old_final is not None) else None,
           'acq': _mean(acqs) if acqs and all(x is not None for x in acqs) else None}
    strata, det_yes, det_total = strata_counts(lock, cells, worlds)
    held_by_stratum = {}
    for st in decide.STRATA:
        num = den = dnum = dden = 0
        for w in worlds:
            for g in idx.get((w, cand, 'HELD', 'final'), []):
                if g.get('stratum') == st:
                    num += g['n_exact']
                    den += g['n']
                    if g.get('determined') is True:
                        dnum += g['n_exact']
                        dden += g['n']
        if den:
            held_by_stratum[st] = {'rate': num / float(den), 'n': den, 'n_determined': dden,
                                   'rate_determined': (dnum / float(dden)) if dden else None}
    # output stats
    stats = cells.get('output_stats') or {}
    prim_stats = [v for k, v in stats.items() if k.split('|')[1:] == [ppool, pck]]
    modal = max((s['modal_fraction'] for s in prim_stats if s.get('modal_fraction') is not None), default=None)
    ent = min((s['entropy_bits'] for s in prim_stats if s.get('entropy_bits') is not None), default=None)
    # comparators and other arms
    comparators, weak, best = comparator_strength(lock, cells, worlds)
    decision_comparators = comparator_strength(lock, cells, worlds, decision=True)[0]
    others = {}
    for a in lock_arms(lock):
        if a == cand:
            continue
        vals = [_rate(idx, w, a, ppool, pck, flt, metric) for w in worlds]
        if vals and all(x is not None for x in vals):
            others[a] = _mean(vals)
    best_other = None
    if others:
        bo = max(sorted(others), key=lambda a: others[a])
        best_other = {'name': bo, 'value': others[bo]}
    held_best = _mean([(decide.ladder_best(cells, w, 'HELD', 'final') or (None, None))[1] for w in worlds])
    k, pairs = common.designs_tried_k(ctx)
    alarms = inert_alarms(cells, lock, worlds)
    out = {
        'schema': SCHEMA_SCORES, 'role': role, 'delivery': ctx.delivery_rel, 'created_utc': common.utc_now(),
        'checker_version': common.CHECKER_VERSION, 'decide_version': decide.DECIDE_VERSION,
        'endpoint_lock_sha256': ctx.endpoint_lock_sha256, 'source_lock_sha256': ctx.source_lock_sha256,
        'n_rows': sr['n_rows'], 'worlds': cells['worlds'], 'evaluated_worlds': worlds,
        'metric': metric, 'primary': cells['primary'],
        'cells': cells['cells'], 'ladder': cells['ladder'], 'ladder_status': cells['ladder_status'],
        'best_shortcut': best, 'strata': strata, 'determinacy': {'determined': det_yes, 'total': det_total},
        'determined_fraction': (det_yes / float(det_total)) if det_total else None,
        'held_exact_own_eos': held, 'held_by_stratum': held_by_stratum, 'held_best_shortcut_value': held_best,
        'headroom': (held - held_best) if (held is not None and held_best is not None) else None,
        'new_final': _mean([_rate(idx, w, cand, 'NEW', 'final', decide.all_items, metric) for w in worlds]),
        'modal_fraction': modal, 'entropy_bits': ent, 'output_stats': stats,
        'G': G, 'G_per_sample': G_per_sample, 'G_per_cpu_h': G_per_cpu_h,
        'G_basis': {'taught_presentations_mean': mean_pres, 'cpu_h': cpu_h, 'cpu_basis': cpu_basis},
        'old': old, 'udi': None, 'designs_tried_k': k, 'designs': pairs, 'inert_alarms': alarms,
        'met_mismatches': cells['met_mismatches'], 'met_mismatch_counts': cells['met_mismatch_counts'],
        'met02_mismatches': cells['met02_mismatches'], 'role_conflicts': cells['role_conflicts'],
        'comparators': comparators, 'decision_comparators': decision_comparators, 'weak_comparator': weak,
        'best_other_arm': best_other,
        'cells_sha256': decide.cells_sha256(cells), 'non_native_rows': cells['non_native_rows'],
    }
    return out


def _write_json(ctx: common.Ctx, rel: str, obj: dict) -> Path:
    if not common.is_v1_delivery(ctx.delivery):
        raise common.MissingInput('legacy delivery: use retro (the scorer never writes into non-v1 directories)')
    path = ctx.path(rel)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(common.dumps_pretty(decide.sanitize(obj)), encoding='utf-8')
    ctx._cache.pop('delivery_files', None)
    return path


def score(ctx: common.Ctx, role: str) -> dict:
    """Score results/<role>/OUTPUTS rows and write results/<role>/SCORES.json (plus VERDICT.json for
    official and TESTBED.json for admission). Returns {'scores', 'paths', 'verdict'?, 'testbed'?}."""
    if role not in ROLES:
        raise common.InvalidInput('role must be one of %s' % '|'.join(ROLES))
    sr = score_role(ctx, role)
    body = summarize(ctx, role, sr)
    out: Dict[str, Any] = {'paths': []}
    if role in LEARNER_ROLES:
        v = evaluate_role(ctx, role)
        body['decision'] = {'verdict': v.get('verdict'), 'shadow_verdict': v.get('shadow_verdict'),
                            'invalid_reason': v.get('invalid_reason'), 'primary': v.get('primary'),
                            'guards': v.get('guards'), 'gates': v.get('gates'), 'reasons': v.get('reasons', [])[:20]}
        if isinstance(ctx.lock.get('udi'), dict):
            try:
                body['udi'] = udi(ctx, role)
            except (common.MissingInput, common.InvalidInput) as e:
                body['udi'] = {'valid': False, 'problems': [str(e)], 'value': None, 'lcb': None}
        if role == 'official':
            verdict = build_verdict(ctx)
            verdict['created_utc'] = common.utc_now()
            out['verdict'] = verdict
            out['paths'].append(_write_json(ctx, 'results/official/VERDICT.json', verdict))
    if role == 'admission':
        try:
            tb = testbed(ctx)
        except common.MissingInput as e:
            tb = {'schema': SCHEMA_TESTBED, 'result': 'MISSING', 'reason': str(e), 'configs': []}
        tb = dict(tb)
        tb['created_utc'] = common.utc_now()
        body['testbed'] = {'result': tb.get('result'), 'reason': tb.get('reason')}
        body['taught_acquisition'] = taught_acquisition(ctx)
        body['shuf'] = shuf_summary(ctx)
        out['testbed'] = tb
        out['paths'].append(_write_json(ctx, 'results/admission/TESTBED.json', tb))
    out['paths'].insert(0, _write_json(ctx, 'results/%s/SCORES.json' % role, body))
    out['scores'] = body
    return out


def cmd_score(args: Any) -> int:
    """MFC score <dir> --role R. Exit 0 when the artifacts were written (whatever the verdict), 2 on missing
    input, 1 on invalid input."""
    ctx = common.make_ctx(args, delivery=args.dir, stage='score', options={'invoked_by': 'score'})
    ok, why = common.v1_status(ctx.delivery)
    if not ok:
        sys.stderr.write('legacy delivery: use retro (%s)\n' % why)
        return common.EXIT_MISSING
    try:
        res = score(ctx, args.role)
    except common.MissingInput as e:
        sys.stderr.write('MISSING: %s\n' % e)
        return common.EXIT_MISSING
    except common.InvalidInput as e:
        sys.stderr.write('FAIL (invalid input): %s\n' % e)
        return common.EXIT_FAIL
    if getattr(args, 'json', False):
        import json
        body = {'scores': res['scores']}
        if 'verdict' in res:
            body['verdict'] = res['verdict']
        if 'testbed' in res:
            body['testbed'] = res['testbed']
        print(json.dumps(decide.sanitize(body), indent=2, sort_keys=True))
    else:
        s = res['scores']
        bs = s.get('best_shortcut') or {}
        line = 'score %s --role %s: %d rows; best shortcut %s %s; G %s' % (
            ctx.delivery_rel, args.role, s['n_rows'], bs.get('name'), _fmt_num(bs.get('value')), _fmt_num(s.get('G')))
        if 'verdict' in res:
            line += '; verdict %s' % res['verdict'].get('verdict')
        if 'testbed' in res:
            line += '; testbed %s' % res['testbed'].get('result')
        if s.get('decision'):
            line += '; decision %s (shadow %s)' % (s['decision'].get('verdict'), s['decision'].get('shadow_verdict'))
        print(line)
        for p in res['paths']:
            print('wrote %s' % Path(p).relative_to(ctx.repo_root).as_posix())
    return common.EXIT_PASS


def _fmt_num(v: Any) -> str:
    return ('%.4f' % v) if decide.is_finite_number(v) else str(v)


# --------------------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------------------

def _attempt_roles(ctx: common.Ctx) -> Set[str]:
    out = set()
    for a in common.read_attempts(ctx):
        for pp in a.get('produced_paths') or []:
            path = pp.get('path') if isinstance(pp, dict) else None
            if isinstance(path, str):
                for r in ROLES:
                    if path.startswith('results/%s/' % r) and path.rsplit('/', 1)[-1] not in (
                            'SCORES.json', 'VERDICT.json', 'TESTBED.json'):
                        out.add(r)
    return out


def scope_roles(ctx: common.Ctx, include_admission: bool = False) -> Tuple[List[str], List[str]]:
    """(roles to check, missing descriptions). Official stage: dev and rehearsal rows (plus official rows if
    present); a role whose worlds the lock lists must have rows. Other stages: every role with rows; a role
    with launch-recorded outputs but no rows is missing."""
    lock = ctx.lock
    roles: List[str] = []
    missing: List[str] = []
    if ctx.stage == 'official':
        for r in ('dev', 'rehearsal'):
            if role_has_rows(ctx, r):
                roles.append(r)
            elif lock_worlds(lock, r):
                missing.append('%s rows absent (lock lists %s worlds)' % (r, r))
        if role_has_rows(ctx, 'official'):
            roles.append('official')
        return roles, missing
    att = _attempt_roles(ctx)
    for r in (['admission'] if include_admission else []) + list(LEARNER_ROLES):
        if role_has_rows(ctx, r):
            roles.append(r)
        elif r in att:
            missing.append('%s outputs were launch-recorded but no rows are readable' % r)
    return roles, missing


def _result(cid: str, problems: List[str], missing: List[str], ok_detail: str, data: Optional[dict] = None,
            warn: bool = False) -> common.Result:
    if problems:
        st = common.WARN if warn else common.FAIL
        return common.Result(cid, st, '; '.join(problems[:20] + missing[:5])[:4000], data=data or {})
    if missing:
        return common.Result(cid, common.MISSING, '; '.join(missing[:10]), data=data or {})
    return common.Result(cid, common.PASS, ok_detail, data=data or {})


CANONICAL_ROW_FILES = tuple('results/%s/OUTPUTS.jsonl' % r for r in ROLES) + ('results/admission/FIXTURES.jsonl',)


def stray_output_rows(ctx: common.Ctx) -> List[str]:
    """Row files under results/<role>/ that look like outputs but are not the canonical OUTPUTS.jsonl: the scorer
    never reads them, so their rows would escape every check."""
    out = []
    for logical, kind, ref, _gz in ctx.row_sources('results/*/*.jsonl'):
        if logical in CANONICAL_ROW_FILES or logical == common.REPLAY_OUTPUTS:
            continue  # replay rows are never scored; RUN-04 reads them
        if 'OUTPUT' in logical.rsplit('/', 1)[-1].upper():
            out.append(logical if kind == 'file' else '%s!%s' % ref)
    return out


@common.check('MET-01')
def check_met_01(ctx: common.Ctx) -> common.Result:
    roles, missing = scope_roles(ctx, include_admission=True)
    problems = []
    stray = stray_output_rows(ctx)
    if stray:
        problems.append('output rows outside results/<role>/OUTPUTS.jsonl[.gz] are never scored: %s' % ', '.join(stray[:10]))
    checked = 0
    nonnative = 0
    for r in roles:
        c = score_role(ctx, r)['cells']
        checked += c['n_rows']
        nonnative += c['non_native_rows']
        if c['met_mismatch_counts']:
            ex = c['met_mismatches'][:3]
            problems.append('%s: %d mismatches %s (e.g. %s)' % (r, sum(c['met_mismatch_counts'].values()),
                                                                c['met_mismatch_counts'], ex))
    if not roles and not missing and not problems:
        return common.Result('MET-01', common.PASS, 'no OUTPUTS rows in scope (nothing opened)')
    return _result('MET-01', problems, missing, '%d rows recomputed from stream_hex in %s; runner emitted_hex/own_eos '
                   'agree; %d rows not on the native path never count' % (checked, ','.join(roles), nonnative))


def _manifest_truth_problems(ctx: common.Ctx, fmt: dict) -> List[str]:
    lock = ctx.lock
    split = lock.get('split') if isinstance(lock.get('split'), dict) else {}
    out = []
    for key in ('held_manifest', 'old_manifest', 'new_manifest'):
        rel = split.get(key)
        if not isinstance(rel, str) or not rel or not ctx.path(rel).is_file():
            continue
        bad = 0
        first = None
        for i, r in enumerate(ctx.read_jsonl(rel)):
            want = addition.expected(r.get('prompt_hex'), fmt)
            if r.get('expected_hex') != want:
                bad += 1
                first = first or (i + 1, r.get('prompt_hex'), r.get('expected_hex'), want)
        if bad:
            out.append('%s: %d expected_hex mismatches (first line %d: %s -> %s, recomputed %s)' % ((rel, bad) + first))
    return out


@common.check('MET-02')
def check_met_02(ctx: common.Ctx) -> common.Result:
    fmt = _fmt(ctx.lock)
    if not addition.is_addition(fmt):
        return common.Result('MET-02', common.NOT_APPLICABLE, 'task.format.kind is %r, not addition' % fmt.get('kind'))
    roles, missing = scope_roles(ctx, include_admission=True)
    problems = _manifest_truth_problems(ctx, fmt)
    checked = 0
    for r in roles:
        c = score_role(ctx, r)['cells']
        checked += c['n_rows']
        if c['met02_mismatch_count']:
            problems.append('%s: %d rows with expected_hex != the recomputed sum (e.g. %s)'
                            % (r, c['met02_mismatch_count'], c['met02_mismatches'][:3]))
    return _result('MET-02', problems, missing, '%d rows: expected_hex equals the normalized sum recomputed from '
                   'prompt_hex (%s)' % (checked, ','.join(roles) or 'no rows in scope'))


def implied_pool_checkpoints(lock: dict) -> List[Tuple[str, str]]:
    """The (pool, checkpoint) cells a lock implies (LAD-01): the cells its rule, guards and gates read, plus the
    fixed SPEC measurement points: the primary (pool, checkpoint); OLD at old_end and final for the OLD guards
    and old_end/old_final (9.5, 9.6); TAUGHT at prefix for acquisition (ADM-ACQ); HELD and NEW at final (G,
    new_final); any gate's (pool, checkpoint); and every other declared pool at final (I4: the final
    checkpoint). SCHEMAS.md defines no separate measurement plan, so this set is what 'implied by the lock'
    means; a pool or checkpoint the lock does not declare is never implied."""
    prim = lock.get('primary') if isinstance(lock.get('primary'), dict) else {}
    nip = prim.get('n_items_per_pool') if isinstance(prim.get('n_items_per_pool'), dict) else {}
    cks = set((lock.get('checkpoints') or {}).keys()) if isinstance(lock.get('checkpoints'), dict) else set()
    pools = set(nip)
    out: Set[Tuple[str, str]] = set()

    def add(pool: Any, ck: Any) -> None:
        if pool in pools and ck in cks:
            out.add((pool, ck))
    add(prim.get('pool'), prim.get('checkpoint'))
    for g in lock.get('guards') or []:
        if isinstance(g, dict):
            gp = g.get('pool') or 'OLD'
            add(gp, 'old_end')
            if g.get('kind', g.get('id')) == 'old_drop':
                add(gp, 'final')
    for gt in lock.get('gates') or []:
        if isinstance(gt, dict):
            add(gt.get('pool'), gt.get('checkpoint'))
    add('OLD', 'old_end')
    add('OLD', 'final')
    add('TAUGHT', 'prefix')
    add('HELD', 'final')
    add('NEW', 'final')
    for pool in pools:
        if not any(pc[0] == pool for pc in out):
            add(pool, 'final')
    return sorted(out)


def _implied_cells(lock: dict, role: str, cells: dict) -> Tuple[List[str], List[str], List[Tuple[str, str]], dict, List[str]]:
    worlds, n = expected_worlds(lock, role, cells)
    problems = []
    if n is not None and len(worlds) != n:
        problems.append('%s: %d worlds with rows, lock.worlds.official.n=%d' % (role, len(worlds), n))
    arms = lock_arms(lock)
    nip = ((lock.get('primary') or {}).get('n_items_per_pool')) or {}
    pcs = implied_pool_checkpoints(lock)
    for pool in sorted(nip):
        if not any(pc[0] == pool for pc in pcs):
            problems.append('pool %s is declared in n_items_per_pool but no declared checkpoint measures it' % pool)
    return worlds, arms, pcs, nip, problems


@common.check('LAD-01')
def check_lad_01(ctx: common.Ctx) -> common.Result:
    lock = ctx.lock
    roles, missing = scope_roles(ctx)
    roles = [r for r in roles if r in LEARNER_ROLES]
    problems: List[str] = []
    n_cells = 0
    split_items = split_manifest_items(ctx, lock) if roles else {}
    for r in roles:
        sr = score_role(ctx, r)
        cells = sr['cells']
        worlds, arms, pcs, nip, pb = _implied_cells(lock, r, cells)
        problems.extend(pb)
        if not arms or not pcs:
            problems.append('lock does not imply cells (arms %r, n_items_per_pool %r, checkpoints %r)'
                            % (arms, nip, lock.get('checkpoints')))
            continue
        for pool in sorted(set(pc[0] for pc in pcs)):
            if isinstance(nip.get(pool), bool) or not isinstance(nip.get(pool), int) or nip[pool] < 1:
                problems.append('n_items_per_pool.%s=%r is not a positive integer' % (pool, nip.get(pool)))
        idx = decide.index_cells(cells['cells'])
        for w in worlds:
            for a in arms:
                for pool, ck in pcs:
                    want = nip.get(pool)
                    n_cells += 1
                    groups = idx.get((w, a, pool, ck))
                    if not groups:
                        problems.append('%s: cell %s/%s/%s/%s missing' % (r, w, a, pool, ck))
                        continue
                    n = sum(g['n'] for g in groups)
                    nr = sum(g['n_rows'] for g in groups)
                    if nr != want or n != want:
                        problems.append('%s: cell %s/%s/%s/%s has %d items with rows (universe %d), lock says %r'
                                        % (r, w, a, pool, ck, nr, n, want))
                    man = split_items.get(pool)
                    if man is not None:
                        got = sr['split_cells'].get((w, a, pool, ck), set())
                        if got != set(man):
                            miss, extra_items = sorted(set(man) - got), sorted(got - set(man))
                            problems.append('%s: cell %s/%s/%s/%s items differ from the locked %s manifest (missing %s%s)'
                                            % (r, w, a, pool, ck, pool, miss[:5],
                                               ', extra %s' % extra_items[:5] if extra_items else ''))
        # extra cells: a world, arm, pool or checkpoint the lock does not declare FAILs; an extra combination of
        # declared pools and checkpoints is allowed but must be complete too
        decl_pools = set(nip)
        decl_cks = set((lock.get('checkpoints') or {}).keys()) if isinstance(lock.get('checkpoints'), dict) else set()
        implied = set(pcs)
        extra = []
        for k in sorted(idx):
            w, a, pool, ck = k
            if w not in worlds or a not in arms or pool not in decl_pools or ck not in decl_cks:
                extra.append(k)
            elif (pool, ck) not in implied:
                n = sum(g['n'] for g in idx[k])
                nr = sum(g['n_rows'] for g in idx[k])
                if nr != nip.get(pool) or n != nip.get(pool):
                    problems.append('%s: extra cell %s/%s/%s/%s has %d items with rows (universe %d), lock says %r'
                                    % (r, w, a, pool, ck, nr, n, nip.get(pool)))
        if extra:
            problems.append('%s: %d cells not implied by the lock, e.g. %s' % (r, len(extra), ['/'.join(k) for k in extra[:5]]))
        if cells['role_conflicts']:
            problems.append('%s: world_role conflicts with lock.worlds: %s' % (r, cells['role_conflicts'][:5]))
    if not roles and not missing:
        return common.Result('LAD-01', common.PASS, 'no learner-arm rows in scope (nothing opened)')
    return _result('LAD-01', problems, missing, '%d lock-implied cells complete with n_items_per_pool items; HELD/OLD/NEW '
                   'cells hold exactly the locked split manifests (%s)' % (n_cells, ','.join(roles)))


def substitute_shortcut(cells: dict, lock: dict) -> dict:
    """Cells with the candidate's primary cells replaced, per world, by the best shortcut's predictions
    (the ladder is kept as computed from the real candidate)."""
    p, problems, nonfinite = decide.parse_params(lock)
    if problems or nonfinite:
        raise common.InvalidInput('lock does not define a decidable rule: %s' % '; '.join(problems + nonfinite))
    new = dict(cells)
    keep = [c for c in cells.get('cells') or [] if not (c.get('arm') == p['candidate_arm'] and c.get('pool') == p['pool']
                                                        and c.get('checkpoint') == p['checkpoint'])]
    subs = {}
    for w in sorted(cells.get('worlds') or {}):
        b = decide.ladder_best(cells, w, p['pool'], p['checkpoint'])
        if b is None:
            continue
        groups = ((cells.get('ladder_cells') or {}).get(decide.lkey(w, p['pool'], p['checkpoint'])) or {}).get(b[0])
        if not groups:
            continue
        subs[w] = b[0]
        for g in groups:
            c = dict(g)
            c.update({'world': w, 'world_role': cells['worlds'].get(w), 'arm': p['candidate_arm'], 'pool': p['pool'],
                      'checkpoint': p['checkpoint'], 'n_rows': g['n']})
            keep.append(c)
    new['cells'] = keep
    new['substituted'] = subs
    return new


@common.check('LAD-02')
def check_lad_02(ctx: common.Ctx) -> common.Result:
    lock = ctx.lock
    roles, missing = scope_roles(ctx)
    roles = [r for r in roles if r in LEARNER_ROLES]
    problems = []
    notes = []
    for r in roles:
        cells = score_role(ctx, r)['cells']
        sub = substitute_shortcut(cells, lock)
        worlds, n = expected_worlds(lock, r, cells)
        v = decide.evaluate(sub, lock, {'inert_alarms': [], 'adm_tb': 'PASS', 'adm_ctrl': 'PASS',
                                        'gates': {g: 'PASS' for g in decide.parse_params(lock)[0].get('gates', [])},
                                        'expected_worlds': worlds, 'expected_n_worlds': n})
        if decide.passes(v):
            problems.append('%s: decide PASSes with the best shortcut substituted for the candidate (%s)'
                            % (r, sub.get('substituted')))
        notes.append('%s -> %s/%s' % (r, v.get('verdict'), v.get('shadow_verdict')))
    if not roles and not missing:
        return common.Result('LAD-02', common.PASS, 'no learner-arm rows in scope (nothing opened)')
    return _result('LAD-02', problems, missing, 'shortcut substitution does not pass: %s' % '; '.join(notes))


@common.check('LAD-03')
def check_lad_03(ctx: common.Ctx) -> common.Result:
    if not role_has_rows(ctx, 'dev'):
        return common.Result('LAD-03', common.MISSING, 'no dev rows: comparator strength on dev cannot be checked')
    cells = score_role(ctx, 'dev')['cells']
    worlds, _n = expected_worlds(ctx.lock, 'dev', cells)
    comparators, weak, best = comparator_strength(ctx.lock, cells, [w for w in worlds if w in cells['worlds']])
    best = best or {}
    if weak:
        return common.Result('LAD-03', common.WARN, 'WEAK_COMPARATOR: %s below the best shortcut %s %.4f on dev (%s)'
                             % (','.join(weak), best.get('name'), best.get('value') or 0.0,
                                {a: comparators[a] for a in weak}), data={'weak_comparator': weak})
    if not best:
        return common.Result('LAD-03', common.MISSING, 'no shortcut ladder on dev (%s)' % cells.get('ladder_status'))
    return common.Result('LAD-03', common.PASS, 'declared comparators %s are at or above the best shortcut %s on dev'
                         % (comparators, best.get('name')))


@common.check('LAD-04')
def check_lad_04(ctx: common.Ctx) -> common.Result:
    roles, missing = scope_roles(ctx)
    roles = [r for r in roles if r in LEARNER_ROLES]
    hits = []
    for r in roles:
        for key, s in sorted((score_role(ctx, r)['cells'].get('output_stats') or {}).items()):
            mf, eb = s.get('modal_fraction'), s.get('entropy_bits')
            if (mf is not None and mf > LAD04_MAX_MODAL) or (eb is not None and eb < LAD04_MIN_ENTROPY):
                hits.append('%s %s: modal %.3f entropy %.3f bit (output %s)' % (r, key, mf or 0.0, eb or 0.0,
                                                                              s.get('modal_output_hex')))
    if hits:
        return common.Result('LAD-04', common.WARN, 'candidate output collapse: %s' % '; '.join(hits[:10]))
    if not roles and not missing:
        return common.Result('LAD-04', common.PASS, 'no learner-arm rows in scope (nothing opened)')
    if missing:
        return common.Result('LAD-04', common.MISSING, '; '.join(missing))
    return common.Result('LAD-04', common.PASS, 'modal fraction <= %.1f and entropy >= %.1f bit in every pool'
                         % (LAD04_MAX_MODAL, LAD04_MIN_ENTROPY))


def _min_items(ctx: common.Ctx) -> int:
    tf = ctx.kb.constraints().get('threshold_floors')
    v = tf.get('min_items_per_stratum') if isinstance(tf, dict) else None
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
        raise common.MissingInput('CONSTRAINTS.threshold_floors.min_items_per_stratum missing')
    return v


VALID_STRATA_BY_LEVEL = {'E3': ('S2', 'S3', 'S4'), 'E4': ('S2', 'S3', 'S4'), 'E2': ('S1', 'S2', 'S3', 'S4')}


@common.check('STR-01')
def check_str_01(ctx: common.Ctx) -> common.Result:
    lock = ctx.lock
    fmt = _fmt(lock)
    if not addition.is_addition(fmt):
        return common.Result('STR-01', common.NOT_APPLICABLE, 'task.format.kind is %r: strata are defined for addition'
                             % fmt.get('kind'))
    pr = lock.get('primary') if isinstance(lock.get('primary'), dict) else {}
    sc = pr.get('strata_counted')
    problems = []
    if not isinstance(sc, list) or not sc or not all(s in decide.STRATA for s in sc):
        problems.append('primary.strata_counted %r must be a non-empty subset of S0..S4' % (sc,))
        sc = [s for s in (sc or []) if s in decide.STRATA] if isinstance(sc, list) else []
    level = lock.get('claim_level_target')
    valid = VALID_STRATA_BY_LEVEL.get(level, decide.STRATA)
    bad = [s for s in sc if s not in valid]
    if bad:
        problems.append('strata %s are not valid counted strata at %s (allowed %s)' % (bad, level, ','.join(valid)))
    need = _min_items(ctx)
    roles, missing = scope_roles(ctx)
    roles = [r for r in roles if r in LEARNER_ROLES]
    notes = []
    for r in roles:
        cells = score_role(ctx, r)['cells']
        worlds, _n = expected_worlds(lock, r, cells)
        st = strata_counts(lock, cells, [w for w in worlds if w in cells['worlds']])[0]
        if st.get('UNCOMPUTED'):
            problems.append('%s: %d primary items have no computable stratum (unparseable prompt or no TAUGHT manifest '
                            'for the candidate)' % (r, st['UNCOMPUTED']))
        for s in sc:
            if st.get(s, 0) < need:
                problems.append('%s: counted stratum %s has %d items (pooled over worlds), min_items_per_stratum %d'
                                % (r, s, st.get(s, 0), need))
        notes.append('%s %s' % (r, st))
    if not roles and not missing and not problems:
        return common.Result('STR-01', common.PASS, 'counted strata %s valid at %s; no learner-arm rows in scope' % (sc, level))
    return _result('STR-01', problems, missing, 'strata computed from content; counted %s valid at %s with >= %d items: %s'
                   % (sc, level, need, '; '.join(notes)))


@common.check('DET-01')
def check_det_01(ctx: common.Ctx) -> common.Result:
    lock = ctx.lock
    if not decide.determinacy_applicable(lock):
        return common.Result('DET-01', common.NOT_APPLICABLE, 'no units block (units.witness) declared')
    units = lock['units']
    problems = []
    if units.get('witness') not in addition.WITNESSES:
        return common.Result('DET-01', common.FAIL, 'units.witness %r not in %s' % (units.get('witness'),
                                                                                  '|'.join(addition.WITNESSES)))
    if not addition.is_addition(_fmt(lock)):
        return common.Result('DET-01', common.FAIL, 'a determinacy witness needs an addition task format')
    need = _min_items(ctx)
    roles, missing = scope_roles(ctx)
    roles = [r for r in roles if r in LEARNER_ROLES]
    notes = []
    p = decide.parse_params(lock)[0]
    counted = set(p.get('strata_counted') or [])
    for r in roles:
        sr = score_role(ctx, r)
        cells = sr['cells']
        idx = decide.index_cells(cells['cells'])
        worlds, _n = expected_worlds(lock, r, cells)
        total = 0
        for w in worlds:
            groups = idx.get((w, p.get('candidate_arm'), p.get('pool'), p.get('checkpoint'))) or []
            det = sum(g['n'] for g in groups if g.get('stratum') in counted and g.get('determined') is True)
            unk = sum(g['n'] for g in groups if g.get('stratum') in counted and g.get('determined') is None)
            if unk:
                problems.append('%s %s: %d counted-stratum items have no computable determinacy' % (r, w, unk))
            if groups and det == 0:
                problems.append('%s %s: no determined counted items' % (r, w))
            total += det
        if total < need:
            problems.append('%s: %d determined counted items (pooled), fewer than %d: the testbed cannot test E3'
                            % (r, total, need))
        notes.append('%s %d determined counted items' % (r, total))
    if not roles and not missing:
        return common.Result('DET-01', common.PASS, 'witness %s declared; no learner-arm rows in scope' % units['witness'])
    return _result('DET-01', problems, missing, 'witness %s: %s' % (units['witness'], '; '.join(notes)))


@common.check('INERT-01')
def check_inert_01(ctx: common.Ctx) -> common.Result:
    roles, missing = scope_roles(ctx)
    roles = [r for r in roles if r in LEARNER_ROLES]
    problems = []
    for r in roles:
        cells = score_role(ctx, r)['cells']
        worlds, _n = expected_worlds(ctx.lock, r, cells)
        for a in inert_alarms(cells, ctx.lock, [w for w in worlds if w in cells['worlds']]):
            problems.append('%s: INVALID INTEGRITY_ALARM: %s' % (r, a['detail']))
    if not roles and not missing:
        return common.Result('INERT-01', common.PASS, 'no learner-arm rows in scope (nothing opened)')
    return _result('INERT-01', problems, missing, 'no identical declared arms and no zero-variance primary contrast (%s)'
                   % ','.join(roles))


@common.check('UDI-01')
def check_udi_01(ctx: common.Ctx) -> common.Result:
    if not udi_applicable(ctx):
        return common.Result('UDI-01', common.NOT_APPLICABLE, 'lock.udi is null and E3 promotion is not intended')
    if not role_has_rows(ctx, 'official'):
        if 'official' in _attempt_roles(ctx):
            return common.Result('UDI-01', common.MISSING, 'official outputs were launch-recorded but no rows are readable')
        return common.Result('UDI-01', common.NOT_APPLICABLE, 'UDI is measured on official worlds; none were opened')
    if not isinstance(ctx.lock.get('udi'), dict):
        return common.Result('UDI-01', common.MISSING, 'E3 promotion intended but lock.udi declares no variants manifest')
    res = udi(ctx, 'official')
    if not res['valid']:
        return common.Result('UDI-01', common.FAIL, 'UDI variants invalid under the witness: %s'
                             % '; '.join(res['problems'][:10]), data={'udi': res})
    if common.sealed_confirm_intended(ctx):
        # SPEC 9.7: for SEALED_CONFIRM the dependent half D is drawn from the seal-derived 'udi' seed. The spec does
        # not define the draw procedure, so the selection cannot be verified here (fail closed).
        return common.Result('UDI-01', common.MISSING, 'UDI %.4f (LCB %.4f) is valid under the witness, but for '
                             'SEALED_CONFIRM the seal-derived draw of D cannot be verified (no draw procedure is '
                             'defined)' % (res['value'], res['lcb']), data={'udi': res})
    return common.Result('UDI-01', common.PASS, 'UDI %.4f, world-level LCB %.4f (%s, alpha %s) over %d variants'
                         % (res['value'], res['lcb'], res['lcb_method'], res['alpha'], len(res['per_variant'])),
                         data={'udi': res})
