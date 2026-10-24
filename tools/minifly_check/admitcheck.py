"""MRK-1 admission and authorization checks (WP4): ADM-01, ADM-TB-CEIL, ADM-TB-FLOOR, ADM-CTRL-ZERO,
ADM-CTRL-ISO, ADM-CTRL-DOSE, ADM-DIR, ADM-FIX, ADM-GATE, ADM-INPUT, ADM-ACQ, ADM-SHUF, ADM-EXPO,
ADM-BUDGET, ADM-REUSE and AUT-01..04 (SPEC 7.4, 8, 12.2). Python 3.9+, standard library only.

The checker reads raw receipts (results/receipts/<world>/<arm>.json), raw TAUGHT manifests, raw
FIXTURES rows and raw admission OUTPUTS rows. ADMISSION.json summaries are never trusted: where a
summary restates something the raw artifacts also carry (worlds, lock shas, receipt coverage, the
input-contrast ratio), any disagreement is FAIL. Testbed, acquisition and SHUF numbers come from the
program scorer (scorer.testbed / taught_acquisition / shuf_summary), never from ADMISSION.

Delivery code is never imported or executed: the write-counter site is located with `ast` only.
The ADM-CTRL checks are called by scorer._adm_ctrl_status as an official precondition, so they never
call the scorer's verdict functions (that would recurse).
"""
from __future__ import annotations

import ast
import base64
import binascii
import gzip
import importlib
import json
import math
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import common
from common import FAIL, MISSING, NOT_APPLICABLE, PASS, InvalidInput, MissingInput, Result, check

FILL = '__FILL__'
AGENTS = ('codex', 'claude')
ADMISSION_REL = 'ADMISSION.json'
AUTHORIZATION_REL = 'AUTHORIZATION.json'
FIXTURES_REL = 'results/admission/FIXTURES.jsonl'
ADMISSION_OUTPUTS_REL = 'results/admission/OUTPUTS.jsonl'
RECEIPTS_PREFIX = 'results/receipts/'
SCHEMA_ADMISSION = 'minifly.admission.v1'
SCHEMA_AUTHORIZATION = 'minifly.authorization.v1'
ROLES = ('admission', 'dev', 'rehearsal', 'official')
CANONICAL_ROW_FILES = tuple('results/%s/OUTPUTS.jsonl' % r for r in ROLES) + (FIXTURES_REL,)
SHA_RE = re.compile(r'^[0-9a-f]{64}$')

ADMISSION_REQUIRED = ('endpoint_lock_sha256', 'source_lock_sha256', 'producer', 'worlds', 'testbed_outputs',
                      'receipts_glob', 'fixtures', 'input_contrast', 'structure_meters', 'parent_admission_sha',
                      'created_utc')
INPUT_CONTRAST_KEYS = ('metric_definition', 'analytic_snr', 'same_input_diff_noise', 'diff_input_same_noise', 'ratio',
                       'prompt_id_loo_acc', 'chance')
STRUCTURE_METERS = ('shared_operand_similarity', 'nonshared_operand_similarity', 'modal_output_fraction',
                    'code_overlap_birth', 'code_overlap_end')
# Relative tolerance for ratio == diff_input_same_noise / same_input_diff_noise (two-decimal rounding allowed).
RATIO_REL_TOL = 0.005

AUTH_STAGE_KEYS = ('implementation', 'technical', 'dev_cycles', 'official', 'push', 'push_lock', 'adoption',
                   'pointer_update')
AUTH_REQUIRED = ('verbatim', 'utc', 'session', 'options_shown', 'recommendation_shown', 'stages')
CAP_KEYS = ('max_records', 'max_cpu_h', 'max_cycles')
# gate / launch stage -> the AUTHORIZATION.stages key that must grant it (AUT-02)
STAGE_GRANT = {'lock': 'implementation', 'testbed': 'implementation', 'technical': 'technical',
               'roster': 'dev_cycles', 'dev': 'dev_cycles', 'rehearsal': 'dev_cycles', 'official': 'official'}
STAGE_ORDER = ('implementation', 'technical', 'dev_cycles', 'official')
PREDICATE_OPS = ('eq', 'ne', 'lt', 'le', 'gt', 'ge', 'in', 'exists')
RECOMMENDED_LABEL_RE = re.compile(r'recommend', re.IGNORECASE)
RECOMMENDED_TEXT_RE = re.compile(r'[(\[]\s*recommended\s*[)\]]', re.IGNORECASE)
FIXTURE_TOLERANCE_KEY = 'fixture_param_tolerances'  # lock.admission_thresholds.<key>: {dotted param: abs tol}


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------

def lazy(name: str):
    """Import a kit module lazily; a missing or broken module raises MissingInput (fail closed)."""
    try:
        return importlib.import_module(name)
    except ImportError as e:
        raise MissingInput('kit module %s unavailable: %s' % (name, e))


def is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def s(v: Any) -> bool:
    return isinstance(v, str) and bool(v.strip()) and v.strip() != FILL


def strip_sha(v: Any) -> Optional[str]:
    if not isinstance(v, str):
        return None
    t = v.strip().lower()
    if t.startswith('sha256:'):
        t = t[7:]
    return t if SHA_RE.match(t) else None


def dig(obj: Any, dotted: str, default: Any = None) -> Any:
    cur = obj
    for part in dotted.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def fill_paths(obj: Any, prefix: str = '') -> List[str]:
    out: List[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(fill_paths(v, '%s.%s' % (prefix, k) if prefix else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(fill_paths(v, '%s[%d]' % (prefix, i)))
    elif obj == FILL:
        out.append(prefix or '<root>')
    return out


def family(agent: Any) -> Optional[str]:
    """'codex' or 'claude' for an identity string (session-qualified forms such as 'claude:<session>'
    included); None when it names neither (same rule as kb.family)."""
    if not isinstance(agent, str):
        return None
    a = agent.strip().lower()
    for fam in AGENTS:
        if a == fam or (a.startswith(fam) and a[len(fam):len(fam) + 1] in (':', '-', '_', ' ', '/', '@')):
            return fam
    return None


def author_family(lock: dict) -> Optional[str]:
    return family(lock.get('author_agent'))


def other_family(fam: Optional[str]) -> Optional[str]:
    if fam not in AGENTS:
        return None
    return 'claude' if fam == 'codex' else 'codex'


def floors(ctx: common.Ctx) -> dict:
    f = ctx.kb.constraints().get('threshold_floors')
    if not isinstance(f, dict):
        raise MissingInput('CONSTRAINTS.threshold_floors missing')
    return f


def floor_or_lock(ctx: common.Ctx, lock_value: Any, floor_key: str, what: str) -> Tuple[float, str]:
    """max(lock value, CONSTRAINTS floor); either may be absent but not both (MissingInput). A non-numeric
    lock value raises InvalidInput."""
    fl = floors(ctx).get(floor_key)
    vals = []
    basis = []
    if lock_value is not None and lock_value != FILL:
        if not is_num(lock_value):
            raise InvalidInput('%s must be a finite number, got %r' % (what, lock_value))
        vals.append(float(lock_value))
        basis.append('lock %s' % lock_value)
    if is_num(fl):
        vals.append(float(fl))
        basis.append('floor %s' % fl)
    if not vals:
        raise MissingInput('%s: neither the lock nor CONSTRAINTS.threshold_floors.%s gives a value' % (what, floor_key))
    return max(vals), 'max(%s)' % ', '.join(basis)


def lock_arms(lock: dict) -> List[dict]:
    return [a for a in (lock.get('arms') or []) if isinstance(a, dict) and isinstance(a.get('id'), str)]


def arm_by_id(lock: dict) -> Dict[str, dict]:
    return {a['id']: a for a in lock_arms(lock)}


def candidate_arm(lock: dict) -> str:
    c = dig(lock, 'primary.candidate_arm')
    if not s(c):
        raise InvalidInput('lock.primary.candidate_arm is not set')
    if c not in arm_by_id(lock):
        raise InvalidInput('lock.primary.candidate_arm %r is not a declared arm' % c)
    return c


def str_list(v: Any) -> List[str]:
    return [x for x in v if isinstance(x, str) and x and x != FILL] if isinstance(v, list) else []


def zero_channels(arm: dict) -> List[str]:
    return str_list(arm.get('zero_channels'))


def lock_worlds(lock: dict, role: str) -> List[str]:
    W = lock.get('worlds') if isinstance(lock.get('worlds'), dict) else {}
    if role == 'official':
        off = W.get('official') if isinstance(W.get('official'), dict) else {}
        return [str(w) for w in (off.get('ids') or []) if isinstance(w, (str, int))]
    return [str(w) for w in (W.get(role) or []) if isinstance(w, (str, int))]


def finish(cid: str, problems: List[str], missing: List[str], ok: str, data: Optional[dict] = None,
           evidence: Optional[List[str]] = None) -> Result:
    if problems:
        return Result(cid, FAIL, '; '.join(problems[:10] + (['MISSING: ' + '; '.join(missing[:3])] if missing else []))[:4000],
                      evidence=(evidence or problems)[:60], data=data or {})
    if missing:
        return Result(cid, MISSING, '; '.join(missing[:10])[:4000], evidence=missing[:60], data=data or {})
    return Result(cid, PASS, ok, evidence=evidence or [], data=data or {})


def tier_of(ctx: common.Ctx) -> int:
    if isinstance(ctx.tier, int):
        return ctx.tier
    try:
        return common.compute_tier(ctx)[0]
    except (MissingInput, InvalidInput, common.CheckerError):
        return 2  # an undeterminable tier is treated as the strictest


# --------------------------------------------------------------------------------------
# Raw artifact readers
# --------------------------------------------------------------------------------------

def receipts(ctx: common.Ctx) -> Dict[Tuple[str, str], Tuple[str, dict]]:
    """{(world, arm): (delivery-relative path, receipt)} for every results/receipts/<world>/<arm>.json.
    A receipt elsewhere under results/receipts/, a non-object, or a world/arm field that disagrees with its
    path raises InvalidInput."""
    key = 'wp4_receipts'
    if key in ctx._cache:
        return ctx._cache[key]
    out: Dict[Tuple[str, str], Tuple[str, dict]] = {}
    for rel in ctx.delivery_files():
        if not rel.startswith(RECEIPTS_PREFIX):
            continue
        rest = rel[len(RECEIPTS_PREFIX):]
        if rest.rsplit('/', 1)[-1] == '.gitkeep':
            continue
        parts = rest.split('/')
        if len(parts) != 2 or not parts[1].endswith('.json'):
            raise InvalidInput('%s is not results/receipts/<world>/<arm>.json' % rel)
        world, arm = parts[0], parts[1][:-len('.json')]
        r = ctx.read_json(rel)
        if not isinstance(r, dict):
            raise InvalidInput('%s must be a JSON object' % rel)
        for f, want in (('world', world), ('arm', arm)):
            if r.get(f) != want:
                raise InvalidInput('%s has %s %r, its path says %r' % (rel, f, r.get(f), want))
        out[(world, arm)] = (rel, r)
    ctx._cache[key] = out
    return out


def fixture_rows(ctx: common.Ctx) -> List[Tuple[str, dict]]:
    """[(label:line, row)] of results/admission/FIXTURES.jsonl (gz and RESULT_BUNDLE members allowed)."""
    key = 'wp4_fixtures'
    if key in ctx._cache:
        return ctx._cache[key]
    if not ctx.row_sources(FIXTURES_REL):
        raise MissingInput('%s absent (run admit.py --technical through MFC launch)' % FIXTURES_REL)
    out = [('%s:%d' % (label, ln), r) for label, ln, r in ctx.iter_rows_with_source(FIXTURES_REL)]
    ctx._cache[key] = out
    return out


def split_fixtures(rows: List[Tuple[str, dict]]) -> Tuple[List[Tuple[str, dict]], List[Tuple[str, dict]], List[str]]:
    """(directional rows, gate rows, malformed descriptions)."""
    dirs, gates, bad = [], [], []
    for label, r in rows:
        has_p, has_g = 'port' in r, 'gate_id' in r
        if has_p and not has_g:
            dirs.append((label, r))
        elif has_g and not has_p:
            gates.append((label, r))
        else:
            bad.append('%s is neither a directional row {port, sign, ...} nor a gate row {gate_id, kind, ...}' % label)
    return dirs, gates, bad


def read_admission(ctx: common.Ctx) -> dict:
    if not ctx.path(ADMISSION_REL).is_file():
        raise MissingInput('%s absent (admit.py --technical writes it)' % ADMISSION_REL)
    a = ctx.read_json(ADMISSION_REL)
    if not isinstance(a, dict):
        raise InvalidInput('%s must be a JSON object' % ADMISSION_REL)
    return a


def read_authorization(ctx: common.Ctx) -> dict:
    if not ctx.path(AUTHORIZATION_REL).is_file():
        raise MissingInput('%s absent: record the owner authorization (templates/AUTHORIZATION.template.json)'
                           % AUTHORIZATION_REL)
    a = ctx.read_json(AUTHORIZATION_REL)
    if not isinstance(a, dict):
        raise InvalidInput('%s must be a JSON object' % AUTHORIZATION_REL)
    return a


def resolve_delivery_or_repo(ctx: common.Ctx, entry: str) -> Tuple[Optional[str], Optional[str]]:
    """(delivery-relative path or None, repo-relative path) of an existing file named delivery- or repo-relative."""
    rel = common.norm_rel(entry)
    if ctx.path(rel).is_file():
        return rel, ctx.repo_rel_of(rel)
    if ctx.repo_path(rel).is_file():
        d = ctx.delivery_rel
        if d and rel.startswith(d + '/'):
            return rel[len(d) + 1:], rel
        return None, rel
    return None, None


# --------------------------------------------------------------------------------------
# Git helpers (strict ordering)
# --------------------------------------------------------------------------------------

def log_index(ctx: common.Ctx) -> Dict[str, int]:
    key = 'wp4_log_index'
    if key not in ctx._cache:
        ctx._cache[key] = {c.sha: i for i, c in enumerate(ctx.git.log())}
    return ctx._cache[key]


def strictly_before(ctx: common.Ctx, a: Optional[common.Commit], b: Optional[common.Commit]) -> bool:
    """a is a proper ancestor of b (the same commit does not count)."""
    if a is None or b is None or a.sha == b.sha:
        return False
    return ctx.git.is_ancestor(a.sha, b.sha)


def first_commit_adding_under(ctx: common.Ctx, prefixes: Sequence[str]) -> Optional[common.Commit]:
    """Earliest commit in which a non-empty file under one of the repo-relative prefixes exists."""
    prefixes = [p for p in prefixes if p]
    if not prefixes:
        return None
    for c in ctx.git.log(list(prefixes)):
        for p in c.paths:
            if not any(common.glob_match(p, pre) for pre in prefixes):
                continue
            data = ctx.git.file_at(c.sha, p)
            if data is None or (p.rsplit('/', 1)[-1] == '.gitkeep' and len(data) == 0):
                continue
            return c
    return None


def output_prefixes(ctx: common.Ctx, role: str) -> List[str]:
    """Delivery-relative prefixes holding outputs of a role: results/<role> and the receipts of its worlds."""
    out = ['results/%s' % role]
    for w in lock_worlds(ctx.lock, role):
        out.append('results/receipts/%s' % w)
    return out


def worktree_outputs(ctx: common.Ctx, role: str) -> List[str]:
    pres = output_prefixes(ctx, role)
    out = []
    for f in ctx.delivery_files():
        if any(common.glob_match(f, p) for p in pres):
            if f.rsplit('/', 1)[-1] == '.gitkeep' and ctx.path(f).stat().st_size == 0:
                continue
            out.append(f)
    return out


def first_output_commit(ctx: common.Ctx, role: str) -> Optional[common.Commit]:
    return first_commit_adding_under(ctx, [ctx.repo_rel_of(p) for p in output_prefixes(ctx, role)])


def committed_before_first_output(ctx: common.Ctx, rel: str, role: str, what: str,
                                  pred: Optional[Callable[[bytes], bool]] = None) -> Tuple[List[str], List[str], str]:
    """(problems, missing, note): the first commit where delivery-relative `rel` exists (and satisfies pred)
    strictly precedes the first commit adding an output of `role`. With outputs present only in the worktree
    the artifact must already be committed."""
    rrel = ctx.repo_rel_of(rel)
    c_art = ctx.git.first_commit_where(rrel, pred if pred is not None else (lambda b: True))
    c_out = first_output_commit(ctx, role)
    wt = worktree_outputs(ctx, role)
    if c_out is None and not wt:
        if c_art is None:
            return [], [], '%s not committed yet; no %s output exists' % (what, role)
        return [], [], '%s committed at %s; no %s output yet' % (what, c_art.sha[:12], role)
    if c_art is None:
        return ['%s is not committed but %s outputs exist (%s)' % (what, role, (c_out.sha[:12] if c_out else
                                                                              ', '.join(wt[:3])))], [], ''
    if c_out is not None and not strictly_before(ctx, c_art, c_out):
        return ['%s commit %s is not a proper ancestor of the first %s output commit %s'
                % (what, c_art.sha[:12], role, c_out.sha[:12])], [], ''
    return [], [], '%s committed at %s before the first %s output%s' % (
        what, c_art.sha[:12], role, (' %s' % c_out.sha[:12]) if c_out else ' (outputs not committed yet)')


# --------------------------------------------------------------------------------------
# ADM-01
# --------------------------------------------------------------------------------------

def _logical_row_path(ctx: common.Ctx, rel: Any) -> Optional[str]:
    if not isinstance(rel, str) or not rel.strip():
        return None
    r = common.norm_rel(rel)
    return r[:-3] if r.endswith('.jsonl.gz') else r


@check('ADM-01')
def check_adm_01(ctx: common.Ctx) -> Result:
    lock = ctx.lock
    adm = read_admission(ctx)
    problems: List[str] = []
    missing: List[str] = []
    notes: List[str] = []
    if adm.get('schema') != SCHEMA_ADMISSION:
        problems.append('schema is %r, not %s' % (adm.get('schema'), SCHEMA_ADMISSION))
    for k in ADMISSION_REQUIRED:
        if k not in adm:
            problems.append('field %s missing' % k)
    fills = fill_paths(adm)
    if fills:
        problems.append('__FILL__ values at %s' % ', '.join(fills[:6]))
    if common.normalize_utc(adm.get('created_utc')) is None:
        problems.append('created_utc %r is not a UTC timestamp' % (adm.get('created_utc'),))
    # both lock shas
    el, sl = ctx.endpoint_lock_sha256, ctx.source_lock_sha256
    if sl is None:
        missing.append('SOURCE_LOCK.json absent (run MFC lock build before admission)')
    if strip_sha(adm.get('endpoint_lock_sha256')) != el:
        problems.append('endpoint_lock_sha256 %s is not the current ENDPOINT_LOCK %s'
                        % (str(adm.get('endpoint_lock_sha256'))[:16], (el or '')[:16]))
    if sl is not None and strip_sha(adm.get('source_lock_sha256')) != sl:
        problems.append('source_lock_sha256 %s is not the current SOURCE_LOCK %s (rerun admission under the '
                        'current closure)' % (str(adm.get('source_lock_sha256'))[:16], sl[:16]))
    # producer admit.py, endpoint-locked
    prod = adm.get('producer') if isinstance(adm.get('producer'), dict) else {}
    admit = dig(lock, 'files.admit')
    efs = lock.get('endpoint_files_sha256') if isinstance(lock.get('endpoint_files_sha256'), dict) else {}
    script = prod.get('script')
    if not s(script):
        problems.append('producer.script is not set')
    elif not s(admit) or common.norm_rel(script) != common.norm_rel(admit):
        problems.append('producer.script %r is not lock.files.admit %r' % (script, admit))
    else:
        sk = common.norm_rel(script)
        locked = None
        for k, v in efs.items():
            if isinstance(k, str) and common.norm_rel(k) == sk:
                locked = strip_sha(v)
        if locked is None:
            problems.append('producer %s is not listed in ENDPOINT_LOCK.endpoint_files_sha256' % sk)
        drel, _rr = resolve_delivery_or_repo(ctx, sk)
        if drel is None:
            missing.append('producer %s does not exist in the delivery' % sk)
        else:
            cur = common.sha256_file(ctx.path(drel))
            if strip_sha(prod.get('sha256')) != cur:
                problems.append('producer.sha256 %s differs from %s on disk %s' % (str(prod.get('sha256'))[:16], sk,
                                                                                   cur[:16]))
            if locked is not None and locked != cur:
                problems.append('%s on disk %s differs from its endpoint-locked sha %s' % (sk, cur[:16], locked[:16]))
    # worlds: technical only
    tech = set(lock_worlds(lock, 'technical'))
    others = set(lock_worlds(lock, 'dev')) | set(lock_worlds(lock, 'rehearsal')) | set(lock_worlds(lock, 'official'))
    worlds = adm.get('worlds')
    if not isinstance(worlds, list) or not worlds or not all(isinstance(w, str) and w for w in worlds):
        problems.append('worlds must be a non-empty list of technical world ids')
        worlds = []
    for w in worlds:
        if w not in tech:
            problems.append('world %s is not a lock.worlds.technical world' % w)
        if w in others:
            problems.append('world %s is a dev/rehearsal/official world (admission is technical only)' % w)
    # receipts: the raw receipts take precedence over the ADMISSION summary
    rc = receipts(ctx)
    glob = adm.get('receipts_glob')
    if not s(glob):
        problems.append('receipts_glob is not set')
    else:
        matched = [(k, v) for k, v in rc.items() if common.glob_match(v[0], glob)]
        if not matched:
            missing.append('receipts_glob %s matches no receipt' % glob)
        for (w, a), (rel, r) in sorted(matched):
            if w not in worlds:
                problems.append('receipt %s (matched by receipts_glob) is for world %s, not an ADMISSION world' % (rel, w))
            if r.get('world_role') != 'technical':
                problems.append('receipt %s has world_role %r, not technical' % (rel, r.get('world_role')))
            if strip_sha(r.get('endpoint_lock_sha256')) != strip_sha(adm.get('endpoint_lock_sha256')):
                problems.append('receipt %s endpoint_lock_sha256 disagrees with ADMISSION' % rel)
        for (w, a), (rel, _r) in sorted(rc.items()):
            if w in worlds and not common.glob_match(rel, glob):
                problems.append('receipt %s of admission world %s is not covered by receipts_glob %s' % (rel, w, glob))
        for w in worlds:
            if not any(k[0] == w for k in rc):
                missing.append('no receipt for admission world %s' % w)
    # row files
    tb = _logical_row_path(ctx, adm.get('testbed_outputs'))
    if tb is None:
        problems.append('testbed_outputs is not set')
    elif tb != ADMISSION_OUTPUTS_REL:
        problems.append('testbed_outputs %s is not %s (the only file the scorer reads)' % (tb, ADMISSION_OUTPUTS_REL))
    elif not ctx.row_sources(ADMISSION_OUTPUTS_REL):
        missing.append('%s has no rows (run the cycle-0 testbed)' % ADMISSION_OUTPUTS_REL)
    fx = _logical_row_path(ctx, adm.get('fixtures'))
    needs_fx = bool(lock.get('ports')) or bool(lock.get('gates'))
    if fx is None:
        if needs_fx:
            problems.append('fixtures is not set although the lock declares ports or gates')
    elif fx != FIXTURES_REL:
        problems.append('fixtures %s is not %s' % (fx, FIXTURES_REL))
    elif needs_fx and not ctx.row_sources(FIXTURES_REL):
        missing.append('%s absent' % FIXTURES_REL)
    # admission rows are technical only
    bad_rows = []
    n_rows = 0
    if ctx.row_sources(ADMISSION_OUTPUTS_REL):
        for label, ln, r in ctx.iter_rows_with_source(ADMISSION_OUTPUTS_REL):
            n_rows += 1
            if r.get('world') not in tech or r.get('world_role') not in (None, 'technical'):
                bad_rows.append('%s:%d world %r role %r' % (label, ln, r.get('world'), r.get('world_role')))
    if bad_rows:
        problems.append('%d admission rows are not on technical worlds (e.g. %s)' % (len(bad_rows), bad_rows[0]))
    pas = adm.get('parent_admission_sha')
    if pas is not None and strip_sha(pas) is None:
        problems.append('parent_admission_sha %r is neither null nor a sha256' % (pas,))
    # committed before the first dev output (the current content must be committed once dev outputs exist)
    if not problems:
        cur_bytes = ctx.path(ADMISSION_REL).read_bytes()
        p2, m2, note = committed_before_first_output(ctx, ADMISSION_REL, 'dev', 'ADMISSION.json')
        problems.extend(p2)
        missing.extend(m2)
        if note:
            notes.append(note)
        if not p2 and (first_output_commit(ctx, 'dev') is not None or worktree_outputs(ctx, 'dev')):
            if ctx.git.first_commit_where(ctx.repo_rel_of(ADMISSION_REL), lambda b: b == cur_bytes) is None:
                problems.append('the current ADMISSION.json content is not committed although dev outputs exist')
    return finish('ADM-01', problems, missing,
                  'ADMISSION.json valid: both lock shas current, producer %s endpoint-locked, technical worlds %s, '
                  '%d admission rows; %s' % (script, ','.join(worlds), n_rows, '; '.join(notes)))


# --------------------------------------------------------------------------------------
# ADM-TB-CEIL / ADM-TB-FLOOR (cycle-0 testbed, scored by the program scorer)
# --------------------------------------------------------------------------------------

def grid_coverage(ctx: common.Ctx, kind: str) -> Tuple[List[str], List[dict]]:
    """(problems, lock grid) for kind 'ceiling' or 'floor': every CONSTRAINTS grid entry must appear in the lock
    grid with the same id and the same value for every CONSTRAINTS key."""
    ckey = 'testbed_%s_grid' % kind
    grid = ctx.kb.constraints().get(ckey)
    if not isinstance(grid, list):
        raise MissingInput('CONSTRAINTS.%s missing' % ckey)
    lgrid = dig(ctx.lock, 'testbed.%s_grid' % kind)
    lgrid = [e for e in lgrid if isinstance(e, dict)] if isinstance(lgrid, list) else []
    by_id: Dict[str, dict] = {}
    problems = []
    for e in lgrid:
        if not s(e.get('id')):
            problems.append('lock.testbed.%s_grid entry without an id' % kind)
            continue
        if e['id'] in by_id:
            problems.append('lock.testbed.%s_grid lists %s twice' % (kind, e['id']))
        by_id[e['id']] = e
    for ce in grid:
        if not isinstance(ce, dict) or not s(ce.get('id')):
            raise InvalidInput('CONSTRAINTS.%s has an entry without an id' % ckey)
        le = by_id.get(ce['id'])
        if le is None:
            problems.append('lock.testbed.%s_grid omits CONSTRAINTS config %s' % (kind, ce['id']))
            continue
        for k, v in ce.items():
            if common.canonical_json(le.get(k)) != common.canonical_json(v):
                problems.append('lock config %s has %s=%r, CONSTRAINTS says %r' % (ce['id'], k, le.get(k), v))
    return problems, lgrid


def _tb_summary(e: dict) -> str:
    return '%s %s (verdict %s/%s, cand %s, d %s)' % (e.get('arm'), e.get('status'), e.get('verdict'),
                                                     e.get('shadow_verdict'), _f(e.get('mean_cand')), _f(e.get('mean_d')))


def _f(v: Any) -> str:
    return ('%.4f' % v) if is_num(v) else str(v)


@check('ADM-TB-CEIL')
def check_adm_tb_ceil(ctx: common.Ctx) -> Result:
    scorer = lazy('scorer')
    cov, lgrid = grid_coverage(ctx, 'ceiling')
    tb = scorer.testbed(ctx)
    configs = tb.get('configs') or []
    ceil = [e for e in configs if e.get('kind') == 'ceiling']
    declared = [e for e in ceil if e.get('declared')]
    undeclared = [e for e in ceil if not e.get('declared')]
    problems = list(cov)
    missing = []
    if not lgrid:
        problems.append('lock.testbed.ceiling_grid is empty')
    for e in undeclared:
        problems.append('rows for undeclared ceiling arm %s (not in lock.testbed.ceiling_grid)' % e.get('arm'))
    for e in declared:
        if e.get('status') == 'MISSING':
            missing.append('ceiling %s not scored: %s' % (e.get('arm'), e.get('reason') or '; '.join(e.get('reasons') or [])))
    passing = [e for e in declared if e.get('status') == 'PASS']
    data = {'testbed_result': tb.get('result'), 'ceilings': [_tb_summary(e) for e in ceil]}
    if not passing and declared and all(e.get('status') == 'FAIL' for e in declared):
        problems.append('CANNOT_TEST_CLAIM_AS_SPECIFIED: every ceiling fails the locked rule (%s)'
                        % '; '.join(_tb_summary(e) for e in declared))
    if problems:
        return finish('ADM-TB-CEIL', problems, missing, '', data=data)
    if missing:
        return finish('ADM-TB-CEIL', [], missing + (['no ceiling passes yet'] if not passing else []), '', data=data)
    if not passing:
        return Result('ADM-TB-CEIL', MISSING, 'no ceiling config passes (%s)' % tb.get('reason'), data=data)
    return Result('ADM-TB-CEIL', PASS, 'ceiling %s passes the locked rule; grid covers CONSTRAINTS (%d configs)'
                  % (','.join(e['arm'] for e in passing), len(declared)), data=data)


@check('ADM-TB-FLOOR')
def check_adm_tb_floor(ctx: common.Ctx) -> Result:
    scorer = lazy('scorer')
    cov, lgrid = grid_coverage(ctx, 'floor')
    tb = scorer.testbed(ctx)
    floors_ = [e for e in (tb.get('configs') or []) if e.get('kind') == 'floor']
    problems = list(cov)
    missing = []
    if not lgrid:
        problems.append('lock.testbed.floor_grid is empty')
    for e in floors_:
        if e.get('status') == 'PASS':
            problems.append('CANNOT_TEST_CLAIM_AS_SPECIFIED: floor %s passes the locked rule (%s)'
                            % (e.get('arm'), _tb_summary(e)))
        elif e.get('status') == 'MISSING':
            missing.append('floor %s not scored: %s' % (e.get('arm'), e.get('reason') or '; '.join(e.get('reasons') or [])))
        if not e.get('declared'):
            problems.append('rows for undeclared floor arm %s (it still counts as a floor)' % e.get('arm'))
    best = tb.get('best_floor') or {}
    data = {'testbed_result': tb.get('result'), 'floors': [_tb_summary(e) for e in floors_], 'best_floor': best}
    if best.get('status') == 'PASS' and not any('passes the locked rule' in p for p in problems):
        problems.append('the best floor %s passes the locked rule' % best.get('arm'))
    return finish('ADM-TB-FLOOR', problems, missing, 'every floor config fails the locked rule (best floor %s, d %s); '
                  'grid covers CONSTRAINTS (%d configs)' % (best.get('arm'), _f(best.get('mean_d')), len(floors_)),
                  data=data)


# --------------------------------------------------------------------------------------
# ADM-CTRL-ZERO / ISO / DOSE (raw receipts)
# --------------------------------------------------------------------------------------

def control_arms(lock: dict) -> List[dict]:
    """Arms with non-empty zero_channels (the ADM-CTRL-ZERO scope), candidate excluded."""
    cand = dig(lock, 'primary.candidate_arm')
    return [a for a in lock_arms(lock) if zero_channels(a) and a['id'] != cand]


def iso_arms(lock: dict) -> List[dict]:
    """ADM-CTRL-ISO scope: arms with zero_channels plus the declared causal_control_arm (by role, never the
    candidate): the causal control must cut only its declared path."""
    cand = dig(lock, 'primary.candidate_arm')
    out = control_arms(lock)
    cc = lock.get('causal_control_arm')
    by = arm_by_id(lock)
    if isinstance(cc, str) and cc in by and cc != cand and all(a['id'] != cc for a in out):
        out.append(by[cc])
    return out


def ctrl_worlds(ctx: common.Ctx, arm_ids: Sequence[str]) -> List[str]:
    """Admission worlds (receipts required there: common.admission_worlds) plus every world holding a receipt
    of one of the arms or of the candidate."""
    rc = receipts(ctx)
    cand = candidate_arm(ctx.lock)
    ws = set(common.admission_worlds(ctx)[0])
    for (w, a) in rc:
        if a in arm_ids or a == cand:
            ws.add(w)
    return sorted(ws)


def _digests(r: dict, phase: str) -> Optional[dict]:
    cd = r.get('channel_digests')
    d = cd.get(phase) if isinstance(cd, dict) else None
    return d if isinstance(d, dict) else None


def native_closure(ctx: common.Ctx) -> Tuple[Dict[str, Optional[str]], str]:
    """({repo-relative path: sha256 or None}, basis) of the native closure: SOURCE_LOCK.native_closure_sha256
    when present, else a fresh lockcheck closure build."""
    key = 'wp4_native_closure'
    if key in ctx._cache:
        return ctx._cache[key]
    nat = None
    basis = ''
    if ctx.path('SOURCE_LOCK.json').is_file():
        nat = ctx.source_lock.get('native_closure_sha256')
        basis = 'SOURCE_LOCK.native_closure_sha256'
    if not isinstance(nat, dict):
        lc = lazy('lockcheck')
        b = lc.build_closure(ctx)
        nat = b.get('native_closure_sha256')
        basis = 'fresh native closure build'
        if not isinstance(nat, dict):
            raise MissingInput('native closure unavailable')
    out = {str(k): strip_sha(v) for k, v in nat.items()}
    ctx._cache[key] = (out, basis)
    return out, basis


def _find_qualname(tree: ast.AST, qual: str) -> bool:
    parts = qual.split('.')
    scope: List[ast.AST] = list(getattr(tree, 'body', []))
    for i, part in enumerate(parts):
        last = i == len(parts) - 1
        nxt = None
        for node in scope:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == part:
                nxt = node
                break
            if not last and isinstance(node, ast.ClassDef) and node.name == part:
                nxt = node
                break
        if nxt is None:
            return False
        if last:
            return isinstance(nxt, (ast.FunctionDef, ast.AsyncFunctionDef))
        scope = list(nxt.body)
    return False


def write_site_problem(ctx: common.Ctx, site: Any, label: str) -> Optional[str]:
    """None when write_counter_site '<delivery-relative path>:<function qualname>' names a function defined in a
    native-closure file whose content still matches the closure sha; else the problem."""
    key = ('wp4_site', repr(site))
    if key in ctx._cache:
        return ctx._cache[key]
    prob = None
    if not s(site) or ':' not in site:
        prob = '%s write_counter_site %r is not <path>:<function>' % (label, site)
    else:
        path, qual = site.rsplit(':', 1)
        qual = qual.strip()
        try:
            rel = common.norm_rel(path.strip())
        except InvalidInput as e:
            rel, prob = None, '%s write_counter_site: %s' % (label, e)
        if rel is not None:
            nat, basis = native_closure(ctx)
            rr = ctx.repo_rel_of(rel)
            if rr not in nat and rel not in nat:
                prob = '%s write_counter_site file %s is not in the native closure (%s)' % (label, rel, basis)
            else:
                want = nat.get(rr, nat.get(rel))
                fp = ctx.repo_path(rr) if rr in nat else ctx.repo_path(rel)
                if not fp.is_file():
                    prob = '%s write_counter_site file %s does not exist' % (label, rel)
                else:
                    data = fp.read_bytes()
                    if want is not None and common.sha256_bytes(data) != want:
                        prob = '%s write_counter_site file %s changed since the closure was recorded' % (label, rel)
                    else:
                        try:
                            tree = ast.parse(data, filename=rel)
                        except (SyntaxError, ValueError) as e:
                            tree, prob = None, '%s write_counter_site file %s does not parse: %s' % (label, rel, e)
                        if tree is not None and not (qual and _find_qualname(tree, qual)):
                            prob = '%s write_counter_site function %s is not defined in %s' % (label, qual, rel)
    ctx._cache[key] = prob
    return prob


@check('ADM-CTRL-ZERO')
def check_adm_ctrl_zero(ctx: common.Ctx) -> Result:
    lock = ctx.lock
    arms = control_arms(lock)
    if not arms:
        return Result('ADM-CTRL-ZERO', NOT_APPLICABLE, 'no arm declares zero_channels')
    cand = candidate_arm(lock)
    rc = receipts(ctx)
    problems: List[str] = []
    missing: List[str] = []
    checked = 0
    worlds = ctrl_worlds(ctx, [a['id'] for a in arms])
    tech_list, tech_basis = common.admission_worlds(ctx)
    tech = set(tech_list)
    if not tech:
        missing.append('no admission (technical) worlds (%s): no technical receipts to read' % tech_basis)
    for a in arms:
        Z = zero_channels(a)
        for w in worlds:
            ra, rw = rc.get((w, a['id'])), rc.get((w, cand))
            if ra is None and rw is None and w not in tech:
                continue
            if ra is None:
                missing.append('receipt results/receipts/%s/%s.json missing' % (w, a['id']))
            if rw is None:
                missing.append('receipt results/receipts/%s/%s.json (candidate W) missing' % (w, cand))
            if ra is None or rw is None:
                continue
            checked += 1
            (rel_a, A), (rel_w, Wr) = ra, rw
            wa, ww = A.get('writes_by_channel'), Wr.get('writes_by_channel')
            ba, pa = _digests(A, 'birth'), _digests(A, 'post_first')
            bw, pw = _digests(Wr, 'birth'), _digests(Wr, 'post_first')
            if not isinstance(wa, dict) or not isinstance(ww, dict):
                problems.append('%s or %s lacks writes_by_channel' % (rel_a, rel_w))
                continue
            if ba is None or pa is None or bw is None or pw is None:
                problems.append('%s or %s lacks channel_digests.birth/post_first' % (rel_a, rel_w))
                continue
            for c in Z:
                v = wa.get(c)
                if not is_int(v):
                    problems.append('%s writes_by_channel[%s] is %r (no write counter on the zeroed channel)' % (rel_a, c, v))
                elif v != 0:
                    problems.append('%s writes_by_channel[%s] = %d on a zeroed channel' % (rel_a, c, v))
                if not s(pa.get(c)) or not s(ba.get(c)):
                    problems.append('%s lacks the %s digest at birth/post_first' % (rel_a, c))
                elif pa.get(c) != ba.get(c):
                    problems.append('%s post_first[%s] differs from birth: the zeroed channel changed' % (rel_a, c))
                vw = ww.get(c)
                if not is_int(vw) or vw <= 0:
                    problems.append('%s writes_by_channel[%s] = %r: the candidate never writes the channel the '
                                    'control zeroes' % (rel_w, c, vw))
                if not s(pw.get(c)) or not s(bw.get(c)):
                    problems.append('%s lacks the %s digest at birth/post_first' % (rel_w, c))
                elif pw.get(c) == bw.get(c):
                    problems.append('%s post_first[%s] equals birth: the treatment is not delivered in W' % (rel_w, c))
            for rel, R in ((rel_a, A), (rel_w, Wr)):
                p = write_site_problem(ctx, R.get('write_counter_site'), rel)
                if p:
                    problems.append(p)
    problems = sorted(set(problems), key=problems.index)
    if problems:
        problems.insert(0, 'TREATMENT_NOT_DELIVERED')
    return finish('ADM-CTRL-ZERO', problems, sorted(set(missing), key=missing.index),
                  '%d (world, control arm) receipt pairs: zeroed channels %s unwritten and unchanged at post_first; the '
                  'candidate writes and changes them; write sites are native-closure functions'
                  % (checked, {a['id']: zero_channels(a) for a in arms}))


@check('ADM-CTRL-ISO')
def check_adm_ctrl_iso(ctx: common.Ctx) -> Result:
    lock = ctx.lock
    arms = iso_arms(lock)
    if not arms:
        return Result('ADM-CTRL-ISO', NOT_APPLICABLE, 'no arm declares zero_channels and no causal_control_arm is declared')
    cand = candidate_arm(lock)
    rc = receipts(ctx)
    tech_list, tech_basis = common.admission_worlds(ctx)
    tech = set(tech_list)
    problems: List[str] = []
    missing: List[str] = []
    if not tech:
        missing.append('no admission (technical) worlds (%s): no technical receipts to read' % tech_basis)
    checked = 0
    for a in arms:
        exempt = set(zero_channels(a)) | set(str_list(a.get('declared_changed')))
        for w in ctrl_worlds(ctx, [a['id']]):
            ra, rw = rc.get((w, a['id'])), rc.get((w, cand))
            if ra is None and rw is None and w not in tech:
                continue
            if ra is None or rw is None:
                missing.append('receipts for %s and %s in world %s are required' % (a['id'], cand, w))
                continue
            checked += 1
            pa, pw = _digests(ra[1], 'post_first'), _digests(rw[1], 'post_first')
            if pa is None or pw is None:
                problems.append('%s or %s lacks channel_digests.post_first' % (ra[0], rw[0]))
                continue
            for c in sorted(set(pa) | set(pw)):
                if c in exempt:
                    continue
                if c not in pa or c not in pw:
                    problems.append('channel %s present in only one of %s, %s at post_first' % (c, ra[0], rw[0]))
                elif pa[c] != pw[c]:
                    problems.append('%s differs from the candidate on channel %s at post_first, which is neither '
                                    'zeroed nor declared_changed (confound)' % (ra[0], c))
    return finish('ADM-CTRL-ISO', problems, sorted(set(missing), key=missing.index),
                  '%d (world, arm) pairs equal the candidate at post_first on every channel outside zero_channels and '
                  'declared_changed (arms %s)' % (checked, ','.join(a['id'] for a in arms)))


@check('ADM-CTRL-DOSE')
def check_adm_ctrl_dose(ctx: common.Ctx) -> Result:
    lock = ctx.lock
    scope = [a for a in lock_arms(lock) if zero_channels(a) or is_num(a.get('min_realized_dose'))]
    if not scope:
        return Result('ADM-CTRL-DOSE', NOT_APPLICABLE, 'no arm declares zero_channels or min_realized_dose')
    fl = floors(ctx).get('min_realized_dose')
    if not is_num(fl):
        raise MissingInput('CONSTRAINTS.threshold_floors.min_realized_dose missing')
    rc = receipts(ctx)
    tech_list, tech_basis = common.admission_worlds(ctx)
    tech = set(tech_list)
    problems: List[str] = []
    missing: List[str] = []
    if not tech:
        missing.append('no admission (technical) worlds (%s): no technical receipts to read' % tech_basis)
    checked = 0
    for a in scope:
        mrd = a.get('min_realized_dose')
        if mrd not in (None, FILL) and not is_num(mrd):
            problems.append('arm %s min_realized_dose %r is not a number' % (a['id'], mrd))
            continue
        thr = max(float(fl), float(mrd)) if is_num(mrd) else float(fl)
        ws = sorted(tech | set(w for (w, arm) in rc if arm == a['id']))
        for w in ws:
            r = rc.get((w, a['id']))
            if r is None:
                missing.append('receipt results/receipts/%s/%s.json missing' % (w, a['id']))
                continue
            checked += 1
            rel, R = r
            dose = R.get('realized_dose')
            if not isinstance(dose, dict) or not dose:
                problems.append('%s has no realized_dose {metric: value}' % rel)
                continue
            for m, v in sorted(dose.items()):
                if not is_num(v):
                    problems.append('%s realized_dose.%s = %r is not a finite number' % (rel, m, v))
                elif v < thr:
                    problems.append('%s realized_dose.%s = %s < %s (max(min_realized_dose, floor))' % (rel, m, v, thr))
    if problems:
        problems.insert(0, 'TREATMENT_NOT_DELIVERED')
    return finish('ADM-CTRL-DOSE', problems, missing, '%d receipts: realized_dose >= max(min_realized_dose, floor %s)'
                  % (checked, fl))


# --------------------------------------------------------------------------------------
# ADM-DIR / ADM-FIX / ADM-GATE (FIXTURES.jsonl)
# --------------------------------------------------------------------------------------

@check('ADM-DIR')
def check_adm_dir(ctx: common.Ctx) -> Result:
    ports = ctx.lock.get('ports')
    ports = [p for p in ports if isinstance(p, dict)] if isinstance(ports, list) else []
    if not ports:
        return Result('ADM-DIR', NOT_APPLICABLE, 'lock.ports is empty')
    dirs, _gates, bad = split_fixtures(fixture_rows(ctx))
    problems = list(bad)
    missing = []
    declared: Set[Tuple[str, str]] = set()
    for p in ports:
        pid = p.get('id')
        signs = p.get('signs')
        if not s(pid) or not isinstance(signs, list) or not signs:
            problems.append('lock port %r needs an id and a non-empty signs list' % (pid,))
            continue
        for sg in signs:
            declared.add((pid, str(sg)))
    n = 0
    for pid, sg in sorted(declared):
        rows = [(lb, r) for lb, r in dirs if r.get('port') == pid and str(r.get('sign')) == sg]
        if not rows:
            missing.append('no directional fixture for port %s sign %s' % (pid, sg))
            continue
        for lb, r in rows:
            n += 1
            d = r.get('target_margin_delta_vs_passive')
            if not is_num(d) or d <= 0:
                problems.append('%s port %s sign %s: target_margin_delta_vs_passive %r is not > 0' % (lb, pid, sg, d))
            if r.get('zero_coef_digest_equals_passive') is not True:
                problems.append('%s port %s sign %s: zero_coef_digest_equals_passive is %r, not true'
                                % (lb, pid, sg, r.get('zero_coef_digest_equals_passive')))
            z = r.get('zero_replay_applications')
            if not is_int(z) or z != 0:
                problems.append('%s port %s sign %s: zero_replay_applications %r != 0' % (lb, pid, sg, z))
    for lb, r in dirs:
        if (r.get('port'), str(r.get('sign'))) not in declared:
            problems.append('%s is a directional fixture for undeclared port/sign %r/%r' % (lb, r.get('port'), r.get('sign')))
    return finish('ADM-DIR', problems, missing, '%d directional fixtures for %d declared (port, sign) pairs: delta > 0, '
                  'zero-coefficient digest equals passive, no zero replays' % (n, len(declared)))


def flatten(obj: Any, prefix: str = '') -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if isinstance(obj, dict) and obj:
        for k, v in obj.items():
            out.update(flatten(v, '%s.%s' % (prefix, k) if prefix else str(k)))
    else:
        out[prefix] = obj
    return out


def _tolerance(tols: Dict[str, float], key: str) -> float:
    parts = key.split('.')
    for i in range(len(parts), 0, -1):
        k = '.'.join(parts[:i])
        if k in tols:
            return tols[k]
    return 0.0


def _values_equal(a: Any, b: Any, tol: float) -> bool:
    if is_num(a) and is_num(b):
        return abs(float(a) - float(b)) <= tol
    try:
        return common.canonical_json(a) == common.canonical_json(b)
    except InvalidInput:
        return False


def production_params(ctx: common.Ctx) -> Tuple[dict, str]:
    """(params.json object, delivery-relative path); its sha must equal SOURCE_LOCK.params_sha256."""
    pe = dig(ctx.lock, 'files.params')
    if not s(pe):
        raise InvalidInput('lock.files.params is not set')
    drel, rr = resolve_delivery_or_repo(ctx, pe)
    if rr is None:
        raise MissingInput('params file %s does not exist' % pe)
    if not ctx.path('SOURCE_LOCK.json').is_file():
        raise MissingInput('SOURCE_LOCK.json absent: the production params sha is unknown')
    want = strip_sha(ctx.source_lock.get('params_sha256'))
    fp = ctx.path(drel) if drel is not None else ctx.repo_path(rr)
    got = common.sha256_file(fp)
    if want is None:
        raise InvalidInput('SOURCE_LOCK.params_sha256 is not a sha256')
    if got != want:
        raise InvalidInput('%s on disk (%s) differs from SOURCE_LOCK.params_sha256 %s: rerun MFC lock build'
                           % (pe, got[:16], want[:16]))
    obj = common.load_json_file(fp)
    if not isinstance(obj, dict):
        raise InvalidInput('%s must be a JSON object' % pe)
    return obj, pe


@check('ADM-FIX')
def check_adm_fix(ctx: common.Ctx) -> Result:
    lock = ctx.lock
    if not lock.get('ports') and not lock.get('gates'):
        return Result('ADM-FIX', NOT_APPLICABLE, 'no ports or gates declared: no fixtures')
    rows = fixture_rows(ctx)
    params, prel = production_params(ctx)
    tols_raw = dig(lock, 'admission_thresholds.%s' % FIXTURE_TOLERANCE_KEY)
    tols: Dict[str, float] = {}
    problems: List[str] = []
    if tols_raw is not None:
        if not isinstance(tols_raw, dict):
            problems.append('admission_thresholds.%s must be {param: abs tolerance}' % FIXTURE_TOLERANCE_KEY)
        else:
            for k, v in tols_raw.items():
                if not is_num(v) or v < 0:
                    problems.append('tolerance for %s is %r (finite >= 0 required)' % (k, v))
                else:
                    tols[str(k)] = float(v)
    prod = flatten(params)
    if not rows:
        return Result('ADM-FIX', MISSING, '%s has no rows' % FIXTURES_REL)
    for lb, r in rows:
        fp = r.get('params')
        if not isinstance(fp, dict):
            problems.append('%s has no params object' % lb)
            continue
        fx = flatten(fp)
        for k, v in prod.items():
            if k not in fx:
                problems.append('%s lacks production param %s' % (lb, k))
            elif not _values_equal(fx[k], v, _tolerance(tols, k)):
                problems.append('%s param %s = %r, production %s has %r (tolerance %s)' % (lb, k, fx[k], prel, v,
                                                                                          _tolerance(tols, k)))
        for k in fx:
            if k not in prod:
                problems.append('%s param %s is not a production param' % (lb, k))
    return finish('ADM-FIX', problems, [], '%d fixture rows use the production params of %s (sha in SOURCE_LOCK)%s'
                  % (len(rows), prel, (' within tolerances %s' % tols) if tols else ' exactly'))


@check('ADM-GATE')
def check_adm_gate(ctx: common.Ctx) -> Result:
    gates = ctx.lock.get('gates')
    gates = [g for g in gates if isinstance(g, dict)] if isinstance(gates, list) else []
    if not gates:
        return Result('ADM-GATE', NOT_APPLICABLE, 'lock.gates is empty')
    _dirs, grows, bad = split_fixtures(fixture_rows(ctx))
    problems = list(bad)
    ids = []
    for g in gates:
        gid = g.get('id')
        if not s(gid):
            problems.append('lock gate %r has no id' % (gid,))
            continue
        if gid in ids:
            problems.append('lock gate %s declared twice' % gid)
        ids.append(gid)
    summary = []
    for gid in ids:
        rows = [(lb, r) for lb, r in grows if r.get('gate_id') == gid]
        neg = pos = 0
        for lb, r in rows:
            kind, res = r.get('kind'), r.get('result')
            if kind not in ('negative', 'positive'):
                problems.append('%s gate %s kind %r is not negative|positive' % (lb, gid, kind))
                continue
            if not isinstance(res, bool):
                problems.append('%s gate %s result %r is not a boolean' % (lb, gid, res))
                continue
            if kind == 'negative':
                if res:
                    problems.append('%s gate %s accepts its negative fixture (result true)' % (lb, gid))
                else:
                    neg += 1
            else:
                if not res:
                    problems.append('%s gate %s rejects its positive fixture (result false)' % (lb, gid))
                else:
                    pos += 1
        if not neg:
            problems.append('gate %s has no negative fixture with result false (a gate that cannot fail)' % gid)
        if not pos:
            problems.append('gate %s has no positive fixture with result true' % gid)
        summary.append('%s %d-/%d+' % (gid, neg, pos))
    for lb, r in grows:
        if r.get('gate_id') not in ids:
            problems.append('%s is a fixture for undeclared gate %r' % (lb, r.get('gate_id')))
    return finish('ADM-GATE', problems, [], 'every plumbing gate rejects a negative and accepts a positive fixture (%s)'
                  % ', '.join(summary))


# --------------------------------------------------------------------------------------
# ADM-REUSE and ADM-INPUT
# --------------------------------------------------------------------------------------

def _parent_rel(lock: dict) -> Optional[str]:
    p = dig(lock, 'lineage.parent')
    if not s(p):
        return None
    d = p.split('@', 1)[0].strip().strip('/')
    return common.norm_rel(d) if d else None


def reuse_status(ctx: common.Ctx) -> Tuple[Optional[bool], List[str], List[str]]:
    """(valid, problems, missing) of an ADMISSION.parent_admission_sha citation. valid is None when nothing is
    cited. Validity: tier 1; lineage.parent is a v1 delivery whose ADMISSION.json has the cited sha; native
    closure, task and split equal the parent's; the parent's admission passed (a consistent non-dry PASS roster
    gate report of the parent with ADM-INPUT PASS)."""
    adm = read_admission(ctx)
    cited = adm.get('parent_admission_sha')
    if cited is None:
        return None, [], []
    problems: List[str] = []
    missing: List[str] = []
    sha = strip_sha(cited)
    if sha is None:
        return False, ['parent_admission_sha %r is not a sha256' % (cited,)], []
    lock = ctx.lock
    prel = _parent_rel(lock)
    if prel is None:
        return False, ['parent_admission_sha is cited but lock.lineage.parent is null'], []
    pd = ctx.repo_path(prel)
    if not common.is_v1_delivery(pd):
        return False, ['lineage.parent %s is not a v1 delivery' % prel], []
    pa = pd / ADMISSION_REL
    if not pa.is_file():
        return False, [], ['parent %s has no ADMISSION.json' % prel]
    if common.sha256_file(pa) != sha:
        problems.append('parent ADMISSION.json sha %s is not the cited %s' % (common.sha256_file(pa)[:16], sha[:16]))
    plock = common.load_json_file(pd / 'ENDPOINT_LOCK.json')
    for k in ('task', 'split'):
        if common.canonical_json(plock.get(k)) != common.canonical_json(lock.get(k)):
            problems.append('lock.%s differs from the parent\'s' % k)
    psl = pd / 'SOURCE_LOCK.json'
    if not psl.is_file() or not ctx.path('SOURCE_LOCK.json').is_file():
        missing.append('SOURCE_LOCK.json of this delivery or of the parent is absent')
    else:
        a = ctx.source_lock.get('native_closure_sha')
        b = common.load_json_file(psl).get('native_closure_sha')
        if not s(a) or a != b:
            problems.append('native_closure_sha %s differs from the parent\'s %s' % (str(a)[:16], str(b)[:16]))
    pctx = common.Ctx(repo_root=ctx.repo_root, delivery=pd, git=ctx.git, kb=ctx.kb, rules=ctx.rules)
    passed = False
    for rep in common.list_gate_reports(pctx):
        if rep.get('stage') != 'roster' or rep.get('dry') or rep.get('verdict') != 'PASS':
            continue
        if any(isinstance(r, dict) and r.get('id') == 'ADM-INPUT' and r.get('status') == PASS
               for r in rep.get('results') or []):
            passed = True
    if not passed:
        problems.append('parent %s has no consistent non-dry PASS roster gate report with ADM-INPUT PASS' % prel)
    return (not problems and not missing), problems, missing


@check('ADM-REUSE')
def check_adm_reuse(ctx: common.Ctx) -> Result:
    tier = tier_of(ctx)
    adm = read_admission(ctx)
    if adm.get('parent_admission_sha') is None:
        return Result('ADM-REUSE', NOT_APPLICABLE, 'ADMISSION.parent_admission_sha is not cited')
    if tier != 1:
        return Result('ADM-REUSE', NOT_APPLICABLE, 'tier %d: ADM-REUSE is T1 only, so the citation is ignored and '
                      'ADM-INPUT and LOCK-12 are enforced' % tier)
    valid, problems, missing = reuse_status(ctx)
    return finish('ADM-REUSE', problems, missing, 'parent admission %s validly cited (closure, task and split equal; '
                  'parent ADM-INPUT passed)' % str(adm.get('parent_admission_sha'))[:16])


def reuse_skips_input(ctx: common.Ctx) -> Tuple[bool, str]:
    """True when ADM-INPUT (and LOCK-12) may be skipped: tier 1, a valid parent citation, and the lock's analytic
    SNR is not 'not_derivable' (which makes ADM-INPUT mandatory)."""
    try:
        if tier_of(ctx) != 1:
            return False, 'tier is not 1'
        if dig(ctx.lock, 'admission_thresholds.analytic_input_snr') == 'not_derivable':
            return False, "analytic_input_snr is 'not_derivable': ADM-INPUT is mandatory"
        valid, problems, missing = reuse_status(ctx)
    except (MissingInput, InvalidInput) as e:
        return False, str(e)
    if valid:
        return True, 'valid ADM-REUSE citation at tier 1'
    return False, '; '.join(problems + missing) or 'no citation'


@check('ADM-INPUT')
def check_adm_input(ctx: common.Ctx) -> Result:
    skip, why = reuse_skips_input(ctx)
    if skip:
        # RULES applies_when starts with 'always' (NOT_APPLICABLE is not declared), so a valid reuse is reported as
        # PASS on the parent's verified ADM-INPUT PASS (ADM-REUSE re-checks closure, task, split and that report).
        return Result('ADM-INPUT', PASS, '[reused parent admission] %s: the parent\'s ADM-INPUT PASS stands for this '
                      'delivery' % why, data={'reused': True})
    lock = ctx.lock
    adm = read_admission(ctx)
    ic = adm.get('input_contrast')
    sm = adm.get('structure_meters')
    problems: List[str] = []
    at = lock.get('admission_thresholds') if isinstance(lock.get('admission_thresholds'), dict) else {}
    cmin, cbasis = floor_or_lock(ctx, at.get('input_contrast_min'), 'input_contrast_min',
                                 'admission_thresholds.input_contrast_min')
    pmar, pbasis = floor_or_lock(ctx, at.get('prompt_id_margin_min'), 'prompt_id_margin_min',
                                 'admission_thresholds.prompt_id_margin_min')
    if not isinstance(ic, dict):
        return Result('ADM-INPUT', FAIL, 'ADMISSION.input_contrast is not an object')
    for k in INPUT_CONTRAST_KEYS:
        if k not in ic:
            problems.append('input_contrast.%s missing' % k)
    ratio, acc, chance = ic.get('ratio'), ic.get('prompt_id_loo_acc'), ic.get('chance')
    same, diff = ic.get('same_input_diff_noise'), ic.get('diff_input_same_noise')
    if not is_num(ratio):
        problems.append('input_contrast.ratio %r is not a finite number' % (ratio,))
    elif ratio < cmin:
        problems.append('INPUT_STATE_NOT_ADMITTED: input_contrast.ratio %s < %s (%s)' % (ratio, cmin, cbasis))
    if not is_num(same) or not is_num(diff) or same <= 0 or diff < 0:
        problems.append('input_contrast components same_input_diff_noise %r / diff_input_same_noise %r must be finite '
                        '(same > 0)' % (same, diff))
    elif is_num(ratio):
        r = float(diff) / float(same)
        if abs(float(ratio) - r) > max(0.005, RATIO_REL_TOL * abs(r)):
            problems.append('input_contrast.ratio %s disagrees with diff_input_same_noise / same_input_diff_noise = %.6f'
                            % (ratio, r))
    if not is_num(chance) or not (0 < chance <= 1):
        problems.append('input_contrast.chance %r must be in (0, 1]' % (chance,))
    if not is_num(acc) or not (0 <= acc <= 1):
        problems.append('input_contrast.prompt_id_loo_acc %r must be in [0, 1]' % (acc,))
    elif is_num(chance) and acc < chance + pmar:
        problems.append('INPUT_STATE_NOT_ADMITTED: prompt_id_loo_acc %s < chance %s + margin %s (%s)'
                        % (acc, chance, pmar, pbasis))
    asnr = ic.get('analytic_snr')
    if not (is_num(asnr) or asnr == 'not_derivable'):
        problems.append("input_contrast.analytic_snr %r must be a number or 'not_derivable'" % (asnr,))
    meters = {}
    if not isinstance(sm, dict):
        problems.append('ADMISSION.structure_meters is not an object (they must be reported)')
    else:
        for k in STRUCTURE_METERS:
            if not is_num(sm.get(k)):
                problems.append('structure_meters.%s %r is not a finite number' % (k, sm.get(k)))
            else:
                meters[k] = sm.get(k)
    return finish('ADM-INPUT', problems, [], 'input contrast ratio %s >= %s and prompt-id LOO accuracy %s >= chance %s + '
                  '%s; structure meters (reported, not gated) %s' % (ratio, cmin, acc, chance, pmar, meters),
                  data={'structure_meters': meters, 'input_contrast': ic})


# --------------------------------------------------------------------------------------
# ADM-ACQ and ADM-SHUF (computed by the program scorer from raw rows and TAUGHT manifests)
# --------------------------------------------------------------------------------------

@check('ADM-ACQ')
def check_adm_acq(ctx: common.Ctx) -> Result:
    scorer = lazy('scorer')
    at = ctx.lock.get('admission_thresholds') if isinstance(ctx.lock.get('admission_thresholds'), dict) else {}
    margin, basis = floor_or_lock(ctx, at.get('taught_acq_margin_min'), 'taught_acq_margin_min',
                                  'admission_thresholds.taught_acq_margin_min')
    acq = scorer.taught_acquisition(ctx)
    per = acq.get('per_world') or []
    problems: List[str] = []
    missing: List[str] = []
    if not per:
        missing.append('no technical worlds with candidate TAUGHT rows at the prefix checkpoint')
    rows = []
    for x in per:
        cand, best = x.get('cand'), x.get('best_shortcut')
        if cand is None or not isinstance(best, dict) or not is_num(best.get('value')):
            missing.append('world %s: candidate TAUGHT/prefix rate or the exposure-independent ladder is unavailable'
                           % x.get('world'))
            continue
        rows.append('%s %.4f vs %s %.4f' % (x.get('world'), cand, best.get('name'), best.get('value')))
        if not cand >= best['value'] + margin:
            problems.append('NOT_ACQUIRING: world %s candidate exact own-EOS on TAUGHT at prefix %.4f < best '
                            'exposure-independent shortcut %s %.4f + margin %s (%s)'
                            % (x.get('world'), cand, best.get('name'), best['value'], margin, basis))
    return finish('ADM-ACQ', problems, missing, 'candidate acquires TAUGHT at prefix above the best exposure-independent '
                  'shortcut + %s: %s' % (margin, '; '.join(rows)), data={'acquisition': acq})


def shuf_band(ctx: common.Ctx) -> Tuple[float, float, str]:
    """Effective shuf_band: the intersection of the lock band and the CONSTRAINTS band (narrower is stricter)."""
    bands = []
    lb = dig(ctx.lock, 'admission_thresholds.shuf_band')
    fb = floors(ctx).get('shuf_band')
    for name, b in (('lock', lb), ('floor', fb)):
        if b is None or b == FILL:
            continue
        if not (isinstance(b, list) and len(b) == 2 and all(is_num(x) for x in b) and b[0] <= b[1]):
            raise InvalidInput('%s shuf_band %r must be [lo, hi]' % (name, b))
        bands.append((float(b[0]), float(b[1]), name))
    if not bands:
        raise MissingInput('no shuf_band in the lock or CONSTRAINTS.threshold_floors')
    lo = max(b[0] for b in bands)
    hi = min(b[1] for b in bands)
    return lo, hi, ' & '.join('%s [%s, %s]' % (b[2], b[0], b[1]) for b in bands)


def _shuf_entries_non_addition(ctx: common.Ctx, taught: dict, shufs: List[str], cand: str) -> Tuple[List[dict], List[str]]:
    """Agreement with the candidate's taught labels (the truth for the same prompt) and prompt-level consistency."""
    out, missing = [], []
    for (w, a), rows in sorted(taught.items()):
        if a not in shufs:
            continue
        truth = {}
        for r in taught.get((w, cand)) or []:
            truth.setdefault(r.get('prompt_hex'), r.get('answer_hex'))
        agree = n = 0
        labs: Dict[Any, Set[str]] = {}
        for r in rows:
            t = truth.get(r.get('prompt_hex'))
            if t is None:
                missing.append('TAUGHT/%s/%s prompt %r has no candidate label (truth unknown for a non-addition task)'
                               % (w, a, r.get('prompt_hex')))
                continue
            n += 1
            agree += 1 if r.get('answer_hex') == t else 0
            labs.setdefault(r.get('prompt_hex'), set()).add(str(r.get('answer_hex')))
        bad = sorted(str(p) for p, l in labs.items() if len(l) > 1)
        out.append({'world': w, 'arm': a, 'n': n, 'agreement': (agree / float(n)) if n else None,
                    'orbit_consistent': not bad, 'inconsistent_orbits': bad[:20],
                    'distinct_true_answers': len(set(truth.values())),
                    'binary_derangement': len(set(truth.values())) == 2 and n > 0 and agree == 0})
    return out, missing


@check('ADM-SHUF')
def check_adm_shuf(ctx: common.Ctx) -> Result:
    lock = ctx.lock
    shufs = [a['id'] for a in lock_arms(lock) if a.get('role') == 'shuf']
    if not shufs:
        return Result('ADM-SHUF', NOT_APPLICABLE, 'no arm has role shuf')
    scorer = lazy('scorer')
    addition = lazy('addition')
    lo, hi, basis = shuf_band(ctx)
    cand = candidate_arm(lock)
    taught = scorer.load_taught(ctx, lock)
    fmt = dig(lock, 'task.format')
    missing: List[str] = []
    if addition.is_addition(fmt):
        entries = scorer.shuf_summary(ctx).get('arms') or []
        how = 'true answers recomputed from the prompts'
    else:
        entries, missing = _shuf_entries_non_addition(ctx, taught, shufs, cand)
        how = "the candidate's taught labels (non-addition task)"
    cand_worlds = sorted(w for (w, a) in taught if a == cand)
    if not cand_worlds:
        missing.append('no TAUGHT manifest for the candidate %s' % cand)
    # testbed-only technical worlds (not admission worlds, no learner receipt) run evaluator arms only, so the
    # SHUF arm has no stream there; every other world with a candidate manifest needs the SHUF manifest
    adm_worlds = set(common.admission_worlds(ctx)[0])
    tech_worlds = set(lock_worlds(lock, 'technical'))
    rc_worlds = set(w for (w, _a) in receipts(ctx))
    need = [w for w in cand_worlds if w in adm_worlds or w not in tech_worlds or w in rc_worlds]
    if not any(w in adm_worlds for w in need):
        missing.append('no admission world carries the candidate TAUGHT manifest (admission worlds %s)'
                       % sorted(adm_worlds))
    for a in shufs:
        for w in need:
            if (w, a) not in taught:
                missing.append('no TAUGHT manifest for SHUF arm %s in world %s' % (a, w))
    problems: List[str] = []
    rows = []
    for e in entries:
        tag = 'TAUGHT/%s/%s' % (e.get('world'), e.get('arm'))
        ag = e.get('agreement')
        if not e.get('n'):
            missing.append('%s has no rows' % tag)
            continue
        rows.append('%s %.3f' % (tag, ag))
        if e.get('binary_derangement'):
            problems.append('%s is a derangement of binary labels (agreement 0): never a valid SHUF' % tag)
        if not (lo <= ag <= hi):
            problems.append('%s agreement with the truth %.4f outside shuf_band [%s, %s] (%s)' % (tag, ag, lo, hi, basis))
        if not e.get('orbit_consistent'):
            problems.append('%s labels are not orbit-consistent (orbits %s)' % (tag, ', '.join(e.get('inconsistent_orbits') or [])[:200]))
    return finish('ADM-SHUF', problems, sorted(set(missing), key=missing.index),
                  'SHUF labels agree with %s within [%s, %s] and are orbit-consistent: %s' % (how, lo, hi, '; '.join(rows)),
                  data={'shuf': entries})


# --------------------------------------------------------------------------------------
# ADM-EXPO (exposure leakage and undeclared streams)
# --------------------------------------------------------------------------------------
# SPEC 7.4 ADM-EXPO, FM-12 and SPEC 20.11 (round 1, INC-HB1-A3): exposure is judged by CONTENT, not by the stream's
# feedback_bearing declaration or by field names. (1) A feedback-bearing row (declared stream flag, or the row's own
# flag) never carries a held prompt (E2) or canonical held orbit (E3), in any string field. (2) No row of ANY
# exposure stream (feedback-bearing or not), of the TAUGHT directory, or of an undeclared TAUGHT/stream/calibration-
# like file carries the expected answer of a held item together with that item's prompt or orbit: every string value
# and key of the row is read as text, as hex and as base64, and byte lists as bytes; a held prompt or orbit anywhere
# in the row plus a value equal to its answer, or one value holding the prompt followed by the answer, is a leak.
# Round-1 repair (verifier probes A5/A6/A4): a string value that is itself JSON (also hex/base64-coded JSON) is parsed
# and read as part of the row; inside one value the answer may follow the prompt or orbit after up to 32 bytes of
# non-digit filler ('3+7 equals 10', '"q": "3+7=", "a": "10"') or precede it as 'c = a+b'; and records are read in
# stream order: a record whose value at some field ends with a held prompt or orbit, followed by a record whose value
# at the same field starts (after up to 16 bytes of non-digit filler) with its answer, is a leak too (a listening
# stream that plays '3+7=' and then '10'). Non-feedback streams may hold held prompts WITHOUT answers (listening,
# presentation). The same scan gives the scorer the held orbits that were taught with their answers (SPEC 9.3: such
# items are S1, never counted).

STREAM_NAME_PARTS = ('taught', 'stream', 'calib', 'curric', 'expos', 'warm', 'listen')
_HEX_STR_RE = re.compile(r'^(?:[0-9A-Fa-f]{2})+$')
_B64_STR_RE = re.compile(r'^[A-Za-z0-9+/]+={0,2}$')
_TEXT_FIELD_SPLIT_RE = re.compile(r'[,\t;|]')
_MAX_EXPO_HITS = 200
_JSON_START_RE = re.compile(r'^\s*[\[{]')
_MAX_EMBEDDED_JSON = 1 << 20
# data files outside manifests/ and results/ that a learner could read (round-1 repair, verifier probe A8)
DATA_EXTS = ('.jsonl', '.json', '.txt', '.tsv', '.csv', '.jsonl.gz', '.json.gz', '.txt.gz')
NON_DELIVERY_DATA_DIRS = ('checks/', 'runlog/', 'review/', 'results/', 'manifests/')


def _stream_decls(lock: dict) -> Tuple[List[Tuple[str, bool]], List[str]]:
    es = lock.get('exposure_streams')
    if not isinstance(es, list) or not es:
        return [], ['lock.exposure_streams is empty: every TAUGHT or stream file must be declared']
    out, bad = [], []
    for i, e in enumerate(es):
        if not isinstance(e, dict) or not s(e.get('path_glob')) or not isinstance(e.get('feedback_bearing'), bool):
            bad.append('exposure_streams[%d] needs path_glob and a boolean feedback_bearing' % i)
            continue
        out.append((common.norm_rel(e['path_glob']), e['feedback_bearing']))
    return out, bad


def _logical_files(ctx: common.Ctx) -> Tuple[List[str], Set[str]]:
    """(every logical delivery path: files with .jsonl.gz shown as .jsonl, plus RESULT_BUNDLE JSONL members;
    the set of logical paths that are JSONL row sources)."""
    rows = set(src[0] for src in ctx.row_sources('**/*.jsonl'))
    files = set(rows)
    for rel in ctx.delivery_files():
        if common.RESULT_BUNDLE_RE.match(rel.rsplit('/', 1)[-1]):
            continue
        files.add(rel[:-3] if rel.endswith('.jsonl.gz') else rel)
    return sorted(files), rows


def _is_stream_candidate(rel: str, excluded: Set[str]) -> bool:
    """A TAUGHT, stream or calibration-like file that must be declared in exposure_streams (SPEC ADM-EXPO, 20.11):
    every file under manifests/ except the split manifests, and under results/ (receipts aside) every non-canonical
    JSONL file and every file whose path names a taught/stream/calibration/curriculum/exposure/warm-up/listening
    part."""
    name = rel.rsplit('/', 1)[-1]
    if rel in excluded or name == '.gitkeep' or name.lower().endswith('.md'):
        return False
    if rel.startswith('manifests/'):
        return True
    if rel.startswith('results/'):
        if rel.startswith('results/receipts/'):
            return False
        if any(any(k in part.lower() for k in STREAM_NAME_PARTS) for part in rel.split('/')):
            return True
        return rel.endswith('.jsonl') and rel not in CANONICAL_ROW_FILES and rel != common.REPLAY_OUTPUTS
    # round-1 repair: a data file elsewhere in the delivery whose path names a stream part (data/warmup.jsonl)
    if rel.startswith(NON_DELIVERY_DATA_DIRS) or not rel.lower().endswith(DATA_EXTS):
        return False
    return any(any(k in part.lower() for k in STREAM_NAME_PARTS) for part in rel.split('/'))


def _decodings(v: Any) -> List[bytes]:
    """Byte readings of one row value: a string as UTF-8 text, as hex and as base64 (when it is valid for either);
    a list of ints 0..255 as bytes."""
    return [d for d, _rich in _decodings_tagged(v)]


def _textlike(b: bytes) -> bool:
    """Bytes a learner could read as text: UTF-8 with printable characters and whitespace only, after trailing control
    bytes (an EOS such as 0x03 or 0x00) are set aside."""
    try:
        t = b.decode('utf-8')
    except UnicodeDecodeError:
        return False
    t = t.rstrip(''.join(chr(c) for c in range(32)))
    return all(ch.isprintable() or ch in '\t\n\r' for ch in t)


def _decodings_tagged(v: Any) -> List[Tuple[bytes, bool]]:
    """_decodings with a 'rich' flag: True for a reading the filler, stream-order and 'c = a+b' rules may use (round-1
    repair). The raw text of a hex-shaped string is the encoding, not what the learner hears ('6b' must not read as
    the answer 6), and a hex or base64 decoding that is not text is noise (base64 of a hex string decodes to random
    bytes), so those readings keep only the exact rules."""
    if isinstance(v, str):
        t = v.strip()
        hex_shaped = bool(_HEX_STR_RE.match(t))
        out = [(v.encode('utf-8', 'surrogatepass'), not hex_shaped)]
        if hex_shaped:
            d = bytes.fromhex(t)
            out.append((d, _textlike(d)))
        if len(t) >= 4 and len(t) % 4 == 0 and _B64_STR_RE.match(t):
            try:
                d = base64.b64decode(t, validate=True)
                out.append((d, _textlike(d)))
            except (ValueError, binascii.Error):
                pass
        return out
    if (isinstance(v, list) and v and len(v) <= 4096
            and all(isinstance(x, int) and not isinstance(x, bool) and 0 <= x <= 255 for x in v)):
        d = bytes(v)
        return [(d, _textlike(d))]
    return []


class TextRecord(list):
    """One line of a text stream: [line, field 1, field 2, ...]. Its items keep their position as their path."""


def _embedded_json(t: str) -> Any:
    """A string that is itself a JSON object or array (a JSON-encoded payload), parsed; otherwise None."""
    if len(t) > _MAX_EMBEDDED_JSON or not _JSON_START_RE.match(t):
        return None
    try:
        o = json.loads(t)   # lenient on purpose: NaN or duplicate keys must not hide an embedded payload
    except (ValueError, RecursionError):
        return None
    return o if isinstance(o, (dict, list)) else None


def _flat_items(obj: Any, out: List[Tuple[tuple, Any]], path: tuple = (), depth: int = 0) -> None:
    """(path, value) of every string value, key and byte list inside obj (one fed record). Keys are listed under
    their object's path + ('<key>',); list elements under the list's path + ('*',); the line and fields of a
    TextRecord under ('<line>',) and ('<field>', j). A string that is itself a JSON object or array (as text, hex or
    base64) is parsed and its content listed under path + ('<json>',) (round-1 repair, verifier probe A5)."""
    if depth > 64:
        return
    if isinstance(obj, TextRecord) and not path:
        for j, x in enumerate(obj):
            if x != '':
                _flat_items(x, out, ('<line>',) if j == 0 else ('<field>', j), depth + 1)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out.append((path + ('<key>',), str(k)))
            _flat_items(v, out, path + (str(k),), depth + 1)
    elif isinstance(obj, list):
        if _decodings(obj) and not all(isinstance(x, str) for x in obj):
            out.append((path, obj))
        for x in obj:
            if isinstance(x, (dict, list, str)):
                _flat_items(x, out, path + ('*',), depth + 1)
    elif isinstance(obj, str):
        out.append((path, obj))
        seen = set()
        for d in _decodings(obj):
            try:
                t = d.decode('utf-8')
            except UnicodeDecodeError:
                continue
            if t in seen:
                continue
            seen.add(t)
            emb = _embedded_json(t)
            if emb is not None:
                _flat_items(emb, out, path + ('<json>',), depth + 1)


def _flat_values(obj: Any, out: List[Any], depth: int = 0) -> None:
    """Every string value, key and byte list inside obj (one fed record), embedded JSON included."""
    items: List[Tuple[tuple, Any]] = []
    _flat_items(obj, items, (), depth)
    out.extend(v for _p, v in items)


def _doc_records(obj: Any) -> List[Any]:
    """Records of a JSON document: the elements of a top-level list; for a top-level object, each object element
    of its list values plus the object without those lists; otherwise the document itself."""
    if isinstance(obj, list):
        return list(obj)
    if isinstance(obj, dict):
        recs: List[Any] = []
        rest = {}
        for k, v in obj.items():
            if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
                recs.extend(v)
            else:
                rest[k] = v
        if rest:
            recs.append(rest)
        return recs
    return [obj]


class HeldIndex(object):
    """Held items of the lock for the exposure scan: keys (canonical orbit for E3 addition, exact prompt otherwise)
    with their item id and expected answer bytes."""

    def __init__(self, ctx: common.Ctx):
        addition = lazy('addition')
        self.addition = addition
        lock = ctx.lock
        self.fmt = dig(lock, 'task.format')
        self.is_add = addition.is_addition(self.fmt)
        self.exact = lock.get('claim_level_target') == 'E2' or not self.is_add
        self.problems: List[str] = []
        self.notes: List[str] = []
        split = lock.get('split') if isinstance(lock.get('split'), dict) else {}
        held_rel = split.get('held_manifest')
        if not s(held_rel):
            raise InvalidInput('lock.split.held_manifest is not set')
        held_rel = _logical_row_path(ctx, held_rel)
        if not ctx.row_sources(held_rel):
            raise MissingInput('held manifest %s absent' % held_rel)
        self.held_rel = held_rel
        self.items: Dict[tuple, str] = {}
        self.answer: Dict[tuple, bytes] = {}
        self.prompts: Dict[bytes, tuple] = {}
        if self.is_add:
            self.sep, self.term = addition._sep_term(self.fmt)
            self.eos = bytes.fromhex(self.fmt['eos_hex']) if s(self.fmt.get('eos_hex')) else b''
            self.rx = re.compile(rb'(?<![0-9])([0-9]{1,64})[ \t]*' + re.escape(self.sep) + rb'[ \t]*([0-9]{1,64})'
                                 rb'(?:[ \t]*(' + re.escape(self.term) + rb')[ \t]*([0-9]{1,65})?)?(?![0-9])')
            # round-1 repair: the answer after non-digit filler, before the equation ('c = a+b'), and the shapes
            # of a value that ends with a prompt or starts with an answer (stream order)
            self.after_rx = re.compile(rb'[^0-9]{0,32}?([0-9]{1,65})(?![0-9])', re.S)
            self.head_rx = re.compile(rb'[^0-9]{0,16}?([0-9]{1,65})(?![0-9])', re.S)
            self.operand_rx = re.compile(rb'[ \t]*' + re.escape(self.sep) + rb'[ \t]*[0-9]')
            self.before_rx = re.compile(rb'(?<![0-9])([0-9]{1,65})[ \t]*' + re.escape(self.term) + rb'[ \t]*$')
        else:
            self.eos = b''
        self.tail_rx = re.compile(rb'[ \t\r\n?.:]*' + (rb'(?:' + re.escape(self.eos) + rb')*' if self.eos else b'')
                                  + rb'[ \t\r\n]*$')
        no_answer = 0
        for label, ln, r in ctx.iter_rows_with_source(held_rel):
            k = self.key_of_hex(r.get('prompt_hex'))
            if k is None:
                self.problems.append('%s:%d held item without prompt_hex' % (label, ln))
                continue
            if not self.exact and k[0] != 'orbit':
                self.problems.append('%s:%d held prompt %r does not parse as addition' % (label, ln, r.get('prompt_hex')))
                continue
            self.items.setdefault(k, str(r.get('item_id')))
            ans = None
            if self.is_add:
                ans = addition.expected_bytes(r.get('prompt_hex'), self.fmt)
            elif isinstance(r.get('expected_hex'), str):
                try:
                    ans = bytes.fromhex(r['expected_hex'])
                except ValueError:
                    ans = None
            if ans:
                self.answer.setdefault(k, ans)
            else:
                no_answer += 1
            try:
                pb = bytes.fromhex(r.get('prompt_hex'))
            except (TypeError, ValueError):
                pb = b''
            if pb:
                self.prompts.setdefault(pb, k)
        if no_answer:
            self.notes.append('%d held item(s) carry no expected answer in %s: only their prompts are checked'
                              % (no_answer, held_rel))

    def key_of_hex(self, ph: Any) -> Optional[tuple]:
        if not isinstance(ph, str):
            return None
        p = ph.strip().lower()
        if not self.exact:
            pr = self.addition.parse(p, self.fmt)
            if pr is not None:
                return ('orbit',) + self.addition.canonical(*pr)
        return ('prompt', p)

    def _answer_form(self, b: bytes) -> bytes:
        t = b.strip()
        while self.eos and t.endswith(self.eos):
            t = t[:-len(self.eos)].rstrip()
        if self.is_add and t and all(48 <= c <= 57 for c in t):
            t = t.lstrip(b'0') or b'0'
        return t

    def _number_after(self, d: bytes, pos: int, rx: 're.Pattern') -> Optional[bytes]:
        """The first digit run at or after pos when only non-digit filler lies before it (rx bounds the filler),
        unless it is the first operand of a following sum ('3+7= 10+4=' holds no answer for 3+7)."""
        m = rx.match(d, pos)
        if not m or self.operand_rx.match(d, m.end(1)):
            return None
        return m.group(1)

    def scan_value(self, d: bytes) -> Tuple[Set[tuple], Set[tuple]]:
        """(held keys whose prompt or orbit occurs in d, held keys whose prompt is followed by its answer in d)."""
        keys, both, _tails = self.scan_value_full(d)
        return keys, both

    def scan_value_full(self, d: bytes, rich: bool = True) -> Tuple[Set[tuple], Set[tuple], Set[tuple]]:
        """(keys present in d, keys answered in d, keys whose prompt or orbit ends d): the answer directly after the
        terminator or, when d is a rich (text) reading, after up to 32 bytes of non-digit filler ('3+7 equals 10') or
        before the equation ('10 = 3+7')."""
        keys: Set[tuple] = set()
        both: Set[tuple] = set()
        tails: Set[tuple] = set()
        if self.is_add:
            for m in self.rx.finditer(d):
                a, b = m.group(1).decode('ascii'), m.group(2).decode('ascii')
                if self.exact:
                    k = ('prompt', (m.group(1) + self.sep + m.group(2) + self.term).hex())
                    if m.group(3) is None:
                        continue  # an exact prompt includes its terminator
                else:
                    k = ('orbit',) + self.addition.canonical(a, b)
                if k not in self.items:
                    continue
                keys.add(k)
                want = self._answer_form(self.answer.get(k, b''))
                if m.group(4) is not None:
                    if want and self._answer_form(m.group(4)) == want:
                        both.add(k)
                    continue
                if not rich:
                    continue
                nxt = self._number_after(d, m.end(), self.after_rx)
                if want and nxt is not None and self._answer_form(nxt) == want:
                    both.add(k)
                    continue
                if not self.exact:
                    pm = self.before_rx.search(d[max(0, m.start() - 80):m.start()])
                    if want and pm and self._answer_form(pm.group(1)) == want:
                        both.add(k)
                        continue
                if self.tail_rx.match(d, m.end()):
                    tails.add(k)
        else:
            t = d.strip()
            for pb, k in self.prompts.items():
                if t == pb or (len(pb) >= 3 and pb in d):
                    keys.add(k)
                    ans = self.answer.get(k)
                    if ans and (pb + ans) in d:
                        both.add(k)
                    elif rich and d.rstrip().endswith(pb):
                        tails.add(k)
        return keys, both, tails

    def heads_of(self, d: bytes, rich: bool = True) -> Set[bytes]:
        """Answer forms a value starts with: the whole value, and for a rich (text) reading (addition) the first digit
        run after up to 16 bytes of non-digit filler that is not a sum's first operand ('10', 'A: 10', '10 next');
        non-addition: the value."""
        out = {self._answer_form(d)}
        if not rich:
            return out
        if self.is_add:
            n = self._number_after(d, 0, self.head_rx)
            if n is not None:
                out.add(self._answer_form(n))
        else:
            out.add(d.lstrip())
        return out

    def answered_by(self, k: tuple, heads: Set[bytes]) -> bool:
        ans = self.answer.get(k)
        if not ans:
            return False
        want = self._answer_form(ans)
        if self.is_add:
            return want in heads
        return any(h == want or h.startswith(ans) for h in heads)

    def scan_items(self, items: Sequence[Tuple[tuple, Any]]) -> Tuple[Set[tuple], Set[tuple], Dict[tuple, Set[tuple]],
                                                                      Dict[tuple, Set[bytes]]]:
        """(keys present, keys present with their answer, per field path the unanswered keys whose prompt or orbit
        ends the field's last value, per field path the answer forms its first value starts with)."""
        present: Set[tuple] = set()
        leaked: Set[tuple] = set()
        answers: Set[bytes] = set()
        tails: Dict[tuple, Set[tuple]] = {}
        heads: Dict[tuple, Set[bytes]] = {}
        for path, v in items:
            tk: Set[tuple] = set()
            hs: Set[bytes] = set()
            for d, rich in _decodings_tagged(v):
                k, both, tl = self.scan_value_full(d, rich)
                present |= k
                leaked |= both
                tk |= tl
                answers.add(self._answer_form(d))
                hs |= self.heads_of(d, rich)
            tails[path] = tk          # the last value at a path ends the field
            heads.setdefault(path, hs)  # the first value at a path starts it
        for k in present - leaked:
            ans = self.answer.get(k)
            if ans and self._answer_form(ans) in answers:
                leaked.add(k)
        return present, leaked, dict((p, ks - leaked) for p, ks in tails.items() if ks - leaked), heads

    def scan_record(self, values: Sequence[Any]) -> Tuple[Set[tuple], Set[tuple]]:
        """(held keys present in the record, held keys present together with their expected answer)."""
        present, leaked, _t, _h = self.scan_items([((), v) for v in values])
        return present, leaked


def _file_records(ctx: common.Ctx, rel: str, row_paths: Set[str]) -> Tuple[Optional[List[Tuple[str, int, Any]]], str]:
    """(records [(label, line, record)], why) of one stream file: JSONL rows, JSON document records, or text lines
    (each line plus its comma/tab/semicolon/pipe fields). None when the content cannot be read as any of these."""
    if rel in row_paths:
        return [(label, ln, r) for label, ln, r in ctx.iter_rows_with_source(rel)], 'jsonl'
    p = ctx.path(rel)
    if not p.is_file():
        return None, 'not a file'
    raw = p.read_bytes()
    if rel.endswith('.gz'):
        try:
            raw = gzip.decompress(raw)
        except (OSError, EOFError) as e:
            return None, 'gzip unreadable (%s)' % e
    try:
        text = raw.decode('utf-8')
    except UnicodeDecodeError:
        return None, 'not UTF-8 text, JSON or JSONL'
    if rel.endswith('.json') or rel.endswith('.json.gz'):
        try:
            obj = common.loads_strict(text, what=rel)
        except InvalidInput as e:
            return None, 'JSON does not parse (%s)' % e
        return [(rel, i + 1, r) for i, r in enumerate(_doc_records(obj))], 'json'
    out: List[Tuple[str, int, Any]] = []
    for i, line in enumerate(text.splitlines(), 1):
        if line.strip():
            out.append((rel, i, TextRecord([line] + [x.strip() for x in _TEXT_FIELD_SPLIT_RE.split(line)])))
    return out, 'text'


def exposure_scan(ctx: common.Ctx) -> dict:
    """The content scan behind ADM-EXPO and the scorer's exposure strata (cached per ctx). Returns {'held':
    HeldIndex, 'problems', 'missing', 'fb_hits', 'leaks', 'leaked_keys', 'n_rows', 'n_fb', 'scanned',
    'undeclared'}. leaked_keys: held keys whose expected answer occurs with the prompt or orbit in some row."""
    key = 'wp4_exposure_scan'
    if key in ctx._cache:
        return ctx._cache[key]
    lock = ctx.lock
    held = HeldIndex(ctx)
    decls, problems = _stream_decls(lock)
    problems = list(held.problems) + problems
    missing: List[str] = []
    split = lock.get('split') if isinstance(lock.get('split'), dict) else {}
    excluded = set()
    for k in ('held_manifest', 'old_manifest', 'new_manifest'):
        v = _logical_row_path(ctx, split.get(k))
        if v:
            excluded.add(v)
    udi = lock.get('udi')
    if isinstance(udi, dict) and s(udi.get('variants_manifest')):
        excluded.add(common.norm_rel(udi['variants_manifest']))
    tdir = common.norm_rel(split.get('taught_manifest') or 'manifests/TAUGHT')
    files, row_paths = _logical_files(ctx)
    undeclared = [f for f in files if _is_stream_candidate(f, excluded)
                  and not any(common.glob_match(f, g) for g, _fb in decls)]
    for f in undeclared:
        problems.append('undeclared TAUGHT/stream file %s (not matched by any exposure_streams path_glob)' % f)
    fb_hits: List[str] = []
    leaks: List[str] = []
    leaked_keys: Dict[tuple, List[str]] = {}
    n_rows = n_fb = 0
    scanned: List[str] = []
    for f in files:
        if f in excluded:
            continue
        matched = [fb for g, fb in decls if common.glob_match(f, g)]
        in_taught = f.startswith(tdir + '/')
        if not matched and not in_taught and f not in undeclared:
            continue
        name = f.rsplit('/', 1)[-1]
        if name == '.gitkeep' or name.lower().endswith('.md'):
            continue
        declared_fb = any(matched) or (f in undeclared)
        recs, how = _file_records(ctx, f, row_paths)
        if recs is None:
            problems.append('exposure stream %s cannot be content-checked: %s (stream files must be JSONL, JSON or '
                            'UTF-8 text)' % (f, how))
            continue
        scanned.append(f)
        prev_tails: Dict[tuple, Set[tuple]] = {}
        prev_where = ''
        for label, ln, r in recs:
            n_rows += 1
            fb = r.get('feedback_bearing') if isinstance(r, dict) else None
            if fb is None:
                fb = declared_fb
            elif not isinstance(fb, bool):
                problems.append('%s:%d feedback_bearing %r is not a boolean' % (label, ln, fb))
                fb = True
            items: List[Tuple[tuple, Any]] = []
            _flat_items(r, items)
            present, leaked, tails, heads = held.scan_items(items)
            # stream order (round-1 repair, verifier probe A6): the previous record's field ended with a held prompt or
            # orbit and this record's same field starts with its answer
            cross: Set[tuple] = set()
            for path, ks in prev_tails.items():
                hs = heads.get(path)
                for k in sorted(ks):
                    if hs and k not in leaked and held.answered_by(k, hs):
                        leaked.add(k)
                        cross.add(k)
                        if len(leaks) < _MAX_EXPO_HITS:
                            leaks.append('%s:%d starts with the answer of held %s %s that %s ends with (stream order)'
                                         % (label, ln, 'prompt' if held.exact else 'orbit', held.items[k], prev_where))
            prev_tails, prev_where = tails, '%s:%d' % (label, ln)
            if fb:
                n_fb += 1
                if isinstance(r, dict):
                    k = held.key_of_hex(r.get('prompt_hex'))
                    if k is None:
                        problems.append('%s:%d feedback-bearing row without prompt_hex' % (label, ln))
                    elif k in held.items:
                        present.add(k)
                for k in sorted(present):
                    if len(fb_hits) < _MAX_EXPO_HITS:
                        fb_hits.append('%s:%d %s %s is in a feedback-bearing row' % (
                            label, ln, 'prompt of held' if held.exact else 'orbit of held', held.items[k]))
            for k in sorted(leaked):
                leaked_keys.setdefault(k, []).append('%s:%d' % (label, ln))
                if k not in cross and len(leaks) < _MAX_EXPO_HITS:
                    leaks.append('%s:%d carries held %s %s together with its answer (%s stream)' % (
                        label, ln, 'prompt' if held.exact else 'orbit', held.items[k],
                        'feedback-bearing' if fb else 'non-feedback'))
    out = {'held': held, 'problems': problems, 'missing': missing, 'fb_hits': fb_hits, 'leaks': leaks,
           'leaked_keys': leaked_keys, 'n_rows': n_rows, 'n_fb': n_fb, 'scanned': scanned, 'undeclared': undeclared}
    ctx._cache[key] = out
    return out


def exposed_held_orbits(ctx: common.Ctx) -> Set[Tuple[str, str]]:
    """Canonical held orbits whose expected answer occurs with the orbit in any exposure stream, TAUGHT file or
    undeclared stream-like file (SPEC 9.3 as amended in 20.11: such items are S1 and never counted). Empty for
    exact-prompt (E2 or non-addition) locks, whose strata are not orbit-based."""
    sc = exposure_scan(ctx)
    return set((k[1], k[2]) for k in sc['leaked_keys'] if k and k[0] == 'orbit')


@check('ADM-EXPO')
def check_adm_expo(ctx: common.Ctx) -> Result:
    sc = exposure_scan(ctx)
    held = sc['held']
    problems = list(sc['problems'])
    if sc['leaks']:
        problems.insert(0, 'held answers exposed in %d row(s) (prompt or orbit with its expected answer, any stream, '
                        'any field): %s' % (sum(len(v) for v in sc['leaked_keys'].values()), '; '.join(sc['leaks'][:5])))
    if sc['fb_hits']:
        problems.insert(0, 'held %s exposed in %d feedback-bearing row(s): %s' % (
            'prompts' if held.exact else 'orbits', len(sc['fb_hits']), '; '.join(sc['fb_hits'][:5])))
    return finish('ADM-EXPO', problems, sc['missing'],
                  '%d held %s; %d rows (%d feedback-bearing) in %d stream files: no held %s in a feedback-bearing row, '
                  'no held answer with its %s in any row; no undeclared TAUGHT/stream files%s'
                  % (len(held.items), 'prompts (E2: exact)' if held.exact else 'canonical orbits', sc['n_rows'],
                     sc['n_fb'], len(sc['scanned']), 'prompt' if held.exact else 'orbit',
                     'prompt' if held.exact else 'orbit', ('; ' + '; '.join(held.notes)) if held.notes else ''),
                  evidence=(sc['fb_hits'] + sc['leaks'])[:60] or None)


# --------------------------------------------------------------------------------------
# ADM-BUDGET
# --------------------------------------------------------------------------------------

@check('ADM-BUDGET')
def check_adm_budget(ctx: common.Ctx) -> Result:
    lock = ctx.lock
    read_authorization(ctx)  # the budget must sit within a recorded authorization
    problems: List[str] = []
    budget = lock.get('budget') if isinstance(lock.get('budget'), dict) else None
    notes = []
    if budget is None:
        problems.append('lock.budget is missing')
    else:
        for key in ('max_records', 'max_cpu_h'):
            v = budget.get(key)
            cap, basis = common.effective_cap(ctx, key)
            if not is_num(v) or v < 0:
                problems.append('lock.budget.%s %r is not a finite number >= 0' % (key, v))
            elif not is_num(cap):
                problems.append('no effective AUTHORIZATION cap for %s (%s)' % (key, basis))
            elif v > cap:
                problems.append('lock.budget.%s %s exceeds the AUTHORIZATION cap %s (%s)' % (key, v, cap, basis))
            else:
                notes.append('%s %s <= %s (%s)' % (key, v, cap, basis))
    pa = dig(lock, 'schedule.paired_atomic')
    if pa is not True:
        problems.append('schedule.paired_atomic is %r, not true (arms of a world must run as one atomic unit)' % (pa,))
    return finish('ADM-BUDGET', problems, [], 'budget within the authorization: %s; schedule.paired_atomic true'
                  % '; '.join(notes))


# --------------------------------------------------------------------------------------
# AUT-01..04
# --------------------------------------------------------------------------------------

def _stage_granted(v: Any) -> bool:
    return v is True or (isinstance(v, dict) and v.get('granted') is True)


@check('AUT-01')
def check_aut_01(ctx: common.Ctx) -> Result:
    a = read_authorization(ctx)
    problems: List[str] = []
    if a.get('schema') != SCHEMA_AUTHORIZATION:
        problems.append('schema is %r, not %s' % (a.get('schema'), SCHEMA_AUTHORIZATION))
    for k in AUTH_REQUIRED:
        if k not in a:
            problems.append('field %s missing' % k)
    fills = fill_paths(a)
    if fills:
        problems.append('__FILL__ values at %s' % ', '.join(fills[:8]))
    if 'verbatim' in a and not s(a.get('verbatim')):
        problems.append("verbatim is empty: record the owner's exact words")
    if 'utc' in a and common.normalize_utc(a.get('utc')) is None:
        problems.append('utc %r is not a UTC timestamp' % (a.get('utc'),))
    if 'session' in a and not s(a.get('session')):
        problems.append('session is empty')
    opts = a.get('options_shown')
    if 'options_shown' in a:
        if not isinstance(opts, list) or not opts:
            problems.append('options_shown must be a non-empty list of the options presented')
        else:
            for i, o in enumerate(opts):
                if not isinstance(o, dict) or not s(o.get('label')) or not s(o.get('text')):
                    problems.append('options_shown[%d] needs a label and a text' % i)
    if 'recommendation_shown' in a and not (a.get('recommendation_shown') is None
                                            or isinstance(a.get('recommendation_shown'), str)):
        problems.append('recommendation_shown must be a string (or null when none was shown)')
    st = a.get('stages')
    if 'stages' in a:
        if not isinstance(st, dict):
            problems.append('stages must be an object of stage booleans')
        else:
            for k in AUTH_STAGE_KEYS:
                v = st.get(k)
                if k not in st:
                    problems.append('stages.%s missing' % k)
                elif k == 'official' and isinstance(v, dict):
                    if not isinstance(v.get('granted'), bool):
                        problems.append('stages.official.granted must be a boolean')
                    if v.get('granted') and not s(v.get('verbatim')):
                        problems.append("stages.official is granted without the owner's GO verbatim")
                elif not isinstance(v, bool):
                    problems.append('stages.%s is %r, not a boolean' % (k, v))
            for k in st:
                if k not in AUTH_STAGE_KEYS:
                    problems.append('unknown stage key stages.%s' % k)
    if 'conditionals' in a and not isinstance(a.get('conditionals'), list):
        problems.append('conditionals must be a list')
    if 'caps' in a and a.get('caps') is not None and not isinstance(a.get('caps'), dict):
        problems.append('caps must be an object')
    return finish('AUT-01', problems, [], 'AUTHORIZATION.json fields present (recorded %s, session %s, %d options)'
                  % (a.get('utc'), a.get('session'), len(opts) if isinstance(opts, list) else 0))


def requested_stage(ctx: common.Ctx) -> Optional[str]:
    st = ctx.options.get('launch_stage') or ctx.stage
    return st if st in STAGE_GRANT else None


@check('AUT-02')
def check_aut_02(ctx: common.Ctx) -> Result:
    a = read_authorization(ctx)
    defaults = common.default_grant_caps(ctx)
    verbatim = a.get('verbatim')
    problems: List[str] = []
    notes: List[str] = []
    caps = a.get('caps') if isinstance(a.get('caps'), dict) else {}
    dapplied = a.get('defaults_applied') is True
    for k in CAP_KEYS:
        v = caps.get(k)
        d = defaults.get(k)
        if v is None:
            if not dapplied:
                problems.append('caps.%s is not recorded and defaults_applied is not true' % k)
            else:
                notes.append('%s default %s' % (k, d))
            continue
        if not is_num(v):
            problems.append('caps.%s %r is not a finite number' % (k, v))
            continue
        if common.number_in_text(v, verbatim):
            notes.append('%s %s (in verbatim)' % (k, v))
        elif dapplied and is_num(d) and v <= d:
            notes.append('%s %s (default cap %s applied)' % (k, v, d))
        else:
            problems.append('caps.%s = %s does not appear literally in the verbatim text (default %s applies; '
                            'record defaults_applied true with caps at or below the defaults, or quote the owner)'
                            % (k, v, d))
    stage = requested_stage(ctx)
    st = a.get('stages') if isinstance(a.get('stages'), dict) else {}
    if stage is None:
        return finish('AUT-02', problems, ['requested stage %r is unknown (lock|technical|roster|official or a launch '
                                           'stage)' % (ctx.options.get('launch_stage') or ctx.stage,)], '')
    key = STAGE_GRANT[stage]
    if stage == 'official' and ctx.dry and ctx.options.get('launch_stage') != 'official':
        key = 'dev_cycles'
        notes.append('dry official gate (rehearsal end): official GO not required yet (official %s)'
                     % ('granted' if _stage_granted(st.get('official')) else 'not granted'))
    v = st.get(key)
    if not _stage_granted(v):
        problems.append('stage %s needs AUTHORIZATION.stages.%s granted (got %r)' % (stage, key, v))
    elif key == 'official':
        go = v if isinstance(v, dict) else {}
        if not (s(go.get('verbatim')) or (isinstance(verbatim, str) and 'official' in verbatim.lower())):
            problems.append("stages.official is granted but neither its GO verbatim nor the owner's verbatim text "
                            "mentions the official run")
        shown = go.get('lock_shas_shown')
        if shown is not None:
            shown_set = set(strip_sha(x) for x in shown) if isinstance(shown, list) else set()
            for name, sha in (('ENDPOINT_LOCK', ctx.endpoint_lock_sha256), ('SOURCE_LOCK', ctx.source_lock_sha256)):
                if sha is not None and sha not in shown_set:
                    problems.append('the official GO lock_shas_shown omits the current %s %s' % (name, sha[:12]))
    notes.append('stage %s covered by stages.%s' % (stage, key))
    return finish('AUT-02', problems, [], '; '.join(notes))


@check('AUT-03')
def check_aut_03(ctx: common.Ctx) -> Result:
    a = read_authorization(ctx)
    opts = a.get('options_shown')
    if not isinstance(opts, list):
        raise InvalidInput('options_shown is not a list')
    problems = []
    for i, o in enumerate(opts):
        if not isinstance(o, dict):
            continue
        lab = o.get('label')
        if isinstance(lab, str) and RECOMMENDED_LABEL_RE.search(lab):
            problems.append("options_shown[%d].label %r carries a recommendation (put it in recommendation_shown)"
                            % (i, lab))
        txt = o.get('text')
        if isinstance(txt, str) and RECOMMENDED_TEXT_RE.search(txt):
            problems.append("options_shown[%d].text marks itself '(Recommended)'" % i)
    return finish('AUT-03', problems, [], '%d option labels carry no recommendation; recommendation_shown is separate'
                  % len(opts))


_MISSING_PTR = object()


def json_pointer(doc: Any, ptr: Any) -> Any:
    """RFC 6901 lookup; _MISSING_PTR when the pointer does not resolve. A malformed pointer raises InvalidInput."""
    if not isinstance(ptr, str):
        raise InvalidInput('pointer %r is not a string' % (ptr,))
    if ptr == '':
        return doc
    if not ptr.startswith('/'):
        raise InvalidInput('pointer %r must start with /' % ptr)
    cur = doc
    for tok in ptr[1:].split('/'):
        tok = tok.replace('~1', '/').replace('~0', '~')
        if isinstance(cur, dict):
            if tok not in cur:
                return _MISSING_PTR
            cur = cur[tok]
        elif isinstance(cur, list):
            if not re.fullmatch(r'0|[1-9][0-9]*', tok) or int(tok) >= len(cur):
                return _MISSING_PTR
            cur = cur[int(tok)]
        else:
            return _MISSING_PTR
    return cur


def _json_eq(a: Any, b: Any) -> bool:
    if is_num(a) and is_num(b):
        return float(a) == float(b)
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    try:
        return common.canonical_json(a) == common.canonical_json(b)
    except InvalidInput:
        return False


def eval_predicate(ctx: common.Ctx, pred: dict) -> Tuple[Optional[bool], str]:
    """(result, explanation). None means the artifact does not exist yet (cannot be evaluated). An invalid
    predicate raises InvalidInput."""
    for k in ('artifact', 'pointer', 'op'):
        if k not in pred:
            raise InvalidInput('predicate lacks %s' % k)
    op = pred.get('op')
    if op not in PREDICATE_OPS:
        raise InvalidInput('predicate op %r is not one of %s' % (op, ' '.join(PREDICATE_OPS)))
    art = pred.get('artifact')
    if not s(art):
        raise InvalidInput('predicate artifact %r is not a path' % (art,))
    rel = common.norm_rel(art)
    if not rel.endswith('.json'):
        raise InvalidInput('predicate artifact %s is not a JSON document' % rel)
    if not ctx.path(rel).is_file():
        return None, 'artifact %s does not exist yet' % rel
    doc = ctx.read_json(rel)
    got = json_pointer(doc, pred.get('pointer'))
    want = pred.get('value')
    where = '%s#%s' % (rel, pred.get('pointer'))
    if op == 'exists':
        res = (got is not _MISSING_PTR) if want is not False else (got is _MISSING_PTR)
        return res, '%s exists=%s' % (where, got is not _MISSING_PTR)
    if got is _MISSING_PTR:
        return False, '%s does not resolve' % where
    if op == 'eq':
        return _json_eq(got, want), '%s = %r (want %r)' % (where, got, want)
    if op == 'ne':
        return not _json_eq(got, want), '%s = %r (want != %r)' % (where, got, want)
    if op == 'in':
        if not isinstance(want, list):
            raise InvalidInput("predicate op 'in' needs a list value")
        return any(_json_eq(got, x) for x in want), '%s = %r (want in %r)' % (where, got, want)
    if not is_num(got) or not is_num(want):
        return False, '%s = %r is not comparable with %r by %s' % (where, got, want, op)
    g, w = float(got), float(want)
    res = {'lt': g < w, 'le': g <= w, 'gt': g > w, 'ge': g >= w}[op]
    return res, '%s = %r %s %r' % (where, got, op, want)


@check('AUT-04')
def check_aut_04(ctx: common.Ctx) -> Result:
    a = read_authorization(ctx)
    conds = a.get('conditionals')
    if not conds:
        return Result('AUT-04', NOT_APPLICABLE, 'AUTHORIZATION.conditionals is empty')
    if not isinstance(conds, list):
        raise InvalidInput('conditionals must be a list')
    af = author_family(ctx.lock)
    nonbenef = other_family(af)
    stage = requested_stage(ctx)
    cur = STAGE_ORDER.index(STAGE_GRANT[stage]) if stage in STAGE_GRANT else len(STAGE_ORDER)
    problems: List[str] = []
    missing: List[str] = []
    notes: List[str] = []
    for i, c in enumerate(conds):
        tag = 'conditionals[%d]' % i
        if not isinstance(c, dict):
            problems.append('%s is not an object' % tag)
            continue
        cst = c.get('stage')
        if cst is not None:
            if cst not in AUTH_STAGE_KEYS:
                problems.append('%s stage %r is not a stage key' % (tag, cst))
                continue
            if cst not in STAGE_ORDER or STAGE_ORDER.index(cst) > cur:
                notes.append('%s deferred: it conditions stage %s' % (tag, cst))
                continue
        pred = c.get('predicate')
        if isinstance(pred, dict):
            try:
                res, why = eval_predicate(ctx, pred)
            except InvalidInput as e:
                problems.append('%s predicate invalid: %s' % (tag, e))
                continue
            if res is None:
                missing.append('%s: %s' % (tag, why))
                continue
            if 'result' in c and c.get('result') is not None and c.get('result') is not res:
                problems.append('%s recorded result %r but the checker evaluates %s (%s)' % (tag, c.get('result'), res, why))
            if not res:
                problems.append('%s is not met: %s' % (tag, why))
            else:
                notes.append('%s holds (%s)' % (tag, why))
            continue
        ev = c.get('evaluated_by')
        fam = family(ev)
        if nonbenef is None:
            problems.append('%s: lock.author_agent %r names no agent family, so no non-beneficiary exists'
                            % (tag, ctx.lock.get('author_agent')))
        elif fam is None:
            problems.append('%s has no checker predicate {artifact, pointer, op, value} and evaluated_by %r names no '
                            'agent (self-judged or unevaluated conditional)' % (tag, ev))
        elif fam != nonbenef:
            problems.append('%s is evaluated by %r, the beneficiary family %s (needs %s)' % (tag, ev, af, nonbenef))
        elif not isinstance(c.get('result'), bool):
            problems.append('%s evaluated by %s records no boolean result' % (tag, ev))
        elif c.get('result') is not True:
            problems.append('%s evaluated by %s is not met (result false)' % (tag, ev))
        elif common.normalize_utc(c.get('utc')) is None:
            problems.append('%s evaluated by %s lacks a utc' % (tag, ev))
        else:
            notes.append('%s met per non-beneficiary %s' % (tag, ev))
    return finish('AUT-04', problems, missing, '; '.join(notes) or 'conditionals hold')
