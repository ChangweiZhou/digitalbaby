"""MRK-1 claims, pointers and errata (WP5): SPEC Sections 7.4 (CLM-*, ERR-01, WLD-*, PTR-*), 8 (CLAIM,
pointer, ERRATA) and 9.9 (headline grammar). Python 3.9+, standard library only.

`MFC claim <dir>` generates CLAIM.json and the headline from the lock, SCORES.json, VERDICT.json and
TESTBED.json (read as files; the scorer is never imported), keeping only the agent-editable fields
(findings_proposed, findings_none_reason, notes), scores the lock predictions, puts the headline on the
first line of REPORT.md and appends a DELIVERY_LEDGER row (and prediction transfer rows) through kb.py.

`MFC lint <files...>` (T0) runs CLM-06, CLM-12 and CLM-16 on arbitrary text files and PTR-01..07 on
pointer files.
"""
from __future__ import annotations

import gzip
import html
import json
import math
import re
import sys
import unicodedata
import zipfile
import zlib
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import common
from common import (CHECKER_ERROR, FAIL, MISSING, NOT_APPLICABLE, PASS, WARN, CheckerError, InvalidInput,  # noqa: F401
                    MissingInput, Result, check)
import kb as kbmod

SCHEMA_CLAIM = 'minifly.claim.v1'
SCHEMA_POINTER = 'minifly.pointer.v1'
LEVELS = ('E0', 'E1', 'E2', 'E3', 'E4')
PRIMARY_GATES = ('PASS', 'FAIL', 'INVALID', 'INCOMPLETE', 'NOT_OPENED', 'CANNOT_TEST_CLAIM_AS_SPECIFIED')
VERDICTS = ('PASS', 'FAIL', 'INVALID', 'INCOMPLETE', 'DEV_NOT_EVIDENCE', 'CANNOT_TEST_CLAIM_AS_SPECIFIED')
INVALID_REASONS = ('INTEGRITY_ALARM', 'TREATMENT_NOT_DELIVERED', 'NONFINITE')
WORLDS_ROLES = ('DEV', 'PILOT', 'MAIN', 'SEALED_CONFIRM')
GATE_TYPES = ('objective', 'component')
REUSE = ('not_tested', 'udi_pass', 'udi_fail')
E3_STRATA = ('S2', 'S3', 'S4')
OBJECTIVE_METRICS = ('exact_own_eos', 'balanced_exact_own_eos')
EDITABLE_FIELDS = ('findings_proposed', 'findings_none_reason', 'notes')
CLM02_EXEMPT = ('findings_proposed', 'notes')
WINKLER_ALPHA = 0.1
EPS = 1e-12

CLAIM_KEYS = (
    'schema', 'delivery', 'protocol_id', 'level_attempted', 'level_earned', 'reuse_specificity', 'primary_gate',
    'dev_shadow_verdict', 'invalid_reason', 'stop_reason', 'gate_type', 'labels', 'worlds_role', 'n_worlds',
    'objective_metric', 'designs_tried_k', 'amendments', 'coprimary_results', 'posthoc_files', 'lineage_parent',
    'family_id', 'native_path_only', 'compute', 'provenance', 'verdict_source', 'predictions_scored', 'findings_proposed',
    'findings_none_reason', 'review', 'review_debt_id', 'authorization', 'confirm_attempt', 'notes',
    'checker_sha256', 'headline')
OBJECTIVE_METRIC_KEYS = ('held_exact_own_eos', 'held_by_stratum', 'determined_fraction', 'old_end', 'old_final',
                         'old_drop', 'new_final', 'best_shortcut', 'best_other_arm', 'headroom', 'G', 'G_per_sample',
                         'G_per_cpu_h')
POINTER_KEYS = ('schema', 'track', 'delivery', 'claim', 'claim_sha256', 'level_earned', 'primary_gate', 'gate_type',
                'worlds_role', 'objective_metric', 'narrowing_rows', 'updated_utc', 'previous_snapshot',
                'user_decision')
POINTER_COPY_FIELDS = ('level_earned', 'primary_gate', 'gate_type', 'worlds_role', 'objective_metric')

SOFT_WORDS = ('largely', 'mostly', 'in part', 'partial', 'partially', 'nearly', 'promising', 'essentially', 'approved',
              'met in part', 'substantially', 'broadly', 'close to', 'encouraging')
SOFT_RE = re.compile(r'(?<![A-Za-z0-9_])(?:%s)(?![A-Za-z0-9_])' % '|'.join(
    re.escape(w).replace('\\ ', ' ').replace(' ', r'\s+') for w in sorted(SOFT_WORDS, key=len, reverse=True)), re.I)
VERDICT_TOKEN_RE = re.compile(r'(?<![A-Za-z0-9_])(?:PASS|FAIL|INVALID|INCOMPLETE|NOT_OPENED|DEV_NOT_EVIDENCE|'
                              r'CANNOT_TEST_CLAIM_AS_SPECIFIED)(?![A-Za-z0-9_])')
PRIMARY_GATE_WORD_RE = re.compile(r'primary[_ ]gate', re.I)
# Generic gate references in prose: the word gate(s) and delivery gate ids such as G1..G99.
GATE_WORD_RE = re.compile(r'(?<![A-Za-z0-9_])(?:gates?|G\d{1,2})(?![A-Za-z0-9_])', re.I)
# confirm wording (SPEC 7.4 CLM-04; SPEC 20.5): after NFKC normalisation, removal of format characters (zero-width
# space/joiners, bidi marks, word joiner, BOM, soft hyphen: Unicode category Cf) and of combining marks, and folding of
# the Cyrillic/Greek letters that look like the Latin letters of 'confirm', every case-insensitive occurrence of
# 'confirm' is a confirm token, taken with its enclosing identifier token [A-Za-z0-9_]+ (so GAIN_CONFIRMED,
# axis_confirmed, NativeConfirmation, reconfirms and con<U+200B>firmed all count). Only the whole tokens below, the
# kit's own enum and schema names, are not confirm wording; matching is case-sensitive (fail closed).
CONFIRM_CORE_RE = re.compile(r'confirm', re.I)
CONFIRM_ALLOWED_TOKENS = frozenset((
    'SEALED_CONFIRM', 'sealed_confirm',     # worlds.official_role_intended / worlds_role value; RULES 'when' value
    'CONFIRM_LEDGER',                       # PROJECT_DOCUMENTS/KB/CONFIRM_LEDGER.jsonl ('jsonl' is its own token)
    'confirm_attempt',                      # CLAIM.json key
    'confirm_alpha_spending',               # CONSTRAINTS.json key
))
_IDENT_CHARS = frozenset('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_')
_CONFUSABLE_FOLD = {
    0x0441: 'c', 0x0421: 'C', 0x03F2: 'c', 0x03F9: 'C',              # Cyrillic es, Greek lunate sigma
    0x043E: 'o', 0x041E: 'O', 0x03BF: 'o', 0x039F: 'O',              # Cyrillic o, Greek omicron
    0x039D: 'N', 0x0456: 'i', 0x0406: 'I', 0x03B9: 'i', 0x0399: 'I',  # Greek Nu, Cyrillic/Greek i
    0x04CF: 'i', 0x04C0: 'I', 0x0131: 'i', 0x041C: 'M', 0x039C: 'M',  # palochka, dotless i, Cyrillic/Greek M
    0x0433: 'r',                                                      # Cyrillic ghe (italic r look-alike)
}
LINEAGE_RE = re.compile(r'(?<![A-Za-z0-9_])(?:native|core|inherited)(?![A-Za-z0-9_])|(?<![A-Za-z0-9_])same\s+fly'
                        r'(?![A-Za-z0-9_])', re.I)
ABS_PATH_RE = re.compile(r'(?<![A-Za-z0-9._~-])(?:/Users/|/home/|/tmp/|/private/var/|/var/folders/|/root/)'
                         r'|(?<![A-Za-z0-9._/-])~/|(?<![A-Za-z0-9])[A-Za-z]:[\\/]')
MD_LINK_RE = re.compile(r'!?\[[^\]\n]*\]\(\s*<?([^)\s>]+)>?(?:\s+"[^"]*")?\s*\)')
URL_SCHEME_RE = re.compile(r'^[A-Za-z][A-Za-z0-9+.-]*:')
NUMBER_RE = re.compile(r'(?<![A-Za-z0-9_])[-+]?\d+(?:\.\d+)?\s*(?:%|pp\b)|(?<![A-Za-z0-9_.])\d*\.\d+(?![A-Za-z0-9_])')
PRIOR_WORK_RE = re.compile(r'(?<![A-Za-z0-9_])(?:prior|previous(?:ly)?|earlier|parent|legacy|predecessor|historical(?:ly)?|'
                           r'past|last delivery|reported in|catalog(?:ue)?)(?![A-Za-z0-9_])'
                           r'|(?<![A-Za-z0-9_])[a-z0-9]+(?:_[a-z0-9.]+)+_(?:design|dev)(?:_v\d+)?_\d{8}(?![A-Za-z0-9_])'
                           r'|(?<![A-Za-z0-9_])v\d+(?:\.\d+)+(?![A-Za-z0-9_])', re.I)
CITATION_RE = re.compile(r'(?<![A-Za-z0-9_-])F-(?:UF-\d{2,}|ARCH-\d{2,}|\d{8}-[a-z][a-z0-9]*-\d+)(?![A-Za-z0-9_])'
                         r'|(?<![A-Za-z0-9_])P\d{3}(?![A-Za-z0-9_])')
HEADLINE_LINE_RE = re.compile(r'^(?:%s) \| earned ' % '|'.join(PRIMARY_GATES))
BOOL_KEY_RE = re.compile(r'(confirm|pass|passes|passed|success|qualified)', re.I)
SHA_RE = re.compile(r'^[0-9a-f]{40}([0-9a-f]{24})?$')

POSTHOC_EXEMPT = ('CLAIM.json', 'REPORT.md', 'ERRATA.jsonl', 'checks/**', 'runlog/**', 'results/*/SCORES.json',
                  'results/official/VERDICT.json', 'results/admission/TESTBED.json')
GATE_RANK = {'FAIL': 0, 'INVALID': 1, 'INCOMPLETE': 1, 'CANNOT_TEST_CLAIM_AS_SPECIFIED': 1, 'NOT_OPENED': 1,
             'PASS': 2}


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------

def _s(v: Any) -> bool:
    return isinstance(v, str) and bool(v.strip())


def _n(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and abs(v) != float('inf')


def _num(v: Any) -> Optional[float]:
    return float(v) if _n(v) else None


def _r(v: Any) -> Any:
    return round(float(v), 10) if _n(v) else None


def _dig(obj: Any, path: str) -> Any:
    cur = obj
    for part in path.split('.'):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def _canon(obj: Any) -> bytes:
    return common.canonical_json(obj)


def _strip_sha(v: Any) -> Any:
    return v[7:] if isinstance(v, str) and v.startswith('sha256:') else v


def _opt_json(ctx: common.Ctx, rel: str) -> Any:
    p = ctx.path(rel)
    if not p.is_file():
        return None
    return common.load_json_file(p)


def _finish(cid: str, problems: List[str], missing: List[str], ok: str, data: Optional[dict] = None,
            warn: Optional[List[str]] = None) -> Result:
    def j(xs):
        return '; '.join(xs[:8]) + (' (+%d more)' % (len(xs) - 8) if len(xs) > 8 else '')
    if problems:
        return Result(cid, FAIL, j(problems), evidence=problems[:50], data=data or {})
    if missing:
        return Result(cid, MISSING, j(missing), evidence=missing[:50], data=data or {})
    if warn:
        return Result(cid, WARN, j(warn), evidence=warn[:50], data=data or {})
    return Result(cid, PASS, ok, data=data or {})


def _check_status(ctx: common.Ctx, rid: str) -> Tuple[str, str]:
    """Status of another registered check on this ctx (cached): PASS only when every result is PASS."""
    key = ('wp5_check', rid)
    if key in ctx._cache:
        return ctx._cache[key]
    try:
        rule = common.rule_index(ctx.rules or common.load_rules()).get(rid)
    except CheckerError as e:
        res = (CHECKER_ERROR, str(e))
        ctx._cache[key] = res
        return res
    if rule is None or rule.get('status') == 'planned':
        res = (CHECKER_ERROR, '%s is %s' % (rid, 'planned' if rule else 'not in RULES.json'))
    else:
        out = common.run_check(ctx, rule)
        bad = [r for r in out if r.status != PASS]
        res = (PASS, out[0].detail if out else '') if not bad else (bad[0].status, bad[0].detail)
    ctx._cache[key] = res
    return res


def _rule_ids(ctx: common.Ctx) -> List[str]:
    try:
        return sorted(common.rule_index(ctx.rules or common.load_rules()))
    except CheckerError:
        return []


def _lock_or_none(ctx: common.Ctx) -> Optional[dict]:
    if ctx.delivery is None:
        return None
    try:
        return ctx.lock
    except (MissingInput, InvalidInput):
        return None


def _same_delivery(a: Any, b: Any) -> bool:
    return kbmod._same_delivery(a, b)


# --------------------------------------------------------------------------------------
# Inputs of claim generation
# --------------------------------------------------------------------------------------

def gate_type_of(lock: dict) -> str:
    """LOCK-03: objective only for an unconditional exact own-EOS metric on own bytes with own EOS."""
    p = lock.get('primary') if isinstance(lock.get('primary'), dict) else {}
    output = p.get('output') if p.get('output') is not None else _dig(lock, 'objective_distance.output')
    ok = p.get('metric') in OBJECTIVE_METRICS and p.get('unconditional') is True and output == 'own_bytes_own_eos'
    return 'objective' if ok else 'component'


def _decision_value(scores: Optional[dict]) -> Optional[str]:
    if not isinstance(scores, dict):
        return None
    dec = scores.get('decision') if isinstance(scores.get('decision'), dict) else {}
    v = dec.get('verdict')
    if v == 'DEV_NOT_EVIDENCE':
        v = dec.get('shadow_verdict')
    if v is not None and v not in VERDICTS:
        raise InvalidInput('SCORES decision verdict %r is not a known verdict' % (v,))
    return v


def inputs(ctx: common.Ctx) -> dict:
    key = 'wp5_inputs'
    if key in ctx._cache:
        return ctx._cache[key]
    lock = ctx.lock
    scores = {}
    for role in ('official', 'rehearsal', 'dev', 'admission'):
        s = _opt_json(ctx, 'results/%s/SCORES.json' % role)
        if s is not None and not isinstance(s, dict):
            raise InvalidInput('results/%s/SCORES.json must be an object' % role)
        scores[role] = s
    verdict = _opt_json(ctx, 'results/official/VERDICT.json')
    if verdict is not None:
        if not isinstance(verdict, dict) or verdict.get('verdict') not in VERDICTS:
            raise InvalidInput('results/official/VERDICT.json verdict %r is not a known verdict'
                               % (verdict.get('verdict') if isinstance(verdict, dict) else verdict,))
    if (verdict is None) != (scores['official'] is None):
        raise InvalidInput('results/official has %s without %s: re-run MFC score <dir> --role official'
                           % (('VERDICT.json', 'SCORES.json') if verdict is not None else ('SCORES.json', 'VERDICT.json')))
    testbed = _opt_json(ctx, 'results/admission/TESTBED.json')
    if scores['official'] is not None:
        mrole = 'official'
    elif scores['dev'] is not None:
        mrole = 'dev'
    elif scores['rehearsal'] is not None:
        mrole = 'rehearsal'
    else:
        mrole = None
    els, sls = ctx.endpoint_lock_sha256, ctx.source_lock_sha256
    for label, obj in (('results/official/VERDICT.json', verdict),
                       ('results/%s/SCORES.json' % mrole, scores.get(mrole) if mrole else None)):
        if obj is None:
            continue
        if _strip_sha(obj.get('endpoint_lock_sha256')) != els:
            raise InvalidInput('%s was produced for ENDPOINT_LOCK %s, not the current %s: re-run MFC score'
                               % (label, str(obj.get('endpoint_lock_sha256'))[:12], str(els)[:12]))
        if _strip_sha(obj.get('source_lock_sha256')) != sls:
            raise InvalidInput('%s was produced for SOURCE_LOCK %s, not the current %s: re-run MFC score'
                               % (label, str(obj.get('source_lock_sha256'))[:12], str(sls)[:12]))
    out = {'lock': lock, 'scores': scores, 'verdict': verdict, 'testbed': testbed, 'metrics_role': mrole,
           'ms': scores.get(mrole) if mrole else None}
    ctx._cache[key] = out
    return out


def _argv_tokens(argv: Any) -> Set[str]:
    toks: Set[str] = set()
    for a in argv or []:
        s = str(a)
        toks.add(s)
        toks.update(t for t in re.split(r'[=,:;\s]+', s) if t)
    return toks


def dev_touched(ctx: common.Ctx, ids: Iterable[str]) -> Dict[str, List[str]]:
    """World ids among `ids` used on technical, dev or rehearsal: lock world lists, non-official OUTPUTS rows
    (by location or world_role), non-official receipts and non-official attempt argv."""
    ids = set(str(i) for i in ids)
    hits: Dict[str, List[str]] = {}
    if not ids:
        return hits

    def add(w, where):
        lst = hits.setdefault(w, [])
        if where not in lst and len(lst) < 5:
            lst.append(where)
    lock = ctx.lock
    worlds = lock.get('worlds') if isinstance(lock.get('worlds'), dict) else {}
    for role in ('technical', 'dev', 'rehearsal'):
        for w in worlds.get(role) or []:
            if str(w) in ids:
                add(str(w), 'lock.worlds.%s' % role)
    for label, lineno, row in ctx.iter_rows_with_source('results/**/OUTPUTS.jsonl'):
        w = row.get('world')
        if not isinstance(w, str) or w not in ids:
            continue
        non_off_dir = any(seg in label for seg in ('results/admission/', 'results/dev/', 'results/rehearsal/'))
        if non_off_dir or row.get('world_role') != 'official':
            add(w, '%s:%d (world_role %s)' % (label, lineno, row.get('world_role')))
    for rel in ctx.delivery_files():
        if rel.startswith('results/receipts/') and rel.endswith('.json'):
            parts = rel.split('/')
            try:
                rc = ctx.read_json(rel)
            except InvalidInput as e:
                # fail closed (SPEC 7.6): an unreadable receipt could belong to any world and any role, so the
                # official worlds cannot be shown fresh; it is never skipped
                raise InvalidInput('receipt %s is not valid JSON, so it cannot be shown that no official world was '
                                   'used on technical/dev/rehearsal (CLM-09, WLD-01): %s' % (rel, e))
            if not isinstance(rc, dict):
                raise InvalidInput('receipt %s is not a JSON object, so its world and world_role are unknown '
                                   '(CLM-09, WLD-01)' % rel)
            w = rc.get('world')
            if (len(parts) > 3 and parts[2] in ids) or w in ids:
                if not isinstance(rc, dict) or rc.get('world_role') != 'official':
                    add(w if w in ids else parts[2], '%s (world_role %s)' % (
                        rel, rc.get('world_role') if isinstance(rc, dict) else None))
    for a in common.read_attempts(ctx):
        if a.get('stage') == 'official':
            continue
        for w in ids & _argv_tokens(a.get('argv')):
            add(w, 'runlog/ATTEMPTS %s argv (stage %s)' % (a.get('attempt_id'), a.get('stage')))
    return hits


def _official_attempts(ctx: common.Ctx) -> List[dict]:
    return [a for a in common.read_attempts(ctx) if a.get('stage') == 'official']


def _floors(ctx: common.Ctx) -> dict:
    f = ctx.kb.constraints().get('threshold_floors')
    if not isinstance(f, dict):
        raise MissingInput('CONSTRAINTS.threshold_floors missing')
    return f


def worlds_role_of(ctx: common.Ctx, inp: dict) -> Tuple[str, List[str], List[str]]:
    """CLM-09: (worlds_role, reasons, evaluated worlds)."""
    lock = ctx.lock
    verdict = inp['verdict']
    if verdict is None:
        ms = inp['ms']
        worlds = [str(w) for w in (ms.get('evaluated_worlds') or [])] if ms else []
        if not worlds:
            # nothing was evaluated, so no world is dev-touched: 0 worlds is fewer than min_worlds (CLM-09)
            return 'PILOT', ['no official verdict and no evaluated worlds: 0 worlds < min_worlds'], worlds
        return 'DEV', ['no official verdict: %s outcomes only' % (inp['metrics_role'] or 'no')], worlds
    wr = verdict.get('world_roles') if isinstance(verdict.get('world_roles'), dict) else {}
    worlds = sorted(str(w) for w in wr)
    if not worlds:
        return 'DEV', ['VERDICT.json names no evaluated worlds with their roles (world_roles)'], worlds
    non_off = [w for w in worlds if wr.get(w) != 'official']
    if non_off:
        return 'DEV', ['worlds %s are not official-role in the verdict' % ','.join(non_off[:5])], worlds
    touched = dev_touched(ctx, worlds)
    if touched:
        return 'DEV', ['official worlds touched on dev/technical/rehearsal: %s' % '; '.join(
            '%s at %s' % (w, ', '.join(v)) for w, v in sorted(touched.items())[:3])], worlds
    if not _official_attempts(ctx):
        return 'DEV', ['no launch-recorded official attempt (MAIN requires the official stage)'], worlds
    intended = _dig(lock, 'worlds.official_role_intended')
    if intended not in ('MAIN', 'SEALED_CONFIRM', 'PILOT'):
        raise InvalidInput('worlds.official_role_intended %r is not MAIN|SEALED_CONFIRM|PILOT' % (intended,))
    req = [v for v in (_floors(ctx).get('min_worlds_nonpilot'), _dig(lock, 'primary.min_worlds')) if _n(v)]
    if not req:
        raise MissingInput('min_worlds unavailable (CONSTRAINTS.threshold_floors.min_worlds_nonpilot, primary.min_worlds)')
    need = max(req)
    if intended == 'PILOT' or len(worlds) < need:
        return 'PILOT', ['%d official worlds%s' % (len(worlds), ' < min_worlds %s' % need if len(worlds) < need
                                                    else ' (PILOT intended)')], worlds
    if intended == 'SEALED_CONFIRM':
        bad = []
        for rid in ('SEAL-01', 'SEAL-02', 'SEAL-03', 'SEAL-04', 'SEAL-05', 'OC-01', 'REV-08'):
            st, det = _check_status(ctx, rid)
            if st != PASS:
                bad.append('%s %s' % (rid, st))
        rev = _opt_json(ctx, 'review/REVIEW.json')
        if not (isinstance(rev, dict) and rev.get('decision') == 'ADMIT'):
            bad.append('REVIEW decision is not ADMIT')
        if bad:
            return 'MAIN', ['SEALED_CONFIRM requires SEAL, OC, REVIEW and replay: %s' % ', '.join(bad)], worlds
        return 'SEALED_CONFIRM', ['seal, OC, review and replay pass'], worlds
    return 'MAIN', ['%d fresh official worlds from a launch-recorded official attempt' % len(worlds)], worlds


def e3_conditions(ctx: common.Ctx, inp: dict) -> Tuple[bool, List[str]]:
    lock = ctx.lock
    p = lock.get('primary') if isinstance(lock.get('primary'), dict) else {}
    why: List[str] = []
    if p.get('pool') != 'HELD':
        why.append('primary pool %r is not HELD' % (p.get('pool'),))
    sc = p.get('strata_counted')
    if not (isinstance(sc, list) and sc and all(s in E3_STRATA for s in sc)):
        why.append('strata_counted %r not within S2..S4' % (sc,))
    roles = {a.get('id'): a.get('role') for a in (lock.get('arms') or []) if isinstance(a, dict)}
    if not any(roles.get(a) == 'no_prior_teaching' for a in (p.get('comparator_arms') or [])):
        why.append('no no_prior_teaching arm among comparator_arms')
    st, det = _check_status(ctx, 'ADM-EXPO')
    if st != PASS:
        why.append('ADM-EXPO %s' % st)
    off = inp['scores']['official'] or {}
    head, margin = off.get('headroom'), p.get('margin_abs')
    if not (_n(head) and _n(margin) and head >= margin - EPS):
        why.append('headroom over the best shortcut %s < margin_abs %s' % (head, margin))
    return (not why), why


def level_of(ctx: common.Ctx, inp: dict, primary_gate: str, gate_type: str) -> Tuple[str, List[str]]:
    """CLM-03 level rules: E2+ needs an objective PASS; E3 needs the E3 conditions; E4 is never earned by a
    single delivery (breadth across task families is not measured by the kit)."""
    target = ctx.lock.get('claim_level_target')
    if target not in LEVELS:
        raise InvalidInput('claim_level_target %r is not E0..E4' % (target,))
    ti = LEVELS.index(target)
    if primary_gate != 'PASS':
        return 'NONE', ['primary gate %s' % primary_gate]
    if gate_type != 'objective':
        return LEVELS[min(ti, 1)], ['component gate: at most E1']
    if ti <= 2:
        return target, ['objective PASS at the target level']
    ok, why = e3_conditions(ctx, inp)
    if not ok:
        return 'E2', ['E3 not earned: %s' % '; '.join(why)]
    return 'E3', ['E3 conditions hold'] + (['E4 is never earned by one delivery'] if target == 'E4' else [])


def udi_certifiable(ctx: common.Ctx) -> Tuple[bool, str]:
    """Whether a UDI result may be claimed: UDI-01 must be registered as non-planned and PASS on this
    delivery. scorer.udi() alone checks the variants against the witness but not the seal-derived draw of
    the dependent set (SPEC 9.7); only UDI-01 can certify the result. While UDI-01 is planned (SPEC 20.2),
    no UDI value is claimed and reuse_specificity stays not_tested."""
    try:
        rule = common.rule_index(ctx.rules or common.load_rules()).get('UDI-01')
    except CheckerError as e:
        return False, 'UDI-01 unavailable: %s' % e
    if rule is None:
        return False, 'UDI-01 is not in RULES.json'
    if rule.get('status') == 'planned':
        return False, 'UDI-01 planned (not yet enforced)'
    st, det = _check_status(ctx, 'UDI-01')
    if st != PASS:
        return False, 'UDI-01 %s: %s' % (st, det)
    return True, 'UDI-01 PASS'


def reuse_of(ctx: common.Ctx, inp: dict) -> str:
    """CLAIM.reuse_specificity: udi_pass/udi_fail only when UDI-01 certifies the UDI result, else not_tested."""
    udi = (inp['scores']['official'] or {}).get('udi')
    if not (isinstance(udi, dict) and udi.get('valid') is True and _n(udi.get('lcb'))):
        return 'not_tested'
    ok, _why = udi_certifiable(ctx)
    if not ok:
        return 'not_tested'
    return 'udi_pass' if udi['lcb'] > 0 else 'udi_fail'


def labels_of(ctx: common.Ctx, inp: dict) -> List[str]:
    lin = ctx.lock.get('lineage') if isinstance(ctx.lock.get('lineage'), dict) else {}
    out: List[str] = []
    if len([c for c in (lin.get('changes_declared') or [])]) > 1:
        out.append('integrated_intervention')
    weak: Set[str] = set()
    for role in ('dev', 'official'):
        s = inp['scores'].get(role)
        if isinstance(s, dict):
            weak.update(str(a) for a in (s.get('weak_comparator') or []))
    if weak:
        out.append('weak_comparator')
    for pr in lin.get('priors_added') or []:
        out.append('priors:%s' % pr)
    return out


def objective_metric_of(inp: dict) -> dict:
    ms = inp['ms'] or {}
    old = ms.get('old') if isinstance(ms.get('old'), dict) else {}
    hbs = ms.get('held_by_stratum') if isinstance(ms.get('held_by_stratum'), dict) else {}
    held = ms.get('held_exact_own_eos')

    def pair(v):
        if isinstance(v, dict) and _s(v.get('name')) and _n(v.get('value')):
            return {'name': v['name'], 'value': _r(v['value'])}
        return None
    return {
        'held_exact_own_eos': _r(held) if _n(held) else 'not_measured',
        'held_by_stratum': {k: _r(v.get('rate') if isinstance(v, dict) else v) for k, v in sorted(hbs.items())},
        'determined_fraction': _r(ms.get('determined_fraction')),
        'old_end': _r(old.get('end')), 'old_final': _r(old.get('final')), 'old_drop': _r(old.get('drop')),
        'new_final': _r(ms.get('new_final')),
        'best_shortcut': pair(ms.get('best_shortcut')),
        'best_other_arm': pair(ms.get('best_other_arm')),
        'headroom': _r(ms.get('headroom')), 'G': _r(ms.get('G')), 'G_per_sample': _r(ms.get('G_per_sample')),
        'G_per_cpu_h': _r(ms.get('G_per_cpu_h')),
    }


def winkler(lo: float, hi: float, x: float, alpha: float = WINKLER_ALPHA) -> float:
    """Winkler interval score of the central (1 - alpha) interval [lo, hi] at observation x."""
    s = hi - lo
    if x < lo:
        s += 2.0 / alpha * (lo - x)
    elif x > hi:
        s += 2.0 / alpha * (x - hi)
    return s


def score_predictions(ctx: common.Ctx, inp: dict, primary_gate: str) -> List[dict]:
    """CLM-10 scoring: hit inside interval90, Winkler score at alpha 0.1, Brier for p_pass. An endpoint that
    was not opened (or not validly resolved: INVALID, INCOMPLETE, CANNOT_TEST) is 'unresolved'."""
    lock = ctx.lock
    preds = lock.get('predictions')
    if preds is None:
        return []
    if not isinstance(preds, list):
        raise InvalidInput('lock.predictions must be a list')
    verdict = inp['verdict'] or {}
    resolved = inp['verdict'] is not None and primary_gate in ('PASS', 'FAIL')
    guards = {g.get('id'): g for g in (lock.get('guards') or []) if isinstance(g, dict)}
    out = []
    for i, p in enumerate(preds):
        if not isinstance(p, dict):
            raise InvalidInput('lock.predictions[%d] is not an object' % i)
        pid, target = p.get('id'), p.get('target')
        ent: Dict[str, Any] = {'id': pid, 'target': target}
        obs, outcome = None, None
        if resolved:
            if target == 'primary':
                obs = _num((verdict.get('primary') or {}).get('mean_cand'))
                outcome = 1 if primary_gate == 'PASS' else 0
            elif isinstance(target, str) and target.startswith('guard:') and target[6:] in guards:
                gid = target[6:]
                gd = (verdict.get('guard_detail') or {}).get(gid) or {}
                kind = guards[gid].get('kind')
                obs = _num(gd.get('mean_drop') if kind == 'old_drop' else gd.get('mean_acq'))
                gs = (verdict.get('guards') or {}).get(gid)
                outcome = 1 if gs == 'PASS' else (0 if gs == 'FAIL' else None)
        if not resolved or (obs is None and outcome is None):
            ent.update({'observed': 'unresolved', 'hit': 'unresolved', 'interval_score': 'unresolved',
                        'brier': 'unresolved', 'status': 'unresolved'})
            out.append(ent)
            continue
        iv = p.get('interval90')
        hit = iscore = None
        if obs is not None and isinstance(iv, list) and len(iv) == 2 and all(_n(x) for x in iv) and iv[0] <= iv[1]:
            hit = bool(iv[0] - EPS <= obs <= iv[1] + EPS)
            iscore = _r(winkler(float(iv[0]), float(iv[1]), obs))
        pp = _num(p.get('p_pass'))
        brier = _r((pp - outcome) ** 2) if (pp is not None and outcome is not None) else None
        br = _num((p.get('base_rate_forecast') or {}).get('value')) if isinstance(p.get('base_rate_forecast'), dict) else None
        ent.update({'observed': _r(obs), 'hit': hit, 'interval_score': iscore, 'brier': brier, 'status': 'scored',
                    'p_pass': pp, 'outcome': outcome, 'base_rate': br,
                    'base_brier': _r((br - outcome) ** 2) if (br is not None and outcome is not None) else None})
        out.append(ent)
    return out


def _launch_recorded(ctx: common.Ctx, rel: str, produced: Dict[str, str]) -> bool:
    """True when `rel` (or its decompressed content / zip members) matches a launch-recorded path+sha256."""
    if produced.get(rel) == common.sha256_file(ctx.path(rel)):
        return True
    if rel.endswith('.gz') and rel[:-3] in produced:
        try:
            with gzip.open(str(ctx.path(rel)), 'rb') as fh:
                return common.sha256_bytes(fh.read()) == produced[rel[:-3]]
        except (OSError, EOFError):
            return False
    if common.RESULT_BUNDLE_RE.match(rel.rsplit('/', 1)[-1]):
        try:
            with zipfile.ZipFile(str(ctx.path(rel))) as zf:
                names = [i for i in zf.infolist() if not i.is_dir()]
                if not names:
                    return False
                zdir = rel.rsplit('/', 1)[0] if '/' in rel else ''
                for info in names:
                    m = common.norm_rel(info.filename)
                    logical = m if (m.startswith('results/') or not zdir) else zdir + '/' + m
                    data = zf.read(info)
                    if logical.endswith('.gz'):
                        logical, data = logical[:-3], gzip.decompress(data)
                    if produced.get(logical) != common.sha256_bytes(data):
                        return False
                return True
        except (zipfile.BadZipFile, OSError, EOFError, InvalidInput):
            return False
    return False


def posthoc_info(ctx: common.Ctx) -> Tuple[List[str], Optional[dict]]:
    """(files added after the official lock, the first official attempt). The official lock is the commit
    the first official attempt ran at (launch requires a clean closure there). Launch-recorded official
    outputs and kit-written files (CLAIM, REPORT, ERRATA, checks/, runlog/, SCORES/VERDICT/TESTBED) are not
    POSTHOC."""
    key = 'wp5_posthoc'
    if key in ctx._cache:
        return ctx._cache[key]
    offs = _official_attempts(ctx)
    if not offs:
        if any(r.startswith('results/official/') for r in ctx.delivery_files()):
            raise MissingInput('results/official/ exists without a launch-recorded official attempt: the official '
                               'lock cannot be established')
        ctx._cache[key] = ([], None)
        return ctx._cache[key]
    first = offs[0]
    head = first.get('git_head')
    if not _s(head):
        raise MissingInput('the official attempt %s records no git_head' % first.get('attempt_id'))
    d = ctx.delivery_rel
    at_lock = set()
    for p in ctx.git.list_files(head, d):
        at_lock.add(p[len(d) + 1:] if d else p)
    produced: Dict[str, str] = {}
    for a in offs:
        for pp in a.get('produced_paths') or []:
            if isinstance(pp, dict) and _s(pp.get('path')):
                produced[common.norm_rel(pp['path'])] = pp.get('sha256')
    out = []
    for rel in ctx.delivery_files():
        if rel in at_lock or any(common.glob_match(rel, g) for g in POSTHOC_EXEMPT):
            continue
        if _launch_recorded(ctx, rel, produced):
            continue
        out.append(rel)
    ctx._cache[key] = (sorted(out), first)
    return ctx._cache[key]


def _verdict_input_patterns(lock: dict) -> List[str]:
    pats = ['results/official/**', 'results/receipts/**', 'manifests/**', 'results/admission/FIXTURES.jsonl',
            'ENDPOINT_LOCK.json', 'SOURCE_LOCK.json']
    files = lock.get('files') if isinstance(lock.get('files'), dict) else {}
    for k in ('admit', 'params'):
        if _s(files.get(k)):
            pats.append(files[k])
    for k in ('native', 'runner', 'evaluator', 'generators', 'assets'):
        for v in files.get(k) or []:
            if _s(v):
                pats.append(v)
    return pats


def _parse_manifest_orbits(path: Path, fmt: dict, addition: Any) -> Set[Tuple[str, str]]:
    out: Set[Tuple[str, str]] = set()
    if path.name.endswith('.gz'):
        with gzip.open(str(path), 'rb') as fh:
            data = fh.read()
    else:
        data = path.read_bytes()
    for _lineno, row in common._iter_jsonl_stream(data.splitlines(), str(path)):
        pr = addition.parse(row.get('prompt_hex'), fmt)
        if pr is not None:
            out.add(addition.canonical(pr[0], pr[1]))
    return out


def held_overlap(ctx: common.Ctx) -> Optional[dict]:
    """WLD-02: canonical held orbits of this delivery that appear in other deliveries' held or taught
    manifests (bounded repo scan). None when not applicable (no addition format or no held manifest)."""
    key = 'wp5_overlap'
    if key in ctx._cache:
        return ctx._cache[key]
    import addition
    lock = ctx.lock
    fmt = _dig(lock, 'task.format')
    held_rel = _dig(lock, 'split.held_manifest') or 'manifests/HELD.jsonl'
    if not addition.is_addition(fmt) or not _s(held_rel) or not ctx.path(held_rel).is_file():
        ctx._cache[key] = None
        return None
    mine: Set[Tuple[str, str]] = set()
    for row in ctx.read_jsonl(held_rel):
        pr = addition.parse(row.get('prompt_hex'), fmt)
        if pr is not None:
            mine.add(addition.canonical(pr[0], pr[1]))
    if not mine:
        ctx._cache[key] = None
        return None
    scan = ctx.repo_scan()
    d = ctx.delivery_rel
    others: Set[Tuple[str, str]] = set()
    sources: List[str] = []
    fmts: Dict[str, dict] = {}
    for f in scan.files:
        rel = scan.rel(f)
        if d and (rel == d or rel.startswith(d + '/')):
            continue
        r2 = '/' + rel
        if not (r2.endswith('/manifests/HELD.jsonl') or r2.endswith('/manifests/HELD.jsonl.gz')
                or ('/manifests/TAUGHT/' in r2 and (rel.endswith('.jsonl') or rel.endswith('.jsonl.gz')))):
            continue
        droot = r2.split('/manifests/')[0].lstrip('/')
        ofmt = fmts.get(droot)
        if ofmt is None:
            ofmt = fmt
            lp = Path(scan.root) / droot / 'ENDPOINT_LOCK.json'
            if lp.is_file():
                try:
                    ol = common.load_json_file(lp)
                    if addition.is_addition(_dig(ol, 'task.format')):
                        ofmt = _dig(ol, 'task.format')
                except (MissingInput, InvalidInput):
                    pass
            fmts[droot] = ofmt
        try:
            got = _parse_manifest_orbits(f, ofmt, addition)
        except (OSError, EOFError, InvalidInput) as e:
            raise MissingInput('held overlap scan could not complete: %s unreadable (%s)' % (rel, e))
        if got & mine:
            sources.append(rel)
        others |= got
    k = len(mine & others)
    res = {'k': k, 'n': len(mine), 'sources': sorted(sources)[:20], 'scope': scan.scope()}
    ctx._cache[key] = res
    return res


def _fmt_pct(v: Any) -> str:
    return _fmt_hundred(v) + '%'


def _fmt_pp(v: Any) -> str:
    return _fmt_hundred(v) + 'pp'


def _fmt_hundred(v: Any) -> str:
    s = '%.1f' % (float(v) * 100.0)
    return '0.0' if s == '-0.0' else s


def _strata_text(lock: dict) -> str:
    sc = _dig(lock, 'primary.strata_counted') or []
    st = set(str(x) for x in sc)
    if st and st <= set(E3_STRATA):
        txt = 'S2+'  # SPEC 9.9 grammar '[S2+ determined, n=<n>]': every counted stratum lies within S2..S4
    elif st:
        txt = ','.join(sorted(st))
    else:
        txt = 'all strata'
    if _dig(lock, 'units.witness') in ('pair_rectangle', 'column_closure'):
        txt += ' determined'
    return txt


def _counted_n(lock: dict, ms: Optional[dict]) -> Any:
    det = (ms or {}).get('determinacy') if isinstance((ms or {}).get('determinacy'), dict) else {}
    if _dig(lock, 'units.witness') in ('pair_rectangle', 'column_closure'):
        return det.get('determined')
    return det.get('total')


def build_headline(claim: dict, lock: dict, inp: dict, overlap: Optional[dict]) -> str:
    """SPEC 9.9 headline grammar."""
    om = claim['objective_metric']
    parts = [claim['primary_gate'], 'earned %s (attempted %s)' % (claim['level_earned'], claim['level_attempted'])]
    held = om.get('held_exact_own_eos')
    if not _n(held):
        # SPEC 9.9: 'When nothing held was measured, the headline reads held exact own-EOS not_measured'; the
        # shortcut comparison and G are defined on the held cells, so they are not printed either
        parts.append('held exact own-EOS not_measured')
    else:
        n = _counted_n(lock, inp['ms'])
        seg = 'held exact own-EOS %s [%s, n=%s]' % (_fmt_pct(held), _strata_text(lock), n if _n(n) else 'not_measured')
        bs = om.get('best_shortcut')
        if isinstance(bs, dict) and _n(bs.get('value')):
            seg += ' vs best shortcut %s %s' % (bs['name'], _fmt_pct(bs['value']))
        else:
            seg += ' vs best shortcut not_measured'
        parts.append(seg)
        parts.append('G %s' % (_fmt_pp(om['G']) if _n(om.get('G')) else 'not_measured'))
    if _n(om.get('old_final')):
        parts.append('OLD final %s (drop %s)' % (_fmt_pct(om['old_final']),
                                                _fmt_pp(om['old_drop']) if _n(om.get('old_drop')) else 'not_measured'))
    elif _n(held):
        parts.append('OLD final not_measured')
    parts.append('%d %s worlds' % (claim['n_worlds'], claim['worlds_role']))
    parts.append('best of k=%d' % claim['designs_tried_k'])
    # optional suffixes
    ms = inp['ms'] or {}
    cand = None
    if inp['verdict'] is not None:
        cand = _num((inp['verdict'].get('primary') or {}).get('mean_cand'))
    if cand is None:
        cand = _num(((ms.get('decision') or {}).get('primary') or {}).get('mean_cand')) if isinstance(ms.get('decision'), dict) else None
    if cand is None:
        cand = _num(held)
    if cand is not None:
        names = set()
        # every arm of the decision comparator (declared comparator_arms plus the implied no_prior_teaching, shuf,
        # frozen and lesion arms, SPEC 20.6) at or above the candidate is named
        dc = dict(ms.get('comparators') or {})
        dc.update(ms.get('decision_comparators') or {})
        for a, v in dc.items():
            if _n(v) and v >= cand - EPS:
                names.add(str(a))
        bo = ms.get('best_other_arm')
        if isinstance(bo, dict) and _n(bo.get('value')) and bo['value'] >= cand - EPS and _s(bo.get('name')):
            names.add(bo['name'])
        ordered = sorted(names)
        bs = ms.get('best_shortcut')
        if isinstance(bs, dict) and _n(bs.get('value')) and bs['value'] >= cand - EPS and _s(bs.get('name')):
            ordered.append(bs['name'])
        if ordered:
            parts.append('%s >= candidate' % ','.join(ordered))
    if claim['coprimary_results']:
        parts.append('amended: co-primary %s' % ','.join(str(c.get('verdict')) for c in claim['coprimary_results']))
    priors = [lab[len('priors:'):] for lab in claim['labels'] if lab.startswith('priors:')]
    if priors:
        parts.append('priors: %s' % ', '.join(priors))
    if 'weak_comparator' in claim['labels']:
        parts.append('WEAK_COMPARATOR')
    if claim['primary_gate'] == 'NOT_OPENED' and claim.get('dev_shadow_verdict'):
        parts.append('dev shadow %s' % claim['dev_shadow_verdict'])
    if overlap and overlap.get('k', 0) > 0:
        parts.append('held overlap %d/%d with other deliveries' % (overlap['k'], overlap['n']))
    return ' | '.join(parts)


def _admission_passed(ctx: common.Ctx) -> bool:
    for rep in common.list_gate_reports(ctx):
        if not rep.get('dry') and rep.get('stage') in ('roster', 'official') and rep.get('verdict') == 'PASS':
            return True
    return False


def e3_basis_of(ctx: common.Ctx, claim: dict) -> Tuple[List[str], bool]:
    lock = ctx.lock
    pool = _dig(lock, 'primary.pool')
    sc = [str(s) for s in (_dig(lock, 'primary.strata_counted') or [])]
    adm = _admission_passed(ctx)
    basis = ['pool %s' % pool, 'strata %s' % (','.join(sc) or 'none'),
             'admission %s' % ('PASS' if adm else 'not passed'), 'worlds_role %s' % claim['worlds_role'],
             'verdict %s' % claim['primary_gate']]
    opened = (pool == 'HELD' and bool(sc) and all(s in E3_STRATA for s in sc) and adm
              and claim['worlds_role'] in ('MAIN', 'SEALED_CONFIRM') and claim['primary_gate'] in ('PASS', 'FAIL'))
    return basis, opened


def _compute_block(ctx: common.Ctx) -> dict:
    recs = 0.0
    for rel in ctx.delivery_files():
        if rel.startswith('results/receipts/') and rel.endswith('.json'):
            rc = ctx.read_json(rel)
            if isinstance(rc, dict) and _n(rc.get('records')):
                recs += float(rc['records'])
    cpu = common.attempts_cpu_s(common.read_attempts(ctx)) / 3600.0
    fam_id, members = common.family_of(ctx)
    return {'records_total_incl_failed': int(recs), 'cpu_h_total_incl_failed': round(cpu, 6),
            'family_cpu_h': round(common.family_cpu_h(ctx, members), 6)}


def _coprimary_results(ctx: common.Ctx, inp: dict) -> List[dict]:
    cps = ctx.lock.get('coprimary') or []
    if not isinstance(cps, list):
        raise InvalidInput('lock.coprimary must be a list')
    given = (inp['verdict'] or {}).get('coprimary_results') or (inp['verdict'] or {}).get('coprimary') or []
    out = []
    for i, e in enumerate(cps):
        label = e.get('protocol_id') if isinstance(e, dict) and e.get('protocol_id') else (e if isinstance(e, str) else 'coprimary[%d]' % i)
        v = None
        for g in given if isinstance(given, list) else []:
            if isinstance(g, dict) and (g.get('protocol_id') == label or g.get('index') == i) and g.get('verdict') in VERDICTS:
                v = g['verdict']
        if inp['verdict'] is None:
            v = 'NOT_OPENED'
        out.append({'endpoint': str(label), 'verdict': v or 'NOT_SCORED'})
    return out


def official_run_provenance(ctx: common.Ctx) -> dict:
    """CLAIM.provenance (SPEC 20.8): whether any official attempt was interrupted (nonzero or killed exit, or recovered
    from its in-flight marker), the resume attempts, the (world, arm) units their rows hold and the units they
    re-ran into results/official_replay/ as the determinism replay."""
    offs = _official_attempts(ctx)
    resumes = [a.get('attempt_id') for a in offs if a.get('resume_of')]

    def units(pattern: str) -> List[List[str]]:
        got: Set[Tuple[str, str]] = set()
        if resumes:
            for r in ctx.iter_rows(pattern):
                if r.get('attempt_id') in resumes:
                    got.add((str(r.get('world')), str(r.get('arm'))))
        return [list(u) for u in sorted(got)]

    def interrupted(a: dict) -> bool:
        ec = a.get('exit_code')
        return bool(a.get('interrupted') or a.get('recovered')) or not (isinstance(ec, int) and not isinstance(ec, bool)
                                                                       and ec == 0)
    return {'interrupted': any(interrupted(a) for a in offs), 'resume_attempts': resumes,
            'resumed_units': units('results/official/OUTPUTS.jsonl'), 'replay_units': units(common.REPLAY_OUTPUTS)}


def _base_claim(ctx: common.Ctx) -> Tuple[dict, dict]:
    """The generated CLAIM (agent-editable fields at their defaults) and generation extras (cached)."""
    key = 'wp5_claim'
    if key in ctx._cache:
        return ctx._cache[key]
    ok, why = common.v1_status(ctx.delivery)
    if not ok:
        raise MissingInput('legacy delivery: use retro (%s)' % why)
    lock = ctx.lock
    inp = inputs(ctx)
    verdict = inp['verdict']
    dev_shadow = _decision_value(inp['scores']['dev']) if inp['scores']['dev'] else _decision_value(inp['scores']['rehearsal'])
    tb = inp['testbed'] if isinstance(inp['testbed'], dict) else {}
    if verdict is not None:
        pg = 'NOT_OPENED' if verdict['verdict'] == 'DEV_NOT_EVIDENCE' else verdict['verdict']
        if verdict['verdict'] == 'DEV_NOT_EVIDENCE' and dev_shadow is None:
            dev_shadow = verdict.get('shadow_verdict')
    elif tb.get('result') == 'CANNOT_TEST_CLAIM_AS_SPECIFIED':
        pg = 'CANNOT_TEST_CLAIM_AS_SPECIFIED'
    else:
        pg = 'NOT_OPENED'
    role, role_why, worlds = worlds_role_of(ctx, inp)
    if role == 'DEV' and verdict is not None and pg in ('PASS', 'FAIL', 'INVALID', 'INCOMPLETE'):
        dev_shadow = pg
        pg = 'NOT_OPENED'
    if pg == 'CANNOT_TEST_CLAIM_AS_SPECIFIED' or tb.get('result') == 'CANNOT_TEST_CLAIM_AS_SPECIFIED':
        stop = 'CANNOT_TEST'
    elif pg == 'INVALID':
        stop = 'INVALID'
    elif verdict is None and any(c.get('decision') == 'stop' for c in common.read_cycles(ctx)):
        stop = 'CYCLE_STOP'
    else:
        stop = None
    gtype = gate_type_of(lock)
    level, level_why = level_of(ctx, inp, pg, gtype)
    posthoc, _off = posthoc_info(ctx)
    fam_id, _members = common.family_of(ctx)
    ms = inp['ms']
    od_native = _dig(lock, 'objective_distance.native_only_path') is True
    if ms is None:
        native_only = None
    else:
        native_only = bool(od_native and ms.get('non_native_rows') == 0)
    amd = []
    for i, a in enumerate(ctx.read_jsonl_optional('AMENDMENTS.jsonl')):
        amd.append({'id': a.get('id') or 'row%d' % (i + 1), 'reason': a.get('reason'),
                    'fields_changed': list(a.get('fields_changed') or []), 'user_decision': a.get('user_decision'),
                    'new_protocol_id': a.get('new_protocol_id')})
    if verdict is not None:
        vsrc = {'decide_version': verdict.get('decide_version'),
                'verdict_sha256': common.sha256_file(ctx.path('results/official/VERDICT.json')),
                'endpoint_lock_sha256': verdict.get('endpoint_lock_sha256'),
                'source_lock_sha256': verdict.get('source_lock_sha256'), 'path': 'results/official/VERDICT.json'}
    else:
        vsrc = {'decide_version': (ms or {}).get('decide_version'), 'verdict_sha256': None,
                'endpoint_lock_sha256': (ms or {}).get('endpoint_lock_sha256'),
                'source_lock_sha256': (ms or {}).get('source_lock_sha256'),
                'path': 'results/%s/SCORES.json' % inp['metrics_role'] if inp['metrics_role'] else None}
    rev = _opt_json(ctx, 'review/REVIEW.json')
    auth = _opt_json(ctx, 'AUTHORIZATION.json')
    d_rel = ctx.delivery_rel
    debts = kbmod.open_debts(ctx.kb, d_rel)
    confirm = None
    for r in ctx.kb.rows('CONFIRM_LEDGER'):
        if _same_delivery(r.get('delivery'), d_rel):
            confirm = {'family_id': r.get('family_id'), 'index': r.get('attempt_index'), 'alpha': r.get('alpha_allocated')}
    k, _pairs = common.designs_tried_k(ctx)
    claim = {
        'schema': SCHEMA_CLAIM,
        'delivery': d_rel,
        'protocol_id': lock.get('protocol_id'),
        'level_attempted': lock.get('claim_level_target'),
        'level_earned': level,
        'reuse_specificity': reuse_of(ctx, inp),
        'primary_gate': pg,
        'dev_shadow_verdict': dev_shadow,
        'invalid_reason': (verdict or {}).get('invalid_reason') if verdict is not None else None,
        'stop_reason': stop,
        'gate_type': gtype,
        'labels': labels_of(ctx, inp),
        'worlds_role': role,
        'n_worlds': len(worlds),
        'objective_metric': objective_metric_of(inp),
        'designs_tried_k': int(k),
        'amendments': amd,
        'coprimary_results': _coprimary_results(ctx, inp),
        'posthoc_files': posthoc,
        'lineage_parent': _dig(lock, 'lineage.parent'),
        'family_id': fam_id,
        'native_path_only': native_only,
        'compute': _compute_block(ctx),
        'provenance': official_run_provenance(ctx),
        'verdict_source': vsrc,
        'predictions_scored': score_predictions(ctx, inp, pg),
        'findings_proposed': [],
        'findings_none_reason': None,
        'review': {'path': 'review/REVIEW.json', 'decision': rev.get('decision')} if isinstance(rev, dict) else None,
        'review_debt_id': debts[-1].get('id') if debts else None,
        'authorization': ({'path': 'AUTHORIZATION.json', 'recommendation_shown': auth.get('recommendation_shown')}
                          if isinstance(auth, dict) else None),
        'confirm_attempt': confirm,
        'notes': '',
        'checker_sha256': common.checker_sha256(),
        'headline': '',
    }
    overlap = held_overlap(ctx)
    claim['headline'] = build_headline(claim, lock, inp, overlap)
    basis, opened = e3_basis_of(ctx, claim)
    extras = {'worlds_role_reasons': role_why, 'level_reasons': level_why, 'e3_basis': basis,
              'opened_held_E3': opened, 'overlap': overlap, 'metrics_role': inp['metrics_role']}
    ctx._cache[key] = (claim, extras)
    return claim, extras


def generate_claim(ctx: common.Ctx, existing: Optional[dict] = None) -> Tuple[dict, dict]:
    """A fresh regeneration with the agent-editable fields of `existing` kept."""
    base, extras = _base_claim(ctx)
    claim = json.loads(json.dumps(base))
    if isinstance(existing, dict):
        for k in EDITABLE_FIELDS:
            if k in existing:
                claim[k] = existing[k]
    return claim, extras


def _read_claim(ctx: common.Ctx) -> dict:
    c = ctx.read_json('CLAIM.json')
    if not isinstance(c, dict):
        raise InvalidInput('CLAIM.json must be an object')
    return c


def claim_problems(c: Any) -> List[str]:
    """CLM-01: schema, required keys, no unknown keys and closed enums."""
    p: List[str] = []
    if not isinstance(c, dict):
        return ['CLAIM.json is not an object']
    if c.get('schema') != SCHEMA_CLAIM:
        p.append('schema %r is not %s' % (c.get('schema'), SCHEMA_CLAIM))
    for k in CLAIM_KEYS:
        if k not in c:
            p.append('missing field %s' % k)
    extra = sorted(set(c) - set(CLAIM_KEYS))
    if extra:
        p.append('unknown fields %s (agents edit only %s)' % (', '.join(extra), ', '.join(EDITABLE_FIELDS)))

    def enum(k, allowed, null=False):
        v = c.get(k)
        if v is None and null:
            return
        if v not in allowed:
            p.append('%s %r not in %s' % (k, v, '|'.join(str(a) for a in allowed)))
    enum('level_attempted', LEVELS)
    enum('level_earned', LEVELS + ('NONE',))
    enum('reuse_specificity', REUSE)
    enum('primary_gate', PRIMARY_GATES)
    enum('dev_shadow_verdict', VERDICTS, null=True)
    enum('invalid_reason', INVALID_REASONS, null=True)
    enum('gate_type', GATE_TYPES)
    enum('worlds_role', WORLDS_ROLES)
    labs = c.get('labels')
    if not isinstance(labs, list) or not all(isinstance(x, str) and (x in ('integrated_intervention', 'weak_comparator')
                                                                      or (x.startswith('priors:') and len(x) > 7))
                                             for x in labs):
        p.append('labels must be a list of integrated_intervention|weak_comparator|priors:<x>')
    for k in ('n_worlds', 'designs_tried_k'):
        v = c.get(k)
        if not (isinstance(v, int) and not isinstance(v, bool) and v >= 0):
            p.append('%s must be an integer >= 0' % k)
    om = c.get('objective_metric')
    if not isinstance(om, dict):
        p.append('objective_metric must be an object')
    else:
        for k in OBJECTIVE_METRIC_KEYS:
            if k not in om:
                p.append('objective_metric.%s missing' % k)
        h = om.get('held_exact_own_eos')
        if not (_n(h) or h == 'not_measured'):
            p.append("objective_metric.held_exact_own_eos must be a number or 'not_measured'")
        for k in ('best_shortcut', 'best_other_arm'):
            v = om.get(k)
            if v is not None and not (isinstance(v, dict) and _s(v.get('name')) and _n(v.get('value'))):
                p.append('objective_metric.%s must be {name, value} or null' % k)
    for k in ('amendments', 'coprimary_results', 'posthoc_files', 'predictions_scored', 'findings_proposed'):
        if not isinstance(c.get(k), list):
            p.append('%s must be a list' % k)
    for i, ps in enumerate(c.get('predictions_scored') or []):
        if not (isinstance(ps, dict) and _s(ps.get('id')) and all(k in ps for k in ('observed', 'hit', 'interval_score',
                                                                                    'brier'))):
            p.append('predictions_scored[%d] must hold {id, observed, hit, interval_score, brier}' % i)
    if not isinstance(c.get('compute'), dict) or not all(k in (c.get('compute') or {}) for k in (
            'records_total_incl_failed', 'cpu_h_total_incl_failed', 'family_cpu_h')):
        p.append('compute must hold records_total_incl_failed, cpu_h_total_incl_failed, family_cpu_h')
    pv = c.get('provenance')
    if not (isinstance(pv, dict) and set(pv) == {'interrupted', 'resume_attempts', 'resumed_units', 'replay_units'}
            and isinstance(pv.get('interrupted'), bool)
            and all(isinstance(pv.get(k), list) for k in ('resume_attempts', 'resumed_units', 'replay_units'))):
        p.append('provenance must be {interrupted: bool, resume_attempts: [], resumed_units: [], replay_units: []}')
    vs = c.get('verdict_source')
    if not isinstance(vs, dict) or not all(k in vs for k in ('decide_version', 'verdict_sha256', 'endpoint_lock_sha256',
                                                             'source_lock_sha256')):
        p.append('verdict_source must hold decide_version, verdict_sha256, endpoint_lock_sha256, source_lock_sha256')
    rv = c.get('review')
    if rv is not None and not (isinstance(rv, dict) and 'path' in rv and 'decision' in rv):
        p.append('review must be {path, decision} or null')
    au = c.get('authorization')
    if au is not None and not (isinstance(au, dict) and 'path' in au and 'recommendation_shown' in au):
        p.append('authorization must be {path, recommendation_shown} or null')
    ca = c.get('confirm_attempt')
    if ca is not None and not (isinstance(ca, dict) and all(k in ca for k in ('family_id', 'index', 'alpha'))):
        p.append('confirm_attempt must be {family_id, index, alpha} or null')
    if not (isinstance(c.get('checker_sha256'), str) and re.fullmatch(r'[0-9a-f]{64}', c['checker_sha256'])):
        p.append('checker_sha256 must be a sha256')
    if not _s(c.get('headline')) or not HEADLINE_LINE_RE.match(str(c.get('headline'))):
        p.append('headline does not follow the SPEC 9.9 grammar')
    if c.get('notes') is not None and not isinstance(c.get('notes'), str):
        p.append('notes must be a string')
    fnr = c.get('findings_none_reason')
    if fnr is not None and not isinstance(fnr, str):
        p.append('findings_none_reason must be a string or null')
    try:
        _canon(c)
    except InvalidInput as e:
        p.append(str(e))
    return p


# --------------------------------------------------------------------------------------
# claim command
# --------------------------------------------------------------------------------------

def _update_report(ctx: common.Ctx, headline: str) -> str:
    p = ctx.path('REPORT.md')
    if not p.is_file():
        p.write_text(headline + '\n', encoding='utf-8')
        return 'created REPORT.md with the headline'
    text = p.read_text(encoding='utf-8')
    lines = text.splitlines(True)
    for i, ln in enumerate(lines):
        if not ln.strip():
            continue
        if ln.strip() == headline:
            return 'REPORT.md first line already equals the headline'
        if HEADLINE_LINE_RE.match(ln.strip()):
            lines[i] = headline + ('\n' if ln.endswith('\n') else '')
            p.write_text(''.join(lines), encoding='utf-8')
            return 'replaced the previous headline on the first line of REPORT.md'
        break
    if not text.strip():
        p.write_text(headline + '\n', encoding='utf-8')
        return 'wrote the headline into the empty REPORT.md'
    # a hand-written first line is never silently pushed down: the author fixes the prose (CLM-05 FAILs until
    # the first non-empty line equals the headline; SPEC 6.3 item 2, 12.2 C22)
    return ('REPORT.md left unchanged: its first non-empty line is not a generated headline (CLM-05 requires it '
            'to equal CLAIM.headline)')


def _append_transfers(ctx: common.Ctx, claim: dict, claim_sha: str) -> List[str]:
    """Transfer rows {prediction_id, delivery, ok|failed} for each finding a resolved prediction cited, and a
    checker narrowing row for a finding with 2+ failed transfers and negative Brier skill (SPEC 11)."""
    lock = ctx.lock
    preds = {p.get('id'): p for p in (lock.get('predictions') or []) if isinstance(p, dict)}
    rows = ctx.kb.rows('FINDINGS')
    findings = {r.get('id'): r for r in rows if (r.get('kind') or 'finding') == 'finding'}
    d_rel = ctx.delivery_rel
    written: List[str] = []
    touched: Set[str] = set()
    for ps in claim.get('predictions_scored') or []:
        if ps.get('status') != 'scored':
            continue
        p = preds.get(ps.get('id')) or {}
        if _n(ps.get('brier')) and _n(ps.get('base_brier')):
            outcome = 'ok' if ps['brier'] < ps['base_brier'] else 'failed'
        elif isinstance(ps.get('hit'), bool):
            outcome = 'ok' if ps['hit'] else 'failed'
        else:
            continue
        for fid in p.get('basis_findings') or []:
            if fid == 'none' or fid not in findings:
                continue
            if any(r.get('kind') == 'transfer' and r.get('refers_to') == fid and r.get('prediction_id') == ps['id']
                   and _same_delivery(r.get('delivery'), d_rel) for r in ctx.kb.rows('FINDINGS')):
                continue
            row = {'kind': 'transfer', 'refers_to': fid, 'prediction_id': ps['id'], 'delivery': d_rel,
                   'outcome': outcome, 'target': ps.get('target'), 'task_family': _dig(lock, 'task.family'),
                   'brier': ps.get('brier'), 'base_brier': ps.get('base_brier'), 'hit': ps.get('hit'),
                   'claim_sha256': claim_sha, 'author_agent': 'checker', 'status': None}
            row = kbmod.assign_id(ctx.kb, 'FINDINGS', row)
            kbmod.append_row(ctx.kb, 'FINDINGS', row)
            written.append(row['id'])
            touched.add(fid)
    for fid in sorted(touched):
        trs = [r for r in ctx.kb.rows('FINDINGS') if r.get('kind') == 'transfer' and r.get('refers_to') == fid]
        failed = [r for r in trs if r.get('outcome') == 'failed']
        bb = [(r.get('brier'), r.get('base_brier')) for r in trs if _n(r.get('brier')) and _n(r.get('base_brier'))]
        if len(failed) < 2 or not bb:
            continue
        mb, mbase = math.fsum(x for x, _y in bb) / len(bb), math.fsum(y for _x, y in bb) / len(bb)
        skill = (1.0 - mb / mbase) if mbase else None
        if skill is None or skill >= 0:
            continue
        if any(r.get('kind') == 'narrowing' and r.get('refers_to') == fid and r.get('author_agent') == 'checker'
               for r in ctx.kb.rows('FINDINGS')):
            continue
        cav = '%d failed transfers and negative Brier skill %.4f against the base rate (rows %s)' % (
            len(failed), skill, ','.join(str(r.get('id')) for r in trs))
        row = {'kind': 'narrowing', 'refers_to': fid, 'claim': 'Narrowing of %s: %s' % (fid, cav), 'caveat': cav,
               'delivery': d_rel, 'author_agent': 'checker', 'status': 'narrowed'}
        row = kbmod.assign_id(ctx.kb, 'FINDINGS', row)
        kbmod.append_row(ctx.kb, 'FINDINGS', row)
        written.append(row['id'])
    return written


def cmd_claim(args: Any) -> int:
    """MFC claim <dir>: write CLAIM.json and the headline, append a DELIVERY_LEDGER row."""
    ctx = common.make_ctx(args, delivery=args.dir, stage='publish', options={'invoked_by': 'claim'})
    ok, why = common.v1_status(ctx.delivery)
    if not ok:
        sys.stderr.write('legacy delivery: use retro (%s)\n' % why)
        return common.EXIT_MISSING
    existing = None
    cp = ctx.path('CLAIM.json')
    if cp.is_file():
        existing = common.load_json_file(cp)
        if not isinstance(existing, dict):
            sys.stderr.write('claim REFUSED: the existing CLAIM.json is not an object (its agent-editable fields '
                             'cannot be preserved)\n')
            return common.EXIT_FAIL
    claim, extras = generate_claim(ctx, existing)
    probs = claim_problems(claim)
    text = common.dumps_pretty(claim)
    sha = common.sha256_bytes(text.encode('utf-8'))
    om = claim['objective_metric']
    row = {'delivery': claim['delivery'], 'utc': common.utc_now(), 'legacy': False,
           'task_family': _dig(ctx.lock, 'task.family'), 'level_earned': claim['level_earned'],
           'primary_gate': claim['primary_gate'], 'gate_type': claim['gate_type'], 'worlds_role': claim['worlds_role'],
           'opened_held_E3': extras['opened_held_E3'], 'e3_basis': extras['e3_basis'],
           'objective_metric': {'held': om.get('held_exact_own_eos'), 'old_final': om.get('old_final'),
                                'G': om.get('G')},
           'family_id': claim['family_id'], 'claim_sha256': sha}
    row_probs = kbmod.row_problems('DELIVERY_LEDGER', row)
    if row_probs:
        sys.stderr.write('claim REFUSED: the DELIVERY_LEDGER row would be invalid: %s\n' % '; '.join(row_probs))
        return common.EXIT_FAIL
    if not ctx.kb.path('DELIVERY_LEDGER.jsonl').is_file():
        raise MissingInput('KB ledger %s absent' % kbmod.ledger_rel('DELIVERY_LEDGER'))
    cp.write_text(text, encoding='utf-8')
    rep = _update_report(ctx, claim['headline'])
    kbmod.append_row(ctx.kb, 'DELIVERY_LEDGER', row)
    transfers = _append_transfers(ctx, claim, sha)
    if getattr(args, 'json', False):
        print(json.dumps({'claim': claim, 'claim_sha256': sha, 'delivery_ledger_row': row, 'report': rep,
                          'kb_rows': transfers, 'schema_problems': probs}, indent=2, ensure_ascii=False))
    else:
        print(claim['headline'])
        print('wrote %s (sha256 %s); %s; DELIVERY_LEDGER row appended%s'
              % (ctx.repo_rel_of('CLAIM.json'), sha[:12], rep,
                 ('; FINDINGS rows %s' % ', '.join(transfers)) if transfers else ''))
        for pr in probs:
            sys.stderr.write('CLAIM.json schema problem (CLM-01 will FAIL): %s\n' % pr)
    return common.EXIT_PASS if not probs else common.EXIT_FAIL


# --------------------------------------------------------------------------------------
# Text scanning (CLM-04/06/07/12/16, lint)
# --------------------------------------------------------------------------------------

def _json_strings(obj: Any, path: str = '') -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.append(('%s/%s#key' % (path, k), str(k)))
            out.extend(_json_strings(v, '%s/%s' % (path, k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(_json_strings(v, '%s/%d' % (path, i)))
    elif isinstance(obj, str):
        out.append((path or '/', obj))
    return out


_BLOCK_START_RE = re.compile(r'^(?:[-*+]\s|\d+[.)]\s|>)')
_SINGLE_LINE_RE = re.compile(r'^(?:#{1,6}\s|\|)')


def text_units(text: str, headline_first: bool) -> List[Tuple[int, str, bool]]:
    """Sentences (line number, sentence, is_headline). The first non-empty line is the headline sentence when
    headline_first; paragraphs are joined across wrapped lines; headings, table rows and fenced code lines
    stand alone; list items start a new block."""
    units: List[Tuple[int, str, bool]] = []
    lines = text.splitlines()
    start = 0
    if headline_first:
        for i, ln in enumerate(lines):
            if ln.strip():
                units.append((i + 1, ln.strip(), True))
                start = i + 1
                break
    blocks: List[Tuple[int, str]] = []
    cur: List[str] = []
    cur_start = 0
    fence = False

    def flush():
        if cur:
            blocks.append((cur_start, ' '.join(cur)))
        del cur[:]
    for i in range(start, len(lines)):
        s = lines[i].strip()
        if s.startswith('```') or s.startswith('~~~'):
            flush()
            fence = not fence
            continue
        if fence:
            if s:
                blocks.append((i + 1, s))
            continue
        if not s:
            flush()
            continue
        if _SINGLE_LINE_RE.match(s):
            flush()
            blocks.append((i + 1, s))
            continue
        if _BLOCK_START_RE.match(s):
            flush()
        if not cur:
            cur_start = i + 1
        cur.append(s)
    flush()
    for ln, b in blocks:
        for sent in re.split(r'(?<=[.!?])\s+', b):
            if sent.strip():
                units.append((ln, sent.strip(), False))
    return units


def _doc(label: str, text: str, headline_first: bool) -> dict:
    return {'label': label, 'text': text, 'headline_first': headline_first}


def _file_docs(path: Path, label: str) -> List[dict]:
    """Text documents of a file: markdown/text as-is (first line = headline sentence), JSON string values."""
    raw = path.read_bytes()
    if path.name.endswith('.json') or path.name.endswith('.jsonl'):
        try:
            if path.name.endswith('.jsonl'):
                objs = [common.loads_strict(ln) for ln in raw.decode('utf-8').splitlines() if ln.strip()]
            else:
                objs = [common.loads_strict(raw)]
            docs = []
            for oi, obj in enumerate(objs):
                for jp, s in _json_strings(obj):
                    docs.append(_doc('%s#%s%s' % (label, ('%d' % (oi + 1)) if len(objs) > 1 else '', jp), s, False))
            return docs
        except (InvalidInput, UnicodeDecodeError):
            pass
    return [_doc(label, raw.decode('utf-8', 'replace'), True)]


def _status_summary_files(ctx: common.Ctx) -> List[str]:
    out = []
    for rel in ctx.delivery_files():
        if rel.startswith('checks/') or rel.startswith('runlog/'):
            continue
        base = rel.rsplit('/', 1)[-1]
        if base.upper().startswith(('STATUS', 'SUMMARY')) and base.lower().endswith(('.md', '.txt', '.json')):
            out.append(rel)
    return out


def publish_docs(ctx: common.Ctx, include_notes: bool = True) -> List[dict]:
    """REPORT.md, STATUS*/SUMMARY* files and the CLAIM headline, notes and proposed finding claims: the texts
    CLM-04/06/07 read."""
    docs: List[dict] = []
    cp = ctx.path('CLAIM.json')
    if cp.is_file():
        c = common.load_json_file(cp)
        if isinstance(c, dict):
            if isinstance(c.get('headline'), str):
                docs.append(_doc('CLAIM.json#headline', c['headline'], True))
            if include_notes and _s(c.get('notes')):
                docs.append(_doc('CLAIM.json#notes', c['notes'], False))
            for i, f in enumerate(c.get('findings_proposed') or []):
                if isinstance(f, dict) and _s(f.get('claim')):
                    docs.append(_doc('CLAIM.json#findings_proposed/%d/claim' % i, f['claim'], False))
    if ctx.path('REPORT.md').is_file():
        docs.extend(_file_docs(ctx.path('REPORT.md'), 'REPORT.md'))
    for rel in _status_summary_files(ctx):
        docs.extend(_file_docs(ctx.path(rel), rel))
    return docs


def _lint_docs(ctx: common.Ctx) -> List[dict]:
    docs: List[dict] = []
    for p in ctx.options.get('lint_files') or []:
        docs.extend(_file_docs(Path(p), _label_of(ctx, Path(p))))
    return docs


def _label_of(ctx: common.Ctx, p: Path) -> str:
    try:
        return p.resolve().relative_to(Path(ctx.repo_root).resolve()).as_posix()
    except ValueError:
        return str(p)


def _lint_mode(ctx: common.Ctx) -> bool:
    return ctx.options.get('lint_files') is not None


def _lint_missing(ctx: common.Ctx) -> None:
    miss = ctx.options.get('lint_missing') or []
    if miss:
        raise MissingInput('lint input not found: %s' % ', '.join(miss))


def _gate_tokens(ctx: common.Ctx) -> 're.Pattern':
    key = 'wp5_gate_re'
    if key in ctx._cache:
        return ctx._cache[key]
    ids = set(_rule_ids(ctx))
    lock = _lock_or_none(ctx)
    if lock:
        for g in (lock.get('gates') or []) + (lock.get('guards') or []):
            if isinstance(g, dict) and _s(g.get('id')):
                ids.add(g['id'])
    if ids:
        rx = re.compile(r'(?<![A-Za-z0-9_])(?:%s)(?![A-Za-z0-9_])' % '|'.join(
            re.escape(i) for i in sorted(ids, key=len, reverse=True)))
    else:
        rx = re.compile(r'(?!x)x')
    ctx._cache[key] = rx
    return rx


def soft_word_hits(ctx: common.Ctx, docs: Sequence[dict]) -> Tuple[List[str], List[str]]:
    """(blocking hits, warning hits) of CLM-06."""
    gate_rx = _gate_tokens(ctx)
    block: List[str] = []
    warn: List[str] = []
    for d in docs:
        for ln, sent, is_head in text_units(d['text'], d['headline_first']):
            words = sorted(set(m.group(0).lower() for m in SOFT_RE.finditer(sent)))
            if not words:
                continue
            gate = bool(gate_rx.search(sent) or VERDICT_TOKEN_RE.search(sent) or PRIMARY_GATE_WORD_RE.search(sent)
                        or GATE_WORD_RE.search(sent))
            msg = '%s:%d %r in %s sentence: %s' % (d['label'], ln, ','.join(words),
                                                   'the headline' if is_head else ('a gate' if gate else 'a'),
                                                   sent[:160])
            (block if (is_head or gate) else warn).append(msg)
    return block, warn


def abs_path_hits(docs: Sequence[dict]) -> List[str]:
    out = []
    for d in docs:
        for i, line in enumerate(d['text'].splitlines(), 1):
            for m in ABS_PATH_RE.finditer(line):
                out.append('%s:%d absolute local path %r' % (d['label'], i, line[max(0, m.start() - 10):m.end() + 30]))
    return out


def broken_links(path: Path, root: Path) -> List[str]:
    """Relative markdown links of `path` that do not resolve (outside fenced code)."""
    out = []
    try:
        text = path.read_text(encoding='utf-8', errors='replace')
    except OSError as e:
        return ['%s unreadable: %s' % (path, e)]
    fence = False
    for i, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if s.startswith('```') or s.startswith('~~~'):
            fence = not fence
            continue
        if fence:
            continue
        for m in MD_LINK_RE.finditer(line):
            tgt = m.group(1)
            if tgt.startswith('#') or URL_SCHEME_RE.match(tgt) or tgt.startswith('//'):
                continue
            t = tgt.split('#', 1)[0].split('?', 1)[0]
            if not t:
                continue
            base = root if t.startswith('/') else path.parent
            cand = (base / t.lstrip('/')).resolve()
            try:
                cand.relative_to(root.resolve())
            except ValueError:
                out.append('%s:%d link %r leaves the repo' % (path.name, i, tgt))
                continue
            if not cand.exists():
                out.append('%s:%d link %r does not resolve' % (path.name, i, tgt))
    return out


def _publish_text_files(ctx: common.Ctx) -> List[str]:
    out = []
    ss = set(_status_summary_files(ctx))
    for rel in ctx.delivery_files():
        if any(rel.startswith(s) for s in ('checks/', 'runlog/', 'manifests/')):
            continue
        if rel.startswith('results/') and rel not in ss:
            continue
        if rel.lower().endswith(('.md', '.txt', '.json')):
            out.append(rel)
    return out


# --------------------------------------------------------------------------------------
# Checks CLM-01..16
# --------------------------------------------------------------------------------------

@check('CLM-01')
def check_clm_01(ctx: common.Ctx) -> Result:
    c = ctx.read_json('CLAIM.json')
    probs = claim_problems(c)
    return _finish('CLM-01', probs, [], 'CLAIM.json schema and enums valid')


@check('CLM-02')
def check_clm_02(ctx: common.Ctx) -> Result:
    if ctx.delivery is not None and not common.is_v1_delivery(ctx.delivery):
        if ctx.stage == 'pointer':
            # SPEC 7.4 PTR-07: a pointer may target a legacy delivery (owner UD + RETRO annotation); a legacy
            # directory has no generated CLAIM by definition (SPEC 1), so there is nothing to regenerate
            return Result('CLM-02', NOT_APPLICABLE, 'legacy pointer target without a generated CLAIM.json: PTR-07 '
                          'governs it')
        raise MissingInput('legacy target has no generated CLAIM.json (PTR-07 governs legacy pointers)')
    cur = _read_claim(ctx)
    fresh, _extras = generate_claim(ctx, cur)
    diffs = []
    for k in sorted(set(cur) | set(fresh)):
        if k in CLM02_EXEMPT:
            continue
        if k not in cur or k not in fresh or _canon(cur.get(k)) != _canon(fresh.get(k)):
            diffs.append('%s: CLAIM %s vs regenerated %s' % (k, json.dumps(cur.get(k), sort_keys=True)[:120],
                                                             json.dumps(fresh.get(k), sort_keys=True)[:120]))
    if diffs:
        return Result('CLM-02', FAIL, 'CLAIM.json differs from a fresh regeneration (run MFC claim; agents edit only '
                      'findings_proposed, findings_none_reason and notes): %s' % '; '.join(diffs[:5]),
                      evidence=diffs[:40])
    return Result('CLM-02', PASS, 'CLAIM.json equals a fresh regeneration: %s' % fresh['headline'])


@check('CLM-03')
def check_clm_03(ctx: common.Ctx) -> Result:
    cur = _read_claim(ctx)
    fresh, extras = generate_claim(ctx, cur)
    probs = []
    for k in ('level_earned', 'gate_type', 'reuse_specificity'):
        if cur.get(k) != fresh.get(k):
            probs.append('%s %r but the level rules give %r (%s)' % (k, cur.get(k), fresh.get(k),
                                                                   '; '.join(extras['level_reasons'])))
    want_ii = 'integrated_intervention' in fresh['labels']
    has_ii = 'integrated_intervention' in (cur.get('labels') or [])
    if want_ii != has_ii:
        probs.append('label integrated_intervention %s (lineage.changes_declared has %d entries)' % (
            'missing' if want_ii else 'not justified', len(_dig(ctx.lock, 'lineage.changes_declared') or [])))
    note = ''
    if common.adoption_intended(ctx) and fresh['level_earned'] == 'E3':
        certifiable, why = udi_certifiable(ctx)
        if not certifiable:
            note = '; E3 promotion not certifiable: %s (reuse_specificity %s, role %s)' % (
                why, fresh['reuse_specificity'], fresh['worlds_role'])
        else:
            promo = fresh['reuse_specificity'] == 'udi_pass' and fresh['worlds_role'] == 'SEALED_CONFIRM'
            note = '; E3 promotion %s (needs UDI LCB > 0 and SEALED_CONFIRM; reuse %s, role %s)' % (
                'earned' if promo else 'NOT earned', fresh['reuse_specificity'], fresh['worlds_role'])
    return _finish('CLM-03', probs, [], 'level %s: %s%s' % (fresh['level_earned'], '; '.join(extras['level_reasons']),
                                                            note))


def confirm_skeleton(text: str) -> str:
    """The text CLM-04 matches on: NFKC, without format characters (Cf) and combining marks (Mn, Me), with the
    confusable Cyrillic/Greek letters of 'confirm' folded to Latin. Line breaks are kept."""
    t = unicodedata.normalize('NFKC', text)
    t = ''.join(ch for ch in unicodedata.normalize('NFKD', t) if unicodedata.category(ch) not in ('Cf', 'Mn', 'Me'))
    return unicodedata.normalize('NFKC', t).translate(_CONFUSABLE_FOLD)


_HTML_COMMENT_RE = re.compile(r'<!--.*?(?:-->|\Z)', re.S)
_HTML_TAG_RE = re.compile(r'</?[A-Za-z][A-Za-z0-9:-]*(?:\s[^<>]*)?/?>')
_MD_MARK_RE = re.compile(r'\*+|`+|~~|__')
# Markdown inline links and images render as their text (round-1 repair, verifier probe D4: '[con](#)firmed',
# 'con[]()firmed'); reference links '[text][ref]' likewise. Nested brackets in the text are not markdown links.
_MD_INLINE_LINK_RE = re.compile(r'!?\[([^\[\]\n]*)\]\([^()\n]*(?:\([^()\n]*\)[^()\n]*)*\)')
_MD_REF_LINK_RE = re.compile(r'!?\[([^\[\]\n]*)\]\[[^\[\]\n]*\]')


def markup_skeleton(text: str, links: bool = False) -> str:
    """Rendered text of markdown/HTML (SPEC 20.12; verifier: CLM-04 markup bypass): HTML entities decoded (repeatedly,
    so &amp;#8203; counts too), HTML comments and tags removed, markdown emphasis and code markers (*, **, `, ~~,
    __) removed. Single underscores stay: they are identifier characters (SEALED_CONFIRM) and never intraword
    emphasis in markdown. With links=True, markdown links and images are also replaced by their bracketed text
    (the link target is not rendered). Then confirm_skeleton (NFKC, format characters, combining marks,
    confusables)."""
    t = text
    for _ in range(3):
        u = html.unescape(t)
        if u == t:
            break
        t = u
    t = _HTML_COMMENT_RE.sub('', t)
    t = _HTML_TAG_RE.sub('', t)
    if links:
        for _ in range(3):
            u = _MD_REF_LINK_RE.sub(r'\1', _MD_INLINE_LINK_RE.sub(r'\1', t))
            if u == t:
                break
            t = u
    t = _MD_MARK_RE.sub('', t)
    return confirm_skeleton(t)


def _confirm_line_tokens(sk: str) -> List[str]:
    out: List[str] = []
    seen: Set[Tuple[int, int]] = set()
    for m in CONFIRM_CORE_RE.finditer(sk):
        a, b = m.start(), m.end()
        while a > 0 and sk[a - 1] in _IDENT_CHARS:
            a -= 1
        while b < len(sk) and sk[b] in _IDENT_CHARS:
            b += 1
        if (a, b) in seen:
            continue
        seen.add((a, b))
        tok = sk[a:b]
        if tok in CONFIRM_ALLOWED_TOKENS:
            continue
        out.append(tok)
    return out


def confirm_tokens(text: str) -> List[Tuple[int, str]]:
    """(line number, identifier token) of every confirm token in `text` that is not an allowed kit name. The text is
    read as rendered (markup_skeleton) first, once with markdown link targets kept (a target named *confirm* still
    counts) and once with links rendered as their text ('[con](#)firmed' is 'confirmed'); a token the second
    reading adds to a line is reported too. Shared with retro.py (SPEC 20.12)."""
    out: List[Tuple[int, str]] = []
    kept = markup_skeleton(text).splitlines()
    rendered = markup_skeleton(text, links=True).splitlines()
    for i, line in enumerate(kept, 1):
        toks = _confirm_line_tokens(confirm_skeleton(line))
        out.extend((i, t) for t in toks)
        if i <= len(rendered) and rendered[i - 1] != line:
            have = list(toks)
            for t in _confirm_line_tokens(confirm_skeleton(rendered[i - 1])):
                if t in have:
                    have.remove(t)
                else:
                    out.append((i, t))
    return out


def _confirm_hits(docs: Sequence[dict]) -> List[str]:
    out = []
    for d in docs:
        for i, tok in confirm_tokens(d['text']):
            out.append('%s:%d %r' % (d['label'], i, tok))
    return out


@check('CLM-04')
def check_clm_04(ctx: common.Ctx) -> Result:
    hits = _confirm_hits(publish_docs(ctx))
    if not hits:
        return Result('CLM-04', NOT_APPLICABLE, 'no confirm* tokens in REPORT, STATUS, SUMMARY, the headline or notes')
    fresh, _x = generate_claim(ctx, None)
    probs = []
    if fresh['worlds_role'] != 'SEALED_CONFIRM':
        probs.append('confirm wording %s but the computed worlds_role is %s' % (', '.join(hits[:5]), fresh['worlds_role']))
    else:
        for rid in ('SEAL-01', 'SEAL-02', 'SEAL-03', 'SEAL-04', 'SEAL-05', 'WLD-01'):
            st, det = _check_status(ctx, rid)
            if st != PASS:
                probs.append('confirm wording requires %s PASS (got %s: %s)' % (rid, st, det[:100]))
    return _finish('CLM-04', probs, [], 'confirm wording on SEALED_CONFIRM with SEAL-* and WLD-01 PASS',
                   data={'hits': hits[:20]})


@check('CLM-05')
def check_clm_05(ctx: common.Ctx) -> Result:
    c = _read_claim(ctx)
    p = ctx.path('REPORT.md')
    if not p.is_file():
        raise MissingInput('REPORT.md absent (its first non-empty line must equal CLAIM.headline)')
    first = next((ln.strip() for ln in p.read_text(encoding='utf-8', errors='replace').splitlines() if ln.strip()), '')
    if first != c.get('headline'):
        return Result('CLM-05', FAIL, 'first non-empty line of REPORT.md %r differs from CLAIM.headline %r'
                      % (first[:160], str(c.get('headline'))[:160]))
    return Result('CLM-05', PASS, 'REPORT.md first line equals the headline')


@check('CLM-06')
def check_clm_06(ctx: common.Ctx) -> Result:
    if _lint_mode(ctx):
        _lint_missing(ctx)
        docs = _lint_docs(ctx)
    else:
        docs = publish_docs(ctx)
    block, warn = soft_word_hits(ctx, docs)
    return _finish('CLM-06', block, [], 'no soft words (%d text units)' % sum(
        len(text_units(d['text'], d['headline_first'])) for d in docs), warn=warn)


def inheritance_demonstrated(ctx: common.Ctx) -> Tuple[bool, str]:
    parent = _dig(ctx.lock, 'lineage.parent')
    if not _s(parent):
        return False, 'no lineage.parent'
    pdir = parent.split('@', 1)[0].strip().strip('/')
    try:
        prel = common.norm_rel(pdir)
    except InvalidInput as e:
        return False, str(e)
    pd = Path(ctx.repo_root) / prel
    if not common.is_v1_delivery(pd) or not (pd / 'SOURCE_LOCK.json').is_file():
        return False, 'parent %s is not a v1 delivery with a SOURCE_LOCK (inheritance cannot be computed)' % prel
    nb = dict(common.closure_jaccard_neighbors(ctx))
    j = nb.get(prel)
    if j is None or j < 0.5:
        return False, 'native-closure Jaccard with parent %s is %s (< 0.5)' % (prel, j)
    return True, 'parent %s shares the native closure (Jaccard %.3f)' % (prel, j)


@check('CLM-07')
def check_clm_07(ctx: common.Ctx) -> Result:
    hits = []
    for d in publish_docs(ctx):
        for i, line in enumerate(d['text'].splitlines(), 1):
            for m in LINEAGE_RE.finditer(line):
                hits.append('%s:%d %r' % (d['label'], i, m.group(0)))
    if not hits:
        return Result('CLM-07', NOT_APPLICABLE, "no native, core, inherited or 'same fly' wording")
    ok, why = inheritance_demonstrated(ctx)
    if ok:
        return Result('CLM-07', PASS, 'lineage wording backed: %s' % why, data={'hits': hits[:20]})
    return Result('CLM-07', FAIL, 'lineage wording %s needs a lineage_parent with inheritance demonstrated: %s'
                  % (', '.join(hits[:5]), why), evidence=hits[:40])


@check('CLM-08')
def check_clm_08(ctx: common.Ctx) -> Result:
    posthoc, off = posthoc_info(ctx)
    if off is None:
        return Result('CLM-08', NOT_APPLICABLE, 'no official run: no file was added after the official lock')
    c = _read_claim(ctx)
    probs = []
    listed = c.get('posthoc_files') if isinstance(c.get('posthoc_files'), list) else None
    if listed is None or sorted(listed) != posthoc:
        probs.append('CLAIM.posthoc_files %s differs from the files added after the official lock %s'
                     % (json.dumps(listed)[:160], json.dumps(posthoc)[:160]))
    pats = _verdict_input_patterns(ctx.lock)
    bad = [f for f in posthoc if any(common.glob_match(f, p) for p in pats)]
    if bad:
        probs.append('files added after the official lock are VERDICT inputs: %s' % ', '.join(bad[:10]))
    return _finish('CLM-08', probs, [], '%d POSTHOC file(s) after the official lock at %s, none a VERDICT input'
                   % (len(posthoc), str(off.get('git_head'))[:12]), data={'posthoc_files': posthoc})


@check('CLM-09')
def check_clm_09(ctx: common.Ctx) -> Result:
    cur = _read_claim(ctx)
    fresh, extras = generate_claim(ctx, cur)
    probs = []
    if cur.get('worlds_role') != fresh['worlds_role']:
        probs.append('worlds_role %r but the computed role is %s (%s)' % (cur.get('worlds_role'), fresh['worlds_role'],
                                                                         '; '.join(extras['worlds_role_reasons'])))
    if cur.get('n_worlds') != fresh['n_worlds']:
        probs.append('n_worlds %r but %d worlds were evaluated' % (cur.get('n_worlds'), fresh['n_worlds']))
    return _finish('CLM-09', probs, [], 'worlds_role %s: %s' % (fresh['worlds_role'],
                                                               '; '.join(extras['worlds_role_reasons'])))


@check('CLM-10')
def check_clm_10(ctx: common.Ctx) -> Result:
    cur = _read_claim(ctx)
    fresh, _x = generate_claim(ctx, cur)
    lock = ctx.lock
    preds = [p for p in (lock.get('predictions') or []) if isinstance(p, dict)]
    ids = [p.get('id') for p in preds]
    scored = {ps.get('id'): ps for ps in (cur.get('predictions_scored') or []) if isinstance(ps, dict)}
    probs: List[str] = []
    missing: List[str] = []
    for pid in ids:
        if pid not in scored:
            probs.append('lock prediction %s has no predictions_scored entry' % pid)
    extra = sorted(set(scored) - set(ids), key=str)
    if extra:
        probs.append('predictions_scored has entries for unregistered predictions %s' % ', '.join(map(str, extra)))
    fresh_by = {ps['id']: ps for ps in fresh['predictions_scored']}
    for pid, ps in scored.items():
        if pid in fresh_by and _canon(ps) != _canon(fresh_by[pid]):
            probs.append('predictions_scored %s differs from the recomputed score %s' % (pid, json.dumps(fresh_by[pid])[:160]))
    # predictions registered in any committed lock version are never deleted
    rel = ctx.repo_rel_of('ENDPOINT_LOCK.json')
    try:
        ever: Dict[str, str] = {}
        for c in ctx.git.log([rel]):
            data = ctx.git.file_at(c.sha, rel)
            if data is None:
                continue
            try:
                old = common.loads_strict(data)
            except InvalidInput:
                continue
            for p in (old.get('predictions') or []) if isinstance(old, dict) else []:
                if isinstance(p, dict) and _s(p.get('id')):
                    ever.setdefault(p['id'], c.sha[:12])
        gone = sorted(set(ever) - set(ids))
        for pid in gone:
            probs.append('prediction %s registered at commit %s was deleted from the lock' % (pid, ever[pid]))
    except MissingInput as e:
        missing.append('lock history unavailable, deleted predictions cannot be ruled out: %s' % e)
    return _finish('CLM-10', probs, missing, '%d lock prediction(s) scored (%s)' % (
        len(ids), ', '.join('%s %s' % (k, v.get('status')) for k, v in sorted(fresh_by.items(), key=lambda kv: str(kv[0])))))


def _source_ok(ctx: common.Ctx, src: Any) -> Optional[str]:
    if not (isinstance(src, str) and kbmod.STRICT_SOURCE_RE.match(src.strip())):
        return 'source %r is not path:line[@sha]' % (src,)
    path, rest = src.strip().split(':', 1)
    line = int(re.match(r'\d+', rest).group(0))
    for base in (Path(ctx.delivery) if ctx.delivery else None, Path(ctx.repo_root)):
        if base is None:
            continue
        try:
            p = base / common.norm_rel(path)
        except InvalidInput as e:
            return str(e)
        if p.is_file():
            n = len(p.read_bytes().splitlines())
            if line < 1 or line > n:
                return 'source %s: line %d outside 1..%d' % (path, line, n)
            return None
    return 'source path %s does not exist (delivery- or repo-relative)' % path


@check('CLM-11')
def check_clm_11(ctx: common.Ctx) -> Result:
    c = _read_claim(ctx)
    fp = c.get('findings_proposed')
    probs = []
    if not isinstance(fp, list):
        return Result('CLM-11', FAIL, 'findings_proposed must be a list')
    if len(fp) > 3:
        probs.append('%d findings_proposed (0-3 allowed)' % len(fp))
    if not fp and not _s(c.get('findings_none_reason')):
        probs.append('zero findings_proposed requires findings_none_reason')
    for i, f in enumerate(fp):
        if not isinstance(f, dict):
            probs.append('findings_proposed[%d] is not an object' % i)
            continue
        if not _s(f.get('claim')):
            probs.append('findings_proposed[%d] lacks a claim' % i)
        cond = f.get('conditions') if isinstance(f.get('conditions'), dict) else {}
        if not _s(cond.get('best_shortcut')):
            probs.append('findings_proposed[%d] lacks conditions.best_shortcut' % i)
        why = _source_ok(ctx, f.get('source'))
        if why:
            probs.append('findings_proposed[%d]: %s' % (i, why))
    return _finish('CLM-11', probs, [], '%d finding(s) proposed%s' % (
        len(fp), '' if fp else ' (reason: %s)' % str(c.get('findings_none_reason'))[:80]))


@check('CLM-12')
def check_clm_12(ctx: common.Ctx) -> Result:
    root = Path(ctx.repo_root)
    if _lint_mode(ctx):
        _lint_missing(ctx)
        files = [Path(p) for p in ctx.options.get('lint_files') or []]
        labels = [_label_of(ctx, p) for p in files]
    else:
        rels = _publish_text_files(ctx)
        files = [ctx.path(r) for r in rels]
        labels = rels
    probs = []
    for p, lab in zip(files, labels):
        raw = p.read_bytes().decode('utf-8', 'replace')
        probs.extend(abs_path_hits([_doc(lab, raw, False)]))
        if p.name.lower().endswith('.md'):
            probs.extend('%s: %s' % (lab, x) for x in broken_links(p, root))
    return _finish('CLM-12', probs, [], 'no absolute local paths; relative links resolve (%d files)' % len(files))


@check('CLM-13')
def check_clm_13(ctx: common.Ctx) -> Result:
    cp = ctx.path('CLAIM.json')
    if not cp.is_file():
        raise MissingInput('CLAIM.json absent (run MFC claim)')
    sha = common.sha256_file(cp)
    c = _read_claim(ctx)
    rows = [r for r in ctx.kb.rows('DELIVERY_LEDGER') if _same_delivery(r.get('delivery'), ctx.delivery_rel)
            and r.get('legacy') is False]
    hit = [r for r in rows if r.get('claim_sha256') == sha]
    if not hit:
        return Result('CLM-13', FAIL, 'no DELIVERY_LEDGER row links CLAIM.json %s (%d row(s) for this delivery): run '
                      'MFC claim after editing findings' % (sha[:12], len(rows)))
    r = hit[-1]
    probs = ['ledger %s %r differs from CLAIM %r' % (k, r.get(k), c.get(k))
             for k in ('level_earned', 'primary_gate', 'gate_type', 'worlds_role', 'family_id') if r.get(k) != c.get(k)]
    return _finish('CLM-13', probs, [], 'DELIVERY_LEDGER row (utc %s) links CLAIM.json %s' % (r.get('utc'), sha[:12]))


def _gate_report_files(ctx: common.Ctx) -> List[str]:
    out = []
    for rel in ctx.delivery_files(refresh=True):
        if rel.startswith('checks/'):
            name = rel.rsplit('/', 1)[-1]
            if (name.startswith('GATE_') or name.startswith('DRY_')) and name.endswith('.json'):
                out.append(rel)
    return out


@check('CLM-14')
def check_clm_14(ctx: common.Ctx) -> Result:
    reps = _gate_report_files(ctx)
    if not reps:
        return Result('CLM-14', FAIL, 'no gate reports under checks/: the lock gate must have run and written one')
    probs: List[str] = []
    missing: List[str] = []
    shas: Dict[str, Optional[str]] = {}
    for rel in reps:
        try:
            rep = ctx.read_json(rel)
        except InvalidInput as e:
            probs.append('%s unreadable: %s' % (rel, e))
            continue
        if not isinstance(rep, dict):
            probs.append('%s is not an object' % rel)
            continue
        bad = common._report_consistent(rep)
        if bad:
            probs.append('%s is not internally consistent (%s)' % (rel, bad))
            continue
        if rep.get('delivery') != ctx.delivery_rel:
            probs.append('%s names delivery %r' % (rel, rep.get('delivery')))
            continue
        if ctx.git_mode == 'real' and rep.get('git_mode') != 'real':
            probs.append('%s was written in %r git mode (corpus reports never count in a real repo)' % (rel, rep.get('git_mode')))
            continue
        head = rep.get('git_head')
        if not _s(head) or (ctx.git_mode == 'real' and not SHA_RE.match(head)):
            probs.append('%s records git_head %r' % (rel, head))
            continue
        if head not in shas:
            try:
                shas[head] = common.checker_sha256_at(ctx.git, head)
            except MissingInput as e:
                missing.append('%s: committed checker at %s unavailable (%s)' % (rel, head[:12], e))
                continue
        want = shas[head]
        got = rep.get('checker_sha256')
        if want is None and ctx.git_mode == 'timeline':
            want = common.EMPTY_KIT_SHA256  # a fixture commit with no kit tree: the committed checker is empty
        if want is not None and got == want:
            continue
        if ctx.git_mode == 'timeline' and got == common.checker_sha256():
            continue  # corpus recursion base: a report written in this run by the running (outer-verified) checker
        if want is None and ctx.git_mode == 'real':
            probs.append('%s: no checker is committed at %s' % (rel, head[:12]))
            continue
        probs.append('%s checker_sha256 %s differs from the %s' % (
            rel, str(got)[:12], ('committed checker at %s (%s)' % (head[:12], want[:12])) if want else
            'running checker (the fixture commits no kit at %s)' % head[:12]))
    return _finish('CLM-14', probs, missing, '%d gate report(s) carry the committed checker sha' % len(reps))


def _written_in_run(ctx: common.Ctx, rel: str) -> bool:
    try:
        rep = ctx.read_json(rel)
    except (MissingInput, InvalidInput):
        return False
    return (isinstance(rep, dict) and common._report_consistent(rep) is None
            and rep.get('checker_sha256') == common.checker_sha256() and rep.get('git_mode') == 'timeline')


def _committed_equal(ctx: common.Ctx, rel: str) -> bool:
    data = ctx.git.file_at(ctx.git.head(), ctx.repo_rel_of(rel))
    p = ctx.path(rel)
    return data is not None and p.is_file() and data == p.read_bytes()


def report_history_problems(ctx: common.Ctx) -> List[str]:
    """The gate, dry and canary reports under checks/ are evidence (SPEC 7.4 CLM-15 'GATE reports'; 3.4 never
    delete failure markers): every report ever committed must still be committed at HEAD and present in the
    worktree, every ATTEMPTS.gate_report must exist, and every CYCLES computed.latest_gate must still hold the
    bytes its sha256 recorded. A shallow clone raises MissingInput."""
    probs: List[str] = []
    pre = ctx.repo_rel_of('checks')
    dpre = ctx.delivery_rel.rstrip('/') + '/'
    head = ctx.git.head()
    seen: Set[str] = set()
    for c in ctx.git.log([pre]):
        for p in c.paths:
            if p.startswith(pre + '/') and p.endswith('.json') and p not in seen and ctx.git.file_at(c.sha, p) is not None:
                seen.add(p)
    for p in sorted(seen):
        drel = p[len(dpre):] if p.startswith(dpre) else p
        if ctx.git.file_at(head, p) is None or not ctx.path(drel).is_file():
            probs.append('committed report %s was deleted (gate, dry and canary reports are never removed)' % drel)
    for a in common.read_attempts(ctx):
        gr = a.get('gate_report')
        if isinstance(gr, str) and gr and not ctx.path(gr).is_file():
            probs.append('ATTEMPTS %s gate_report %s is absent' % (a.get('attempt_id'), gr))
    for i, r in enumerate(common.read_cycles(ctx), 1):
        lg = (r.get('computed') or {}).get('latest_gate') if isinstance(r.get('computed'), dict) else None
        if not isinstance(lg, dict) or not isinstance(lg.get('path'), str) or not lg.get('path'):
            continue
        rp = ctx.path(lg['path'])
        if not rp.is_file():
            probs.append('cycle %s latest_gate %s is absent' % (r.get('k', i), lg['path']))
        elif isinstance(lg.get('sha256'), str) and common.sha256_file(rp) != lg['sha256']:
            probs.append('cycle %s latest_gate %s no longer holds the bytes the close recorded' % (r.get('k', i), lg['path']))
    return probs


@check('CLM-15')
def check_clm_15(ctx: common.Ctx) -> Result:
    lock = ctx.lock
    probs: List[str] = report_history_problems(ctx)
    files: List[str] = ['ENDPOINT_LOCK.json', 'SOURCE_LOCK.json']
    srcs = ctx.row_sources('results/official/OUTPUTS.jsonl')
    official_run = bool(srcs or _official_attempts(ctx) or ctx.path('results/official/VERDICT.json').is_file())
    if not official_run:
        # an honest exit before any official run (e.g. CANNOT_TEST at cycle 0): there are no official OUTPUTS,
        # receipts or GATE_official to publish; the subset is the locks that exist, every gate report and the
        # admission/testbed OUTPUTS the claim rests on
        files = ['ENDPOINT_LOCK.json'] + (['SOURCE_LOCK.json'] if ctx.path('SOURCE_LOCK.json').is_file() else [])
        for _logical, kind, ref, _gz in ctx.row_sources('results/admission/OUTPUTS.jsonl'):
            f = ref if kind == 'file' else ref[0]
            if f not in files:
                files.append(f)
        reps = _gate_report_files(ctx)
        files.extend(reps)
        return _subset_result(ctx, files, reps, probs, 'no official run: ')
    if not srcs:
        probs.append('no official OUTPUTS (results/official/OUTPUTS.jsonl[.gz] or RESULT_BUNDLE*.zip)')
    for logical, kind, ref, _gz in srcs:
        f = ref if kind == 'file' else ref[0]
        if f not in files:
            files.append(f)
    off_worlds = set(str(w) for w in (_dig(lock, 'worlds.official.ids') or []))
    pools = set([_dig(lock, 'primary.pool')] + [g.get('pool') for g in (lock.get('guards') or []) if isinstance(g, dict)])
    pools.discard(None)
    seen: Set[Tuple[str, str]] = set()
    for row in ctx.iter_rows('results/official/OUTPUTS.jsonl'):
        if isinstance(row.get('world'), str):
            off_worlds.add(row['world'])
            seen.add((row['world'], row.get('pool')))
    for w in sorted(off_worlds):
        for pool in sorted(pools):
            if (w, pool) not in seen:
                probs.append('official OUTPUTS lack counted pool %s for world %s' % (pool, w))
    arms = [a.get('id') for a in (lock.get('arms') or []) if isinstance(a, dict) and _s(a.get('id'))]
    tdir = _dig(lock, 'split.taught_manifest') or 'manifests/TAUGHT'
    for w in sorted(off_worlds):
        for a in arms:
            rc = 'results/receipts/%s/%s.json' % (w, a)
            files.append(rc)
            t = '%s/%s/%s.jsonl' % (tdir.rstrip('/'), w, a)
            files.append(t + '.gz' if (not ctx.path(t).is_file() and ctx.path(t + '.gz').is_file()) else t)
    # the runtime closures of the official attempts (RUN-03) and any determinism-replay rows of a resumed official
    # run (RUN-04) are part of what a reader needs to verify the official histories
    for a in _official_attempts(ctx):
        rc = 'runlog/RUNTIME_CLOSURE_%s.json' % a.get('attempt_id')
        if any(isinstance(e, dict) and e.get('path') == rc for e in (a.get('produced_paths') or [])) and rc not in files:
            files.append(rc)
    for _logical, kind, ref, _gz in ctx.row_sources(common.REPLAY_OUTPUTS):
        f = ref if kind == 'file' else ref[0]
        if f not in files:
            files.append(f)
    reps = _gate_report_files(ctx)
    if not any(r.rsplit('/', 1)[-1].startswith('GATE_official') for r in reps):
        probs.append('no GATE_official report in checks/')
    files.extend(reps)
    return _subset_result(ctx, files, reps, probs, '')


def _subset_result(ctx: common.Ctx, files: List[str], reps: List[str], probs: List[str], prefix: str) -> Result:
    total = 0
    in_run = []
    for f in files:
        p = ctx.path(f)
        if not p.is_file():
            probs.append('%s absent' % f)
            continue
        total += p.stat().st_size
        if not _committed_equal(ctx, f):
            if ctx.git_mode == 'timeline' and f in reps and _written_in_run(ctx, f):
                in_run.append(f)  # corpus recursion base: written by the running checker during this selftest
                continue
            probs.append('%s is not committed as is' % f)
    cap = ctx.kb.constraints().get('verification_subset_max_mb')
    if not _n(cap):
        raise MissingInput('CONSTRAINTS.verification_subset_max_mb missing')
    if total > float(cap) * 1024 * 1024:
        probs.append('verification subset %.1f MB exceeds the %s MB cap: mode digest_plus_sample (per-(world, arm) '
                     'digests plus a reviewer-seeded full-row sample) is not implemented in checker %s'
                     % (total / 1048576.0, cap, common.CHECKER_VERSION))
    return _finish('CLM-15', probs, [], '%sverification subset committed: %d files, %.2f MB (cap %s MB)%s'
                   % (prefix, len(files), total / 1048576.0, cap,
                      ('; %d gate report(s) written in this corpus run' % len(in_run)) if in_run else ''))


def prior_number_hits(docs: Sequence[dict]) -> List[str]:
    out = []
    for d in docs:
        for ln, sent, _h in text_units(d['text'], d['headline_first']):
            if NUMBER_RE.search(sent) and PRIOR_WORK_RE.search(sent) and not CITATION_RE.search(sent):
                out.append('%s:%d %s' % (d['label'], ln, sent[:160]))
    return out


@check('CLM-16')
def check_clm_16(ctx: common.Ctx) -> Result:
    if _lint_mode(ctx):
        _lint_missing(ctx)
        docs = _lint_docs(ctx)
    else:
        docs = []
        for rel in _publish_text_files(ctx):
            if rel.lower().endswith('.md'):
                docs.extend(_file_docs(ctx.path(rel), rel))
    if not any(PRIOR_WORK_RE.search(d['text']) and NUMBER_RE.search(d['text']) for d in docs):
        return Result('CLM-16', NOT_APPLICABLE, 'no numbers about prior work in the scanned texts')
    hits = prior_number_hits(docs)
    return _finish('CLM-16', hits, [], 'numbers about prior work cite an F-id or catalog id')


# --------------------------------------------------------------------------------------
# CLM-17 (structural, enforced) and CLM-17P (free-prose lint aid, advisory): overclaim in agent-written text
# (SPEC 20.13 as amended by 20.20; round 1, INC-HB1-C2, FM-22)
# --------------------------------------------------------------------------------------
# The generated fields (headline, primary_gate, level_earned) are checked by CLM-01/02/05. Round-1 repair part 3
# (verifier: guessing overclaims in free prose with regexes both misses paraphrases and false-alarms on honest text)
# splits the check by mechanism:
#   CLM-17 (enforced) judges only text with a closed vocabulary:
#     (a) CLAIM.notes, CLAIM.findings_none_reason and every string leaf of each findings_proposed entry name no
#         E-level token above level_earned, in any phrasing, negated or not (LEVEL_TOKEN_RE read as rendered:
#         markup_skeleton, entities, look-alike letters, 'E 3', 'E-three', 'E-III', 'evidence level 3', 'the third
#         evidence level', 'level 3 of the evidence ladder'); the attempted level lives only in the generated
#         level_attempted. findings_proposed[].level, when present, is a level <= level_earned.
#     (b) machine-readable status surfaces: the REPORT.md body, STATUS*/SUMMARY*/MEMO* anywhere in the delivery and
#         NOTES*/ANALYSIS*/README*/RESULTS* outside results/ and manifests/. Only their key/value shapes are read
#         (status_pairs): 'key: value' / 'key = value' segments, short 'key - value' segments, markdown table cells
#         (row key and column header), one-line 'question? answer' pairs and JSON leaves. A key that names the level
#         or the gate must carry the CLAIM value or a negation; an E-level above level_earned in a key with an
#         affirmative value, or as a short unqualified value, FAILs (judge_status_pair).
#   CLM-17P (advisory: WARN, never FAIL) reads free prose (prose_flags). Since round-1 repair part 4 its level side is
#   a token rule, not a vocabulary: every clause of a non-question sentence that names an E-level above level_earned
#   without a qualifier is flagged, whatever its verb (a token on a design or process noun is left out). Its gate
#   side (a met primary, rule, criterion, endpoint or objective; a met guard) still needs a MET word. Every WARN row
#   lists the flagged sentence; at T2, REV-09 makes the reviewer adjudicate each one
#   (REVIEW.fm22_prose_adjudications). Free-prose overclaim is a judgment check with a lint aid, not a mechanical
#   guarantee: a sentence the lint does not flag publishes unless the reviewer adds it.

# Look-alike capital/small E (Cyrillic Ie, Greek Epsilon, Latin open E); NFKC already maps fullwidth, mathematical
# and script letters and digits.
_LEVEL_FOLD = {0x0415: 'E', 0x0435: 'e', 0x0395: 'E', 0x03B5: 'e', 0x0190: 'E', 0x025B: 'e'}
LEVEL_WORDS = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
               'zeroth': 0, 'first': 1, 'second': 2, 'third': 3, 'fourth': 4, 'i': 1, 'ii': 2, 'iii': 3, 'iv': 4}
_LEVEL_SEP = '[  \\-‐-―]?'
_LEVEL_DASH = '[\\-‐-―]'
_LEVEL_NUM = r'(?:([0-4])(?![0-9A-Za-z])|(zero|one|two|three|four)\b)'
_LEVEL_ORD = r'(zeroth|first|second|third|fourth)'
# Round-1 repair part 4 (verifier non-blocking: token gaps): Roman numerals after a dash ('E-III', and 'e-iii' in a
# normalised key), 'evidence tier/rung/grade 3', an ordinal before 'evidence level' ('the third evidence level') and a
# level or ordinal rung of 'the (evidence) ladder' ('level 3 of the evidence ladder', 'the third rung of the ladder')
# are level tokens too.
LEVEL_TOKEN_RE = re.compile(
    r'(?<![A-Za-z0-9])(?:E%(sep)s%(num)s'
    r'|(?-i:[Ee]%(dash)s(I{1,3}|IV|i{1,3}|iv))(?![0-9A-Za-z])'
    r'|(?:evidence|E)%(sep)s(?:level|tier|rung|grade)\s+%(num)s'
    r'|%(ord)s\s+(?:evidence|E)%(sep)s(?:level|tier|rung|grade)\b'
    r'|(?:level|tier|rung|step|grade)\s+%(num)s\s+(?:of|on|in)\s+the\s+(?:evidence|E)?%(sep)sladder\b'
    r'|%(ord)s\s+(?:level|tier|rung|step|grade)\s+(?:of|on|in)\s+the\s+(?:evidence|E)?%(sep)sladder\b)'
    % {'sep': _LEVEL_SEP, 'dash': _LEVEL_DASH, 'num': _LEVEL_NUM, 'ord': _LEVEL_ORD}, re.I)
CLM17_ACHIEVE_RE = re.compile(
    r'\b(?:reach(?:ed|es|ing)?|achiev(?:e|ed|es|ing)|earn(?:ed|s|ing)?|attain(?:ed|s|ing|ment)?|met|meets?|'
    r'meeting|demonstrat(?:e|ed|es|ing)|establish(?:ed|es|ing)?|clear(?:ed|s|ing)?|pass(?:ed|es|ing)?|'
    r'qualif(?:y|ies|ied|ying)|secur(?:e|ed|es)|obtain(?:ed|s)?|deliver(?:ed|s)?|supported|warranted|substantiated|'
    r'proven|hit(?:s|ting)?|exceed(?:s|ed|ing)?|surpass(?:es|ed|ing)?|cross(?:es|ed|ing)?|unlock(?:s|ed|ing)?|'
    r'grant(?:s|ed)?|accomplish(?:es|ed|ing)?|wins?|won|beat(?:s|en)?)\b'
    r'|\b(?:ma(?:ke|kes|de|king)\s+it|g(?:et|ets|ot|otten|etting))\s+(?:up\s+)?to\b'
    r'|\barriv(?:e|es|ed|ing)\s+at\b'
    r'|\bholds(?=\s*$|\s+(?:up|firm|true|for|on|across|in|at|here|everywhere|throughout|again)\b)'
    r'|\b(?:is|are|was|were|sits?|stands?|performs?|operates?|now|lands?|landed)\s+at\b'
    r'|\bat\s+(?:the\s+|an?\s+)?E-?[0-4]\b|\blevel\s+(?:is|was|=)\b|\b(?:counts?|counted|ranks?|ranked)\s+as\b'
    r'|(?<![A-Za-z0-9])E-?[0-4][- ]?(?:grade|level|worthy|quality|class|calib(?:re|er)|strength|standard)\b'
    r'|\b(?:is|are|was|were|becomes?|became)\s+(?:now\s+)?(?:an?|the)\s+E-?[0-4](?:[- ]?level)?\s+'
    r'(?:learner|model|system|fly|agent|composer|generali[sz]er)\b'
    r'|\b(?:result|outcome)\s+(?:is|was|=)\s+(?:an?\s+|the\s+)?E-?[0-4]\b'
    # a level token as the predicate of a copula ('the evidence level for X is E3', "W's composition is E3")
    r'|\b(?:is|are|was|were|be|been|becomes?|became|remains?|remained|stays?|stayed)\s+'
    r'(?:(?:now|already|clearly|firmly|solidly|officially|squarely|fully|thus|therefore|then)\s+)?'
    r'(?:an?\s+|the\s+)?E-?[0-4](?![0-9A-Za-z])', re.I)
CLM17_PASSVERB_RE = re.compile(r'^(?:pass(?:ed|es|ing)?|clear(?:ed|s|ing)?)$', re.I)
CLM17_PROCESS_CHECK_RE = re.compile(
    r'\b(?:lock|roster|admission|testbed|technical|rehearsal|plumbing|smoke|unit|schema|lint|integrity|preflight|'
    r'launch|canary|review|dev|cycle[- ]?0)\s+(?:gates?|checks?|tests?|screens?|stages?|bundles?)\b'
    r'|\b(?:lock|roster|admission|testbed|rehearsal|review|canary|lint|plumbing|schema|preflight|audit)\b', re.I)
# a level token used attributively on a process or design noun ('the E3 design', 'E1 plumbing checks', 'E3 strata')
_LEVEL_ATTR_NOUNS = (r'design|lock|plan|attempt|delivery|stage|gates?|plumbing|checks?|tests?|roster|admission|testbed|'
                     r'review|cycle|bundle|setup|protocol|spec|pipeline|endpoint|strata|stratum|items?|worlds?|pool|'
                     r'requirements?|conditions?|criteria|criterion|rule|question|hypothesis|target|goal|candidate')
CLM17_LEVEL_ATTR_RE = re.compile(r'^[- ]?(?:level[- ]?)?(?:%s)\b' % _LEVEL_ATTR_NOUNS, re.I)
# CLM-17P's token rule (round-1 repair part 4) exempts only a level token on a design or process noun: the work, not
# its outcome. Outcome nouns of _LEVEL_ATTR_NOUNS (delivery, endpoint, rule, criteria, requirements, conditions,
# candidate) do not exempt ('this is an E3 delivery', 'W fulfils the E3 requirements' are flagged).
_PROCESS_ATTR_NOUNS = (r'design|lock|plan|attempt|stage|gates?|plumbing|checks?|tests?|roster|admission|testbed|'
                       r'review|cycle|bundle|setup|protocol|spec|pipeline|strata|stratum|items?|worlds?|pool|'
                       r'question|hypothes[ie]s|target|goal|pre-?registration|prereg')
CLM17_PROCESS_ATTR_RE = re.compile(r'^[- ]?(?:level[- ]?)?(?:%s)\b' % _PROCESS_ATTR_NOUNS, re.I)
CLM17_LEVEL_PP_RE = re.compile(r'\b(?:for|of|on|in|under|behind)\s+(?:(?:the|this|that|our|its|an?)\s+)?$', re.I)
CLM17_AFFIRM_RE = re.compile(r'\bis\s+(?:yes|y|true|done|ok|okay|complete|completed|achieved|reached|met)\s*$', re.I)
CLM17_CHECKMARKS = ('✓', '✔', '☑', '✅')
CLM17_QUALIFIER_RE = re.compile(
    r"\b(?:attempt(?:ed|s|ing)?|target(?:ed|s|ing)?|goals?|plan(?:ned|s|ning)?|aim(?:ed|s|ing)?|intend(?:ed|s)?|"
    r"towards?|not|no|never|neither|nor|fail(?:ed|s|ing|ure)?|below|would|could|might|if|unless|cannot|without|"
    r"requir(?:e|es|ed|ing)|need(?:ed|s)?|must|should|expect(?:ed|s)?|predict(?:ed|s|ion)?|whether|unearned|"
    r"short\s+of|fell\s+short|falls?\s+short|out\s+of\s+reach|beyond\s+reach|unmet|unreached|miss(?:ed|es|ing)?|"
    r"lacks?|lacking)\b|n[’']t\b", re.I)
CLM17_COND_RE = re.compile(r'\b(?:when|whenever|iff|once|provided|assuming)\b', re.I)
# Round-1 repair part 4 (SPEC 20.20 (b)): negative-polarity intensifiers and double negations are removed before the
# qualifier search, so they do not qualify the assertion they strengthen ('not just E2 but E3', 'no doubt W reached
# E3', 'W never failed to reach E3', 'no longer short of E3'). Removing words only makes qualifiers rarer, so this
# tightens CLM-17 (b) values and CLM-17P alike.
CLM17_INTENSIFIER_RE = re.compile(
    r"\b(?:not|n[’']t)\s+(?:just|merely|only|simply|even\s+close\s+to\s+(?:failing|missing))\b"
    r"|\b(?:no|without(?:\s+(?:a|any))?|beyond(?:\s+(?:any|all|a))?)\s+(?:doubts?|question)\b"
    r"|\bnothing\s+(?:short\s+of|less\s+than|but)\b|\bno\s+less\s+than\b|\b(?:there\s+is\s+)?no\s+denying\b"
    r"|\b(?:cannot|can\s*not|can[’']t)\s+be\s+(?:denied|doubted|questioned|clearer|disputed)\b|\bundeniabl[ey]\b"
    r"|\b(?:not|never|n[’']t)\s+(?:once\s+)?(?:fail(?:ed|s|ing)?|miss(?:ed|es|ing)?)\s+to\b"
    r"|\bno\s+longer\s+(?:below|short\s+of|missing|lacking|out\s+of\s+reach|fail(?:s|ing)?)\b", re.I)
# Qualifier classes CLM-17P also reads (requirement, intention, plan and deferral words like those of
# CLM17_QUALIFIER_RE); kept out of the shared CLM17_QUALIFIER_RE so that CLM-17 (b), which is enforced, does not lose
# a FAIL.
CLM17P_QUALIFIER_RE = re.compile(r'\b(?:demand(?:s|ed|ing)?|hypothes(?:is|es|i[sz]ed?)|hop(?:e|es|ed|ing)|'
                                 r'aspir(?:e|es|ed|ing|ations?)|next|future|await(?:s|ed|ing)?|'
                                 r'wait(?:s|ed|ing)?\s+(?:for|until|on))\b', re.I)
# a clause that opens with the bare form of an achievement verb is an imperative or a plan ('Goal: reach E3 next')
CLM17_IMPERATIVE_RE = re.compile(r'^\s*(?:to\s+)?(?:reach|achieve|earn|attain|meet|demonstrate|establish|clear|pass|'
                                 r'qualify|secure|obtain|deliver)\b', re.I)
CLM17_NOUN_ASSERTED_RE = re.compile(
    r'\b(?:target|goal|plan|aim)s?\s+(?:(?:is|was|were|are|has\s+been|have\s+been|got|gets)\s+)?(?:fully\s+|now\s+)?'
    r'(?:reached|met|achieved|attained|cleared|passed|satisfied|hit|fulfill?ed)\b', re.I)
CLM17_CONFORMITY_RE = re.compile(
    r'\b(?:exactly\s+|just\s+|precisely\s+)?as\s+(?:was\s+|had\s+been\s+|we\s+)?(?:planned|expected|predicted|'
    r'intended|hoped|targeted|designed|anticipated|forecast|projected|registered|locked|aimed)\b'
    r'|\b(?:in\s+line|consistent)\s+with\s+(?:the\s+|our\s+)?(?:plan|prediction|predictions|expectations?|forecasts?|'
    r'target|goal)s?\b|\baccording\s+to\s+(?:the\s+)?plan\b|\bon\s+target\b'
    r'|\bwithout\s+(?:any\s+)?(?:exception|caveats?|qualifications?|issues?|problems?|reservations?)\b', re.I)
CLM17_GATE_SUBJECT_RE = re.compile(
    r'\b(?:primary|(?:decision\s+)?rules?|criteri(?:on|a)|endpoints?|objective|the\s+result|the\s+outcome)\b', re.I)
CLM17_GUARD_RE = re.compile(r'\bguards?\b', re.I)
CLM17_MET_RE = re.compile(
    r'\b(?:met|meets?|clear(?:ed|s)?|pass(?:ed|es)?|satisf(?:y|ies|ied)|reach(?:ed|es)?|successful(?:ly)?|'
    r'succeed(?:ed|s)?|achiev(?:ed|es)|fulfill?(?:ed|s)?|holds|hit|accomplish(?:ed|es)?|wins?|won|attained|'
    r'exceed(?:ed|s)?|surpass(?:ed|es)?)\b'
    r'|\bpositive\b(?!\s+controls?)'
    r'|\bsuccess\b(?!\s+(?:criteri|condition|threshold|rule|definition|bar\b))', re.I)
CLM17_SPLIT_RE = re.compile(
    r'[;:|()\[\]{}]|\s[-–—]+\s|,?\s+\b(?:but|while|whereas|although|though|however|yet|because|since|so|'
    r'and|with|whilst|which)\b', re.I)
CLM17_PRESENT_RE = re.compile(
    r'\b(?:here|now|currently|today|this\s+(?:delivery|run|design|report|cycle|learner|attempt|work|study))\b', re.I)
# key/value segments: 'key: value' or 'key = value' inside one line segment
CLM17_KV_SEG_SPLIT_RE = re.compile(r'[;|]|,\s')
CLM17_KV_DASH_RE = re.compile(r'^\s*(?:[-*+>]\s+|\d+[.)]\s+|#{1,6}\s+)*(?P<k>[^:=]{1,40}?)\s+[-–—]+\s+(?P<v>[^:=]{1,40}?)'
                              r'\s*[.!?]?\s*$')
CLM17_KV_DASH_MAX_WORDS = 3
CLM17_KV_RE = re.compile(r'^\s*(?:[-*+>]\s*|\d+[.)]\s+|#{1,6}\s+)*(?P<k>[^:=]{1,80}?)\s*(?::|(?<![<>!=])=(?!=))\s*'
                         r'(?P<v>[^:=]{1,120}?)\s*[.!?]?\s*$')
CLM17_WORD_RE = re.compile(r'[A-Za-z0-9][A-Za-z0-9./%+-]*')
CLM17_TABLE_SEP_RE = re.compile(r'^\s*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)*\|?\s*$')
CLM17_KV_MAX_WORDS = 6
GUARD_KIND_WORDS = {'old_drop': ('drop', 'retention', 'forgetting', 'retained'),
                    'old_acq': ('acq', 'acquisition', 'acquired')}


def _agent_words(text: str, links: bool = False) -> str:
    """markup_skeleton plus identifier splitting (snake_case and CamelCase become words) and the look-alike E fold,
    for clause and key/value matching."""
    t = markup_skeleton(text, links=links)
    t = re.sub(r'(?<=[a-z0-9])(?=[A-Z][a-z])', ' ', t)
    return t.replace('_', ' ').translate(_LEVEL_FOLD)


def _n_words(t: str) -> int:
    return len(CLM17_WORD_RE.findall(t))


def _cites_prior(t: str) -> bool:
    return bool(CITATION_RE.search(t) or PRIOR_WORK_RE.search(t))


def _token_level(m: 're.Match') -> int:
    for g in m.groups():
        if g is not None:
            return int(g) if g.isdigit() else LEVEL_WORDS[g.lower()]
    return -1


def level_tokens(text: str) -> List[Tuple[int, int, str]]:
    """(line, level, token) of every E-level token of a text read as rendered (once as written, once with markdown
    links rendered as their text), look-alike E letters folded and identifiers split ('E_3', 'level_E3')."""
    out: List[Tuple[int, int, str]] = []
    seen: Set[Tuple[int, int, str]] = set()
    for links in (False, True):
        for i, line in enumerate(_agent_words(text, links=links).splitlines(), 1):
            for m in LEVEL_TOKEN_RE.finditer(line):
                k = (i, _token_level(m), m.group(0))
                if k not in seen:
                    seen.add(k)
                    out.append(k)
    return out


def _earned_int(level_earned: Any) -> int:
    """E0..E4 -> 0..4; NONE (or anything else) -> -1."""
    if isinstance(level_earned, str) and re.fullmatch(r'E[0-4]', level_earned.strip()):
        return int(level_earned.strip()[1])
    return -1


def _level_value(v: Any) -> Optional[int]:
    """A findings_proposed[].level value: 'E0'..'E4' (as rendered) -> 0..4, 'NONE' -> -1, anything else None."""
    if not isinstance(v, str):
        return None
    t = ' '.join(_agent_words(v).split()).strip(' .')
    if t.upper() == 'NONE':
        return -1
    m = LEVEL_TOKEN_RE.fullmatch(t)
    return _token_level(m) if m else None


# --- CLM-17 (a): the agent-edited CLAIM fields --------------------------------------------------------------------

CLAIM_AGENT_TEXT_FIELDS = ('notes', 'findings_none_reason')


def _finding_strings(f: dict) -> List[Tuple[str, str]]:
    """(path, text) of every non-empty string leaf of one findings_proposed entry except its 'level' (and the key
    names of its objects, which are agent-written too), the claim first."""
    out: List[Tuple[str, str]] = []

    def walk(o: Any, path: List[str]) -> None:
        if isinstance(o, dict):
            for k, v in o.items():
                if not path and k == 'level':
                    continue
                if isinstance(k, str) and LEVEL_TOKEN_RE.search(_agent_words(k)):
                    out.append(('/'.join(path + [k]) + ' (key)', k))
                walk(v, path + [str(k)])
        elif isinstance(o, list):
            for j, v in enumerate(o):
                walk(v, path + [str(j)])
        elif isinstance(o, str) and o.strip():
            out.append(('/'.join(path), o))
    walk(f, [])
    out.sort(key=lambda x: x[0] != 'claim')
    return out


def claim_field_problems(claim: Optional[dict], level_earned: Any, label: str = 'CLAIM.json') -> Tuple[List[str], int]:
    """(problems, number of fields read) of CLM-17 (a): CLAIM.notes, findings_none_reason and every string leaf of
    each findings_proposed entry (its claim, conditions values, extra keys and key names; _finding_strings) name no
    E-level token above level_earned in any phrasing; findings_proposed[].level, when present, is a level <=
    level_earned."""
    if not isinstance(claim, dict):
        return [], 0
    earned = _earned_int(level_earned)
    out: List[str] = []
    n = 0

    def scan(lab: str, text: str) -> None:
        above = [(ln, lv, tok) for ln, lv, tok in level_tokens(text) if lv > earned]
        if above:
            out.append('%s names %s above level_earned %s (%s); agent text never names a level above level_earned, '
                       'not even negated: the attempted level is the generated level_attempted'
                       % (lab, ', '.join(sorted(set('E%d' % lv for _l, lv, _t in above))), level_earned,
                          '; '.join('line %d %r' % (ln, tok) for ln, _lv, tok in above[:4])))
    for k in CLAIM_AGENT_TEXT_FIELDS:
        if _s(claim.get(k)):
            n += 1
            scan('%s#%s' % (label, k), claim[k])
    fp = claim.get('findings_proposed')
    for i, f in enumerate(fp if isinstance(fp, list) else []):
        if not isinstance(f, dict):
            continue
        # round-1 repair part 4 (verifier non-blocking): every agent-written string leaf of the entry (claim,
        # conditions values, extra keys), not only its claim; 'level' is judged as a level below
        for path, text in _finding_strings(f):
            n += 1
            scan('%s#findings_proposed/%d/%s' % (label, i, path), text)
        if f.get('level') is not None:
            n += 1
            lv = _level_value(f.get('level'))
            if lv is None:
                out.append('%s#findings_proposed/%d/level %r is not E0..E4 or NONE' % (label, i, f.get('level')))
            elif lv > earned:
                out.append('%s#findings_proposed/%d/level %s is above level_earned %s' % (label, i, f['level'],
                                                                                        level_earned))
    return out, n


# --- CLM-17 (b): key/value shapes of machine-readable status surfaces -----------------------------------------------

STATUS_SHORT_VALUE_WORDS = 6     # a level value or an E-level key's answer is judged up to this length
STATUS_GATE_VALUE_WORDS = 8      # an achievement verb in a gate key's value is judged up to this length
# keys that name level_earned (normalised: lower case, markup and punctuation removed, context words stripped)
LEVEL_KEYS = frozenset((
    'level', 'evidence level', 'level earned', 'earned', 'earned level', 'e level', 'e-level',
    'evidence level earned', 'earned evidence level', 'level reached', 'reached level', 'level achieved',
    'achieved level', 'level attained', 'attained level', 'claimed level', 'level claimed', 'level of evidence',
    'evidence level reached', 'evidence level achieved', 'level obtained'))
CORE_EARNED_KEYS = LEVEL_KEYS - {'level'}
# keys that name the primary gate or the outcome
GATE_KEYS = frozenset((
    'primary gate', 'gate', 'verdict', 'outcome', 'result', 'status', 'primary', 'primary result', 'primary outcome',
    'primary verdict', 'primary status', 'primary gate result', 'primary gate status', 'gate result', 'gate status',
    'primary endpoint', 'endpoint', 'primary endpoint result', 'primary endpoint status', 'decision',
    'primary decision', 'bottom line', 'conclusion', 'headline result', 'headline verdict', 'decision rule result',
    'rule result', 'objective', 'primary objective'))
# words that open a key as context only ('Delivery status', 'Final level', JSON 'claim.level_earned')
_KEY_CONTEXT_RE = re.compile(r'^(?:(?:summary|status|claim|delivery|this delivery|overall|final|official|headline|'
                             r'current|the|our|this|its)\s+)+')
# a key that describes the design, the plan or other work, not the outcome of this delivery
KEY_QUALIFIER_RE = re.compile(r'\b(?:next|design(?:ed)?|protocol|lock(?:ed)?|hypothes[ie]s|question|prior|previous|'
                              r'parent|earlier|legacy|endpoint|objective|threshold|bar|ladder|definition|meaning|scope|task|'
                              r'focus|topic)\b',
                              re.I)
GATE_VALUE_RE = re.compile(r'(pass|fail|invalid|incomplete|not opened|cannot test claim as specified)(?![a-z])')
NEG_SYMBOLS = frozenset(('✗', '✘', '❌', '×', '—', '–', '-', '--', '∅'))
NEG_VALUE_RE = re.compile(r"(?:no|not|none|nope|false|never|fail(?:ed|s|ing|ure)?|missed|unmet|unreached|unearned|"
                          r"n/a|na|nil|negative|nothing)(?![a-z])")
# a plain zero count or share is a negative answer ('0/8 worlds', '0 of 8', '0%'); a count followed by anything but
# counting nouns is not ('0/8 worlds failed' says the opposite)
ZERO_COUNT_RE = re.compile(r'0(?:\.0+)?\s*(?:/\s*[0-9]+|of\s+[0-9]+|out\s+of\s+[0-9]+|%)'
                           r'(?:\s+(?:main|official|held|dev|worlds?|items?|sums?|runs?|cases?|arms?|seeds?))*\s*$')
AFF_WORDS = frozenset(('yes', 'y', 'true', 'met', 'reached', 'pass', 'passed', 'passes', 'passing', 'achieved',
                       'attained', 'earned', 'done', 'ok', 'okay', 'complete', 'completed', 'success',
                       'successful', 'succeeded', 'satisfied', 'cleared', 'hit', 'holds', 'confirmed', 'positive',
                       'won', 'unlocked', 'granted', 'secured', 'obtained', 'demonstrated', 'established',
                       'fulfilled', 'fulfiled', 'accomplished', 'affirmative',
                       # round-1 repair part 4 (verifier non-blocking: two-line answers 'Absolutely.', 'Indeed.')
                       'absolutely', 'indeed', 'certainly', 'definitely', 'surely', 'sure', 'yep', 'yeah', 'yup',
                       'affirmed', 'correct'))
TASK_STATE_WORDS = frozenset(('done', 'ok', 'okay', 'complete', 'completed'))   # 'Status: done' is about the work
_NEG_INSIDE_RE = re.compile(r"\b(?:not|no|never)\b|n[’']t\b", re.I)
_PAIR_STRIP_RE = re.compile(r"[*`~#>\[\]()\"“”‘’]+")


def _nk(k: Optional[str]) -> str:
    """Normalised key: lower case, markup and brackets removed, list markers and a 'Q:' label dropped, trailing
    punctuation (and '?') removed."""
    t = _PAIR_STRIP_RE.sub(' ', k or '').lower()
    t = re.sub(r'^\s*(?:[-+]|\d+[.)])\s+', '', t)
    t = re.sub(r'^\s*(?:q|question)\s*[:.]\s*', '', t)
    t = ' '.join(t.split())
    return t.strip(' .,;:!?=')


def _ck(k: str) -> str:
    """A normalised key without its context words ('delivery status' -> 'status', 'final level' -> 'level')."""
    c = _KEY_CONTEXT_RE.sub('', k).strip()
    return c or k


def _nv(v: Optional[str]) -> str:
    """Normalised value: lower case, markup removed, an 'A:' label and surrounding punctuation dropped."""
    t = re.sub(r"[*`~]+", ' ', v or '').lower()
    t = ' '.join(t.split()).strip(' .,;:!"\'')
    t = re.sub(r'^(?:a|answer)\s*[:.]\s*', '', t)
    return t


def _neg_value(v: str) -> bool:
    return v in NEG_SYMBOLS or v in ('n', '0', 'x̸') or bool(NEG_VALUE_RE.match(v) or ZERO_COUNT_RE.match(v))


def _aff_value(v: str, task_words: bool = True) -> bool:
    if any(v.startswith(c) for c in CLM17_CHECKMARKS):
        return True
    m = re.match(r'[a-z]+', v)
    head = m.group(0) if m else ''
    if head not in AFF_WORDS or (not task_words and head in TASK_STATE_WORDS):
        return False
    return not _NEG_INSIDE_RE.search(v)


def _levels_in(t: str, attributive_ok: bool = True) -> List[int]:
    """Levels of the E-level tokens of a normalised text; with attributive_ok a token that qualifies a design or
    process noun ('E3 design', 'E3 strata') is left out."""
    out = []
    for m in LEVEL_TOKEN_RE.finditer(t):
        if attributive_ok and CLM17_LEVEL_ATTR_RE.match(t[m.end():]):
            continue
        out.append(_token_level(m))
    return out


def _key_qualified(k: str) -> bool:
    t = CLM17_NOUN_ASSERTED_RE.sub(' ', k)
    return bool(CLM17_QUALIFIER_RE.search(t) or KEY_QUALIFIER_RE.search(t) or _cites_prior(k))


def _kv_seg(seg: str) -> Optional[Tuple[str, str]]:
    m = CLM17_KV_RE.match(seg or '')
    if m and 0 < _n_words(m.group('k')) <= CLM17_KV_MAX_WORDS and _n_words(m.group('v')) > 0:
        return m.group('k'), m.group('v')
    m = CLM17_KV_DASH_RE.match(seg or '')
    if (m and len(re.findall(r'\s[-–—]+\s', seg)) == 1 and 0 < _n_words(m.group('k')) <= CLM17_KV_DASH_MAX_WORDS
            and 0 < _n_words(m.group('v')) <= CLM17_KV_DASH_MAX_WORDS):
        return m.group('k'), m.group('v')
    return None


def _table_cells(line: str) -> List[str]:
    s = line.strip()
    if s.startswith('|'):
        s = s[1:]
    if s.endswith('|'):
        s = s[:-1]
    return [c.strip() for c in s.split('|')]


def _is_table_line(s: str) -> bool:
    s = s.strip()
    return s.startswith('|') and s.count('|') >= 2


def _pair(line: int, key: str, header: Optional[str], value: str, shape: str, text: str) -> dict:
    return {'line': line, 'key': key, 'header': header, 'value': value, 'shape': shape, 'text': text}


def _text_pairs(text: str) -> List[dict]:
    """Key/value shapes of one normalised text: markdown table cells (keyed by the row's first cell and, under a header
    row, by the column header; a first-column cell is a value only under an earned-level header), 'key: value' /
    'key = value' segments (cut at ; | and ', ' inside a sentence), short 'key - value' segments, and 'question?
    answer' pairs (on one line, or a question line followed by a short answer line)."""
    out: List[dict] = []
    lines = text.splitlines()
    i = 0
    prev_q: Optional[Tuple[int, str]] = None
    while i < len(lines):
        s = lines[i].strip()
        if _is_table_line(s):
            j = i
            while j < len(lines) and _is_table_line(lines[j]):
                j += 1
            block = [(k + 1, lines[k].strip()) for k in range(i, j)]
            header: Optional[List[str]] = None
            rows = block
            if len(block) > 1 and CLM17_TABLE_SEP_RE.match(block[1][1]) and '-' in block[1][1]:
                header, rows = _table_cells(block[0][1]), block[2:]
                if not rows:   # a header-only table: its one row is data
                    header, rows = None, block[:1]
            for ln, r in rows:
                if CLM17_TABLE_SEP_RE.match(r):
                    continue
                cells = _table_cells(r)
                for c, val in enumerate(cells):
                    if c == 0:
                        if header and _ck(_nk(header[0])) in CORE_EARNED_KEYS:
                            out.append(_pair(ln, header[0], None, val, 'table', r))
                        continue
                    hd = header[c] if header and c < len(header) else None
                    out.append(_pair(ln, cells[0], hd, val, 'table', r))
            prev_q = None
            i = j
            continue
        ln = i + 1
        i += 1
        if not s:
            continue
        if prev_q is not None and '?' not in s and _n_words(s) <= STATUS_SHORT_VALUE_WORDS:
            out.append(_pair(ln, prev_q[1], None, s, 'qa', prev_q[1] + ' ' + s))
        sents = [x for x in re.split(r'(?<=[.!?])\s+', s) if x.strip()]
        prev_q = (ln, sents[-1]) if sents and sents[-1].rstrip().endswith('?') else None
        for k, sent in enumerate(sents):
            if sent.rstrip().endswith('?'):
                if k + 1 < len(sents) and _n_words(sents[k + 1]) <= STATUS_SHORT_VALUE_WORDS:
                    out.append(_pair(ln, sent, None, sents[k + 1], 'qa', s))
                continue
            for part in re.split(r'[;|]', sent):
                segs = re.split(r',\s', part)
                for q, seg in enumerate(segs):
                    kv = _kv_seg(seg)
                    if not kv:
                        continue
                    out.append(_pair(ln, kv[0], None, kv[1], 'kv', s))
                    # round-1 repair part 4: each following comma segment that holds no key/value shape continues
                    # the value, for the gate rules only ('Decision: proceed, the rule is met')
                    for nxt in segs[q + 1:]:
                        if _kv_seg(nxt):
                            break
                        if nxt.strip():
                            ext = _pair(ln, kv[0], None, nxt, 'kv_ext', s)
                            ext['head'] = kv[1]
                            out.append(ext)
    return out


def status_pairs(text: str) -> List[dict]:
    """Key/value shapes of a markdown/text status surface (_text_pairs), read as rendered: once as written and once
    with markdown links rendered as their text; duplicates dropped."""
    out: List[dict] = []
    seen: Set[Tuple[Any, ...]] = set()
    for links in (False, True):
        for p in _text_pairs(_agent_words(text, links=links)):
            k = (p['line'], p['key'], p['header'], p['value'])
            if k not in seen:
                seen.add(k)
                out.append(p)
    return out


def json_status_pairs(obj: Any) -> List[dict]:
    """Key/value shapes of a JSON status surface: every scalar leaf, keyed by its last two path keys (identifiers
    split into words), true/false as 'true'/'false'; null leaves are skipped."""
    out: List[dict] = []

    def walk(o: Any, path: List[str]) -> None:
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, path + [str(k)])
        elif isinstance(o, list):
            for v in o:
                walk(v, path)
        elif o is not None and path:
            val = 'true' if o is True else 'false' if o is False else str(o)
            key = _agent_words(' '.join(path[-2:]))
            out.append(_pair(0, key, None, _agent_words(val), 'json', '%s: %s' % ('/'.join(path), val)))
    walk(obj, [])
    return out


def _status_basis(level_earned: Any, primary_gate: Any, guards: Dict[str, Any], lock: Optional[dict]) -> dict:
    cand = str(_dig(lock or {}, 'primary.candidate_arm') or '').lower()
    cand_words = ' '.join(_agent_words(cand).lower().split())
    return {'earned': _earned_int(level_earned), 'level_earned': level_earned, 'gate': primary_gate,
            'guards': guards or {}, 'names': _guard_names(lock), 'candidate': cand_words,
            'row_subjects': {'', 'primary', 'overall', 'this delivery', 'delivery', 'claim', 'headline', 'this'}
            | ({cand} if cand else set()) | ({cand_words} if cand_words else set())}


def _status_value_claims(v: str, b: dict) -> bool:
    """A 'status' or 'decision' value (normalised) that pairs a MET word with an outcome subject: the primary, the rule,
    a criterion, an endpoint, the objective or the result (CLM17_GATE_SUBJECT_RE), an E-level above level_earned, or
    the candidate arm when no process check is named ('W passes' claims, 'W passed review' does not). A process check
    named before the first MET word is that verb's subject ('lock gate passed for the primary design', 'admission
    passed')."""
    m = CLM17_MET_RE.search(v)
    if not m:
        return False
    if CLM17_PROCESS_CHECK_RE.search(v[:m.start()]):
        return False
    if CLM17_GATE_SUBJECT_RE.search(v) or any(lv > b['earned'] for lv in _levels_in(v)):
        return True
    cand = b.get('candidate')
    if not cand or not re.search(r'(?<![a-z0-9])%s(?![a-z0-9])' % re.escape(cand), v):
        return False
    return not CLM17_PROCESS_CHECK_RE.search(v)   # 'W passed review': the candidate passed a process check


def _guards_named(keys: Sequence[str], names: Dict[str, Tuple[str, ...]]) -> Tuple[bool, List[str]]:
    """(the keys are about guards, the guard ids they name)."""
    low = ' '.join(keys)
    named = [gid for gid, words in names.items() if any(re.search(r'\b%s\b' % re.escape(w), low) for w in words)]
    return bool(named or CLM17_GUARD_RE.search(low)), named


def judge_status_pair(p: dict, b: dict) -> Optional[str]:
    """CLM-17 (b) verdict on one key/value shape: the reason it FAILs, or None.

    In order: a key naming the attempted level, a target, a plan, the design or prior work (or citing prior work) is
    not judged for levels. A level key (LEVEL_KEYS) carries level_earned at the head of its value, or a negation; a
    column header naming the level carries a level <= level_earned or a negation. A key about guards takes no PASS or
    affirmative value while VERDICT says the guard did not PASS. A gate key (GATE_KEYS; as a column header only with
    the delivery, 'primary' or the candidate arm as row key) carries primary_gate, a negation or no gate word; on a
    non-PASS primary it takes no affirmative value and no short unqualified achievement phrase; for 'status' and
    'decision' only a phrase that pairs the achievement word with an outcome subject counts (_status_value_claims:
    'Status: primary met' FAILs, 'Status: admission passed' does not), and a value continued over comma segments
    ('Decision: proceed, the rule is met') is read by these gate rules too (_judge_ext_pair). A key naming the primary, the rule, a criterion, an endpoint or the objective takes no
    affirmative value or short unqualified achievement phrase on a non-PASS primary. An E-level above level_earned in
    an unqualified key takes no affirmative value, and a short unqualified value is no E-level above level_earned."""
    row = _nk(p['key'])
    hd = _nk(p['header']) if p.get('header') else ''
    v = _nv(p['value'])
    if not v or not row and not hd:
        return None
    if p.get('shape') == 'kv_ext':
        return _judge_ext_pair(p, b, row, v)
    keys = [x for x in (row, hd) if x]
    if any(_cites_prior(x) for x in keys) or _cites_prior(v):
        return None
    earned, gate = b['earned'], b['gate']
    gate_n = str(gate or '').lower().replace('_', ' ')
    qual = any(_key_qualified(x) for x in keys)
    attempted = any('attempt' in x for x in keys)
    neg, aff = _neg_value(v), _aff_value(v)
    tag = '%s%s' % (' '.join(p['key'].split()), (' / ' + ' '.join(p['header'].split())) if p.get('header') else '')
    shown = ' '.join(p['value'].split())
    row_c, hd_c = _ck(row), (_ck(hd) if hd else '')
    subject_row = row_c in b['row_subjects']
    # level keys
    if not qual and not attempted and (row_c in LEVEL_KEYS or hd_c in LEVEL_KEYS):
        head_none = re.match(r'none(?![a-z])', v) is not None   # the level NONE, not a negation
        if neg and not head_none:
            return None
        if row_c in LEVEL_KEYS:
            # the value is the level: level_earned at its head, any other level only behind a qualifier
            m = LEVEL_TOKEN_RE.search(v)
            if head_none:
                head_lv, rest = -1, v[4:]
            elif m and not v[:m.start()].strip():
                head_lv, rest = _token_level(m), v[m.end():]
            else:
                return ('%r gives %r: a level key carries level_earned %s at its head, or a negation'
                        % (tag, shown, b['level_earned']))
            if head_lv != earned:
                return '%r says %s, level_earned is %s' % (tag, shown, b['level_earned'])
            above = [lv for lv in _levels_in(rest) if lv > earned]
            if above and not CLM17_QUALIFIER_RE.search(rest):
                return '%r adds E%d above level_earned %s without a qualifier' % (tag, max(above), b['level_earned'])
            return None
        lv_all = [_token_level(m) for m in LEVEL_TOKEN_RE.finditer(v)]
        if not lv_all and not head_none:
            return ('%r gives %r: a level column carries a level <= level_earned %s, or a negation'
                    % (tag, shown, b['level_earned']))
        if lv_all and max(lv_all) > earned and not CLM17_QUALIFIER_RE.search(v):
            return '%r says %s, above level_earned %s' % (tag, shown, b['level_earned'])
        return None
    # keys about guards
    about_guard, named = _guards_named(keys, b['names'])
    if about_guard and not qual:
        gm = GATE_VALUE_RE.match(v)
        if (gm and gm.group(1) == 'pass') or (aff and not gm):
            pool = named or sorted(b['names']) or ['<any guard>']
            bad = [g for g in pool if str(b['guards'].get(g)) != 'PASS']
            if bad:
                return '%r says %s; VERDICT says %s' % (tag, shown, ', '.join('%s %s' % (g, b['guards'].get(g))
                                                                            for g in bad))
        return None
    # gate keys
    gk = row_c if row_c in GATE_KEYS else (hd_c if hd_c in GATE_KEYS and subject_row else None)
    if gk is not None:
        gm = GATE_VALUE_RE.match(v)
        if gm:
            if gm.group(1) not in ('fail', gate_n):
                return '%r says %s, CLAIM.primary_gate is %s' % (tag, shown, gate)
            return None
        if neg:
            return None
        if gate != 'PASS':
            if _aff_value(v, task_words=gk not in ('status', 'decision')):
                return '%r says %s while primary_gate is %s' % (tag, shown, gate)
            # round-1 repair part 4 (verifier blocking issue 1): a 'status' or 'decision' value is judged when it pairs
            # a MET word with an outcome subject ('Status: primary met', 'Decision: W passes the locked rule');
            # free status text without one ('Status: admission passed, roster next') is not
            if (_n_words(v) <= STATUS_GATE_VALUE_WORDS and CLM17_MET_RE.search(v) and not _qualified(v, True)
                    and (gk not in ('status', 'decision') or _status_value_claims(v, b))):
                return '%r calls the outcome met (%r) while primary_gate is %s' % (tag, shown, gate)
    else:
        subjects = [row] + ([hd] if hd and subject_row else [])
        if gate != 'PASS' and not neg and any(CLM17_GATE_SUBJECT_RE.search(x) for x in subjects):
            if aff:
                return '%r says %s while primary_gate is %s' % (tag, shown, gate)
            if _n_words(v) <= STATUS_GATE_VALUE_WORDS and CLM17_MET_RE.search(v) and not _qualified(v, True):
                return '%r calls it met (%r) while primary_gate is %s' % (tag, shown, gate)
    if attempted or qual:
        return None
    # an E-level above level_earned in the key, answered affirmatively
    klev = [lv for x in keys for lv in _levels_in(x)]
    if klev and max(klev) > earned and aff:
        return '%r answers %s for E%d above level_earned %s' % (tag, shown, max(klev), b['level_earned'])
    # a short value that is an E-level above level_earned
    vlev = _levels_in(v)
    if (vlev and max(vlev) > earned and _n_words(v) <= STATUS_SHORT_VALUE_WORDS and not neg
            and not _qualified(v, True)):
        return '%r gives E%d above level_earned %s as its value (%r)' % (tag, max(vlev), b['level_earned'], shown)
    return None


def _judge_ext_pair(p: dict, b: dict, row: str, v: str) -> Optional[str]:
    """A comma segment that continues a key's value ('Decision: proceed, the rule is met' gives the pair Decision /
    'the rule is met', shape kv_ext): only the gate rules read it, on a non-PASS primary, under a gate key or a key
    naming the primary, the rule, a criterion, an endpoint or the objective, and only when the head value is no gate
    word or negation ('Conclusion: FAIL, although ...' elaborates an honest gate). The segment FAILs when it is a short
    unqualified achievement phrase that pairs its MET word with an outcome subject (_status_value_claims), so 'Result:
    see the table, although the OLD acquisition guard passed' is not judged here."""
    gate = b['gate']
    if gate == 'PASS' or _cites_prior(row) or _cites_prior(v):
        return None
    row_c = _ck(row)
    if row_c not in GATE_KEYS and not CLM17_GATE_SUBJECT_RE.search(row):
        return None
    head = _nv(p.get('head'))
    if GATE_VALUE_RE.match(head) or _neg_value(head) or GATE_VALUE_RE.match(v) or _neg_value(v):
        return None
    if (_n_words(v) <= STATUS_GATE_VALUE_WORDS and CLM17_MET_RE.search(v) and not _qualified(v, True)
            and _status_value_claims(v, b)):
        return '%r continues with %r while primary_gate is %s' % (' '.join(p['key'].split()),
                                                                  ' '.join(p['value'].split()), gate)
    return None


def status_pair_problems(docs: Sequence[dict], b: dict) -> List[str]:
    out: List[str] = []
    for d in docs:
        pairs = d.get('pairs')
        if pairs is None:
            pairs = status_pairs(d['text'])
        for p in pairs:
            why = judge_status_pair(p, b)
            if why:
                out.append('%s:%s %s' % (d['label'], p['line'] or '-', why))
    return out


# --- CLM-17P: free-prose clause heuristics (advisory lint aid) -------------------------------------------------------

def _clauses_of(sent: str) -> List[str]:
    """Clauses of one sentence. A sentence citing prior work is skipped, unless it also points at this delivery: then
    it is cut at commas too and only the clauses that cite nothing and name no prior work are kept. A sentence with a
    key/value segment whose joined clause is read (_kv_clause) is taken segment by segment without that segment, so
    'E3 reached: no' is never judged as the bare key 'E3 reached'."""
    cites = CITATION_RE.search(sent) and PRIOR_WORK_RE.search(sent)
    if cites and not CLM17_PRESENT_RE.search(sent):
        return []
    segs = CLM17_KV_SEG_SPLIT_RE.split(sent)
    kvs = [seg for seg in segs if _kv_seg(seg) and _kv_clause(*_kv_seg(seg))]
    pieces = [seg for seg in segs if seg not in kvs] if kvs else [sent]
    out = []
    for piece in pieces:
        for cl in CLM17_SPLIT_RE.split(piece or ''):
            for part in (re.split(r',\s*', cl or '') if cites else [cl]):
                part = (part or '').strip(' ,.;')
                if part and not (cites and _cites_prior(part)):
                    out.append(part)
    return out


def _kv_clause(key: str, value: str) -> Optional[str]:
    key, value = key.strip(' *_.,'), value.strip(' *_.,')
    if value in CLM17_CHECKMARKS:
        value = 'yes'
    if not key or not value or _n_words(key) == 0 or _n_words(value) == 0:
        return None
    if _n_words(key) > CLM17_KV_MAX_WORDS or _n_words(value) > CLM17_KV_MAX_WORDS:
        return None
    return '%s is %s' % (key, value)


def _kv_clauses(text: str) -> List[Tuple[int, str, str, str]]:
    """(line, '<key> is <value>', line text, value) for the key/value shapes of a normalised text (_text_pairs; a cell under
    a header row gives a clause for its row key and one for its column header). A pair whose row key or header is
    qualified (_key_qualified: the attempted level, a target, a plan, the design, the endpoint, prior work, a
    negation) gives no clause: CLM-17 (b) judges such pairs structurally."""
    out: List[Tuple[int, str, str, str]] = []
    for p in _text_pairs(text):
        if p['shape'] == 'kv_ext':
            # a comma segment continuing a key/value line is read as a clause of its own (a key/value block line
            # is not read as a sentence)
            cl = p['value'].strip(' *_.,')
            cites = CITATION_RE.search(p['text']) and PRIOR_WORK_RE.search(p['text'])
            if _n_words(cl) and not _cites_prior(cl) and not (cites and not CLM17_PRESENT_RE.search(p['text'])):
                out.append((p['line'], cl, p['text'], None))   # prior work is skipped as in _clauses_of
            continue
        if p['shape'] == 'qa':
            continue
        keys = [p['key']] + ([p['header']] if p.get('header') else [])
        if any(_key_qualified(_nk(k)) for k in keys):
            continue   # 'Endpoint: E3 composition', 'Next: E3', '| level attempted | E3 |' under any header: a design,
            #            plan or attempted-level key (judged by CLM-17 (b)), for the row key and the header alike
        if _neg_value(_nv(p['value'])):
            continue   # 'E3 earned: false', 'Reached E3: N', '| E1 | — |', '0/8 worlds' under 'E3 reached': a negative
            #            value asserts nothing (CLM-17 (b) judges the pair structurally)
        for k in keys:
            cl = _kv_clause(k, p['value'])
            if cl:
                out.append((p['line'], cl, p['text'], p['value']))
    return out


def _is_kv_line(s: str) -> bool:
    s = s.strip()
    if not s or s.startswith('|'):
        return False
    m = CLM17_KV_RE.match(s)
    return bool(m and 0 < _n_words(m.group('k')) <= CLM17_KV_MAX_WORDS and _n_words(m.group('v')) > 0)


def _kv_blocks_standalone(text: str) -> str:
    """The text with every line of a run of two or more consecutive 'key: value' lines (a STATUS-style block) marked
    as a single-line unit, so text_units does not join one record with the next into one sentence. Line numbers are
    kept."""
    lines = text.split('\n')
    kv = [_is_kv_line(x) for x in lines]
    out = []
    for i, x in enumerate(lines):
        if kv[i] and ((i > 0 and kv[i - 1]) or (i + 1 < len(lines) and kv[i + 1])):
            out.append('| ' + x.strip())
        else:
            out.append(x)
    return '\n'.join(out)


def overclaim_clauses(text: str) -> List[Tuple[int, str, bool, str, Optional[str]]]:
    """(line, clause, joined, sentence, value) of every clause of one text, after normalisation (as written, and with
    markdown links rendered as their text); joined marks a '<key> is <value>' clause built from a key/value shape (its
    sentence is the source line, value its value; None for a free clause). Questions ('...?') assert nothing and are
    skipped; a table row or a key/value line is read only as joined clauses. Sentences citing prior work are skipped
    (see _clauses_of)."""
    out: List[Tuple[int, str, bool, str, Optional[str]]] = []
    seen: Set[Tuple[int, str]] = set()
    for links in (False, True):
        words = _agent_words(text, links=links)
        found: List[Tuple[int, str, bool, str, Optional[str]]] = []
        for ln, sent, _head in text_units(_kv_blocks_standalone(words), False):
            if sent.startswith('|') or sent.rstrip().endswith('?'):
                continue
            found.extend((ln, cl, False, sent, None) for cl in _clauses_of(sent))
        for ln, cl, line, value in _kv_clauses(words):
            if CITATION_RE.search(cl) and PRIOR_WORK_RE.search(cl):
                continue
            found.append((ln, cl, True, line, value))
        for ln, cl, joined, sent, value in found:
            if (ln, cl) not in seen:
                seen.add((ln, cl))
                out.append((ln, cl, joined, sent, value))
    return out


# a key/value value that is a measurement: a number first ('85.2% exact', '0.852', '41.3pp', '62.5% [S2+, n=80]')
CLM17_MEASURE_RE = re.compile(r'^\s*[~≈<>=±+-]*\s*\d')


def _qualified(clause: str, joined: bool = False, prose: bool = False) -> bool:
    """True when the clause carries an explicit qualifier. Conformity phrases ('as planned', 'without exception'),
    negative-polarity intensifiers and double negations ('not just', 'no doubt', 'never failed to') and a noun
    qualifier asserted as reached ('the E3 target was reached', 'goal met') are removed first, so they do not qualify
    their own assertion. In a joined key/value clause conditional words (when, iff, once, ...) qualify too: 'Decision
    rule: W passes when LCB > 0' defines the rule. With prose (CLM-17P) the requirement, intention and plan words of
    CLM17P_QUALIFIER_RE qualify as well."""
    t = CLM17_NOUN_ASSERTED_RE.sub(' ', CLM17_INTENSIFIER_RE.sub(' ', CLM17_CONFORMITY_RE.sub(' ', clause)))
    return bool(CLM17_QUALIFIER_RE.search(t) or CLM17_IMPERATIVE_RE.match(t)
                or (joined and CLM17_COND_RE.search(t)) or (prose and CLM17P_QUALIFIER_RE.search(t)))


def _named_levels(cl: str) -> List[int]:
    """Levels of the E-level tokens of a clause, leaving out a token on a design or process noun (CLM17_PROCESS_ATTR_RE:
    'the E3 design', 'E1 plumbing checks', 'E3 strata')."""
    return [_token_level(m) for m in LEVEL_TOKEN_RE.finditer(cl) if not CLM17_PROCESS_ATTR_RE.match(cl[m.end():])]


def _level_asserted(cl: str, joined: bool = False) -> bool:
    """CLM-17P (a) shape: an achievement word or shape (in a joined key/value clause also an affirmative value: 'E3:
    yes'), unless every achievement word is pass/clear and its subject is a process check named before it ('the E3
    lock gate passed', 'admission passed for the E3 design'): each level token then qualifies a process noun (E3
    design, E1 plumbing checks, E3 lock) and sits in that subject or after a preposition (for/of/on/in/under) behind
    the verb, never as its object ('the lock passed E3', 'the review passed W for E3' still assert). A level token
    that is the predicate of a copula counts only when no design or process noun follows it ('this is an E3 design'
    asserts nothing)."""
    if joined and CLM17_AFFIRM_RE.search(cl):
        return True
    ms = [m for m in CLM17_ACHIEVE_RE.finditer(cl)
          if not (LEVEL_TOKEN_RE.search(m.group(0)) and CLM17_LEVEL_ATTR_RE.match(cl[m.end():]))]
    if not ms:
        return False
    if all(CLM17_PASSVERB_RE.match(m.group(0)) for m in ms):
        first = ms[0].start()
        if CLM17_PROCESS_CHECK_RE.search(cl[:first]) and all(
                CLM17_LEVEL_ATTR_RE.match(cl[m.end():])
                and (m.start() < first or CLM17_LEVEL_PP_RE.search(cl[first:m.start()][-24:]))
                for m in LEVEL_TOKEN_RE.finditer(cl)):
            return False
    return True


def _guard_names(lock: Optional[dict]) -> Dict[str, Tuple[str, ...]]:
    out: Dict[str, Tuple[str, ...]] = {}
    for g in ((lock or {}).get('guards') or []):
        if isinstance(g, dict) and _s(g.get('id')):
            words = (g['id'].replace('_', ' ').lower(),) + GUARD_KIND_WORDS.get(str(g.get('kind')), ())
            out[g['id']] = words
    return out


def prose_flags(docs: Sequence[dict], level_earned: Any, primary_gate: Any, guards: Dict[str, Any],
                lock: Optional[dict]) -> List[dict]:
    """CLM-17P: {label, line, sentence, clause, reason} of every free-prose clause that (a) names an E-level above
    level_earned without a qualifier (the token rule, round-1 repair part 4: no achievement word is needed; a token on
    a design or process noun is left out, _named_levels), or holds such a token on a process noun with an achievement
    word or shape ('W passed the E3 gate'), or (b) on a non-PASS primary calls the primary, the rule, an endpoint, the
    objective or the result met, or calls a guard met that VERDICT says did not PASS. Questions, sentences citing
    prior work and key/value shapes with a qualified key or a negative value assert nothing (overclaim_clauses). A
    lint aid: the reviewer adjudicates every flag at T2 (REV-09)."""
    earned = _earned_int(level_earned)
    names = _guard_names(lock)
    out: List[dict] = []

    def flag(d: dict, ln: int, cl: str, sent: str, reason: str) -> None:
        out.append({'label': d['label'], 'line': ln, 'sentence': ' '.join(sent.split()), 'clause': cl[:200],
                    'reason': reason})
    for d in docs:
        for ln, cl, joined, sent, value in overclaim_clauses(d['text']):
            joined = joined or bool(d.get('joined'))
            value = value if value is not None else d.get('value')
            if _qualified(cl, joined, prose=True):
                continue
            named = _named_levels(cl)
            if joined and value is not None and CLM17_MEASURE_RE.match(value):
                # 'Held E3 composition: 85.2% exact' measures the E3 task; only a level in the value names one (an
                # achievement word on the key, 'E3 reached: 8/8 worlds', is still read below)
                named = _named_levels(value)
            if named and max(named) > earned:
                flag(d, ln, cl, sent, 'names E%d above level_earned %s without a qualifier' % (max(named), level_earned))
                continue
            levels = [_token_level(m) for m in LEVEL_TOKEN_RE.finditer(cl)]
            if levels and max(levels) > earned and _level_asserted(cl, joined):
                flag(d, ln, cl, sent, 'asserts E%d above level_earned %s' % (max(levels), level_earned))
                continue
            if primary_gate == 'PASS' or not (CLM17_MET_RE.search(cl) or (joined and CLM17_AFFIRM_RE.search(cl))):
                continue
            if CLM17_GATE_SUBJECT_RE.search(cl):
                flag(d, ln, cl, sent, 'describes the primary/rule/endpoint/result as met while primary_gate is %s'
                     % primary_gate)
                continue
            if CLM17_GUARD_RE.search(cl):
                low = cl.lower()
                named = [gid for gid, words in names.items() if any(re.search(r'\b%s\b' % re.escape(w), low)
                                                                    for w in words)]
                pool = named or sorted(names) or ['<any guard>']
                bad = [gid for gid in pool if str(guards.get(gid)) != 'PASS']
                if bad:
                    flag(d, ln, cl, sent, 'describes guard(s) %s as met; VERDICT says %s (primary_gate %s)'
                         % (', '.join(pool), ', '.join('%s %s' % (g, guards.get(g)) for g in bad), primary_gate))
    return out


def overclaim_problems(docs: Sequence[dict], level_earned: Any, primary_gate: Any, guards: Dict[str, Any],
                       lock: Optional[dict]) -> List[str]:
    """prose_flags as one line each: '<label>:<line> <reason>: <clause> | sentence: <sentence>'."""
    return ['%s:%d %s: %r | sentence: %s' % (f['label'], f['line'], f['reason'], f['clause'][:140], f['sentence'])
            for f in prose_flags(docs, level_earned, primary_gate, guards, lock)]


def _json_clause_docs(obj: Any, label: str) -> List[dict]:
    """JSON agent text for CLM-17P: each string value as text, and each scalar leaf as the clause '<key words> is
    <value>' ('... is not true' for false)."""
    docs: List[dict] = []

    def joined(lab: str, text: str, value: Any) -> dict:
        d = _doc(lab, text, False)
        d['joined'] = True   # a '<key> is <value>' clause: read like a markdown key/value line
        d['value'] = value if isinstance(value, str) else json.dumps(value)
        return d

    def walk(o: Any, path: List[str]):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, path + [str(k)])
        elif isinstance(o, list):
            for v in o:
                walk(v, path)
        else:
            words = ' '.join(path[-2:]) if path else ''
            lab = '%s#/%s' % (label, '/'.join(path))
            if isinstance(o, str):
                # a short string under a key is read as its joined clause only, and a string under a qualified key
                # ('level_attempted', 'next_step', 'target') not at all as text: {"level_attempted": "E3"} asserts
                # nothing (round-1 repair part 4, for the token rule)
                if not words or not (_key_qualified(_nk(_agent_words(words)))
                                     or _n_words(_agent_words(o)) <= CLM17_KV_MAX_WORDS):
                    docs.append(_doc(lab, o, False))
                if words and not _neg_value(_nv(_agent_words(o))):   # {"E3_earned": "false"} asserts nothing
                    docs.append(joined(lab, '%s is %s.' % (words, o), o))
            elif isinstance(o, bool):
                docs.append(joined(lab, '%s is %s.' % (words, 'true' if o else 'not true'), o))
            elif o is not None:
                docs.append(joined(lab, '%s is %s.' % (words, o), o))
    walk(obj, [])
    return docs


AGENT_FILE_PREFIXES = ('STATUS', 'SUMMARY', 'MEMO')
# Round-1 repair (verifier: scope): agent notes and analyses inside the delivery, outside the generated results/
# and the learner's manifests/.
AGENT_NOTE_PREFIXES = ('NOTES', 'NOTE_', 'ANALYSIS', 'README', 'RESULTS')
AGENT_SKIP_DIRS = ('checks/', 'runlog/', 'review/')
AGENT_OUTPUT_DIRS = ('results/', 'manifests/')


def _agent_files(ctx: common.Ctx) -> List[str]:
    """The delivery's status surfaces (CLM-17 (b), CLM-17P): STATUS*/SUMMARY*/MEMO* anywhere outside checks/,
    runlog/ and review/; NOTES*/ANALYSIS*/README*/RESULTS* also outside results/ and manifests/ (.md, .txt, .json)."""
    out = []
    for rel in ctx.delivery_files():
        if rel.startswith(AGENT_SKIP_DIRS):
            continue
        base = rel.rsplit('/', 1)[-1]
        if not base.lower().endswith(('.md', '.txt', '.json')):
            continue
        up = base.upper()
        if up.startswith(AGENT_FILE_PREFIXES) or (up.startswith(AGENT_NOTE_PREFIXES)
                                                   and not rel.startswith(AGENT_OUTPUT_DIRS)):
            out.append(rel)
    return out


def _prose_files(ctx: common.Ctx) -> List[str]:
    """CLM-17P's wider prose scope: the status surfaces plus every other .md/.txt file of the delivery outside
    checks/, runlog/, review/, results/ and manifests/ (REPORT.md is read as its body)."""
    out = list(_agent_files(ctx))
    for rel in ctx.delivery_files():
        if rel in out or rel == 'REPORT.md' or rel.startswith(AGENT_SKIP_DIRS + AGENT_OUTPUT_DIRS):
            continue
        if rel.lower().endswith(('.md', '.txt')):
            out.append(rel)
    return out


def _agent_file_docs(path: Path, label: str) -> List[dict]:
    """CLM-17P documents of a file: JSON as string values and joined leaf clauses, anything else as text."""
    raw = path.read_bytes()
    if path.name.lower().endswith('.json'):
        try:
            return _json_clause_docs(common.loads_strict(raw), label)
        except InvalidInput:
            pass
    return [_doc(label, raw.decode('utf-8', 'replace'), False)]


def _status_file_docs(path: Path, label: str) -> List[dict]:
    """CLM-17 (b) documents of a status surface: a parseable JSON file as its leaf pairs, anything else as text."""
    raw = path.read_bytes()
    if path.name.lower().endswith('.json'):
        try:
            d = _doc(label, '', False)
            d['pairs'] = json_status_pairs(common.loads_strict(raw))
            return [d]
        except InvalidInput:
            pass
    return [_doc(label, raw.decode('utf-8', 'replace'), False)]


def _report_body(text: str) -> str:
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if ln.strip():
            return '\n'.join(lines[i + 1:])
    return ''


def claim_text_docs(claim: Optional[dict], label: str = 'CLAIM.json') -> List[dict]:
    """The agent-edited CLAIM texts as documents (notes, findings_none_reason and every string leaf of each
    findings_proposed entry but its level)."""
    docs: List[dict] = []
    if isinstance(claim, dict):
        for k in CLAIM_AGENT_TEXT_FIELDS:
            if _s(claim.get(k)):
                docs.append(_doc('%s#%s' % (label, k), claim[k], False))
        fp = claim.get('findings_proposed')
        for i, f in enumerate(fp if isinstance(fp, list) else []):
            if isinstance(f, dict):
                for path, text in _finding_strings(f):
                    docs.append(_doc('%s#findings_proposed/%d/%s' % (label, i, path), text, False))
    return docs


def agent_text_docs(ctx: common.Ctx, delivery: Optional[Path] = None, claim: Optional[dict] = None) -> List[dict]:
    """CLM-17P's free-prose documents: the REPORT body, the agent-edited CLAIM texts and the delivery's other prose
    files (_prose_files)."""
    docs = claim_text_docs(claim)
    if ctx.path('REPORT.md').is_file():
        docs.append(_doc('REPORT.md', _report_body(ctx.path('REPORT.md').read_text('utf-8', 'replace')), False))
    for rel in _prose_files(ctx):
        docs.extend(_agent_file_docs(ctx.path(rel), rel))
    return docs


def status_surface_docs(ctx: common.Ctx) -> List[dict]:
    """CLM-17 (b)'s documents: the REPORT body and the delivery's status surfaces (_agent_files)."""
    docs: List[dict] = []
    if ctx.path('REPORT.md').is_file():
        docs.append(_doc('REPORT.md', _report_body(ctx.path('REPORT.md').read_text('utf-8', 'replace')), False))
    for rel in _agent_files(ctx):
        docs.extend(_status_file_docs(ctx.path(rel), rel))
    return docs


def _claim_basis(ctx: common.Ctx) -> Tuple[Any, Any, Dict[str, Any], str]:
    """(level_earned, primary_gate, VERDICT guards, basis): the fresh regeneration when it can be computed, else the
    committed CLAIM.json."""
    key = 'wp5_clm17_basis'
    if key in ctx._cache:
        return ctx._cache[key]
    try:
        fresh, _x = generate_claim(ctx, None)
        guards = ((inputs(ctx).get('verdict') or {}).get('guards')) or {}
        res = (fresh.get('level_earned'), fresh.get('primary_gate'), guards, 'fresh claim')
    except (MissingInput, InvalidInput) as e:
        c = _read_claim(ctx)
        res = (c.get('level_earned'), c.get('primary_gate'), {}, 'committed CLAIM.json (regeneration: %s)' % e)
    ctx._cache[key] = res
    return res


DELIVERY_REF_RE = re.compile(r'(?<![A-Za-z0-9_.-])([A-Za-z0-9][A-Za-z0-9_.-]*_(?:design|dev)_v[0-9]+_[0-9]{8})'
                             r'(?![A-Za-z0-9_])')


def _delivery_basis(d: Path) -> Tuple[dict, Optional[dict], dict]:
    """(CLAIM, ENDPOINT_LOCK, VERDICT) of a v1 delivery directory, for lint."""
    claim = common.load_json_file(d / 'CLAIM.json')
    lock = common.load_json_file(d / 'ENDPOINT_LOCK.json')
    vp = d / 'results/official/VERDICT.json'
    verdict = common.load_json_file(vp) if vp.is_file() else {}
    if not isinstance(claim, dict):
        raise InvalidInput('%s/CLAIM.json is not an object' % d.name)
    return claim, (lock if isinstance(lock, dict) else None), (verdict if isinstance(verdict, dict) else {})


def _named_delivery_texts(ctx: common.Ctx, text: str) -> Dict[str, str]:
    """For a file outside every delivery (round-1 repair, verifier: project status documents): the parts of its text
    that name exactly one v1 delivery of this repo, per delivery, with the name replaced by 'this delivery'. Prose
    is taken per sentence (a sentence naming several deliveries per clause); a table row is taken with its header."""
    root = Path(ctx.repo_root)
    known: Dict[str, bool] = {}

    def is_del(name: str) -> bool:
        if name not in known:
            d = root / name
            known[name] = (d / 'CLAIM.json').is_file() and common.is_v1_delivery(d)
        return known[name]

    def names(t: str) -> List[str]:
        return sorted(set(n for n in DELIVERY_REF_RE.findall(t) if is_del(n)))

    out: Dict[str, List[str]] = {}

    def put(name: str, part: str) -> None:
        out.setdefault(name, []).append(DELIVERY_REF_RE.sub(lambda m: 'this delivery' if m.group(1) == name
                                                             else m.group(1), part))
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith('|') and s.count('|') >= 2:
            j = i
            while j < len(lines) and lines[j].strip().startswith('|'):
                j += 1
            block = [x.strip() for x in lines[i:j]]
            head = block[:2] if len(block) > 1 and CLM17_TABLE_SEP_RE.match(block[1]) else []
            for row in block[len(head):]:
                for n in names(row):
                    put(n, '\n'.join(head + [row]))
            i = j
            continue
        para = []
        while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith('|'):
            para.append(lines[i].strip())
            i += 1
        if not para:
            i += 1
            continue
        for sent in re.split(r'(?<=[.!?])\s+', ' '.join(para)):
            ns = names(sent)
            if len(ns) == 1:
                put(ns[0], sent)
            elif len(ns) > 1:
                for cl in re.split(r',\s*', sent):
                    for part in CLM17_SPLIT_RE.split(cl):
                        pn = names(part or '')
                        if len(pn) == 1:
                            put(pn[0], part)
    return dict((k, '\n\n'.join(v)) for k, v in out.items())


def _lint_units(ctx: common.Ctx) -> Tuple[List[Tuple[dict, Optional[dict], dict, List[Tuple[str, dict]], List[dict],
                                                     List[dict]]], List[str], List[str]]:
    """Lint mode for CLM-17 and CLM-17P: [(claim, lock, verdict, [(label, linted CLAIM)], status docs, prose docs)],
    problems,
    skipped labels. A file inside a v1 delivery is judged against that delivery's CLAIM.json; a file outside every
    delivery is judged, part by part, against each delivery it names (_named_delivery_texts)."""
    units = []
    probs: List[str] = []
    skipped: List[str] = []
    for p in [Path(x) for x in ctx.options.get('lint_files') or []]:
        d = next((a for a in p.resolve().parents if (a / 'CLAIM.json').is_file() and (a / 'ENDPOINT_LOCK.json').is_file()),
                 None)
        label = _label_of(ctx, p)
        if d is None:
            parts = _named_delivery_texts(ctx, p.read_text('utf-8', 'replace'))
            if not parts:
                skipped.append(label)
                continue
            for name in sorted(parts):
                try:
                    claim, lock, verdict = _delivery_basis(Path(ctx.repo_root) / name)
                except (MissingInput, InvalidInput) as e:
                    probs.append('%s: the claim of %s cannot be read (%s)' % (label, name, e))
                    continue
                doc = _doc('%s (about %s)' % (label, name), parts[name], False)
                units.append((claim, lock, verdict, [], [doc], [doc]))
            continue
        try:
            claim, lock, verdict = _delivery_basis(d)
        except (MissingInput, InvalidInput) as e:
            probs.append('%s: its delivery claim cannot be read (%s)' % (label, e))
            continue
        if p.name == 'CLAIM.json':
            units.append((claim, lock, verdict, [(label, claim)], [], claim_text_docs(claim, label)))
        elif p.name == 'REPORT.md':
            doc = _doc(label, _report_body(p.read_text('utf-8', 'replace')), False)
            units.append((claim, lock, verdict, [], [doc], [doc]))
        else:
            units.append((claim, lock, verdict, [], _status_file_docs(p, label), _agent_file_docs(p, label)))
    return units, probs, skipped


@check('CLM-17')
def check_clm_17(ctx: common.Ctx) -> Result:
    if _lint_mode(ctx):
        _lint_missing(ctx)
        units, probs, skipped = _lint_units(ctx)
        n = 0
        for claim, lock, verdict, claims, sdocs, _pd in units:
            for lab, c in claims:
                pr, k = claim_field_problems(c, claim.get('level_earned'), lab)
                probs.extend(pr)
                n += k
            if sdocs:
                n += len(sdocs)
                probs.extend(status_pair_problems(sdocs, _status_basis(claim.get('level_earned'),
                                                                       claim.get('primary_gate'),
                                                                       verdict.get('guards') or {}, lock)))
        if n == 0 and not probs:
            return Result('CLM-17', NOT_APPLICABLE, 'no lint input inside, or naming, a v1 delivery with CLAIM.json '
                          '(%s)' % (', '.join(skipped[:5]) or 'none'))
        return _finish('CLM-17', probs, [], 'no CLAIM text names a level above level_earned and no status key/value '
                       'shape contradicts the claim (%d texts%s)' % (n, '; outside and naming no delivery: %s'
                                                                     % ', '.join(skipped[:5]) if skipped else ''))
    level, gate, guards, basis = _claim_basis(ctx)
    claim = None
    if ctx.path('CLAIM.json').is_file():
        try:
            claim = _read_claim(ctx)
        except (MissingInput, InvalidInput):
            claim = None
    probs, n = claim_field_problems(claim, level)
    sdocs = status_surface_docs(ctx)
    if n == 0 and not any(d['text'].strip() or d.get('pairs') for d in sdocs):
        return Result('CLM-17', NOT_APPLICABLE, 'no agent-written CLAIM text and no status surface (REPORT body, '
                      'STATUS/SUMMARY/MEMO, NOTES/ANALYSIS/README/RESULTS)')
    probs.extend(status_pair_problems(sdocs, _status_basis(level, gate, guards, _lock_or_none(ctx))))
    return _finish('CLM-17', probs, [], 'CLAIM notes, findings_none_reason and findings name no level above %s; '
                   'status key/value shapes carry the claim (primary_gate %s; %d CLAIM texts, %d status surfaces; '
                   'basis %s)' % (level, gate, n, len(sdocs), basis))


def prose_flags_of(ctx: common.Ctx) -> Tuple[List[dict], List[dict]]:
    """(CLM-17P flags, the prose documents read) of a delivery; shared with REV-09."""
    level, gate, guards, _basis = _claim_basis(ctx)
    claim = None
    if ctx.path('CLAIM.json').is_file():
        try:
            claim = _read_claim(ctx)
        except (MissingInput, InvalidInput):
            claim = None
    docs = agent_text_docs(ctx, claim=claim)
    return prose_flags(docs, level, gate, guards, _lock_or_none(ctx)), docs


def _flag_line(f: dict) -> str:
    return '%s:%s %s | sentence: %s' % (f['label'], f['line'], f['reason'], f['sentence'])


@check('CLM-17P')
def check_clm_17p(ctx: common.Ctx) -> Result:
    if _lint_mode(ctx):
        _lint_missing(ctx)
        units, probs, skipped = _lint_units(ctx)
        flags: List[dict] = []
        n = 0
        for claim, lock, verdict, _claims, _sdocs, pdocs in units:
            n += len(pdocs)
            flags.extend(prose_flags(pdocs, claim.get('level_earned'), claim.get('primary_gate'),
                                     verdict.get('guards') or {}, lock))
        if n == 0 and not probs:
            return Result('CLM-17P', NOT_APPLICABLE, 'no lint input inside, or naming, a v1 delivery with CLAIM.json')
    else:
        flags, docs = prose_flags_of(ctx)
        probs = []
        n = len(docs)
        if not any(d['text'].strip() for d in docs):
            return Result('CLM-17P', NOT_APPLICABLE, 'no agent-written prose (REPORT body, CLAIM texts, status or '
                          'other prose files)')
    lines = probs + [_flag_line(f) for f in flags]
    if lines:
        return Result('CLM-17P', FAIL, 'lint aid, judged by the reviewer (REV-09 at T2): %d sentence(s) flagged: %s'
                      % (len(flags), ' || '.join(lines[:8]) + (' (+%d more)' % (len(lines) - 8) if len(lines) > 8 else '')),
                      evidence=lines, data={'flagged': flags})
    return Result('CLM-17P', PASS, 'no free-prose clause flagged (%d texts); free-prose overclaim is a judgment check, '
                  'this lint is only an aid' % n)


# --------------------------------------------------------------------------------------
# ERR-01, WLD-01, WLD-02
# --------------------------------------------------------------------------------------

def _gate_of(data: Optional[bytes]) -> Optional[str]:
    if data is None:
        return None
    try:
        obj = common.loads_strict(data)
    except InvalidInput:
        return 'UNPARSEABLE'
    return obj.get('primary_gate') if isinstance(obj, dict) else 'UNPARSEABLE'


@check('ERR-01')
def check_err_01(ctx: common.Ctx) -> Result:
    rel = ctx.repo_rel_of('CLAIM.json')
    seq: List[Tuple[str, str]] = []
    for c in ctx.git.log([rel]):
        g = _gate_of(ctx.git.file_at(c.sha, rel))
        if g is not None:
            seq.append((c.sha[:12], g))
    if not seq:
        return Result('ERR-01', NOT_APPLICABLE, 'CLAIM.json has never been committed (not yet published)')
    wp = ctx.path('CLAIM.json')
    if wp.is_file():
        g = _gate_of(wp.read_bytes())
        if g != seq[-1][1]:
            seq.append(('worktree', g))
    changes = [(seq[i - 1], seq[i]) for i in range(1, len(seq)) if seq[i][1] != seq[i - 1][1]]
    if not changes:
        return Result('ERR-01', PASS, 'primary_gate %s unchanged since the first publish at %s' % (seq[0][1], seq[0][0]))
    errata = ctx.read_jsonl_optional('ERRATA.jsonl')
    probs: List[str] = []
    try:
        expected = kbmod.kit_expected()
    except (MissingInput, InvalidInput):
        expected = None
    used: Set[int] = set()
    for (a_at, a), (b_at, b) in changes:
        idx = next((i for i, r in enumerate(errata) if i not in used and r.get('old_primary_gate') == a
                    and r.get('new_primary_gate') == b), None)
        if idx is None:
            probs.append('primary_gate %s -> %s at %s without an ERRATA row' % (a, b, b_at))
            continue
        used.add(idx)
        r = errata[idx]
        cls = r.get('classification')
        up = GATE_RANK.get(b, 0) > GATE_RANK.get(a, 0)
        for k in ('id', 'diff_summary'):
            if not _s(r.get(k)):
                probs.append('ERRATA row %s lacks %s' % (r.get('id'), k))
        if common.normalize_utc(r.get('utc')) is None:
            probs.append('ERRATA row %s utc %r invalid' % (r.get('id'), r.get('utc')))
        if cls not in ('BUG_FIX', 'POST_HOC'):
            probs.append('ERRATA row %s classification %r is not BUG_FIX|POST_HOC' % (r.get('id'), cls))
        elif cls == 'POST_HOC' and up:
            probs.append('ERRATA row %s: POST_HOC never upgrades a verdict (%s -> %s)' % (r.get('id'), a, b))
        elif cls == 'BUG_FIX' and up:
            fx = r.get('fixture_id')
            if not _s(fx) or expected is None or fx not in expected or not (kbmod.corpus_dir() / str(fx)).is_dir():
                probs.append('ERRATA row %s: a BUG_FIX upgrade needs a corpus fixture id (got %r)' % (r.get('id'), fx))
            so = r.get('reviewer_signoff')
            if not _s(so):
                probs.append('ERRATA row %s: a BUG_FIX upgrade needs a reviewer sign-off' % r.get('id'))
            elif '/' in so or so.endswith(('.json', '.md')):
                if not (ctx.path(so).is_file() or (Path(ctx.repo_root) / common.norm_rel(so)).is_file()):
                    probs.append('ERRATA row %s: reviewer_signoff %s does not exist' % (r.get('id'), so))
    return _finish('ERR-01', probs, [], '%d primary_gate change(s), each with a valid ERRATA row: %s' % (
        len(changes), ', '.join('%s->%s' % (a[1], b[1]) for a, b in changes)))


def official_world_ids(ctx: common.Ctx) -> Set[str]:
    lock = ctx.lock
    ids = set(str(w) for w in (_dig(lock, 'worlds.official.ids') or []) if _s(w))
    for row in ctx.iter_rows('results/official/OUTPUTS.jsonl'):
        if _s(row.get('world')):
            ids.add(row['world'])
    for rel in ctx.delivery_files():
        if rel.startswith('results/receipts/') and rel.endswith('.json'):
            rc = ctx.read_json(rel)
            if isinstance(rc, dict) and rc.get('world_role') == 'official' and _s(rc.get('world')):
                ids.add(rc['world'])
    return ids


def _scan_bytes_for(fh: Any, rx: 're.Pattern', maxlen: int) -> Set[str]:
    found: Set[str] = set()
    tail = b''
    while True:
        chunk = fh.read(1 << 20)
        if not chunk:
            break
        buf = tail + chunk
        for m in rx.finditer(buf):
            found.add(m.group(1).decode('utf-8', 'replace'))
        tail = buf[-(maxlen + 2):]
    return found


def repo_world_hits(ctx: common.Ctx, ids: Set[str]) -> Tuple[Dict[str, List[str]], str]:
    """Quoted occurrences of world ids in JSON files outside this delivery (bounded scan, SPEC 20.4)."""
    key = ('wp5_world_scan', tuple(sorted(ids)))
    if key in ctx._cache:
        return ctx._cache[key]
    rx = re.compile(b'"(' + b'|'.join(re.escape(i.encode('utf-8')) for i in sorted(ids, key=len, reverse=True)) + b')"')
    maxlen = max(len(i.encode('utf-8')) for i in ids)
    scan = ctx.repo_scan()
    d = ctx.delivery_rel
    hits: Dict[str, List[str]] = {}
    n = 0
    for f in scan.files:
        rel = scan.rel(f)
        if d and (rel == d or rel.startswith(d + '/')):
            continue
        if not (rel.endswith('.json') or rel.endswith('.jsonl') or rel.endswith('.jsonl.gz')):
            continue
        n += 1
        try:
            if rel.endswith('.gz'):
                with gzip.open(str(f), 'rb') as fh:
                    got = _scan_bytes_for(fh, rx, maxlen)
            else:
                with open(f, 'rb') as fh:
                    got = _scan_bytes_for(fh, rx, maxlen)
        except (OSError, EOFError, zlib.error) as e:
            raise MissingInput('repo scan could not complete: %s unreadable (%s)' % (rel, e))
        for w in got:
            lst = hits.setdefault(w, [])
            if len(lst) < 5:
                lst.append(rel)
    res = (hits, '%s; %d JSON/JSONL files searched for quoted world ids' % (scan.scope(), n))
    ctx._cache[key] = res
    return res


@check('WLD-01')
def check_wld_01(ctx: common.Ctx) -> Result:
    ids = official_world_ids(ctx)
    if not ids:
        return Result('WLD-01', NOT_APPLICABLE, 'no official world ids (listed or in official rows)')
    probs: List[str] = []
    missing: List[str] = []
    for w, where in sorted(dev_touched(ctx, ids).items()):
        probs.append('official world %s used on dev/technical/rehearsal: %s' % (w, ', '.join(where)))
    hits, scope = repo_world_hits(ctx, ids)
    for w, where in sorted(hits.items()):
        probs.append('official world %s occurs elsewhere in the repo: %s' % (w, ', '.join(where)))
    lock = ctx.lock
    try:
        lrel = ctx.repo_rel_of('ENDPOINT_LOCK.json')
        cur = ctx.path('ENDPOINT_LOCK.json').read_bytes()
        lock_c = ctx.git.first_commit_where(lrel, lambda b: b == cur)
        out_c = ctx.git.first_commit_adding(ctx.repo_rel_of('results/official') + '/**')
        if lock_c is None:
            probs.append('the current ENDPOINT_LOCK.json is not committed')
        elif out_c is not None and (out_c.sha == lock_c.sha or not ctx.git.is_ancestor(lock_c.sha, out_c.sha)):
            probs.append('official outputs first committed at %s, not after the lock commit %s' % (out_c.sha[:12], lock_c.sha[:12]))
        sealed = _dig(lock, 'worlds.official_role_intended') == 'SEALED_CONFIRM' or _dig(lock, 'worlds.official.mode') == 'sealed'
        if sealed:
            rv = ctx.git.first_commit_adding(ctx.repo_rel_of('review/SEAL_REVEAL.json'))
            if rv is None:
                probs.append('sealed worlds but review/SEAL_REVEAL.json is not committed')
            elif out_c is not None and (out_c.sha == rv.sha or not ctx.git.is_ancestor(rv.sha, out_c.sha)):
                probs.append('official outputs first committed at %s, not after the reveal %s' % (out_c.sha[:12], rv.sha[:12]))
    except MissingInput as e:
        missing.append('commit order of lock, reveal and official outputs unavailable: %s' % e)
    return _finish('WLD-01', probs, missing, '%d official world id(s) fresh: no dev use, nowhere else in the repo, '
                   'outputs after the lock (%s)' % (len(ids), scope), data={'official_worlds': sorted(ids)})


@check('WLD-02')
def check_wld_02(ctx: common.Ctx) -> Result:
    ov = held_overlap(ctx)
    if ov is None:
        return Result('WLD-02', NOT_APPLICABLE, 'no canonical held items (no addition format or held manifest)')
    if ov['k'] > 0:
        return Result('WLD-02', FAIL, '%d of %d canonical held items overlap earlier deliveries\' held or taught sets '
                      '(%s); reported in the headline' % (ov['k'], ov['n'], ', '.join(ov['sources'][:5])),
                      evidence=ov['sources'], data=ov)
    return Result('WLD-02', PASS, 'no overlap of %d canonical held items with other deliveries (%s)'
                  % (ov['n'], ov['scope']), data=ov)


# --------------------------------------------------------------------------------------
# Pointers PTR-01..07
# --------------------------------------------------------------------------------------

def _pointer(ctx: common.Ctx) -> Optional[dict]:
    key = 'wp5_pointer'
    if key in ctx._cache:
        return ctx._cache[key]
    ptr = ctx.options.get('pointer')
    if not ptr:
        return None
    p = Path(ptr)
    cands = [p] if p.is_absolute() else [Path.cwd() / p, Path(ctx.repo_root) / p]
    path = next((c for c in cands if c.is_file()), None)
    if path is None:
        raise MissingInput('pointer file %s not found' % ptr)
    raw = path.read_bytes()
    obj = common.loads_strict(raw, what=str(ptr))
    try:
        rel = path.resolve().relative_to(Path(ctx.repo_root).resolve()).as_posix()
    except ValueError:
        rel = None
    info = {'path': path, 'rel': rel, 'raw': raw, 'obj': obj, 'label': rel or str(path)}
    ctx._cache[key] = info
    return info


def _ptr_target(ctx: common.Ctx, obj: Any) -> Tuple[Optional[str], Optional[Path]]:
    d = obj.get('delivery') if isinstance(obj, dict) else None
    if not _s(d):
        return None, None
    rel = common.norm_rel(d)
    return rel, Path(ctx.repo_root) / rel


def _na_no_pointer(cid: str) -> Result:
    return Result(cid, NOT_APPLICABLE, 'no pointer file given (pointer stage or lint with a pointer file)')


def _ud_scopes(ctx: common.Ctx, uid: Any) -> Optional[List[str]]:
    if not _s(uid):
        return None
    ud = ctx.kb.user_decision(uid)
    if not isinstance(ud, dict):
        return None
    return [str(s) for s in (ud.get('scope') or [])]


@check('PTR-01')
def check_ptr_01(ctx: common.Ctx) -> Result:
    info = _pointer(ctx)
    if info is None:
        return _na_no_pointer('PTR-01')
    obj = info['obj']
    probs: List[str] = []
    missing: List[str] = []
    if not isinstance(obj, dict) or obj.get('schema') != SCHEMA_POINTER:
        return Result('PTR-01', FAIL, '%s: schema %r is not %s' % (info['label'], obj.get('schema') if isinstance(obj, dict)
                                                                 else type(obj).__name__, SCHEMA_POINTER))
    for k in POINTER_KEYS:
        if k not in obj:
            probs.append('missing field %s' % k)
    extra = sorted(set(obj) - set(POINTER_KEYS) - {'retro_annotation'})
    if extra:
        probs.append('unknown fields %s' % ', '.join(extra))
    if not _s(obj.get('track')):
        probs.append('track must be a non-empty string')
    if common.normalize_utc(obj.get('updated_utc')) is None:
        probs.append('updated_utc %r is not a UTC time' % (obj.get('updated_utc'),))
    if not isinstance(obj.get('narrowing_rows'), list):
        probs.append('narrowing_rows must be a list')
    # authorization
    drel, ddir = _ptr_target(ctx, obj)
    auth_ok = False
    why_auth = []
    if ddir is not None and (ddir / 'AUTHORIZATION.json').is_file():
        auth = common.load_json_file(ddir / 'AUTHORIZATION.json')
        pu = (auth.get('stages') or {}).get('pointer_update') if isinstance(auth, dict) else None
        auth_ok = pu is True or (isinstance(pu, dict) and pu.get('granted') is True)
        if not auth_ok:
            why_auth.append('AUTHORIZATION.stages.pointer_update is not true')
    else:
        why_auth.append('no AUTHORIZATION.json at the target')
    if not auth_ok:
        scopes = _ud_scopes(ctx, obj.get('user_decision'))
        names = set([drel, drel.rsplit('/', 1)[-1]]) if drel else set()
        if scopes is not None and any(s in ('adoption:%s' % n for n in names) for s in scopes):
            auth_ok = True
        else:
            why_auth.append('user_decision %r is not a UD with scope adoption:<delivery>' % (obj.get('user_decision'),))
    if not auth_ok:
        probs.append('pointer update not authorized: %s' % '; '.join(why_auth))
    # previous pointer snapshotted in the same commit
    rel = info['rel']
    track = obj.get('track') if _s(obj.get('track')) else ''
    hist_dir = '%s_POINTER_HISTORY' % track.upper()
    snap = obj.get('previous_snapshot')
    if rel is None:
        probs.append('pointer file lies outside the repo')
    else:
        try:
            head = ctx.git.head()
            committed = ctx.git.file_at(head, rel)
            if committed is not None and committed == info['raw']:
                hist = ctx.git.log([rel])
                if len(hist) >= 2:
                    last, prev = hist[-1], hist[-2]
                    prev_bytes = ctx.git.file_at(prev.sha, rel)
                    if not _s(snap):
                        probs.append('previous_snapshot missing although %s had an earlier version' % rel)
                    else:
                        srel = common.norm_rel(snap)
                        if srel not in last.paths:
                            probs.append('previous snapshot %s is not in the pointer commit %s' % (srel, last.sha[:12]))
                        elif ctx.git.file_at(last.sha, srel) != prev_bytes:
                            probs.append('previous snapshot %s does not equal the previous pointer at %s' % (srel, prev.sha[:12]))
            elif committed is not None:
                if not _s(snap):
                    probs.append('previous_snapshot missing although %s is committed at HEAD' % rel)
                else:
                    srel = common.norm_rel(snap)
                    sp = Path(ctx.repo_root) / srel
                    if not sp.is_file() or sp.read_bytes() != committed:
                        probs.append('previous snapshot %s does not equal the committed pointer' % srel)
                    elif ctx.git.file_at(head, srel) is not None:
                        probs.append('previous snapshot %s is already committed: it must land in the same commit as the '
                                     'pointer change' % srel)
            if _s(snap) and hist_dir not in common.norm_rel(snap).split('/'):
                probs.append('previous_snapshot %s is not under %s/' % (snap, hist_dir))
        except MissingInput as e:
            missing.append('pointer history unavailable: %s' % e)
        except InvalidInput as e:
            probs.append(str(e))
    return _finish('PTR-01', probs, missing, '%s: schema v1, authorized, previous pointer snapshotted' % info['label'])


@check('PTR-02')
def check_ptr_02(ctx: common.Ctx) -> Result:
    info = _pointer(ctx)
    if info is None:
        return _na_no_pointer('PTR-02')
    obj = info['obj']
    if not isinstance(obj, dict):
        return Result('PTR-02', FAIL, '%s is not a JSON object' % info['label'])
    drel, ddir = _ptr_target(ctx, obj)
    if drel is None:
        return Result('PTR-02', FAIL, 'pointer names no delivery')
    if ctx.delivery is not None and not _same_delivery(drel, ctx.delivery_rel):
        return Result('PTR-02', FAIL, 'pointer delivery %s is not the gated delivery %s' % (drel, ctx.delivery_rel))
    if obj.get('claim') is None and not common.is_v1_delivery(ddir):
        return Result('PTR-02', NOT_APPLICABLE, 'legacy target %s without a CLAIM (PTR-07 governs)' % drel)
    probs = []
    if not _s(obj.get('claim')):
        return Result('PTR-02', FAIL, 'pointer names no claim')
    crel = common.norm_rel(obj['claim'])
    cp = Path(ctx.repo_root) / crel
    if not cp.is_file():
        raise MissingInput('referenced claim %s not found' % crel)
    if crel != (drel + '/CLAIM.json' if drel else 'CLAIM.json'):
        probs.append('claim %s is not %s/CLAIM.json' % (crel, drel))
    raw = cp.read_bytes()
    sha = common.sha256_bytes(raw)
    if _strip_sha(obj.get('claim_sha256')) != sha:
        probs.append('claim_sha256 %s differs from the CLAIM bytes %s' % (str(obj.get('claim_sha256'))[:12], sha[:12]))
    claim = common.loads_strict(raw, what=crel)
    if not isinstance(claim, dict):
        return Result('PTR-02', FAIL, '%s is not an object' % crel)
    if not _same_delivery(claim.get('delivery'), drel):
        probs.append('claim delivery %r is not %s' % (claim.get('delivery'), drel))
    for k in POINTER_COPY_FIELDS:
        if k not in obj or k not in claim or _canon(obj.get(k)) != _canon(claim.get(k)):
            probs.append('%s differs from the CLAIM (%s vs %s)' % (k, json.dumps(obj.get(k))[:80], json.dumps(claim.get(k))[:80]))
    return _finish('PTR-02', probs, [], 'pointer fields are byte copies of %s (%s)' % (crel, sha[:12]))


def _bool_keys(obj: Any, path: str = '') -> List[str]:
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = '%s/%s' % (path, k)
            if isinstance(v, bool) and BOOL_KEY_RE.search(str(k)):
                out.append('%s: %s' % (p, json.dumps(v)))
            out.extend(_bool_keys(v, p))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(_bool_keys(v, '%s/%d' % (path, i)))
    return out


@check('PTR-03')
def check_ptr_03(ctx: common.Ctx) -> Result:
    info = _pointer(ctx)
    if info is None:
        return _na_no_pointer('PTR-03')
    hits = _bool_keys(info['obj'])
    return _finish('PTR-03', ['%s bare boolean %s' % (info['label'], h) for h in hits], [],
                   '%s: no bare boolean confirm/pass/success/qualified keys' % info['label'])


@check('PTR-04')
def check_ptr_04(ctx: common.Ctx) -> Result:
    info = _pointer(ctx)
    if info is None:
        return _na_no_pointer('PTR-04')
    obj = info['obj']
    quoted: Set[str] = set()
    for _jp, s in _json_strings(obj):
        quoted.update(kbmod.F_ID_FIND_RE.findall(s))
    rows = ctx.kb.rows('FINDINGS')
    row_ids = set(r.get('id') for r in rows)
    cited = set(str(x) for x in (obj.get('narrowing_rows') or [])) if isinstance(obj, dict) else set()
    probs = []
    for fid in sorted(quoted):
        if fid in row_ids and not any(r.get('id') == fid and (r.get('kind') or 'finding') == 'finding' for r in rows):
            continue  # a narrowing row id quoted as such
        st = kbmod.finding_state(ctx.kb, fid, ctx.git, ctx.rules)
        if st is None:
            probs.append('quoted finding %s is not in FINDINGS' % fid)
            continue
        if st['status'] in ('narrowed', 'superseded'):
            need = [r for r in st['narrowing_rows'] if r not in cited]
            if need:
                probs.append('quoted finding %s is %s; cite narrowing row(s) %s in narrowing_rows (caveat: %s)'
                             % (fid, st['status'], ', '.join(need), ' | '.join(st['caveats'])[:160]))
    for r in sorted(cited):
        if r not in row_ids:
            probs.append('narrowing_rows cites %s, which is not a FINDINGS row' % r)
    return _finish('PTR-04', probs, [], '%d quoted finding(s); narrowed or superseded ones cite their narrowing rows'
                   % len(quoted))


@check('PTR-05')
def check_ptr_05(ctx: common.Ctx) -> Result:
    info = _pointer(ctx)
    if info is None:
        return _na_no_pointer('PTR-05')
    docs = [_doc(info['label'], info['raw'].decode('utf-8', 'replace'), False)]
    return _finish('PTR-05', abs_path_hits(docs), [], '%s: no absolute local paths' % info['label'])


def _dir_date(name: str) -> str:
    m = re.search(r'_(\d{8})$', name)
    return m.group(1) if m else ''


@check('PTR-06')
def check_ptr_06(ctx: common.Ctx) -> Result:
    info = _pointer(ctx)
    if info is None:
        return _na_no_pointer('PTR-06')
    drel, ddir = _ptr_target(ctx, info['obj'])
    if drel is None or not common.is_v1_delivery(ddir):
        return Result('PTR-06', NOT_APPLICABLE, 'target %s is not a v1 delivery' % drel)
    lock = common.load_json_file(ddir / 'ENDPOINT_LOCK.json')
    fam = _dig(lock, 'task.family')
    mine = (_dir_date(ddir.name), str(common.normalize_utc(lock.get('created_utc')) or ''))
    newer = []
    for other in common.v1_deliveries(ctx):
        if _same_delivery(other, drel):
            continue
        try:
            ol = common.load_json_file(Path(ctx.repo_root) / other / 'ENDPOINT_LOCK.json')
        except (MissingInput, InvalidInput):
            continue
        if _dig(ol, 'task.family') != fam:
            continue
        theirs = (_dir_date(Path(other).name), str(common.normalize_utc(ol.get('created_utc')) or ''))
        if theirs > mine:
            newer.append(other)
    if newer:
        return Result('PTR-06', FAIL, 'newer v1 deliveries in the same track (task family %s): %s' % (fam, ', '.join(newer[:5])))
    return Result('PTR-06', PASS, 'no newer v1 delivery with task family %s' % fam)


@check('PTR-07')
def check_ptr_07(ctx: common.Ctx) -> Result:
    info = _pointer(ctx)
    if info is None:
        return _na_no_pointer('PTR-07')
    obj = info['obj']
    drel, ddir = _ptr_target(ctx, obj)
    if drel is None:
        return Result('PTR-07', FAIL, 'pointer names no delivery')
    if common.is_v1_delivery(ddir):
        return Result('PTR-07', NOT_APPLICABLE, 'target %s is a v1 delivery' % drel)
    probs = []
    scopes = _ud_scopes(ctx, obj.get('user_decision') if isinstance(obj, dict) else None)
    name = drel.rsplit('/', 1)[-1]
    if scopes is None:
        probs.append('legacy target %s needs an owner UD (user_decision %r not in USER_DECISIONS)'
                     % (drel, obj.get('user_decision') if isinstance(obj, dict) else None))
    elif not any(name in s for s in scopes):
        probs.append('UD %s scope %s does not name %s' % (obj.get('user_decision'), scopes, name))
    paths = [s for _jp, s in _json_strings(obj) if s.startswith(common.KB_REL + '/RETRO/')]
    if isinstance(obj, dict) and _s(obj.get('retro_annotation')):
        paths.insert(0, obj['retro_annotation'])
    ok_retro = False
    for p in paths:
        try:
            fp = Path(ctx.repo_root) / common.norm_rel(p)
        except InvalidInput:
            continue
        if fp.is_file() and common.norm_rel(p).startswith(common.KB_REL + '/RETRO/') and name in fp.read_text(
                encoding='utf-8', errors='replace'):
            ok_retro = True
    if not ok_retro:
        probs.append('legacy target %s needs a %s/RETRO annotation path naming it (retro_annotation)' % (drel, common.KB_REL))
    return _finish('PTR-07', probs, [], 'legacy target %s with an owner UD and a RETRO annotation' % drel)


# --------------------------------------------------------------------------------------
# lint (T0)
# --------------------------------------------------------------------------------------

def is_pointer_file(path: Path) -> bool:
    name = path.name
    if not name.endswith('.json'):
        return False
    try:
        obj = common.load_json_file(path)
    except (MissingInput, InvalidInput):
        return bool(re.search(r'POINTER|^CURRENT_', name))
    if isinstance(obj, dict) and obj.get('schema') == SCHEMA_POINTER:
        return True
    return bool(re.search(r'POINTER|^CURRENT_', name)) and 'HISTORY' not in str(path.parent)


def _advisory(rule: dict, r: Result) -> Result:
    if rule.get('status') == 'advisory' and r.status in (FAIL, MISSING):
        r.detail = '[advisory rule; was %s] %s' % (r.status, r.detail)
        r.status = WARN
    return r


def cmd_lint(args: Any) -> int:
    """MFC lint <files...>: CLM-06, CLM-12, CLM-16 on every file; PTR-01..07 on pointer files (T0)."""
    ctx = common.make_ctx(args, delivery=None, stage='lint', options={'invoked_by': 'lint'})
    files: List[Path] = []
    missing: List[str] = []
    for f in args.files:
        p = Path(f)
        cands = [p] if p.is_absolute() else [Path.cwd() / p, Path(ctx.repo_root) / p]
        hit = next((c for c in cands if c.is_file()), None)
        if hit is None:
            missing.append(str(f))
        else:
            files.append(hit.resolve())
    ctx.options['lint_files'] = [str(p) for p in files]
    ctx.options['lint_missing'] = missing
    ctx.tier = 0
    rules = ctx.rules or common.load_rules()
    idx = common.rule_index(rules)
    pointers = [p for p in files if is_pointer_file(p)]
    results: List[Result] = []
    for rid in common.bundle_ids(rules, 'lint'):
        rule = idx[rid]
        if rid.startswith('PTR-'):
            if not pointers:
                results.append(Result(rid, NOT_APPLICABLE, 'no pointer file among the lint inputs'))
                continue
            for pf in pointers:
                ctx.options['pointer'] = str(pf)
                ctx._cache.pop('wp5_pointer', None)
                for r in common.run_check(ctx, rule):
                    results.append(_advisory(rule, r))
            ctx.options.pop('pointer', None)
            ctx._cache.pop('wp5_pointer', None)
            continue
        for r in common.run_check(ctx, rule):
            results.append(_advisory(rule, r))
    rc = common.exit_code(results)
    if getattr(args, 'json', False):
        print(json.dumps({'command': 'lint', 'files': [_label_of(ctx, p) for p in files], 'missing': missing,
                          'pointer_files': [_label_of(ctx, p) for p in pointers],
                          'results': [r.to_dict() for r in results], 'exit_code': rc,
                          'verdict': common.VERDICT_BY_EXIT[rc]}, indent=2, ensure_ascii=False))
    else:
        print(common.format_results(results))
        print('lint: %s (exit %d) over %d file(s), %d pointer file(s)' % (common.VERDICT_BY_EXIT[rc], rc, len(files),
                                                                       len(pointers)))
    return rc
