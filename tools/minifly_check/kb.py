"""MRK-1 knowledge ledgers (WP5): SPEC Sections 8 (KB ledgers) and 11 (knowledge ledger, R-levels,
prediction scoring, KB score, backtest), plus checks KB-01..05. Python 3.9+, standard library only.

What lives here:
- row schemas of the seven PROJECT_DOCUMENTS/KB ledgers (KB-01) and `MFC ledger append`, which validates
  a row, assigns its id (F-<yyyymmdd>-<agent>-<n>, UD-<yyyymmdd>-<n>, I-<yyyymmdd>-<n>, DEBT-<yyyymmdd>-<n>)
  and appends exactly one line;
- the FINDINGS fold: current status (order-independent, so union merges are safe), caveats, narrowing
  rows and the computed R-level of a finding;
- deliveries_since_last_e3(kb): the computed E3 cadence counter that resets only on an admitted, scored,
  main-world held-E3 primary (SPEC 6.2 item 10, 11);
- `MFC kb status|score|backtest-snapshot|backtest-score`.

Never a score of an agent or of the kit: calibration numbers are KB diagnostics only (SPEC 11).
"""
from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
import re
import sys
import tarfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import common
from common import (FAIL, MISSING, NOT_APPLICABLE, PASS, WARN, CheckerError, InvalidInput, MissingInput,  # noqa: F401
                    Result, check)

KB_REL = common.KB_REL
LEDGERS = ('FINDINGS', 'INCIDENTS', 'USER_DECISIONS', 'DELIVERY_LEDGER', 'CONFIRM_LEDGER', 'REVIEW_DEBT',
           'CANARY_COMMITMENTS')
APPENDABLE = ('FINDINGS', 'INCIDENTS', 'USER_DECISIONS', 'REVIEW_DEBT', 'CANARY_COMMITMENTS')

FINDING_KINDS = ('finding', 'activation', 'narrowing', 'supersession', 'retraction', 'transfer', 'reconciliation')
FINDING_STATUSES = ('proposed', 'provisional', 'active', 'unverifiable_in_repo', 'narrowed', 'superseded', 'retracted')
RECONCILIATION_STATUSES = ('CONFIRMED', 'CONFIRMED_WITH_CORRECTION', 'REFUTED', 'NEW_DEFECT', 'SCOPE_LIMITED',
                           'UNVERIFIED')
NEW_FINDING_STATUSES = ('proposed', 'provisional', 'unverifiable_in_repo')
TRANSFER_OUTCOMES = ('ok', 'failed')
R2_BASES = ('rederived_metric', 'changed_presentation')
LEVELS = ('E0', 'E1', 'E2', 'E3', 'E4')
PRIMARY_GATES = ('PASS', 'FAIL', 'INVALID', 'INCOMPLETE', 'NOT_OPENED', 'CANNOT_TEST_CLAIM_AS_SPECIFIED')
GATE_TYPES = ('objective', 'component')
WORLDS_ROLES = ('DEV', 'PILOT', 'MAIN', 'SEALED_CONFIRM')
TASK_FAMILIES = ('addition_multi_digit', 'addition_single_digit', 'addition_component', 'other_arithmetic',
                 'non_arithmetic')

F_ID_RE = re.compile(r'^F-(?:UF-\d{2,}|ARCH-\d{2,}|\d{8}-[a-z][a-z0-9]*-\d+)$')
F_ID_FIND_RE = re.compile(r'(?<![A-Za-z0-9_-])F-(?:UF-\d{2,}|ARCH-\d{2,}|\d{8}-[a-z][a-z0-9]*-\d+)(?![A-Za-z0-9_])')
UD_ID_RE = re.compile(r'^UD-(?:\d{8}-\d+|\d{4,})$')
FM_RE = re.compile(r'^FM-\d\d$')
HEX64_RE = re.compile(r'^[0-9a-f]{64}$')
FAM_RE = re.compile(r'^FAM-[0-9a-f]{8}$')
# A strict source reference for rows appended through `ledger append`: path:line[-line][,..][@sha].
STRICT_SOURCE_RE = re.compile(r'^[^\s:@;]+:\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*(?:@[0-9A-Za-z][0-9A-Za-z._/-]*)?$')

LAPLACE_MIN_N = 5
BASE_RATE_BAND = (0.1, 0.9)
BOOTSTRAP_DEFAULT = 2000


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------

def _s(v: Any) -> bool:
    return isinstance(v, str) and bool(v.strip())


def _num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and abs(v) != float('inf')


def _int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def ledger_rel(name: str) -> str:
    return '%s/%s.jsonl' % (KB_REL, name)


def family(agent: Any) -> Optional[str]:
    """Agent family of an identity string: 'codex' or 'claude' (session-qualified forms such as
    'claude:<session>' or 'claude-digest' included); None when it names neither."""
    if not isinstance(agent, str):
        return None
    a = agent.strip().lower()
    for fam in ('codex', 'claude'):
        if a == fam or (a.startswith(fam) and a[len(fam):len(fam) + 1] in (':', '-', '_', ' ', '/', '@')):
            return fam
    return None


def trailer_family(commit: common.Commit) -> Optional[str]:
    fams = set(f for f in (family(v) for v in commit.trailer_values('Acting-Agent')) if f)
    if len(fams) == 1:
        return fams.pop()
    return None if not fams else 'ambiguous'


def utc_key(v: Any) -> str:
    return common.normalize_utc(v) or ''


def _canon(row: Any) -> bytes:
    return common.canonical_json(row)


def _lines_canon(data: Optional[bytes], label: str) -> List[bytes]:
    """Canonical JSON of each non-blank line of a ledger version (raw stripped line when it does not parse)."""
    out: List[bytes] = []
    if not data:
        return out
    for raw in data.decode('utf-8', 'replace').splitlines():
        line = raw.strip()
        if not line:
            continue
        try:
            out.append(_canon(common.loads_strict(line, what=label)))
        except InvalidInput:
            out.append(line.encode('utf-8'))
    return out


def _raw_lines(data: Optional[bytes]) -> List[str]:
    if not data:
        return []
    return [ln.strip() for ln in data.decode('utf-8', 'replace').splitlines() if ln.strip()]


def corpus_dir() -> Path:
    return common.KIT_DIR / 'corpus'


def kit_expected() -> dict:
    """tools/minifly_check/EXPECTED.json of the running kit (fixtures live in the kit, not in a fixture repo)."""
    obj = common.load_json_file(common.KIT_DIR / 'EXPECTED.json')
    if not isinstance(obj, dict):
        raise InvalidInput('EXPECTED.json must be an object')
    return obj


# --------------------------------------------------------------------------------------
# Row schemas (KB-01)
# --------------------------------------------------------------------------------------

def row_problems(name: str, row: Any) -> List[str]:
    """Schema problems of one ledger row (empty when valid). Extra fields are allowed; required fields,
    types and enums are checked (SPEC 8, SCHEMAS.md Section 18)."""
    p: List[str] = []
    if not isinstance(row, dict):
        return ['row is not a JSON object']

    def need_str(k, allow_null=False):
        v = row.get(k)
        if v is None and allow_null and k in row:
            return
        if not _s(v):
            p.append('%s must be a non-empty string%s' % (k, ' or null' if allow_null else ''))

    def need_utc(k, allow_null=False):
        v = row.get(k)
        if v is None and allow_null and k in row:
            return
        if common.normalize_utc(v) is None:
            p.append('%s must be a UTC time (YYYYMMDDTHHMMSSZ), got %r' % (k, v))

    def need_bool(k):
        if not isinstance(row.get(k), bool):
            p.append('%s must be true or false' % k)

    if name == 'FINDINGS':
        fid = row.get('id')
        if not (isinstance(fid, str) and F_ID_RE.match(fid)):
            p.append('id %r is not F-<yyyymmdd>-<agent>-<n>, F-UF-xx or F-ARCH-nn' % (fid,))
        kind = row.get('kind')
        if kind not in FINDING_KINDS:
            p.append('kind %r not in %s' % (kind, '|'.join(FINDING_KINDS)))
        need_str('author_agent')
        need_str('delivery')
        need_utc('utc')
        st = row.get('status')
        if kind == 'finding':
            if row.get('refers_to') is not None:
                p.append('a finding row has refers_to null')
            for k in ('claim', 'level', 'source'):
                need_str(k)
            for k in ('conditions', 'effect'):
                if not isinstance(row.get(k), dict):
                    p.append('%s must be an object' % k)
            if st not in FINDING_STATUSES:
                p.append('status %r not in %s' % (st, '|'.join(FINDING_STATUSES)))
            if row.get('caveat') is not None and not isinstance(row.get('caveat'), str):
                p.append('caveat must be a string or null')
        elif kind in FINDING_KINDS:
            rt = row.get('refers_to')
            if not (isinstance(rt, str) and F_ID_RE.match(rt)):
                p.append('%s row refers_to %r is not a finding id' % (kind, rt))
            if st is not None and st not in FINDING_STATUSES and not (kind == 'reconciliation'
                                                                      and st in RECONCILIATION_STATUSES):
                p.append('status %r is not a known status' % (st,))
            if kind in ('narrowing', 'supersession', 'retraction') and not (_s(row.get('caveat')) or _s(row.get('claim'))):
                p.append('%s row needs a caveat or claim stating the change' % kind)
            if kind == 'activation' and not (_s(row.get('evidence')) or _s(row.get('source'))):
                p.append('activation row needs evidence (or source) of the re-derivation')
            if kind == 'transfer':
                need_str('prediction_id')
                if row.get('outcome') not in TRANSFER_OUTCOMES:
                    p.append('transfer outcome %r not in ok|failed' % (row.get('outcome'),))
            if row.get('holds_under') is not None and row.get('holds_under') not in R2_BASES:
                p.append('holds_under %r not in %s' % (row.get('holds_under'), '|'.join(R2_BASES)))
    elif name == 'INCIDENTS':
        need_str('id')
        kind = row.get('kind')
        if kind not in ('report', 'closure'):
            p.append('kind %r not in report|closure' % (kind,))
        fm = row.get('fm_class')
        if not (isinstance(fm, str) and FM_RE.match(fm)):
            p.append('fm_class %r is not FM-xx' % (fm,))
        need_str('status')
        if 'delivery' not in row:
            p.append('delivery missing')
        if kind == 'report':
            for k in ('symptom', 'evidence', 'found_by_agent'):
                need_str(k)
            need_utc('found_utc')
            need_bool('caught_at_discovery')
            need_bool('heldout_eligible')
            v = row.get('checker_sha_at_discovery')
            if v is not None and not _s(v):
                p.append('checker_sha_at_discovery must be a string or null')
        elif kind == 'closure':
            for k in ('check_id', 'fixture_id'):
                if k not in row:
                    p.append('closure row lacks %s' % k)
            if common.normalize_utc(row.get('utc')) is None and common.normalize_utc(row.get('found_utc')) is None:
                p.append('closure row needs utc (or found_utc)')
            rt = row.get('refers_to')
            if rt is not None and not _s(rt):
                p.append('refers_to must be an incident id or null')
    elif name == 'USER_DECISIONS':
        uid = row.get('id')
        if not (isinstance(uid, str) and UD_ID_RE.match(uid)):
            p.append('id %r is not UD-<yyyymmdd>-<n>' % (uid,))
        need_utc('utc')
        for k in ('verbatim', 'session', 'channel', 'recorded_by'):
            need_str(k)
        sc = row.get('scope')
        if not (isinstance(sc, list) and sc and all(_s(x) for x in sc)):
            p.append('scope must be a non-empty list of strings')
        if not isinstance(row.get('numbers_restated'), list):
            p.append('numbers_restated must be a list')
    elif name == 'DELIVERY_LEDGER':
        need_str('delivery')
        need_utc('utc')
        leg = row.get('legacy')
        if not isinstance(leg, bool):
            p.append('legacy must be true or false')
        oh = row.get('opened_held_E3')
        if not (oh is True or oh is False or oh == 'unknown'):
            p.append("opened_held_E3 must be true, false or 'unknown'")
        if not isinstance(row.get('e3_basis'), list):
            p.append('e3_basis must be a list')
        om = row.get('objective_metric')
        if not (isinstance(om, dict) and all(k in om for k in ('held', 'old_final', 'G'))):
            p.append('objective_metric must be {held, old_final, G}')
        if leg is True:
            if oh is True:
                p.append('legacy rows never set opened_held_E3 true')
        elif leg is False:
            if row.get('task_family') not in TASK_FAMILIES:
                p.append('task_family %r not in %s' % (row.get('task_family'), '|'.join(TASK_FAMILIES)))
            if row.get('level_earned') not in LEVELS + ('NONE',):
                p.append('level_earned %r not in E0..E4|NONE' % (row.get('level_earned'),))
            if row.get('primary_gate') not in PRIMARY_GATES:
                p.append('primary_gate %r not in %s' % (row.get('primary_gate'), '|'.join(PRIMARY_GATES)))
            if row.get('gate_type') not in GATE_TYPES:
                p.append('gate_type %r not in objective|component' % (row.get('gate_type'),))
            if row.get('worlds_role') not in WORLDS_ROLES:
                p.append('worlds_role %r not in %s' % (row.get('worlds_role'), '|'.join(WORLDS_ROLES)))
            if not isinstance(oh, bool):
                p.append('v1 rows set opened_held_E3 to true or false')
            if not (isinstance(row.get('family_id'), str) and FAM_RE.match(row['family_id'])):
                p.append('family_id %r is not FAM-<sha8>' % (row.get('family_id'),))
            if not (isinstance(row.get('claim_sha256'), str) and HEX64_RE.match(row['claim_sha256'])):
                p.append('claim_sha256 must be a sha256')
    elif name == 'CONFIRM_LEDGER':
        need_str('family_id')
        need_str('delivery')
        need_utc('utc')
        need_str('seal_reveal_commit')
        if not (_int(row.get('attempt_index')) and row['attempt_index'] >= 0):
            p.append('attempt_index must be an integer >= 0')
        a = row.get('alpha_allocated')
        if not (_num(a) and 0 < a <= 1):
            p.append('alpha_allocated must be a number in (0, 1]')
    elif name == 'REVIEW_DEBT':
        for k in ('id', 'delivery', 'reason', 'recorded_by'):
            need_str(k)
        need_utc('utc')
        if 'repaid_by' not in row:
            p.append('repaid_by missing (null while open)')
        elif row.get('repaid_by') is not None and not _s(row.get('repaid_by')):
            p.append('repaid_by must be a non-empty string or null')
    elif name == 'CANARY_COMMITMENTS':
        for k in ('batch_id', 'author_identity', 'checker_sha_under_test'):
            need_str(k)
        need_utc('utc')
        for k in ('bundle_sha256', 'key_sha256'):
            if not (isinstance(row.get(k), str) and HEX64_RE.match(row[k])):
                p.append('%s must be a sha256' % k)
        n, nc = row.get('n'), row.get('n_clean')
        if not (_int(n) and n > 0):
            p.append('n must be a positive integer')
        if not (_int(nc) and nc >= 0 and (not _int(n) or nc <= n)):
            p.append('n_clean must be an integer in [0, n]')
        if not isinstance(row.get('fm_coverage'), list):
            p.append('fm_coverage must be a list')
        if 'revealed_utc' not in row:
            p.append('revealed_utc missing (null until reveal)')
        elif row.get('revealed_utc') is not None and common.normalize_utc(row.get('revealed_utc')) is None:
            p.append('revealed_utc must be a UTC time or null')
    else:
        p.append('unknown ledger %s' % name)
    return p


# --------------------------------------------------------------------------------------
# Review debt, incidents, closures
# --------------------------------------------------------------------------------------

def _same_delivery(a: Any, b: Any) -> bool:
    if not isinstance(a, str) or not isinstance(b, str):
        return False
    a, b = a.strip().strip('/'), b.strip().strip('/')
    return a == b or a.rsplit('/', 1)[-1] == b.rsplit('/', 1)[-1]


def open_debts(kb: common.KB, delivery: Optional[str] = None) -> List[dict]:
    """REVIEW_DEBT rows that are open (no row with the same id and repaid_by set)."""
    rows = kb.rows('REVIEW_DEBT')
    repaid = set(r.get('id') for r in rows if r.get('repaid_by'))
    out = [r for r in rows if not r.get('repaid_by') and r.get('id') not in repaid]
    if delivery is not None:
        out = [r for r in out if _same_delivery(r.get('delivery'), delivery)]
    return out


def closure_problems(row: dict, rule_idx: Dict[str, dict], expected: Optional[dict],
                     missing: Optional[List[str]] = None) -> List[str]:
    """KB-05 for one closure: check_id in RULES (not planned) and a fixture whose EXPECTED.must_fire
    includes that check (with the fixture directory present in the kit corpus). When EXPECTED.json is
    unavailable the fixture cannot be verified: noted in `missing` (or as a problem when no list is given)."""
    p: List[str] = []
    cid, fx = row.get('check_id'), row.get('fixture_id')
    if not _s(cid):
        p.append('no check_id')
    elif cid not in rule_idx:
        p.append('check_id %s is not in RULES.json' % cid)
    elif rule_idx[cid].get('status') == 'planned':
        p.append('check_id %s is planned (not enforced)' % cid)
    if not _s(fx):
        p.append('no fixture_id')
    elif expected is None:
        if missing is not None:
            missing.append('fixture %s cannot be verified: EXPECTED.json unavailable' % fx)
        else:
            p.append('EXPECTED.json unavailable')
    else:
        ent = expected.get(fx)
        if not isinstance(ent, dict):
            p.append('fixture %s has no EXPECTED.json entry' % fx)
        else:
            mf = ent.get('must_fire') or []
            if not isinstance(mf, list) or cid not in mf:
                p.append('EXPECTED[%s].must_fire does not include %s' % (fx, cid))
            if not (corpus_dir() / str(fx)).is_dir():
                p.append('fixture directory corpus/%s is absent' % fx)
    return p


def incident_closure_target(row: dict) -> Optional[str]:
    rt = row.get('refers_to')
    return rt if _s(rt) else row.get('id')


def open_incidents(kb: common.KB, rule_idx: Optional[Dict[str, dict]] = None,
                   expected: Optional[dict] = None) -> List[dict]:
    """Report rows without an effective closure (a closure row that satisfies KB-05)."""
    rows = kb.rows('INCIDENTS')
    if rule_idx is None:
        try:
            rule_idx = common.rule_index(common.load_rules())
        except CheckerError:
            rule_idx = {}
    if expected is None:
        try:
            expected = kit_expected()
        except (MissingInput, InvalidInput):
            expected = None
    closed = set()
    for r in rows:
        if r.get('kind') == 'closure' and not closure_problems(r, rule_idx, expected):
            closed.add(incident_closure_target(r))
    return [r for r in rows if r.get('kind') == 'report' and r.get('id') not in closed]


# --------------------------------------------------------------------------------------
# Source-delivery eligibility (KB-04) and the FINDINGS fold
# --------------------------------------------------------------------------------------

def source_delivery_status(kb: common.KB, delivery: Any, git: Optional[common.GitView] = None,
                           rules: Optional[dict] = None) -> Tuple[bool, List[str]]:
    """KB-04 conditions on a finding's source delivery: a v1 delivery whose generated CLAIM (linked by a
    DELIVERY_LEDGER claim_sha256) has worlds_role MAIN or SEALED_CONFIRM, computed tier 2, no open debt."""
    key = ('src_status', str(delivery))
    if key in kb._cache:
        return kb._cache[key]
    reasons: List[str] = []
    root = Path(kb.repo_root)
    if not _s(delivery):
        res = (False, ['finding has no source delivery'])
        kb._cache[key] = res
        return res
    try:
        rel = common.norm_rel(delivery)
    except InvalidInput as e:
        res = (False, [str(e)])
        kb._cache[key] = res
        return res
    d = root / rel
    if not common.is_v1_delivery(d):
        reasons.append('source %s is not a v1 delivery (%s)' % (delivery, common.v1_status(d)[1]))
    else:
        cp = d / 'CLAIM.json'
        if not cp.is_file():
            reasons.append('source %s has no CLAIM.json' % delivery)
        else:
            try:
                claim = common.load_json_file(cp)
                role = claim.get('worlds_role') if isinstance(claim, dict) else None
                if role not in ('MAIN', 'SEALED_CONFIRM'):
                    reasons.append('source worlds_role %r is not MAIN or SEALED_CONFIRM' % (role,))
                sha = common.sha256_file(cp)
                linked = [r for r in kb.rows('DELIVERY_LEDGER') if _same_delivery(r.get('delivery'), rel)
                          and r.get('claim_sha256') == sha]
                if not linked:
                    reasons.append('source CLAIM.json %s is not linked by a DELIVERY_LEDGER row (run MFC claim)' % sha[:12])
            except (MissingInput, InvalidInput) as e:
                reasons.append('source CLAIM.json unreadable: %s' % e)
            try:
                sctx = common.Ctx(repo_root=root, delivery=d, git=git or common.GitView(), kb=kb,
                                  rules=rules or common.load_rules(), stage='publish')
                tier, _why = common.compute_tier(sctx)
                if tier < 2:
                    reasons.append('source delivery computed tier %d < 2' % tier)
            except (MissingInput, InvalidInput, CheckerError, NotImplementedError) as e:
                reasons.append('source tier undeterminable: %s' % e)
        try:
            debts = open_debts(kb, rel)
            if debts:
                reasons.append('source has open review debt %s' % ','.join(str(r.get('id')) for r in debts))
        except (MissingInput, InvalidInput) as e:
            reasons.append('REVIEW_DEBT unreadable: %s' % e)
    res = (not reasons, reasons)
    kb._cache[key] = res
    return res


def activation_problems(kb: common.KB, finding: dict, row: dict, git: Optional[common.GitView] = None,
                        rules: Optional[dict] = None, check_source: bool = True) -> List[str]:
    """Field-level KB-03 (family) and KB-04 (source) problems of one activation/reconciliation row."""
    p: List[str] = []
    fa, ra = family(finding.get('author_agent')), family(row.get('author_agent'))
    if fa is None:
        p.append('finding author %r is of no known family' % (finding.get('author_agent'),))
    if ra is None:
        p.append('%s author %r is of no known family' % (row.get('kind'), row.get('author_agent')))
    if fa is not None and fa == ra:
        p.append('%s by %s of a finding authored by %s (same family %s)' % (
            row.get('kind'), row.get('author_agent'), finding.get('author_agent'), fa))
    if check_source and row.get('kind') == 'activation':
        ok, why = source_delivery_status(kb, finding.get('delivery'), git, rules)
        if not ok:
            p.extend(why)
    return p


def _finding_family(kb: common.KB, finding: dict) -> Optional[str]:
    if _s(finding.get('task_family')):
        return finding['task_family']
    try:
        rows = [r for r in kb.rows('DELIVERY_LEDGER') if _same_delivery(r.get('delivery'), finding.get('delivery'))]
    except (MissingInput, InvalidInput):
        rows = []
    fams = [r.get('task_family') for r in rows if _s(r.get('task_family'))]
    return fams[-1] if fams else None


def finding_state(kb: common.KB, fid: str, git: Optional[common.GitView] = None,
                  rules: Optional[dict] = None) -> Optional[dict]:
    """The fold of SPEC 11 over the finding row and the rows referring to it. None when no finding row.

    Status: the most restrictive of the finding's own status and the statuses implied by the rows
    referring to it (proposed/provisional < active < unverifiable_in_repo < narrowed < superseded <
    retracted), independent of row order. A finding's own 'active' status counts only through a valid
    activation row (other family, KB-03; T2 MAIN/SEALED_CONFIRM source without debt, KB-04); invalid
    activations are ignored and listed.

    R-level: R1 a valid activation (or other-family reconciliation) with evidence; R2 a non-author row with
    holds_under rederived_metric|changed_presentation; R3 a transfer row 'ok' from another delivery; R4 an
    'ok' transfer on a different task family or substrate. Only R3+ is called transferable."""
    rows = kb.rows('FINDINGS')
    base = None
    refs: List[dict] = []
    for r in rows:
        kind = r.get('kind') or 'finding'
        if kind == 'finding' and r.get('id') == fid:
            base = r
        elif r.get('refers_to') == fid and kind != 'finding':
            refs.append(r)
    if base is None:
        return None
    notes: List[str] = []
    own = base.get('status') or 'proposed'
    if own == 'active':
        own = 'provisional'
        notes.append("the finding row's own status 'active' is ignored: activation needs a valid activation row")
    statuses = [own]
    caveats = [base['caveat']] if _s(base.get('caveat')) else []
    narrowing_rows: List[str] = []
    valid_acts: List[str] = []
    invalid_acts: List[dict] = []
    r1 = r2 = False
    transfers = {'ok': [], 'failed': []}
    fa = family(base.get('author_agent'))
    for r in refs:
        kind = r.get('kind')
        rid = r.get('id')
        if kind == 'activation':
            pr = activation_problems(kb, base, r, git, rules)
            if pr:
                invalid_acts.append({'id': rid, 'problems': pr})
                continue
            valid_acts.append(rid)
            statuses.append('active')
            if _s(r.get('evidence')) or _s(r.get('source')):
                r1 = True
        elif kind in ('narrowing', 'supersession', 'retraction'):
            statuses.append(common.FINDING_KIND_STATUS[kind])
            if kind in ('narrowing', 'supersession'):
                narrowing_rows.append(rid)
            if _s(r.get('caveat')):
                caveats.append(r['caveat'])
        elif kind == 'reconciliation':
            st = r.get('status')
            other = fa is not None and family(r.get('author_agent')) not in (None, fa)
            if isinstance(st, str) and st in common.FINDING_STATUS_RANK and st != 'active':
                statuses.append(st)
            if other and (_s(r.get('evidence')) or _s(r.get('source'))) and st not in ('REFUTED', 'retracted'):
                r1 = True
        elif kind == 'transfer':
            if r.get('outcome') in TRANSFER_OUTCOMES:
                transfers[r['outcome']].append(r)
        if (kind in ('activation', 'reconciliation') and r.get('holds_under') in R2_BASES
                and fa is not None and family(r.get('author_agent')) not in (None, fa)
                and r.get('outcome') != 'failed'):
            r2 = True
    status = max(statuses, key=lambda s: common.FINDING_STATUS_RANK.get(s, 2))
    ok_later = [t for t in transfers['ok'] if not _same_delivery(t.get('delivery'), base.get('delivery'))]
    r3 = bool(ok_later)
    bf = _finding_family(kb, base)
    bs = (base.get('conditions') or {}).get('substrate') if isinstance(base.get('conditions'), dict) else None
    r4 = any((_s(t.get('task_family')) and bf and t['task_family'] != bf)
             or (_s(t.get('substrate')) and _s(bs) and t['substrate'] != bs) for t in ok_later)
    level = 4 if r4 else 3 if r3 else 2 if r2 else 1 if r1 else 0
    return {'id': fid, 'status': status, 'r_level': 'R%d' % level,
            'transferable': level >= 3 and status not in ('superseded', 'retracted'),
            'caveats': caveats, 'narrowing_rows': narrowing_rows, 'valid_activations': valid_acts,
            'invalid_activations': invalid_acts, 'transfers_ok': len(transfers['ok']),
            'transfers_failed': len(transfers['failed']), 'delivery': base.get('delivery'),
            'author_agent': base.get('author_agent'), 'claim': base.get('claim'), 'notes': notes}


def finding_status(kb: common.KB, fid: str, git: Optional[common.GitView] = None,
                   rules: Optional[dict] = None) -> Optional[str]:
    st = finding_state(kb, fid, git, rules)
    return st['status'] if st else None


# --------------------------------------------------------------------------------------
# E3 cadence (SPEC 6.2 item 10; LOCK-09 calls deliveries_since_last_e3)
# --------------------------------------------------------------------------------------

_STRATA_RANGE_RE = re.compile(r's\s*(\d)\s*(?:-|\.\.|to)\s*s?\s*(\d)')
_STRATA_ONE_RE = re.compile(r's\s*(\d)')


def _basis_strata(entry: str) -> Optional[List[int]]:
    e = entry.lower()
    if 'strat' not in e:
        return None
    nums: List[int] = []
    for a, b in _STRATA_RANGE_RE.findall(e):
        lo, hi = int(a), int(b)
        if lo > hi:
            return []
        nums.extend(range(lo, hi + 1))
    rest = _STRATA_RANGE_RE.sub(' ', e)
    nums.extend(int(x) for x in _STRATA_ONE_RE.findall(rest))
    return nums


def opened_held_e3_valid(row: dict) -> Tuple[bool, List[str]]:
    """True only for a v1 row with opened_held_E3 true whose e3_basis shows pool HELD, counted strata within
    S2..S4 and admission passed, with worlds_role MAIN|SEALED_CONFIRM and primary_gate PASS|FAIL."""
    why: List[str] = []
    if row.get('opened_held_E3') is not True:
        why.append('opened_held_E3 is %r' % (row.get('opened_held_E3'),))
    if row.get('legacy') is not False:
        why.append('legacy rows never reset the cadence')
    basis = [str(x) for x in (row.get('e3_basis') or [])] if isinstance(row.get('e3_basis'), list) else []
    if not any(re.search(r'\bpool\W+held\b', b.lower()) for b in basis):
        why.append('e3_basis does not show pool HELD')
    nums = [n for s in (_basis_strata(b) for b in basis) if s is not None for n in s]
    if not nums or not all(2 <= n <= 4 for n in nums):
        why.append('e3_basis does not show counted strata within S2..S4')
    if not any(re.search(r'\badmission\W+(pass|passed|admitted)\b', b.lower()) for b in basis):
        why.append('e3_basis does not show admission passed')
    if row.get('worlds_role') not in ('MAIN', 'SEALED_CONFIRM'):
        why.append('worlds_role %r is not MAIN|SEALED_CONFIRM' % (row.get('worlds_role'),))
    if row.get('primary_gate') not in ('PASS', 'FAIL'):
        why.append('primary_gate %r is not PASS|FAIL (an honest exit does not reset the cadence)'
                   % (row.get('primary_gate'),))
    return (not why), why


def e3_cadence(kb: common.KB) -> dict:
    """Deliveries counted since the last valid held-E3 opening. One entry per delivery (DELIVERY_LEDGER
    holds one row per `claim` run): positioned at its earliest row, judged by its latest row."""
    rows = kb.rows('DELIVERY_LEDGER')
    cons = kb.constraints()
    counts_legacy = cons.get('e3_cadence_counts_legacy') is True
    first: Dict[str, Tuple[str, int]] = {}
    latest: Dict[str, Tuple[Tuple[str, int], dict]] = {}
    for i, r in enumerate(rows):
        d = r.get('delivery')
        if not _s(d):
            continue
        k = (utc_key(r.get('utc')), i)
        if d not in first or k < first[d]:
            first[d] = k
        if d not in latest or k >= latest[d][0]:
            latest[d] = (k, r)
    entries = sorted(((first[d], d, latest[d][1]) for d in latest), key=lambda x: x[0])
    last = None
    for j, (_k, d, r) in enumerate(entries):
        if opened_held_e3_valid(r)[0]:
            last = j
    after = entries[last + 1:] if last is not None else entries
    counted = [d for _k, d, r in after if r.get('legacy') is not True or counts_legacy]
    return {'deliveries_since_last_e3': len(counted), 'counted': counted,
            'last_e3_delivery': entries[last][1] if last is not None else None,
            'counts_legacy': counts_legacy, 'e3_cadence_max': cons.get('e3_cadence_max')}


def deliveries_since_last_e3(kb: common.KB) -> int:
    """Count of v1 DELIVERY_LEDGER deliveries (legacy too when CONSTRAINTS.e3_cadence_counts_legacy) since the
    last row with a valid opened_held_E3 (see opened_held_e3_valid)."""
    return int(e3_cadence(kb)['deliveries_since_last_e3'])


# --------------------------------------------------------------------------------------
# Appending rows (ledger append, claim)
# --------------------------------------------------------------------------------------

def _yyyymmdd(row: dict) -> str:
    u = common.normalize_utc(row.get('utc')) or common.utc_now()
    return u[:8]


def _agent_token(agent: Any) -> str:
    fam = family(agent)
    if fam:
        return fam
    tok = re.sub(r'[^a-z0-9]', '', str(agent or '').lower())
    if not tok or not tok[0].isalpha():
        raise InvalidInput('author_agent %r cannot form a finding id' % (agent,))
    return tok[:16]


def _next_n(existing: Iterable[str], prefix: str) -> int:
    n = 0
    rx = re.compile('^' + re.escape(prefix) + r'(\d+)$')
    for i in existing:
        m = rx.match(str(i)) if isinstance(i, str) else None
        if m:
            n = max(n, int(m.group(1)))
    return n + 1


def assign_id(kb: common.KB, name: str, row: dict) -> dict:
    """Fill in the id (and utc) of a new row; an explicit id is kept."""
    row = dict(row)
    if not row.get('utc'):
        row['utc'] = common.utc_now()
    if name == 'CANARY_COMMITMENTS':
        return row
    if _s(row.get('id')):
        return row
    rows = kb.rows(name)
    ids = [r.get('id') for r in rows]
    day = _yyyymmdd(row)
    if name == 'FINDINGS':
        prefix = 'F-%s-%s-' % (day, _agent_token(row.get('author_agent')))
    elif name == 'USER_DECISIONS':
        prefix = 'UD-%s-' % day
    elif name == 'INCIDENTS':
        if row.get('kind') == 'closure' and _s(row.get('refers_to')):
            prefix = '%s-CL' % row['refers_to']
        else:
            prefix = 'I-%s-' % day
    elif name == 'REVIEW_DEBT':
        prefix = 'DEBT-%s-' % day
    else:
        raise InvalidInput('%s rows are written by the kit, not by ledger append' % name)
    row['id'] = '%s%d' % (prefix, _next_n(ids, prefix))
    return row


def _write_line(path: Path, row: dict) -> None:
    line = common.canonical_json(row) + b'\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    prefix = b''
    if path.is_file() and path.stat().st_size > 0:
        with open(path, 'rb') as fh:
            fh.seek(-1, os.SEEK_END)
            if fh.read(1) != b'\n':
                prefix = b'\n'
    with open(path, 'ab') as fh:
        fh.write(prefix + line)


def append_row(kb: common.KB, name: str, row: dict) -> dict:
    """Validate (KB-01 schema and key uniqueness) and append exactly one line to KB/<name>.jsonl."""
    if name not in LEDGERS:
        raise InvalidInput('unknown ledger %s' % name)
    probs = row_problems(name, row)
    if probs:
        raise InvalidInput('%s row invalid: %s' % (name, '; '.join(probs)))
    path = kb.path(name + '.jsonl')
    if not path.is_file():
        raise MissingInput('ledger %s does not exist (create the empty file through governance)' % path)
    existing = kb.rows(name)
    keyfn = common.LEDGER_KEYS.get(name, lambda r: r.get('id'))
    k = keyfn(row)
    canon = _canon(row)
    if k is not None:
        hk = _canon(list(k) if isinstance(k, tuple) else k)
        for r in existing:
            rk = keyfn(r)
            if rk is not None and _canon(list(rk) if isinstance(rk, tuple) else rk) == hk:
                if _canon(r) == canon:
                    return row  # identical line already present (idempotent)
                raise InvalidInput('%s already has a row with key %r (append-only; ids are unique)' % (name, k))
    _write_line(path, row)
    kb._cache.pop('rows:' + name, None)
    for key in [k2 for k2 in kb._cache if isinstance(k2, tuple) and k2[:1] == ('src_status',)]:
        kb._cache.pop(key, None)
    return row


def append_problems(ctx: common.Ctx, name: str, row: dict) -> List[str]:
    """Semantic rules for a row appended through `ledger append` (beyond the schema)."""
    kb = ctx.kb
    p: List[str] = []
    if name == 'FINDINGS':
        kind = row.get('kind')
        if kind == 'finding':
            if row.get('status') not in NEW_FINDING_STATUSES:
                p.append("a new finding is appended with status proposed (got %r); only the other agent family "
                         "activates it through an activation row" % (row.get('status'),))
            if not (isinstance(row.get('source'), str) and STRICT_SOURCE_RE.match(row['source'])):
                p.append('source %r is not path:line[@sha]' % (row.get('source'),))
            cond = row.get('conditions') if isinstance(row.get('conditions'), dict) else {}
            if not _s(cond.get('best_shortcut')):
                p.append('conditions.best_shortcut is required')
        else:
            target = None
            for r in kb.rows('FINDINGS'):
                if r.get('id') == row.get('refers_to') and (r.get('kind') or 'finding') == 'finding':
                    target = r
            if target is None:
                p.append('refers_to %r is not a finding in FINDINGS' % (row.get('refers_to'),))
            elif kind in ('activation', 'reconciliation'):
                p.extend(activation_problems(kb, target, row, ctx.git, ctx.rules))
    elif name == 'INCIDENTS':
        if row.get('kind') == 'closure' or row.get('status') == 'closed':
            try:
                exp = kit_expected()
            except (MissingInput, InvalidInput) as e:
                exp = None
                p.append('EXPECTED.json unavailable: %s' % e)
            p.extend('closure: %s' % x for x in closure_problems(row, common.rule_index(ctx.rules), exp))
            tgt = incident_closure_target(row)
            if row.get('kind') == 'closure' and not any(r.get('id') == tgt and r.get('kind') == 'report'
                                                        for r in kb.rows('INCIDENTS')):
                p.append('closure refers to %r, which is not a report row' % (tgt,))
    elif name == 'REVIEW_DEBT':
        if row.get('repaid_by'):
            if not any(r.get('id') == row.get('id') and not r.get('repaid_by') for r in kb.rows('REVIEW_DEBT')):
                p.append('repayment row names %r, which is not an open debt' % (row.get('id'),))
        elif row.get('id') and any(r.get('id') == row.get('id') for r in kb.rows('REVIEW_DEBT')):
            p.append('debt id %r already exists' % (row.get('id'),))
    elif name == 'CANARY_COMMITMENTS':
        if row.get('revealed_utc') and not any(r.get('batch_id') == row.get('batch_id') for r in
                                               kb.rows('CANARY_COMMITMENTS')):
            p.append('reveal row for batch %r without a commitment row' % (row.get('batch_id'),))
    return p


def cmd_ledger(args: Any) -> int:
    """MFC ledger append --ledger L --row ROW.json: validate, assign the id, append one line."""
    if getattr(args, 'action', None) != 'append':
        sys.stderr.write('ledger: only "append" is supported\n')
        return common.EXIT_ERROR
    name = args.ledger
    if name not in APPENDABLE:
        sys.stderr.write('ledger append: %s is written by the kit (claim, seal reveal), not by hand\n' % name)
        return common.EXIT_FAIL
    ctx = common.make_ctx(args, delivery=None, stage='ledger', options={'invoked_by': 'ledger'})
    rp = Path(args.row)
    if not rp.is_absolute() and not rp.is_file():
        rp = Path(ctx.repo_root) / args.row
    row = common.load_json_file(rp)
    if not isinstance(row, dict):
        sys.stderr.write('ledger append: %s is not a JSON object\n' % args.row)
        return common.EXIT_FAIL
    row = assign_id(ctx.kb, name, row)
    probs = row_problems(name, row) + append_problems(ctx, name, row)
    if probs:
        sys.stderr.write('ledger append REFUSED (%s): %s\n' % (name, '; '.join(probs)))
        return common.EXIT_FAIL
    row = append_row(ctx.kb, name, row)
    if getattr(args, 'json', False):
        print(json.dumps(row, indent=2, ensure_ascii=False))
    else:
        print('appended %s row %s to %s' % (name, row.get('id') or row.get('batch_id'), ledger_rel(name)))
    return common.EXIT_PASS


# --------------------------------------------------------------------------------------
# KB score (SPEC 11) and the one-shot backtest
# --------------------------------------------------------------------------------------

def latest_delivery_rows(kb: common.KB, v1_only: bool = True) -> Dict[str, dict]:
    out: Dict[str, Tuple[Tuple[str, int], dict]] = {}
    for i, r in enumerate(kb.rows('DELIVERY_LEDGER')):
        d = r.get('delivery')
        if not _s(d) or (v1_only and r.get('legacy') is not False):
            continue
        k = (utc_key(r.get('utc')), i)
        if d not in out or k >= out[d][0]:
            out[d] = (k, r)
    return {d: v[1] for d, v in out.items()}


def laplace(k: int, n: int) -> float:
    return (k + 1.0) / (n + 2.0)


def _mean(xs: Sequence[float]) -> Optional[float]:
    xs = list(xs)
    return math.fsum(xs) / len(xs) if xs else None


def kb_score(kb: common.KB) -> dict:
    """Hit rate and Brier skill against the Laplace base rate per task family (labelled INSUFFICIENT_N when
    fewer than 5 scored deliveries, DEGENERATE_BASE when the base rate lies outside [0.1, 0.9]); values are
    reported in every case, as diagnostics only. Per-finding transfer counts from FINDINGS."""
    root = Path(kb.repo_root)
    per_family: Dict[str, dict] = {}
    excluded: List[dict] = []
    for d, row in sorted(latest_delivery_rows(kb).items()):
        fam = row.get('task_family') or 'unknown'
        ent = per_family.setdefault(fam, {'deliveries': [], 'primary': [], 'intervals': []})
        cp = root / common.norm_rel(d) / 'CLAIM.json'
        if not cp.is_file():
            excluded.append({'delivery': d, 'reason': 'CLAIM.json absent'})
            continue
        if common.sha256_file(cp) != row.get('claim_sha256'):
            excluded.append({'delivery': d, 'reason': 'CLAIM.json sha differs from the ledger claim_sha256'})
            continue
        claim = common.load_json_file(cp)
        gate = row.get('primary_gate')
        if gate not in ('PASS', 'FAIL'):
            excluded.append({'delivery': d, 'reason': 'primary_gate %s is unresolved' % gate})
            continue
        o = 1 if gate == 'PASS' else 0
        ent['deliveries'].append({'delivery': d, 'outcome': o})
        # SPEC 8 entries are {id, observed, hit, interval_score, brier}; target and p_pass come from the lock
        # prediction with that id when the entry does not repeat them (claim adds them; older claims may not)
        lock_preds: Dict[str, dict] = {}
        lp = root / common.norm_rel(d) / 'ENDPOINT_LOCK.json'
        if lp.is_file():
            try:
                lk = common.load_json_file(lp)
            except InvalidInput:
                lk = None
            for pr in (lk.get('predictions') or []) if isinstance(lk, dict) else []:
                if isinstance(pr, dict) and _s(pr.get('id')):
                    lock_preds[pr['id']] = pr
        for ps in claim.get('predictions_scored') or []:
            if not isinstance(ps, dict):
                continue
            st = ps.get('status')
            if st is None:
                st = 'scored' if (_num(ps.get('observed')) or isinstance(ps.get('hit'), bool)) else 'unresolved'
            if st != 'scored':
                continue
            lpred = lock_preds.get(ps.get('id')) or {}
            target = ps.get('target', lpred.get('target'))
            p_pass = ps.get('p_pass', lpred.get('p_pass'))
            if target == 'primary' and _num(p_pass):
                ent['primary'].append({'delivery': d, 'id': ps.get('id'), 'p': float(p_pass), 'o': o})
            if isinstance(ps.get('hit'), bool):
                ent['intervals'].append({'delivery': d, 'id': ps.get('id'), 'hit': ps['hit']})
    families = {}
    for fam, ent in sorted(per_family.items()):
        n = len(ent['deliveries'])
        k = sum(x['outcome'] for x in ent['deliveries'])
        base = laplace(k, n)
        bp = _mean([(x['p'] - x['o']) ** 2 for x in ent['primary']])
        bb = _mean([(base - x['o']) ** 2 for x in ent['primary']])
        skill = (1.0 - bp / bb) if (bp is not None and bb) else None
        hit = _mean([1.0 if x['hit'] else 0.0 for x in ent['intervals']])
        if n < LAPLACE_MIN_N:
            label = 'INSUFFICIENT_N'
        elif not (BASE_RATE_BAND[0] <= base <= BASE_RATE_BAND[1]):
            label = 'DEGENERATE_BASE'
        else:
            label = 'OK'
        families[fam] = {'n_scored_deliveries': n, 'n_pass': k, 'laplace_base_rate': base,
                         'n_primary_predictions': len(ent['primary']), 'brier': bp, 'brier_base': bb,
                         'brier_skill': skill, 'n_interval_predictions': len(ent['intervals']), 'hit_rate': hit,
                         'label': label}
    # program-level summary over every family (diagnostic; the per-family labels govern interpretation)
    allp = [x for ent in per_family.values() for x in ent['primary']]
    brier_mean = _mean([(x['p'] - x['o']) ** 2 for x in allp])
    brier_mean = round(brier_mean, 6) if brier_mean is not None else None  # summary shown to 6 decimals
    skill_txt = '; '.join('%s: %s (n=%d, brier_skill %s)' % (
        fam, v['label'], v['n_scored_deliveries'],
        ('%.6g' % v['brier_skill']) if v['brier_skill'] is not None else 'n/a') for fam, v in sorted(families.items()))
    transfers: Dict[str, Dict[str, int]] = {}
    for r in kb.rows('FINDINGS'):
        if r.get('kind') == 'transfer' and r.get('outcome') in TRANSFER_OUTCOMES:
            t = transfers.setdefault(str(r.get('refers_to')), {'ok': 0, 'failed': 0})
            t[r['outcome']] += 1
    return {'schema': 'minifly.kb_score.v1', 'utc': common.utc_now(), 'families': families,
            'n_primary_predictions': len(allp), 'brier_mean': brier_mean, 'skill': skill_txt or 'INSUFFICIENT_N (n=0)',
            'finding_transfers': dict(sorted(transfers.items())), 'excluded': excluded,
            'base_rate_method': 'Laplace (k+1)/(n+2) over scored (PASS|FAIL) v1 deliveries of the task family, '
                                'one per delivery (latest DELIVERY_LEDGER row)',
            'note': 'KB diagnostic only: calibration is never an agent score or a kit score (SPEC 11)'}


def _read_forecasts(path: Path) -> Dict[str, float]:
    if not path.is_file():
        raise MissingInput('forecast file %s not found' % path)
    if path.name.endswith('.jsonl'):
        rows = common.load_jsonl_file(path)
    else:
        obj = common.load_json_file(path)
        if isinstance(obj, dict) and isinstance(obj.get('forecasts'), list):
            rows = obj['forecasts']
        elif isinstance(obj, list):
            rows = obj
        elif isinstance(obj, dict):
            rows = [{'delivery': k, 'p_pass': v} for k, v in obj.items()]
        else:
            raise InvalidInput('%s: forecasts must be a list or {forecasts: [...]}' % path)
    out: Dict[str, float] = {}
    for i, r in enumerate(rows):
        if not isinstance(r, dict) or not _s(r.get('delivery')) or not _num(r.get('p_pass')) \
                or not (0.0 <= float(r['p_pass']) <= 1.0):
            raise InvalidInput('%s row %d must be {delivery, p_pass in [0, 1]}' % (path, i))
        if r['delivery'] in out:
            raise InvalidInput('%s forecasts %s twice' % (path, r['delivery']))
        out[r['delivery']] = float(r['p_pass'])
    return out


def _read_outcomes(path: Optional[Path], kb: common.KB) -> Dict[str, int]:
    if path is not None:
        obj = common.load_json_file(path)
        items = obj.get('outcomes') if isinstance(obj, dict) and isinstance(obj.get('outcomes'), dict) else obj
        if not isinstance(items, dict):
            raise InvalidInput('%s: outcomes must be {delivery: PASS|FAIL|1|0}' % path)
        out = {}
        for d, v in items.items():
            if v in ('PASS', 1, True):
                out[d] = 1
            elif v in ('FAIL', 0, False):
                out[d] = 0
        return out
    out = {}
    for d, r in latest_delivery_rows(kb).items():
        if r.get('primary_gate') in ('PASS', 'FAIL'):
            out[d] = 1 if r['primary_gate'] == 'PASS' else 0
    return out


def backtest_score(kb: common.KB, contract: Dict[str, float], kbf: Dict[str, float], outcomes: Dict[str, int],
                   as_of: Optional[str] = None, n_boot: int = BOOTSTRAP_DEFAULT, seed: int = 0) -> dict:
    """Brier skill of the contract-only and KB-informed forecasts against Laplace, KB value =
    Brier(contract-only) - Brier(KB) with a paired bootstrap 95% CI. Always labelled LEAK_RISK."""
    ds = sorted(d for d in set(contract) & set(kbf) if d in outcomes)
    unresolved = sorted((set(contract) | set(kbf)) - set(ds))
    if not ds:
        raise MissingInput('no delivery has both forecasts and a resolved PASS|FAIL outcome')
    k = n = 0
    for d, r in latest_delivery_rows(kb).items():
        if d in ds or r.get('primary_gate') not in ('PASS', 'FAIL'):
            continue
        if as_of and utc_key(r.get('utc')) > as_of:
            continue
        n += 1
        k += 1 if r['primary_gate'] == 'PASS' else 0
    base = laplace(k, n)
    o = [outcomes[d] for d in ds]
    pc = [contract[d] for d in ds]
    pk = [kbf[d] for d in ds]

    def brier(ps, idx):
        return math.fsum((ps[i] - o[i]) ** 2 for i in idx) / len(idx)
    idx = list(range(len(ds)))
    b_c, b_k = brier(pc, idx), brier(pk, idx)
    b_base = math.fsum((base - x) ** 2 for x in o) / len(o)
    rng = random.Random(seed)
    vals = []
    for _ in range(int(n_boot)):
        s = [rng.randrange(len(ds)) for _i in idx]
        vals.append(brier(pc, s) - brier(pk, s))
    vals.sort()
    lo = vals[int(math.floor(0.025 * (len(vals) - 1)))] if vals else None
    hi = vals[int(math.ceil(0.975 * (len(vals) - 1)))] if vals else None
    return {'schema': 'minifly.kb_backtest.v1', 'label': 'LEAK_RISK', 'as_of': as_of, 'n': len(ds),
            'deliveries': ds, 'unresolved_or_unpaired': unresolved,
            'laplace_base_rate': base, 'laplace_basis': {'k': k, 'n': n},
            'brier_contract_only': b_c, 'brier_kb': b_k, 'brier_base': b_base,
            'skill_contract_only': (1.0 - b_c / b_base) if b_base else None,
            'skill_kb': (1.0 - b_k / b_base) if b_base else None,
            'kb_value': b_c - b_k, 'kb_value_ci95': [lo, hi], 'bootstrap': {'n': int(n_boot), 'seed': seed,
                                                                             'method': 'paired, deliveries resampled'},
            'note': 'LEAK_RISK: the container holds the full history; a falsifiable early signal, not a ritual'}


def _parse_as_of(text: Any) -> str:
    if not isinstance(text, str):
        raise InvalidInput('--as-of is required (e.g. 2026-10-15T00:00Z)')
    s = text.strip()
    if re.fullmatch(r'\d{4}-\d{2}-\d{2}', s):
        s += 'T00:00Z'
    u = common.normalize_utc(s)
    if u is None:
        raise InvalidInput('--as-of %r is not a UTC time (YYYY-MM-DDTHH:MMZ)' % text)
    return u


def _safe_extract(tar_bytes: bytes, out: Path) -> int:
    n = 0
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode='r:') as tf:
        for m in tf.getmembers():
            name = m.name
            if name.startswith('/') or '..' in Path(name).parts:
                raise InvalidInput('archive member %r escapes the output dir' % name)
            if m.isdir():
                (out / name).mkdir(parents=True, exist_ok=True)
                continue
            if not m.isfile():
                continue  # symlinks and devices are never materialized
            dest = out / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            fh = tf.extractfile(m)
            dest.write_bytes(fh.read() if fh else b'')
            n += 1
    return n


ARCHIVE_BATCH_BYTES = 64 * 1024 * 1024
ARCHIVE_BATCH_PATHS = 400


def _archive_batches(root: Path, sha: str, paths: Sequence[str]) -> Iterable[List[str]]:
    """Path batches of at most ARCHIVE_BATCH_BYTES blob bytes (sizes from cat-file --batch-check), so that git
    archive output held in memory stays bounded on a large repository."""
    if not paths:
        return
    req = ''.join('%s:%s\n' % (sha, p) for p in paths).encode('utf-8')
    proc = common.run_git(root, ['cat-file', '--batch-check'], input_bytes=req, timeout=600)
    sizes = []
    for p, line in zip(paths, proc.stdout.decode('utf-8', 'replace').splitlines()):
        parts = line.split()
        sizes.append(int(parts[2]) if len(parts) == 3 and parts[2].isdigit() else 0)
    batch: List[str] = []
    total = 0
    for p, sz in zip(paths, sizes):
        if batch and (total + sz > ARCHIVE_BATCH_BYTES or len(batch) >= ARCHIVE_BATCH_PATHS):
            yield batch
            batch, total = [], 0
        batch.append(p)
        total += sz
    if batch:
        yield batch


def backtest_snapshot(ctx: common.Ctx, as_of: str, out: Path) -> dict:
    """git archive of the last commit at or before as_of into `out` (outside the repo)."""
    root = Path(ctx.repo_root).resolve()
    outp = Path(out).resolve()
    try:
        outp.relative_to(root)
        raise InvalidInput('--out %s lies inside the repo; the snapshot must be written outside it' % outp)
    except ValueError:
        pass
    if outp.exists() and (not outp.is_dir() or any(outp.iterdir())):
        raise InvalidInput('--out %s exists and is not an empty directory' % outp)
    commits = [c for c in ctx.git.log() if c.utc and c.utc <= as_of]
    if not commits:
        raise MissingInput('no commit at or before %s' % as_of)
    c = max(enumerate(commits), key=lambda ic: (ic[1].utc, ic[0]))[1]
    outp.mkdir(parents=True, exist_ok=True)
    if isinstance(ctx.git, common.RealGit):
        n = 0
        for batch in _archive_batches(root, c.sha, ctx.git.list_files(c.sha)):
            proc = common.run_git(root, ['archive', '--format=tar', c.sha, '--'] + batch, allow_archive=True,
                                  timeout=600)
            n += _safe_extract(proc.stdout, outp)
    else:
        n = 0
        for p in ctx.git.list_files(c.sha):
            data = ctx.git.file_at(c.sha, p)
            if data is None:
                continue
            dest = outp / common.norm_rel(p)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            n += 1
    meta = {'schema': 'minifly.kb_backtest_snapshot.v1', 'as_of': as_of, 'commit': c.sha, 'commit_utc': c.utc,
            'files': n, 'label': 'LEAK_RISK', 'created_utc': common.utc_now()}
    (outp / '_BACKTEST_SNAPSHOT.json').write_text(common.dumps_pretty(meta), encoding='utf-8')
    return meta


# --------------------------------------------------------------------------------------
# kb status
# --------------------------------------------------------------------------------------

def kb_status(ctx: common.Ctx) -> dict:
    kb = ctx.kb
    out: Dict[str, Any] = {'schema': 'minifly.kb_status.v1', 'utc': common.utc_now()}
    try:
        out['cadence'] = e3_cadence(kb)
        cm = out['cadence'].get('e3_cadence_max')
        out['cadence']['due'] = bool(_num(cm) and out['cadence']['deliveries_since_last_e3'] >= cm)
    except (MissingInput, InvalidInput) as e:
        out['cadence'] = {'error': str(e)}
    try:
        rule_idx = common.rule_index(ctx.rules or common.load_rules())
    except CheckerError:
        rule_idx = {}
    try:
        exp = kit_expected()
    except (MissingInput, InvalidInput):
        exp = None
    try:
        out['open_incidents'] = [{'id': r.get('id'), 'fm_class': r.get('fm_class'), 'delivery': r.get('delivery'),
                                  'symptom': r.get('symptom')} for r in open_incidents(kb, rule_idx, exp)]
    except (MissingInput, InvalidInput) as e:
        out['open_incidents'] = {'error': str(e)}
    narrowed, invalid = [], []
    try:
        for r in kb.rows('FINDINGS'):
            if (r.get('kind') or 'finding') != 'finding':
                continue
            st = finding_state(kb, r['id'], ctx.git, ctx.rules)
            if st is None:
                continue
            if st['status'] in ('narrowed', 'superseded', 'retracted', 'unverifiable_in_repo'):
                narrowed.append({'id': st['id'], 'status': st['status'], 'caveats': st['caveats'],
                                 'narrowing_rows': st['narrowing_rows'], 'r_level': st['r_level']})
            if st['invalid_activations']:
                invalid.append({'id': st['id'], 'invalid_activations': st['invalid_activations']})
        out['narrowed_findings'] = narrowed
        out['invalid_activations_ignored'] = invalid
    except (MissingInput, InvalidInput) as e:
        out['narrowed_findings'] = {'error': str(e)}
    try:
        out['open_debts'] = [{'id': r.get('id'), 'delivery': r.get('delivery'), 'reason': r.get('reason')}
                             for r in open_debts(kb)]
    except (MissingInput, InvalidInput) as e:
        out['open_debts'] = {'error': str(e)}
    try:
        rows = latest_delivery_rows(kb)
        sealed = [{'delivery': d, 'utc': r.get('utc'), 'G': (r.get('objective_metric') or {}).get('G')}
                  for d, r in rows.items() if r.get('worlds_role') == 'SEALED_CONFIRM']
        main = [{'delivery': d, 'utc': r.get('utc'), 'G': (r.get('objective_metric') or {}).get('G')}
                for d, r in rows.items() if r.get('worlds_role') == 'MAIN']
        out['sealed_G_series'] = sorted(sealed, key=lambda x: utc_key(x['utc']))
        out['main_G_unsealed'] = sorted(main, key=lambda x: utc_key(x['utc']))
    except (MissingInput, InvalidInput) as e:
        out['sealed_G_series'] = {'error': str(e)}
    return out


def _print_status(st: dict) -> None:
    c = st.get('cadence') or {}
    if 'error' in c:
        print('cadence: unavailable (%s)' % c['error'])
    else:
        print('cadence: %d deliveries since the last held-E3 opening (max %s)%s; last E3 delivery: %s'
              % (c['deliveries_since_last_e3'], c.get('e3_cadence_max'), ' -> DUE' if c.get('due') else '',
                 c.get('last_e3_delivery') or 'none'))
    oi = st.get('open_incidents')
    if isinstance(oi, list):
        print('open incidents: %d' % len(oi))
        for r in oi:
            print('  %s %s %s: %s' % (r['id'], r['fm_class'], r['delivery'], str(r['symptom'])[:120]))
    nf = st.get('narrowed_findings')
    if isinstance(nf, list):
        print('narrowed/superseded/retracted/unverifiable findings: %d' % len(nf))
        for r in nf:
            print('  %s %s (%s) rows %s: %s' % (r['id'], r['status'], r['r_level'], ','.join(r['narrowing_rows']) or '-',
                                                ' | '.join(str(x) for x in r['caveats'])[:200]))
    od = st.get('open_debts')
    if isinstance(od, list):
        print('open review debts: %d' % len(od))
        for r in od:
            print('  %s %s: %s' % (r['id'], r['delivery'], r['reason']))
    sg = st.get('sealed_G_series')
    if isinstance(sg, list):
        print('sealed G series (the only progress metric): %s' % (', '.join('%s G=%s' % (x['delivery'], x['G'])
                                                                           for x in sg) or 'empty'))
        print('MAIN G (unsealed, not progress): %s' % (', '.join('%s G=%s' % (x['delivery'], x['G'])
                                                                for x in st.get('main_G_unsealed') or []) or 'empty'))


def _write_out(path: Optional[str], obj: dict) -> Optional[Path]:
    if not path:
        return None
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(common.dumps_pretty(obj), encoding='utf-8')
    return p


def _extra_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog='minifly_check.py kb', add_help=False)
    ap.add_argument('--contract')
    ap.add_argument('--kb', dest='kb_forecasts')
    ap.add_argument('--outcomes')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--boot', type=int, default=BOOTSTRAP_DEFAULT)
    return ap


def cmd_kb(args: Any) -> int:
    """MFC kb {status|score|backtest-snapshot|backtest-score} [--as-of T] [--out PATH]
    [--contract F --kb F [--outcomes F] [--seed N] [--boot N]]."""
    try:
        ex, unknown = _extra_parser().parse_known_args(list(getattr(args, 'extra', None) or []))
    except SystemExit:
        return common.EXIT_ERROR
    if unknown:
        sys.stderr.write('kb: unrecognized arguments: %s\n' % ' '.join(unknown))
        return common.EXIT_ERROR
    ctx = common.make_ctx(args, delivery=None, stage='kb', options={'invoked_by': 'kb'})
    action = args.action
    as_json = bool(getattr(args, 'json', False))
    if action == 'status':
        st = kb_status(ctx)
        _write_out(getattr(args, 'out', None), st)
        if as_json:
            print(json.dumps(st, indent=2, ensure_ascii=False, default=str))
        else:
            _print_status(st)
        return common.EXIT_PASS
    if action == 'score':
        sc = kb_score(ctx.kb)
        _write_out(getattr(args, 'out', None), sc)
        if as_json:
            print(json.dumps(sc, indent=2, ensure_ascii=False))
        else:
            for fam, v in sc['families'].items():
                print('%s: n=%d base=%.4f brier=%s skill=%s hit_rate=%s [%s]' % (
                    fam, v['n_scored_deliveries'], v['laplace_base_rate'], v['brier'], v['brier_skill'],
                    v['hit_rate'], v['label']))
            print('finding transfers: %s' % json.dumps(sc['finding_transfers']))
            print(sc['note'])
        return common.EXIT_PASS
    if action == 'backtest-snapshot':
        if not getattr(args, 'out', None):
            sys.stderr.write('kb backtest-snapshot requires --out DIR (outside the repo)\n')
            return common.EXIT_MISSING
        meta = backtest_snapshot(ctx, _parse_as_of(getattr(args, 'as_of', None)), Path(args.out))
        print(json.dumps(meta, indent=2) if as_json else 'snapshot of %s (%s) -> %s [LEAK_RISK]'
              % (meta['commit'], meta['commit_utc'], args.out))
        return common.EXIT_PASS
    if action == 'backtest-score':
        if not ex.contract or not ex.kb_forecasts:
            sys.stderr.write('kb backtest-score requires --contract FILE and --kb FILE\n')
            return common.EXIT_MISSING
        as_of = _parse_as_of(args.as_of) if getattr(args, 'as_of', None) else None
        res = backtest_score(ctx.kb, _read_forecasts(Path(ex.contract)), _read_forecasts(Path(ex.kb_forecasts)),
                             _read_outcomes(Path(ex.outcomes) if ex.outcomes else None, ctx.kb), as_of=as_of,
                             n_boot=ex.boot, seed=ex.seed)
        _write_out(getattr(args, 'out', None), res)
        if as_json:
            print(json.dumps(res, indent=2))
        else:
            print('LEAK_RISK backtest: n=%d Brier contract-only=%.4f KB=%.4f base=%.4f; KB value %.4f CI95 %s'
                  % (res['n'], res['brier_contract_only'], res['brier_kb'], res['brier_base'], res['kb_value'],
                     res['kb_value_ci95']))
        return common.EXIT_PASS
    sys.stderr.write('kb: unknown action %r\n' % action)
    return common.EXIT_ERROR


# --------------------------------------------------------------------------------------
# Scope of KB-03/KB-04 and line provenance
# --------------------------------------------------------------------------------------

def _head_lines(ctx: common.Ctx, name: str) -> Optional[set]:
    """Canonical rows of the ledger at HEAD; None when HEAD cannot be read."""
    try:
        data = ctx.git.file_at(ctx.git.head(), ledger_rel(name))
    except (MissingInput, InvalidInput, NotImplementedError):
        return None
    return set(_lines_canon(data, ledger_rel(name)))


def _line_commits(ctx: common.Ctx, name: str) -> Dict[bytes, common.Commit]:
    """Canonical row -> first commit whose version of the ledger contains it (MissingInput when shallow)."""
    key = ('wp5_line_commits', name)
    if key in ctx._cache:
        return ctx._cache[key]
    rel = ledger_rel(name)
    out: Dict[bytes, common.Commit] = {}
    for c in ctx.git.log([rel]):
        for line in _lines_canon(ctx.git.file_at(c.sha, rel), rel):
            out.setdefault(line, c)
    ctx._cache[key] = out
    return out


def _scoped_rows(ctx: common.Ctx, kinds: Sequence[str]) -> Tuple[List[Tuple[dict, str]], str]:
    """FINDINGS rows of `kinds` that this publish answers for: rows about this delivery's findings (or naming
    this delivery), and rows not yet committed at HEAD (being added now). Without a delivery: every row."""
    rows = ctx.kb.rows('FINDINGS')
    findings = {r.get('id'): r for r in rows if (r.get('kind') or 'finding') == 'finding'}
    if ctx.delivery is None:
        return [(r, 'all rows') for r in rows if r.get('kind') in kinds], 'every row (no delivery)'
    d_rel, d_name = ctx.delivery_rel, ctx.delivery_name
    head = _head_lines(ctx, 'FINDINGS')
    out = []
    for r in rows:
        if r.get('kind') not in kinds:
            continue
        tgt = findings.get(r.get('refers_to'))
        mine = (tgt is not None and _same_delivery(tgt.get('delivery'), d_rel)) or _same_delivery(r.get('delivery'), d_rel) \
            or (tgt is not None and _same_delivery(tgt.get('delivery'), d_name))
        new = head is None or _canon(r) not in head
        if mine or new:
            out.append((r, 'about this delivery' if mine else 'not committed at HEAD'))
    scope = ('rows about findings of %s plus rows not committed at HEAD%s' %
             (d_rel, '' if head is not None else ' (HEAD unreadable: every row)'))
    return out, scope


# --------------------------------------------------------------------------------------
# Checks KB-01..05
# --------------------------------------------------------------------------------------

def _finish(cid: str, problems: List[str], missing: List[str], ok: str, data: Optional[dict] = None) -> Result:
    if problems:
        return Result(cid, FAIL, '; '.join(problems[:8]) + (' (+%d more)' % (len(problems) - 8) if len(problems) > 8 else ''),
                      evidence=problems[:50], data=data or {})
    if missing:
        return Result(cid, MISSING, '; '.join(missing[:8]), evidence=missing[:50], data=data or {})
    return Result(cid, PASS, ok, data=data or {})


@check('KB-01')
def check_kb_01(ctx: common.Ctx) -> Result:
    problems: List[str] = []
    missing: List[str] = []
    counts = {}
    fids = set()
    incident_ids = set()
    for name in LEDGERS:
        try:
            rows = ctx.kb.rows(name)
        except MissingInput as e:
            missing.append('%s: %s' % (name, e))
            continue
        except InvalidInput as e:
            problems.append('%s: %s' % (name, e))
            continue
        counts[name] = len(rows)
        for i, r in enumerate(rows):
            for pr in row_problems(name, r):
                problems.append('%s row %d (%s): %s' % (name, i + 1, r.get('id') or r.get('batch_id') or r.get('delivery'), pr))
            if name == 'FINDINGS' and (r.get('kind') or 'finding') == 'finding':
                fids.add(r.get('id'))
            if name == 'INCIDENTS' and r.get('kind') == 'report':
                incident_ids.add(r.get('id'))
    try:
        for r in ctx.kb.rows('FINDINGS'):
            if r.get('kind') not in (None, 'finding') and r.get('refers_to') not in fids:
                problems.append('FINDINGS %s refers_to %r, which is not a finding row' % (r.get('id'), r.get('refers_to')))
        for r in ctx.kb.rows('INCIDENTS'):
            if r.get('kind') == 'closure' and incident_closure_target(r) not in incident_ids:
                problems.append('INCIDENTS closure %s refers to %r, which is not a report row'
                                % (r.get('id'), incident_closure_target(r)))
    except (MissingInput, InvalidInput):
        pass
    return _finish('KB-01', problems, missing, 'ledger rows valid: %s' % ', '.join('%s %d' % kv for kv in counts.items()),
                   data={'rows': counts})


@check('KB-02')
def check_kb_02(ctx: common.Ctx) -> Result:
    problems: List[str] = []
    missing: List[str] = []
    for name in LEDGERS:
        try:
            ctx.kb.rows(name)  # ids unique (duplicate keys with different content raise)
        except MissingInput as e:
            missing.append('%s: %s' % (name, e))
            continue
        except InvalidInput as e:
            problems.append('%s: %s' % (name, e))
    if ctx.git.is_shallow():
        missing.append('shallow clone: ledger history cannot be verified append-only (git fetch --unshallow)')
        return _finish('KB-02', problems, missing, '')
    checked = 0
    for name in LEDGERS:
        rel = ledger_rel(name)
        prev: Optional[set] = None
        prev_sha = None
        for c in ctx.git.log([rel]):
            cur = set(_raw_lines(ctx.git.file_at(c.sha, rel)))
            if prev is not None:
                gone = prev - cur
                if gone:
                    problems.append('%s: commit %s removed or edited %d line(s) present at %s'
                                    % (name, c.sha[:12], len(gone), str(prev_sha)[:12]))
            prev, prev_sha = cur, c.sha
            checked += 1
        try:
            head_data = ctx.git.file_at(ctx.git.head(), rel)
        except MissingInput as e:
            missing.append('%s: %s' % (name, e))
            continue
        wt = ctx.kb.path(name + '.jsonl')
        cur = set(_raw_lines(wt.read_bytes())) if wt.is_file() else set()
        gone = set(_raw_lines(head_data)) - cur
        if gone:
            problems.append('%s: the worktree removes or edits %d committed line(s)' % (name, len(gone)))
    return _finish('KB-02', problems, missing, 'ledgers append-only over %d ledger versions; keys unique' % checked)


@check('KB-03')
def check_kb_03(ctx: common.Ctx) -> Result:
    rows = ctx.kb.rows('FINDINGS')
    findings = {r.get('id'): r for r in rows if (r.get('kind') or 'finding') == 'finding'}
    scoped, scope = _scoped_rows(ctx, ('activation', 'reconciliation'))
    problems: List[str] = []
    missing: List[str] = []
    self_active = []
    if ctx.delivery is not None:
        d_rel = ctx.delivery_rel
        self_active = [r for r in findings.values() if r.get('status') == 'active' and _same_delivery(r.get('delivery'), d_rel)]
    for r in self_active:
        problems.append('finding %s of this delivery is written with status active by its author %s (self-activation; '
                        'append status proposed and let the other family activate it)' % (r.get('id'), r.get('author_agent')))
    if not scoped and not self_active:
        return Result('KB-03', NOT_APPLICABLE, 'no activation or reconciliation rows in scope (%s)' % scope)
    head = _head_lines(ctx, 'FINDINGS')
    commits: Optional[Dict[bytes, common.Commit]] = None
    for r, why in scoped:
        rid = r.get('id')
        tgt = findings.get(r.get('refers_to'))
        if tgt is None:
            problems.append('%s %s refers to %r, which is not a finding' % (r.get('kind'), rid, r.get('refers_to')))
            continue
        problems.extend('%s: %s' % (rid, p) for p in activation_problems(ctx.kb, tgt, r, check_source=False))
        canon = _canon(r)
        if head is not None and canon not in head:
            missing.append('%s %s is not committed: commit it with an Acting-Agent trailer of the %s family'
                           % (r.get('kind'), rid, family(r.get('author_agent')) or 'other'))
            continue
        try:
            if commits is None:
                commits = _line_commits(ctx, 'FINDINGS')
        except MissingInput as e:
            missing.append('%s: cannot find the commit that introduced it (%s)' % (rid, e))
            continue
        c = commits.get(canon)
        if c is None:
            missing.append('%s: no commit introduces this row' % rid)
            continue
        tf = trailer_family(c)
        if tf is None:
            problems.append('%s: introducing commit %s has no Acting-Agent trailer' % (rid, c.sha[:12]))
        elif tf == 'ambiguous':
            problems.append('%s: introducing commit %s names both agent families' % (rid, c.sha[:12]))
        else:
            if tf != family(r.get('author_agent')):
                problems.append('%s: author_agent %s but commit %s is Acting-Agent %s' % (
                    rid, r.get('author_agent'), c.sha[:12], tf))
            if tf == family(tgt.get('author_agent')):
                problems.append('%s: committed by the finding author family %s (commit %s)' % (rid, tf, c.sha[:12]))
    return _finish('KB-03', problems, missing, '%d activation/reconciliation row(s) by the other family (%s)'
                   % (len(scoped), scope))


@check('KB-04')
def check_kb_04(ctx: common.Ctx) -> Result:
    rows = ctx.kb.rows('FINDINGS')
    findings = {r.get('id'): r for r in rows if (r.get('kind') or 'finding') == 'finding'}
    scoped, scope = _scoped_rows(ctx, ('activation',))
    if not scoped:
        return Result('KB-04', NOT_APPLICABLE, 'no activation rows in scope (%s)' % scope)
    problems: List[str] = []
    for r, _why in scoped:
        tgt = findings.get(r.get('refers_to'))
        if tgt is None:
            problems.append('activation %s refers to %r, which is not a finding' % (r.get('id'), r.get('refers_to')))
            continue
        ok, why = source_delivery_status(ctx.kb, tgt.get('delivery'), ctx.git, ctx.rules)
        if not ok:
            problems.append('activation %s of %s: %s' % (r.get('id'), tgt.get('id'), '; '.join(why)))
    return _finish('KB-04', problems, [], '%d activation(s) of findings from T2 MAIN/SEALED_CONFIRM deliveries '
                   'without open debt (%s)' % (len(scoped), scope))


@check('KB-05')
def check_kb_05(ctx: common.Ctx) -> Result:
    rows = ctx.kb.rows('INCIDENTS')
    closures = [r for r in rows if r.get('kind') == 'closure' or r.get('status') == 'closed']
    if not closures:
        return Result('KB-05', NOT_APPLICABLE, 'no incident closure rows')
    rule_idx = common.rule_index(ctx.rules or common.load_rules())
    try:
        expected: Optional[dict] = kit_expected()
    except MissingInput:
        expected = None
    problems: List[str] = []
    missing: List[str] = []
    for r in closures:
        miss: List[str] = []
        for p in closure_problems(r, rule_idx, expected, miss):
            problems.append('%s (%s): %s' % (r.get('id'), r.get('check_id'), p))
        missing.extend('%s: %s' % (r.get('id'), m) for m in miss)
    return _finish('KB-05', problems, missing, '%d closure(s), each with a RULES check and a fixture that must fire it'
                   % len(closures))
