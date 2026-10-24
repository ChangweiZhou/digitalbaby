"""MRK-1 seal protocol (WP2): `MFC seal {commit-nonce|reveal|derive|verify} <dir>`, SEAL-01..05 and
CONF-01 (SPEC 10).

    MFC seal commit-nonce <dir> [--nonce-file PATH] [--agent codex|claude] [--session S]
    MFC seal reveal <dir> [--nonce-file PATH]
    MFC seal derive <dir>
    MFC seal verify <dir>

Derivation (normative, SPEC 10 step 5; all strings UTF-8):
    nonce        = secrets.token_hex(32)                    (64 lowercase hex characters)
    nonce_sha256 = sha256(nonce)
    key          = sha256(nonce || owner_go_verbatim)
    seed_i       = int.from_bytes(HMAC-SHA256(key, L1 || '|' || L2 || '|' || purpose || '|' || i)[:8], 'big')
    world id i   = str(seed_i) for purpose 'confirm_worlds', i = 0 .. worlds.official.n - 1
L1 and L2 are the hex sha256 of ENDPOINT_LOCK.json and SOURCE_LOCK.json; purposes 'udi' and 'replay'
give the UDI dependent-set and replay seeds (i = 0). The nonce lives only in a private file outside
the repository until `seal reveal` writes review/SEAL_REVEAL.json.

The SEAL checks apply only when worlds.official_role_intended or CLAIM.worlds_role is SEALED_CONFIRM
(otherwise NOT_APPLICABLE, as RULES.json declares). At `gate --stage official --dry` (rehearsal
end, before the seal protocol has run) a seal artifact that does not exist yet is reported WARN
(deferred); everything that does exist is checked, and the non-dry official gate enforces all.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import common
import lockcheck
from common import FAIL, NOT_APPLICABLE, PASS, WARN, InvalidInput, MissingInput, Result, check
from lockcheck import strip_sha

SEAL_COMMIT_REL = 'review/SEAL_COMMIT.json'
SEAL_REVEAL_REL = 'review/SEAL_REVEAL.json'
SCHEMA_SEAL_COMMIT = 'minifly.seal_commit.v1'
SCHEMA_SEAL_REVEAL = 'minifly.seal_reveal.v1'
SEAL_IDS = ('SEAL-01', 'SEAL-02', 'SEAL-03', 'SEAL-04', 'SEAL-05', 'CONF-01')
PURPOSE_WORLDS = 'confirm_worlds'
NONCE_RE = re.compile(r'^[0-9a-f]{64}$')
DEFAULT_ALPHA_SPENDING = 'geometric_0.05_2^-j'
SPENDING_RE = re.compile(r'^geometric_([0-9]*\.?[0-9]+(?:[eE]-?[0-9]+)?)_2\^-j$')
WORLD_FLAG_RE = re.compile(r'^--?(?:only[-_])?worlds?(?:[-_]ids?)?$')


# --------------------------------------------------------------------------------------
# Derivation
# --------------------------------------------------------------------------------------

def nonce_sha256(nonce: str) -> str:
    return hashlib.sha256(nonce.encode('utf-8')).hexdigest()


def derive_key(nonce: str, go_verbatim: str) -> bytes:
    return hashlib.sha256(nonce.encode('utf-8') + go_verbatim.encode('utf-8')).digest()


def derive_seed(key: bytes, l1: str, l2: str, purpose: str, i: int) -> int:
    msg = ('%s|%s|%s|%d' % (l1, l2, purpose, i)).encode('utf-8')
    return int.from_bytes(hmac.new(key, msg, hashlib.sha256).digest()[:8], 'big')


def derive_seeds(nonce: str, go_verbatim: str, l1: str, l2: str, purpose: str, n: int) -> List[int]:
    key = derive_key(nonce, go_verbatim)
    return [derive_seed(key, l1, l2, purpose, i) for i in range(n)]


WORLD_ID_FORMAT = 'w_seal_%016x'  # official world id i = 'w_seal_' + 16 lowercase hex digits of world_seed_i


def world_id_of(seed: int) -> str:
    """SPEC 10 step 5 defines world_seed_i (the first 8 HMAC bytes, big-endian); the id is its 16-hex form
    with the 'w_seal_' prefix (SCHEMAS.md section 13)."""
    return WORLD_ID_FORMAT % seed


def derive_world_id_list(nonce: str, go_verbatim: str, l1: str, l2: str, n: int) -> List[str]:
    return [world_id_of(s) for s in derive_seeds(nonce, go_verbatim, l1, l2, PURPOSE_WORLDS, n)]


def alpha_schedule(cons: dict) -> Tuple[Callable[[int], float], str]:
    spec = cons.get('confirm_alpha_spending', DEFAULT_ALPHA_SPENDING) if isinstance(cons, dict) else None
    if spec is None:
        spec = DEFAULT_ALPHA_SPENDING
    m = SPENDING_RE.match(str(spec).strip()) if isinstance(spec, str) else None
    if not m:
        raise InvalidInput('CONSTRAINTS.confirm_alpha_spending %r is not geometric_<a>_2^-j' % (spec,))
    base = float(m.group(1))
    if not (0 < base <= 1):
        raise InvalidInput('confirm_alpha_spending base %s is not in (0, 1]' % base)
    return (lambda j: base * (2.0 ** -int(j))), str(spec)


# --------------------------------------------------------------------------------------
# Artifacts
# --------------------------------------------------------------------------------------

def _read_obj(ctx: common.Ctx, rel: str) -> Optional[dict]:
    p = ctx.path(rel)
    if not p.is_file():
        return None
    obj = ctx.read_json(rel)
    if not isinstance(obj, dict):
        raise InvalidInput('%s must be an object' % rel)
    return obj


def load_seal_commit(ctx: common.Ctx) -> Optional[dict]:
    return _read_obj(ctx, SEAL_COMMIT_REL)


def load_seal_reveal(ctx: common.Ctx) -> Optional[dict]:
    return _read_obj(ctx, SEAL_REVEAL_REL)


def load_go(ctx: common.Ctx) -> Tuple[Optional[dict], str]:
    """(GO record, note): AUTHORIZATION.stages.official as {granted, verbatim, utc, lock_shas_shown}."""
    auth = _read_obj(ctx, 'AUTHORIZATION.json')
    if auth is None:
        return None, 'no AUTHORIZATION.json'
    st = auth.get('stages')
    off = st.get('official') if isinstance(st, dict) else None
    if isinstance(off, dict):
        return off, 'AUTHORIZATION.stages.official'
    if off is True:
        return None, 'AUTHORIZATION.stages.official is a bare true: the GO needs {granted, verbatim, utc, lock_shas_shown}'
    return None, 'AUTHORIZATION.stages.official is %r' % (off,)


def seal_commit_problems(sc: dict) -> List[str]:
    out = []
    if sc.get('schema') != SCHEMA_SEAL_COMMIT:
        out.append('SEAL_COMMIT schema is %r, not %s' % (sc.get('schema'), SCHEMA_SEAL_COMMIT))
    for k in ('endpoint_lock_sha256', 'source_lock_sha256', 'nonce_sha256'):
        if strip_sha(sc.get(k)) is None:
            out.append('SEAL_COMMIT.%s is not a sha256' % k)
    ra = sc.get('reviewer_agent')
    if not isinstance(ra, str) or ra.split(':', 1)[0].strip().lower() not in lockcheck.AGENTS:
        out.append('SEAL_COMMIT.reviewer_agent %r is not codex|claude' % (ra,))
    if common.normalize_utc(sc.get('utc')) is None:
        out.append('SEAL_COMMIT.utc %r is not a UTC timestamp' % (sc.get('utc'),))
    return out


def seal_reveal_problems(sr: dict) -> List[str]:
    out = []
    if sr.get('schema') != SCHEMA_SEAL_REVEAL:
        out.append('SEAL_REVEAL schema is %r, not %s' % (sr.get('schema'), SCHEMA_SEAL_REVEAL))
    n = sr.get('nonce')
    if not isinstance(n, str) or not NONCE_RE.match(n):
        out.append('SEAL_REVEAL.nonce is not 64 lowercase hex characters')
    if common.normalize_utc(sr.get('utc')) is None:
        out.append('SEAL_REVEAL.utc %r is not a UTC timestamp' % (sr.get('utc'),))
    return out


def go_problems(go: dict) -> List[str]:
    out = []
    if go.get('granted') is not True:
        out.append('GO granted is %r, not true' % (go.get('granted'),))
    if not isinstance(go.get('verbatim'), str) or not go.get('verbatim').strip():
        out.append('GO verbatim is empty')
    if common.normalize_utc(go.get('utc')) is None:
        out.append('GO utc %r is not a UTC timestamp' % (go.get('utc'),))
    if not isinstance(go.get('lock_shas_shown'), list):
        out.append('GO lock_shas_shown must be a list')
    return out


def go_names_shas(go: dict, l1: Optional[str], l2: Optional[str]) -> List[str]:
    out = []
    text = str(go.get('verbatim') or '').lower()
    shown = set(strip_sha(x) for x in (go.get('lock_shas_shown') or []))
    for name, s in (('ENDPOINT_LOCK', l1), ('SOURCE_LOCK', l2)):
        if not s:
            out.append('no current %s sha' % name)
            continue
        if s not in text:
            out.append('the GO verbatim does not name the %s sha %s' % (name, s[:12]))
        if s not in shown:
            out.append('GO lock_shas_shown lacks the %s sha %s' % (name, s[:12]))
    return out


def official_outputs_exist(ctx: common.Ctx) -> bool:
    if lockcheck.worktree_outputs(ctx, 'official'):
        return True
    try:
        return lockcheck.first_output_commit(ctx, 'official') is not None
    except MissingInput:
        return False


def deferred(ctx: common.Ctx) -> bool:
    """Dry official gate before the seal protocol: artifacts that do not exist yet are deferred."""
    return bool(ctx.dry) and not official_outputs_exist(ctx)


def derived_world_ids(ctx: common.Ctx) -> Optional[List[str]]:
    """Seal-derived official world ids, or None while the seal is incomplete. A nonce that does not
    match SEAL_COMMIT raises InvalidInput."""
    sc = load_seal_commit(ctx)
    sr = load_seal_reveal(ctx)
    go, _note = load_go(ctx)
    if sc is None or sr is None or go is None:
        return None
    nonce = sr.get('nonce')
    if not isinstance(nonce, str) or not NONCE_RE.match(nonce):
        raise InvalidInput('SEAL_REVEAL.nonce is malformed')
    if nonce_sha256(nonce) != strip_sha(sc.get('nonce_sha256')):
        raise InvalidInput('the revealed nonce does not match SEAL_COMMIT.nonce_sha256')
    n = lockcheck.dig(ctx.lock, 'worlds.official.n')
    l1, l2 = ctx.endpoint_lock_sha256, ctx.source_lock_sha256
    if not lockcheck.is_int(n) or n <= 0 or not l1 or not l2 or not isinstance(go.get('verbatim'), str):
        return None
    return derive_world_id_list(nonce, go['verbatim'], l1, l2, n)


def file_commit(ctx: common.Ctx, drel: str) -> Optional[common.Commit]:
    """First commit holding the current bytes of a delivery-relative file."""
    p = ctx.path(drel)
    if not p.is_file():
        return None
    cur = p.read_bytes()
    return ctx.git.first_commit_where(ctx.repo_rel_of(drel), lambda b: b == cur)


def go_commit(ctx: common.Ctx, go: dict) -> Optional[common.Commit]:
    verb = go.get('verbatim')

    def pred(o: Any) -> bool:
        st = o.get('stages') if isinstance(o, dict) else None
        off = st.get('official') if isinstance(st, dict) else None
        return isinstance(off, dict) and off.get('granted') is True and off.get('verbatim') == verb
    return lockcheck.content_commit(ctx, ctx.repo_rel_of('AUTHORIZATION.json'), pred)


def confirm_rows(ctx: common.Ctx) -> List[dict]:
    return ctx.kb.rows('CONFIRM_LEDGER')


def my_confirm_row(ctx: common.Ctx, rows: Sequence[dict]) -> Optional[dict]:
    ids = lockcheck.delivery_ids(ctx)
    mine = [r for r in rows if r.get('delivery') in ids and lockcheck.is_int(r.get('attempt_index'))]
    if not mine:
        return None
    return max(mine, key=lambda r: r['attempt_index'])


def ledger_row_commit(ctx: common.Ctx, row: dict) -> Optional[common.Commit]:
    rel = common.KB_REL + '/CONFIRM_LEDGER.jsonl'
    key = (row.get('family_id'), row.get('attempt_index'), row.get('delivery'))

    def pred(b: bytes) -> bool:
        try:
            rows = lockcheck.jsonl_rows_from_bytes(b, rel)
        except InvalidInput:
            return False
        return any((r.get('family_id'), r.get('attempt_index'), r.get('delivery')) == key for r in rows)
    return lockcheck.content_commit(ctx, rel, pred, parse_json=False)


def _na(cid: str) -> Optional[Result]:
    return Result(cid, NOT_APPLICABLE, 'SEALED_CONFIRM is not intended (worlds.official_role_intended / CLAIM.worlds_role)')


def _finish(cid: str, problems: List[str], deferred_items: List[str], ok_detail: str,
            data: Optional[dict] = None) -> Result:
    if problems:
        return Result(cid, FAIL, '; '.join(problems[:8]), evidence=problems[:40], data=data or {})
    if deferred_items:
        return Result(cid, WARN, '[dry official gate: deferred until the seal protocol runs; enforced at gate/launch '
                      '--stage official] %s' % '; '.join(deferred_items), data=data or {})
    return Result(cid, PASS, ok_detail, data=data or {})


# --------------------------------------------------------------------------------------
# SEAL-01..05, CONF-01
# --------------------------------------------------------------------------------------

@check('SEAL-01')
def check_seal_01(ctx: common.Ctx) -> Result:
    """SEALED_CONFIRM requires SEAL_COMMIT, GO and SEAL_REVEAL."""
    if not common.sealed_confirm_intended(ctx):
        return _na('SEAL-01')
    problems: List[str] = []
    missing: List[str] = []
    sc = load_seal_commit(ctx)
    sr = load_seal_reveal(ctx)
    go, note = load_go(ctx)
    if sc is None:
        missing.append('%s absent' % SEAL_COMMIT_REL)
    else:
        problems.extend(seal_commit_problems(sc))
    if go is None:
        missing.append('owner GO absent (%s)' % note)
    else:
        problems.extend(go_problems(go))
    if sr is None:
        missing.append('%s absent' % SEAL_REVEAL_REL)
    else:
        problems.extend(seal_reveal_problems(sr))
    if missing and not deferred(ctx):
        problems = missing + problems
        missing = []
    return _finish('SEAL-01', problems, missing, 'SEAL_COMMIT, GO and SEAL_REVEAL present and well-formed')


@check('SEAL-02')
def check_seal_02(ctx: common.Ctx) -> Result:
    """SEAL_COMMIT precedes GO, GO precedes REVEAL, REVEAL precedes the first official output; SEAL_COMMIT
    is on a reviewer-namespace branch (the reviewer is the other agent family)."""
    if not common.sealed_confirm_intended(ctx):
        return _na('SEAL-02')
    lock = ctx.lock
    fam = lockcheck.reviewer_family(lock)
    problems: List[str] = []
    missing: List[str] = []
    notes: List[str] = []
    if fam is None:
        problems.append('lock.author_agent %r is not codex|claude: no reviewer namespace' % lock.get('author_agent'))
    sc = load_seal_commit(ctx)
    sr = load_seal_reveal(ctx)
    go, gnote = load_go(ctx)
    c_sc = c_go = c_sr = None
    if sc is None:
        missing.append('no SEAL_COMMIT')
    else:
        ra = str(sc.get('reviewer_agent') or '').split(':', 1)[0].strip().lower()
        if fam and ra != fam:
            problems.append('SEAL_COMMIT.reviewer_agent %r is not the reviewer family %s (single-agent mode cannot '
                            'seal)' % (sc.get('reviewer_agent'), fam))
        c_sc = file_commit(ctx, SEAL_COMMIT_REL)
        if c_sc is None:
            problems.append('SEAL_COMMIT.json (current content) is not committed')
        elif fam:
            br = lockcheck.in_namespace(ctx.git.remote_branches_containing(c_sc.sha), fam)
            if not br:
                problems.append('SEAL_COMMIT commit %s is in no origin/%s/* branch' % (c_sc.sha[:12], fam))
            else:
                notes.append('SEAL_COMMIT %s on %s' % (c_sc.sha[:12], br[0]))
    if go is None:
        missing.append('no GO (%s)' % gnote)
    else:
        c_go = go_commit(ctx, go)
        if c_go is None:
            problems.append('the GO (AUTHORIZATION.stages.official) is not committed')
        elif c_sc is not None and not lockcheck.strictly_before(ctx, c_sc, c_go):
            problems.append('GO commit %s does not follow the SEAL_COMMIT commit %s' % (c_go.sha[:12], c_sc.sha[:12]))
        elif c_go is not None:
            notes.append('GO %s' % c_go.sha[:12])
    if sr is None:
        missing.append('no SEAL_REVEAL')
    else:
        c_sr = file_commit(ctx, SEAL_REVEAL_REL)
        if c_sr is None:
            problems.append('SEAL_REVEAL.json (current content) is not committed')
        else:
            if c_go is None:
                problems.append('SEAL_REVEAL committed at %s with no committed GO before it' % c_sr.sha[:12])
            elif not lockcheck.strictly_before(ctx, c_go, c_sr):
                problems.append('SEAL_REVEAL commit %s does not follow the GO commit %s (reveal before GO)'
                                % (c_sr.sha[:12], c_go.sha[:12]))
            if c_sc is not None and not lockcheck.strictly_before(ctx, c_sc, c_sr):
                problems.append('SEAL_REVEAL commit %s does not follow SEAL_COMMIT %s' % (c_sr.sha[:12], c_sc.sha[:12]))
            notes.append('REVEAL %s' % c_sr.sha[:12])
    c_out = lockcheck.first_output_commit(ctx, 'official')
    outputs = official_outputs_exist(ctx)
    if outputs:
        if c_sr is None:
            problems.append('official outputs exist without a committed SEAL_REVEAL before them')
        elif c_out is not None and not lockcheck.strictly_before(ctx, c_sr, c_out):
            problems.append('the first official output commit %s does not follow the REVEAL commit %s'
                            % (c_out.sha[:12], c_sr.sha[:12]))
    if missing and not deferred(ctx):
        problems = missing + problems
        missing = []
    return _finish('SEAL-02', problems, missing, 'ordering SEAL_COMMIT < GO < REVEAL < official outputs holds (%s)'
                   % ', '.join(notes))


@check('SEAL-03')
def check_seal_03(ctx: common.Ctx) -> Result:
    """The nonce hash matches, and the GO verbatim names the lock shas."""
    if not common.sealed_confirm_intended(ctx):
        return _na('SEAL-03')
    problems: List[str] = []
    missing: List[str] = []
    l1, l2 = ctx.endpoint_lock_sha256, ctx.source_lock_sha256
    if not l2:
        problems.append('SOURCE_LOCK.json is absent: the seal binds both lock shas')
    sc = load_seal_commit(ctx)
    sr = load_seal_reveal(ctx)
    go, gnote = load_go(ctx)
    if sc is None:
        missing.append('no SEAL_COMMIT')
    else:
        if strip_sha(sc.get('endpoint_lock_sha256')) != l1:
            problems.append('SEAL_COMMIT endpoint_lock_sha256 %s is not the current ENDPOINT_LOCK %s'
                            % (str(sc.get('endpoint_lock_sha256'))[:12], (l1 or '')[:12]))
        if strip_sha(sc.get('source_lock_sha256')) != l2:
            problems.append('SEAL_COMMIT source_lock_sha256 %s is not the current SOURCE_LOCK %s'
                            % (str(sc.get('source_lock_sha256'))[:12], (l2 or '')[:12]))
    if go is None:
        missing.append('no GO (%s)' % gnote)
    else:
        problems.extend(go_names_shas(go, l1, l2))
    if sr is None:
        missing.append('no SEAL_REVEAL')
    elif sc is not None:
        nonce = sr.get('nonce')
        if not isinstance(nonce, str) or not NONCE_RE.match(nonce):
            problems.append('SEAL_REVEAL.nonce is malformed')
        elif nonce_sha256(nonce) != strip_sha(sc.get('nonce_sha256')):
            problems.append('sha256(revealed nonce) %s does not match SEAL_COMMIT.nonce_sha256 %s'
                            % (nonce_sha256(nonce)[:12], str(sc.get('nonce_sha256'))[:12]))
    review = lockcheck.read_review(ctx)
    rs = review.get('seal') if isinstance(review, dict) else None
    if isinstance(rs, dict) and sc is not None:
        if rs.get('nonce_sha256') is not None and strip_sha(rs.get('nonce_sha256')) != strip_sha(sc.get('nonce_sha256')):
            problems.append('REVIEW.seal.nonce_sha256 differs from SEAL_COMMIT')
        if rs.get('revealed_nonce') is not None and sr is not None and rs.get('revealed_nonce') != sr.get('nonce'):
            problems.append('REVIEW.seal.revealed_nonce differs from SEAL_REVEAL')
    if missing and not deferred(ctx):
        problems = missing + problems
        missing = []
    return _finish('SEAL-03', problems, missing, 'nonce hash matches SEAL_COMMIT; the GO verbatim names L1 %s and L2 %s'
                   % ((l1 or '')[:12], (l2 or '')[:12]))


def world_tokens(argv: Sequence[Any]) -> List[str]:
    out: List[str] = []
    take = False
    for tok in argv or []:
        t = str(tok)
        if take:
            out.extend(p for p in re.split(r'[,\s]+', t) if p)
            take = False
            continue
        if t.startswith('-') and '=' in t and WORLD_FLAG_RE.match(t.split('=', 1)[0]):
            out.extend(p for p in re.split(r'[,\s]+', t.split('=', 1)[1]) if p)
            continue
        if WORLD_FLAG_RE.match(t):
            take = True
    return out


def observed_official_worlds(ctx: common.Ctx) -> Dict[str, List[str]]:
    seen: Dict[str, List[str]] = {}
    for label, row in lockcheck.current_outcome_rows(ctx):
        lab = label.split('!', 1)[-1] if '!' in label else label
        official = row.get('world_role') == 'official' or lab.startswith('results/official/') \
            or label.startswith('results/official/')
        if official and row.get('world') is not None:
            seen.setdefault(str(row.get('world')), []).append(label)
    return seen


@check('SEAL-04')
def check_seal_04(ctx: common.Ctx) -> Result:
    """Official world ids equal the seal-derived ones."""
    if not common.sealed_confirm_intended(ctx):
        return _na('SEAL-04')
    lock = ctx.lock
    problems: List[str] = []
    missing: List[str] = []
    mode = lockcheck.dig(lock, 'worlds.official.mode')
    if mode != 'sealed':
        problems.append('worlds.official.mode is %r: SEALED_CONFIRM worlds must be sealed (derived after GO)' % (mode,))
    n = lockcheck.dig(lock, 'worlds.official.n')
    if not lockcheck.is_int(n) or n <= 0:
        problems.append('worlds.official.n %r is not a positive integer' % (n,))
    derived: Optional[List[str]] = None
    try:
        derived = derived_world_ids(ctx)
    except InvalidInput as e:
        problems.append('cannot derive the official worlds: %s' % e)
    observed = observed_official_worlds(ctx)
    if derived is None and not problems:
        missing.append('seal not revealed yet: official worlds not derivable')
    data: Dict[str, Any] = {'derived': derived or []}
    if derived is not None:
        dset = set(derived)
        listed = [str(x) for x in (lockcheck.dig(lock, 'worlds.official.ids') or [])]
        if listed and set(listed) != dset:
            problems.append('worlds.official.ids %s differ from the derived ids' % ','.join(sorted(set(listed) - dset)[:5]
                                                                                         or listed[:5]))
        if observed:
            extra = sorted(set(observed) - dset)
            lack = sorted(dset - set(observed))
            if extra:
                problems.append('official rows use non-derived world(s) %s' % ', '.join(extra[:5]))
            if lack:
                problems.append('derived world(s) %s have no official rows' % ', '.join(lack[:5]))
        argv = ctx.options.get('pending_argv')
        if argv:
            toks = world_tokens(argv)
            bad = sorted(set(toks) - dset)
            if bad:
                problems.append('the official launch argv names non-derived world(s) %s' % ', '.join(bad[:5]))
            data['argv_worlds'] = toks
    if missing and not deferred(ctx):
        problems = missing + problems
        missing = []
    ok = 'official worlds equal the %d seal-derived ids%s' % (len(derived or []), (
        ' (%d observed in official rows)' % len(observed)) if observed else ' (no official rows yet; launch must use '
                                                                           'exactly these)')
    return _finish('SEAL-04', problems, missing, ok, data=data)


@check('SEAL-05')
def check_seal_05(ctx: common.Ctx) -> Result:
    """The CONFIRM_LEDGER row (appended by seal reveal) precedes outcomes."""
    if not common.sealed_confirm_intended(ctx):
        return _na('SEAL-05')
    problems: List[str] = []
    missing: List[str] = []
    rows = confirm_rows(ctx)
    row = my_confirm_row(ctx, rows)
    if row is None:
        missing.append('no CONFIRM_LEDGER row for this delivery (seal reveal appends it)')
    else:
        c_row = ledger_row_commit(ctx, row)
        outputs = official_outputs_exist(ctx)
        if c_row is None:
            if outputs or not ctx.dry:
                problems.append('the CONFIRM_LEDGER row (family %s, attempt %s) is not committed'
                                % (row.get('family_id'), row.get('attempt_index')))
            else:
                missing.append('CONFIRM_LEDGER row not committed yet')
        else:
            c_out = lockcheck.first_output_commit(ctx, 'official')
            if c_out is not None and not lockcheck.strictly_before(ctx, c_row, c_out):
                problems.append('the CONFIRM_LEDGER row commit %s does not precede the first official output %s'
                                % (c_row.sha[:12], c_out.sha[:12]))
    if missing and not deferred(ctx):
        problems = missing + problems
        missing = []
    return _finish('SEAL-05', problems, missing, 'CONFIRM_LEDGER row (family %s, attempt %s) committed before any '
                   'official outcome' % ((row or {}).get('family_id'), (row or {}).get('attempt_index')))


@check('CONF-01')
def check_conf_01(ctx: common.Ctx) -> Result:
    """Lock alpha <= the CONFIRM_LEDGER alpha_allocated (geometric family spending)."""
    if not common.sealed_confirm_intended(ctx):
        return _na('CONF-01')
    lock = ctx.lock
    problems: List[str] = []
    missing: List[str] = []
    agg = lockcheck.dig(lock, 'primary.aggregator') or {}
    alpha = agg.get('alpha') if isinstance(agg, dict) else None
    if not isinstance(agg, dict) or agg.get('type') != 'paired_lcb' or not lockcheck.is_num(alpha):
        return Result('CONF-01', FAIL, 'SEALED_CONFIRM needs an alpha-bearing aggregator (paired_lcb with alpha); got %r'
                      % (agg,))
    sched, spec = alpha_schedule(ctx.kb.constraints())
    fam_id, _members = common.family_of(ctx)
    rows = confirm_rows(ctx)
    fam_rows = [r for r in rows if r.get('family_id') == fam_id]
    idxs = sorted(r.get('attempt_index') for r in fam_rows if lockcheck.is_int(r.get('attempt_index')))
    if len(idxs) != len(fam_rows) or idxs != list(range(len(idxs))):
        problems.append('family %s CONFIRM_LEDGER attempt_index values %s are not 0..%d'
                        % (fam_id, idxs, len(fam_rows) - 1))
    row = my_confirm_row(ctx, rows)
    data: Dict[str, Any] = {'family_id': fam_id, 'lock_alpha': alpha, 'spending': spec}
    if row is None:
        j = len(fam_rows)
        allowed = sched(j)
        data['prospective_attempt_index'] = j
        if alpha > allowed + 1e-15:
            problems.append('lock alpha %s exceeds the %s allocation %s for attempt j=%d of family %s'
                            % (alpha, spec, allowed, j, fam_id))
        else:
            missing.append('no CONFIRM_LEDGER row yet; lock alpha %s <= prospective allocation %s (j=%d)'
                           % (alpha, allowed, j))
    else:
        j = row.get('attempt_index')
        alloc = row.get('alpha_allocated')
        data.update({'attempt_index': j, 'alpha_allocated': alloc})
        if row.get('family_id') != fam_id:
            problems.append('CONFIRM_LEDGER row family %r is not the computed family %s' % (row.get('family_id'), fam_id))
        if not lockcheck.is_num(alloc):
            problems.append('alpha_allocated %r is not a number' % (alloc,))
        else:
            if alloc > sched(j) + 1e-15:
                problems.append('alpha_allocated %s exceeds the %s schedule %s at j=%d' % (alloc, spec, sched(j), j))
            if alpha > alloc + 1e-15:
                problems.append('lock alpha %s exceeds alpha_allocated %s (attempt j=%d of family %s)'
                                % (alpha, alloc, j, fam_id))
    if missing and not deferred(ctx):
        problems = ['no CONFIRM_LEDGER row for this delivery (seal reveal appends it)'] + problems
        missing = []
    return _finish('CONF-01', problems, missing, 'lock alpha %s <= alpha_allocated %s (family %s, j=%s)'
                   % (alpha, (row or {}).get('alpha_allocated'), fam_id, (row or {}).get('attempt_index')), data=data)


# --------------------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------------------

def _out(msg: str) -> None:
    sys.stdout.write(msg.rstrip('\n') + '\n')


def _err(msg: str) -> None:
    sys.stderr.write(msg.rstrip('\n') + '\n')


def parse_extra(extra: Sequence[str]) -> Tuple[Dict[str, str], List[str]]:
    opts: Dict[str, str] = {}
    bad: List[str] = []
    it = list(extra or [])
    i = 0
    while i < len(it):
        tok = it[i]
        key, val = (tok.split('=', 1) + [None])[:2] if tok.startswith('--') else (tok, None)
        if key in ('--agent', '--session'):
            if val is None:
                if i + 1 >= len(it):
                    bad.append(tok)
                    break
                val = it[i + 1]
                i += 1
            opts[key[2:]] = val
        else:
            bad.append(tok)
        i += 1
    return opts, bad


def default_nonce_path(delivery_name: str, l1: str) -> Path:
    return Path.home() / ('.minifly_nonce_%s_%s' % (delivery_name, l1[:8]))


def nonce_path_problem(ctx: common.Ctx, p: Path) -> Optional[str]:
    rp = Path(os.path.realpath(str(p)))
    roots = [Path(ctx.repo_root).resolve()]
    try:
        roots.append(common.git_toplevel(common.KIT_DIR))
    except (MissingInput, common.CheckerError):
        pass
    for r in roots:
        try:
            rp.relative_to(r)
            return 'the private nonce path %s lies inside the repository %s; keep it outside every repo' % (rp, r)
        except ValueError:
            continue
    return None


def _append_ledger(ctx: common.Ctx, row: dict) -> Path:
    p = ctx.kb.path('CONFIRM_LEDGER.jsonl')
    if not p.is_file():
        raise MissingInput('KB ledger %s is missing' % p)
    data = p.read_bytes()
    with open(p, 'ab') as fh:
        if data and not data.endswith(b'\n'):
            fh.write(b'\n')
        fh.write(common.canonical_json(row) + b'\n')
        fh.flush()
        os.fsync(fh.fileno())
    return p


def cmd_commit_nonce(ctx: common.Ctx, args: Any, opts: Dict[str, str]) -> int:
    lock = ctx.lock
    l1, l2 = ctx.endpoint_lock_sha256, ctx.source_lock_sha256
    if not l2:
        _err('seal commit-nonce: SOURCE_LOCK.json (the final SOURCE_LOCK, L2) is required')
        return common.EXIT_MISSING
    agent = opts.get('agent') or os.environ.get('MINIFLY_AGENT')
    fam = agent.split(':', 1)[0].strip().lower() if agent else None
    if fam not in lockcheck.AGENTS:
        _err('seal commit-nonce: give the reviewer family with --agent codex|claude or MINIFLY_AGENT')
        return common.EXIT_MISSING
    author = lockcheck.author_family(lock)
    if author is None:
        _err('seal commit-nonce REFUSED: lock.author_agent %r is not codex|claude' % lock.get('author_agent'))
        return common.EXIT_FAIL
    if fam == author:
        _err('seal commit-nonce REFUSED: the reviewer (%s) is the author family; single-agent mode cannot seal' % fam)
        return common.EXIT_FAIL
    if official_outputs_exist(ctx):
        _err('seal commit-nonce REFUSED: official outputs already exist')
        return common.EXIT_FAIL
    review = lockcheck.read_review(ctx)
    acks = [a for a in ((review or {}).get('source_lock_acks') or []) if isinstance(a, dict)]
    if not isinstance(review, dict) or strip_sha(review.get('reviewed_endpoint_lock_sha256')) != l1 \
            or not any(strip_sha(a.get('sha')) == l2 for a in acks):
        _err('seal commit-nonce REFUSED: review/REVIEW.json must first attest ENDPOINT_LOCK %s '
             '(reviewed_endpoint_lock_sha256) and ack SOURCE_LOCK %s (source_lock_acks) (SPEC 10 step 2)'
             % (l1[:12], l2[:12]))
        return common.EXIT_FAIL
    p = Path(args.nonce_file).expanduser() if getattr(args, 'nonce_file', None) else default_nonce_path(
        ctx.delivery_name, l1)
    why = nonce_path_problem(ctx, p)
    if why:
        _err('seal commit-nonce REFUSED: %s' % why)
        return common.EXIT_FAIL
    if p.exists():
        _err('seal commit-nonce REFUSED: %s exists; never overwrite a nonce (pass a new --nonce-file to redo the seal)'
             % p)
        return common.EXIT_FAIL
    if not p.parent.is_dir():
        _err('seal commit-nonce: directory %s does not exist' % p.parent)
        return common.EXIT_MISSING
    nonce = secrets.token_hex(32)
    fd = os.open(str(p), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w', encoding='utf-8') as fh:
        fh.write(nonce + '\n')
    sc = {'schema': SCHEMA_SEAL_COMMIT, 'delivery': ctx.delivery_name, 'endpoint_lock_sha256': l1,
          'source_lock_sha256': l2, 'nonce_sha256': nonce_sha256(nonce), 'reviewer_agent': fam,
          'reviewer_session': opts.get('session') or os.environ.get('MINIFLY_SESSION') or '', 'utc': common.utc_now()}
    out = ctx.path(SEAL_COMMIT_REL)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(common.dumps_pretty(sc), encoding='utf-8')
    _out('seal commit-nonce: wrote %s (nonce_sha256 %s); the nonce is in %s (mode 600).'
         % (ctx.repo_rel_of(SEAL_COMMIT_REL), sc['nonce_sha256'][:12], p))
    _out('Next: commit and push %s on your origin/%s/* branch; never commit, paste or relay the nonce. The owner GO '
         '(AUTHORIZATION.stages.official, naming both lock shas) must be committed after this commit.'
         % (SEAL_COMMIT_REL, fam))
    return common.EXIT_PASS


def cmd_reveal(ctx: common.Ctx, args: Any) -> int:
    sc = load_seal_commit(ctx)
    if sc is None:
        _err('seal reveal: %s is absent (run seal commit-nonce first)' % SEAL_COMMIT_REL)
        return common.EXIT_MISSING
    probs = seal_commit_problems(sc)
    if probs:
        _err('seal reveal REFUSED: %s' % '; '.join(probs))
        return common.EXIT_FAIL
    l1, l2 = ctx.endpoint_lock_sha256, ctx.source_lock_sha256
    if strip_sha(sc.get('endpoint_lock_sha256')) != l1 or strip_sha(sc.get('source_lock_sha256')) != l2:
        _err('seal reveal REFUSED: the locks changed since SEAL_COMMIT (start again at step 1)')
        return common.EXIT_FAIL
    p = Path(args.nonce_file).expanduser() if getattr(args, 'nonce_file', None) else default_nonce_path(
        ctx.delivery_name, l1)
    why = nonce_path_problem(ctx, p)
    if why:
        _err('seal reveal REFUSED: %s' % why)
        return common.EXIT_FAIL
    if not p.is_file():
        _err('seal reveal: nonce file %s not found (if lost, repeat steps 2-4 with a fresh commit-nonce)' % p)
        return common.EXIT_MISSING
    nonce = p.read_text(encoding='utf-8').strip()
    if not NONCE_RE.match(nonce) or nonce_sha256(nonce) != strip_sha(sc.get('nonce_sha256')):
        _err('seal reveal REFUSED: the nonce in %s does not match SEAL_COMMIT.nonce_sha256' % p)
        return common.EXIT_FAIL
    go, gnote = load_go(ctx)
    if go is None:
        _err('seal reveal REFUSED: no owner GO (%s); never reveal before GO' % gnote)
        return common.EXIT_FAIL
    probs = go_problems(go) + go_names_shas(go, l1, l2)
    if probs:
        _err('seal reveal REFUSED: GO defective: %s' % '; '.join(probs))
        return common.EXIT_FAIL
    fam = lockcheck.reviewer_family(ctx.lock)
    c_sc = file_commit(ctx, SEAL_COMMIT_REL)
    if c_sc is None:
        _err('seal reveal REFUSED: %s is not committed' % SEAL_COMMIT_REL)
        return common.EXIT_FAIL
    if fam is None or not lockcheck.in_namespace(ctx.git.remote_branches_containing(c_sc.sha), fam):
        _err('seal reveal REFUSED: SEAL_COMMIT commit %s is in no origin/%s/* branch' % (c_sc.sha[:12], fam))
        return common.EXIT_FAIL
    c_go = go_commit(ctx, go)
    if c_go is None or not lockcheck.strictly_before(ctx, c_sc, c_go):
        _err('seal reveal REFUSED: the GO is not committed after the SEAL_COMMIT commit %s (reveal before GO)'
             % c_sc.sha[:12])
        return common.EXIT_FAIL
    if official_outputs_exist(ctx):
        _err('seal reveal REFUSED: official outputs already exist')
        return common.EXIT_FAIL
    sr_old = load_seal_reveal(ctx)
    if isinstance(sr_old, dict) and sr_old.get('nonce') == nonce:
        _err('seal reveal REFUSED: this nonce is already revealed')
        return common.EXIT_FAIL
    sched, spec = alpha_schedule(ctx.kb.constraints())
    fam_id, _members = common.family_of(ctx)
    rows = confirm_rows(ctx)
    j = sum(1 for r in rows if r.get('family_id') == fam_id)
    try:
        head = ctx.git.head()
    except MissingInput:
        head = None
    row = {'family_id': fam_id, 'delivery': ctx.delivery_rel, 'attempt_index': j, 'alpha_allocated': sched(j),
           'seal_reveal_commit': head, 'utc': common.utc_now()}
    out = ctx.path(SEAL_REVEAL_REL)
    out.write_text(common.dumps_pretty({'schema': SCHEMA_SEAL_REVEAL, 'nonce': nonce, 'utc': row['utc']}),
                   encoding='utf-8')
    lp = _append_ledger(ctx, row)
    try:
        lrel = lp.resolve().relative_to(Path(ctx.repo_root).resolve()).as_posix()
    except ValueError:
        lrel = str(lp)
    _out('seal reveal: wrote %s and appended CONFIRM_LEDGER row (family %s, attempt_index %d, alpha_allocated %s, %s)'
         % (ctx.repo_rel_of(SEAL_REVEAL_REL), fam_id, j, row['alpha_allocated'], spec))
    _out('Next: commit %s and %s together now, on a branch that descends from the GO commit %s; then '
         '`MFC seal derive` gives the official world ids.' % (ctx.repo_rel_of(SEAL_REVEAL_REL), lrel, c_go.sha[:12]))
    return common.EXIT_PASS


def cmd_derive(ctx: common.Ctx, args: Any) -> int:
    sc, sr = load_seal_commit(ctx), load_seal_reveal(ctx)
    go, gnote = load_go(ctx)
    if sc is None or sr is None or go is None:
        _err('seal derive: needs SEAL_COMMIT, SEAL_REVEAL and the GO (%s)' % gnote)
        return common.EXIT_MISSING
    try:
        ids = derived_world_ids(ctx)
    except InvalidInput as e:
        _err('seal derive REFUSED: %s' % e)
        return common.EXIT_FAIL
    if ids is None:
        _err('seal derive: worlds.official.n must be a positive integer and both locks present')
        return common.EXIT_FAIL
    l1, l2 = ctx.endpoint_lock_sha256, ctx.source_lock_sha256
    nonce, verb = sr['nonce'], go['verbatim']
    n = len(ids)
    out = {'endpoint_lock_sha256': l1, 'source_lock_sha256': l2, 'n': n, 'purpose': PURPOSE_WORLDS,
           'confirm_worlds': ids, 'confirm_seeds': derive_seeds(nonce, verb, l1, l2, PURPOSE_WORLDS, n),
           'udi_seed': derive_seeds(nonce, verb, l1, l2, 'udi', 1)[0],
           'replay_seed': derive_seeds(nonce, verb, l1, l2, 'replay', 1)[0]}
    print(json.dumps(out, indent=2))
    return common.EXIT_PASS


def cmd_verify(ctx: common.Ctx, args: Any) -> int:
    idx = common.rule_index(ctx.rules)
    try:
        ctx.tier = common.compute_tier(ctx)[0]
    except (MissingInput, InvalidInput, common.CheckerError):
        ctx.tier = 2
    results: List[Result] = []
    for cid in SEAL_IDS:
        results.extend(common.run_check(ctx, idx[cid]))
    if getattr(args, 'json', False):
        rc = common.exit_code(results)
        print(json.dumps({'results': [r.to_dict() for r in results], 'exit_code': rc,
                          'verdict': common.VERDICT_BY_EXIT[rc]}, indent=2))
    else:
        print(common.format_results(results))
    rc = common.exit_code(results)
    _out('seal verify: %s (exit %d)' % (common.VERDICT_BY_EXIT[rc], rc))
    return rc


def cmd_seal(args: Any) -> int:
    """MFC seal {commit-nonce|reveal|derive|verify} <dir> [--nonce-file PATH] [--agent A] [--session S]."""
    opts, bad = parse_extra(getattr(args, 'extra', None) or [])
    if bad:
        _err('seal: unrecognized arguments: %s' % ' '.join(bad))
        return common.EXIT_ERROR
    action = args.action
    ctx = common.make_ctx(args, delivery=args.dir, stage='official', options={'invoked_by': 'seal %s' % action})
    ok, why = common.v1_status(ctx.delivery)
    if not ok:
        _err('seal REFUSED: legacy delivery: use retro (%s)' % why)
        return common.EXIT_MISSING
    if action == 'verify':
        return cmd_verify(ctx, args)
    if not common.sealed_confirm_intended(ctx):
        _err('seal %s REFUSED: SEALED_CONFIRM is not intended (worlds.official_role_intended)' % action)
        return common.EXIT_FAIL
    if action == 'commit-nonce':
        return cmd_commit_nonce(ctx, args, opts)
    if action == 'reveal':
        return cmd_reveal(ctx, args)
    if action == 'derive':
        return cmd_derive(ctx, args)
    _err('seal: unknown action %r' % action)
    return common.EXIT_ERROR
