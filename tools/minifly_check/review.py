"""MRK-1 independent-review checks (WP4): REV-01..08, DEBT-01..02 and `MFC review` (SPEC 5, 6.4, 7.4, 8).

    MFC review replay-plan <dir> --seed HEX [--n 4]   # seeded (world, arm) choice for the post-run replay
    MFC review check <dir>                            # REV-01..07 (or DEBT-01), REV-08 when applicable, DEBT-02

Identity is never taken from git author names (Codex commits as the owner's identity). A commit counts as the
reviewer's only when every `Acting-Agent:` trailer names the reviewer family (the family opposite to
lock.author_agent) and the commit is contained in a remote branch of that family's namespace
(origin/<family>/*). review/REVIEW.json may be changed afterwards by anyone else only in
findings[].reconciled_by_author (the author's reconciliation); any other change outside the reviewer's
commits is FAIL. Python 3.9+, standard library only.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import admitcheck
import common
from admitcheck import dig, family, finish, is_int, is_num, other_family, s, strip_sha, strictly_before
from common import FAIL, MISSING, NOT_APPLICABLE, PASS, WARN, InvalidInput, MissingInput, Result, check

REVIEW_REL = 'review/REVIEW.json'
SCHEMA_REVIEW = 'minifly.review.v1'
SCHEMA_CANARY_REPORT = 'minifly.canary_report.v1'
SCHEMA_REPLAY_PLAN = 'minifly.replay_plan.v1'
REQUIRED_KINDS = ('rerun_admission', 'reject_input', 'trace_number', 'shortcut_attempt')
CHECK_KINDS = REQUIRED_KINDS + ('oracle_upstream', 'replay')
COMPARED_KINDS = ('rerun_admission',)  # SPEC 6.4: rerun_admission compares digests (REV-03 reads match)
SEVERITIES = ('BLOCKER', 'MAJOR', 'MINOR', 'NOTE')
STATUSES = ('OPEN', 'CONFIRMED', 'CONFIRMED_WITH_CORRECTION', 'REFUTED', 'NEW_DEFECT', 'SCOPE_LIMITED')
LEGACY_STATUS = {  # V51 legacy map (SPEC 6.4)
    'CONFIRMED_WITH_QUANTITATIVE_CORRECTION': 'CONFIRMED_WITH_CORRECTION',
    'RISK_CONFIRMED_UNDER_ACTUAL_NATIVE_MASKS': 'CONFIRMED_WITH_CORRECTION',
    'MATCHED_EVENT_SENSITIVITY_SUPPORTS_CALIBRATION_FRAME_DEFECT': 'CONFIRMED_WITH_CORRECTION',
    'COUNTERFACTUAL_FAILURE_REPRODUCED': 'CONFIRMED',
    'SPECIFICATION_IMPLEMENTATION_MISMATCH': 'NEW_DEFECT',
    'LATENT_INTEGRATION_DEFECT': 'NEW_DEFECT',
    'CLAUDE_HISTORY_STATEMENT_CORRECTED': 'REFUTED',
}
AUTHOR_EDITABLE = ('reconciled_by_author',)  # finding fields the delivery author may change after the review
# OUTPUTS fields of the canonical row digest (attempt_id is provenance and differs between runs)
DIGEST_FIELDS = ('world', 'world_role', 'arm', 'pool', 'checkpoint', 'item_id', 'prompt_hex', 'expected_hex',
                 'stream_hex', 'emitted_hex', 'own_eos', 'answer_path', 'native_closure_sha', 'params_sha256')
REV07_EXTS = ('.json', '.jsonl', '.jsonl.gz', '.md', '.py')  # SPEC 20.4 bounded-scan file types
MIN_REPLAY_PAIRS = 4


# --------------------------------------------------------------------------------------
# REVIEW.json and identity
# --------------------------------------------------------------------------------------

def read_review(ctx: common.Ctx) -> dict:
    if not ctx.path(REVIEW_REL).is_file():
        raise MissingInput('%s absent: no cross-family review recorded' % REVIEW_REL)
    r = ctx.read_json(REVIEW_REL)
    if not isinstance(r, dict):
        raise InvalidInput('%s must be a JSON object' % REVIEW_REL)
    return r


def families(ctx: common.Ctx) -> Tuple[str, str]:
    af = admitcheck.author_family(ctx.lock)
    if af is None:
        raise InvalidInput('lock.author_agent %r is not codex|claude' % (ctx.lock.get('author_agent'),))
    return af, other_family(af)


def in_namespace(branches: Sequence[str], fam: str) -> List[str]:
    rx = re.compile(r'^(?:remotes/)?origin/%s/' % re.escape(fam))
    return [b for b in branches if rx.match(str(b))]


def trailer_families(c: common.Commit) -> List[Optional[str]]:
    return [family(v) for v in c.trailer_values('Acting-Agent')]


def is_family_commit(ctx: common.Ctx, c: Optional[common.Commit], fam: str) -> Tuple[bool, str]:
    """(ok, why): every Acting-Agent trailer names `fam` and the commit is in an origin/<fam>/* branch."""
    if c is None:
        return False, 'no commit'
    tf = trailer_families(c)
    if not tf:
        return False, 'commit %s has no Acting-Agent trailer' % c.sha[:12]
    if any(f != fam for f in tf):
        return False, 'commit %s Acting-Agent %s is not %s' % (c.sha[:12], c.trailer_values('Acting-Agent'), fam)
    br = in_namespace(ctx.git.remote_branches_containing(c.sha), fam)
    if not br:
        return False, 'commit %s is in no origin/%s/* branch' % (c.sha[:12], fam)
    return True, '%s in %s' % (c.sha[:12], ','.join(br[:2]))


def _strip_author_fields(obj: Any) -> Any:
    if not isinstance(obj, dict):
        return obj
    out = dict(obj)
    fs = out.get('findings')
    if isinstance(fs, list):
        out['findings'] = [({k: v for k, v in f.items() if k not in AUTHOR_EDITABLE} if isinstance(f, dict) else f)
                           for f in fs]
    return out


def _parse_or_none(data: Optional[bytes]) -> Any:
    if data is None:
        return None
    try:
        return common.loads_strict(data, what=REVIEW_REL)
    except InvalidInput:
        return '<unparseable>'


def review_history_problems(ctx: common.Ctx, rf: str) -> Tuple[List[str], Optional[common.Commit]]:
    """(problems, introducing commit): later versions of REVIEW.json that are not the reviewer's own commits may
    change only findings[].reconciled_by_author; the worktree counts as one more unattributed version."""
    rel = ctx.repo_rel_of(REVIEW_REL)
    versions = [(c, ctx.git.file_at(c.sha, rel)) for c in ctx.git.log([rel])]
    intro = None
    for c, d in versions:
        if d is not None:
            intro = c
            break
    problems: List[str] = []
    prev = None
    started = False
    for c, d in versions:
        if not started:
            if d is None:
                continue
            started = True
            prev = d
            continue
        if d == prev:
            continue
        ok, _why = is_family_commit(ctx, c, rf)
        if not ok and _strip_author_fields(_parse_or_none(d)) != _strip_author_fields(_parse_or_none(prev)):
            problems.append('commit %s (Acting-Agent %s) changed reviewer-owned fields of REVIEW.json; only the '
                            "reviewer's own commits may (the author edits only findings[].reconciled_by_author)"
                            % (c.sha[:12], c.trailer_values('Acting-Agent') or 'none'))
        prev = d
    wt = ctx.path(REVIEW_REL).read_bytes() if ctx.path(REVIEW_REL).is_file() else None
    if started and wt is not None and wt != prev:
        if _strip_author_fields(_parse_or_none(wt)) != _strip_author_fields(_parse_or_none(prev)):
            problems.append('the worktree REVIEW.json differs from the last committed version in reviewer-owned '
                            'fields (uncommitted, unattributed edit)')
    return problems, intro


@check('REV-01')
def check_rev_01(ctx: common.Ctx) -> Result:
    review = read_review(ctx)
    af, rf = families(ctx)
    problems: List[str] = []
    missing: List[str] = []
    notes: List[str] = []
    fam = family(review.get('reviewer_agent'))
    if fam is None:
        problems.append('reviewer_agent %r names no agent family' % (review.get('reviewer_agent'),))
    elif fam == af:
        problems.append('reviewer_agent %r is the author family %s: a same-family review is debt, not review'
                        % (review.get('reviewer_agent'), af))
    if family(review.get('author_agent')) != af:
        problems.append('REVIEW.author_agent %r is not the lock author family %s' % (review.get('author_agent'), af))
    if review.get('delivery') not in (ctx.delivery_name, ctx.delivery_rel):
        problems.append('REVIEW.delivery %r names another delivery' % (review.get('delivery'),))
    lrel = ctx.repo_rel_of('ENDPOINT_LOCK.json')
    c_lock = ctx.git.first_commit_adding(lrel)
    if c_lock is None:
        missing.append('ENDPOINT_LOCK.json is not committed')
    else:
        lf = trailer_families(c_lock)
        if not lf:
            problems.append('the lock commit %s has no Acting-Agent trailer (author family cannot be corroborated)'
                            % c_lock.sha[:12])
        elif any(f != af for f in lf):
            problems.append('the lock commit %s says Acting-Agent %s, lock.author_agent says %s'
                            % (c_lock.sha[:12], c_lock.trailer_values('Acting-Agent'), af))
    hist, intro = review_history_problems(ctx, rf)
    if intro is None:
        missing.append('review/REVIEW.json is not committed')
    else:
        ok, why = is_family_commit(ctx, intro, rf)
        if not ok:
            problems.append('the commit introducing REVIEW.json is not the reviewer family %s: %s' % (rf, why))
        else:
            notes.append('introduced by %s' % why)
        rb = review.get('reviewer_branch')
        if s(rb):
            norm = rb.strip()
            norm = norm if norm.startswith(('origin/', 'remotes/')) else 'origin/' + norm
            norm = norm[len('remotes/'):] if norm.startswith('remotes/') else norm
            br = [b[len('remotes/'):] if b.startswith('remotes/') else b
                  for b in ctx.git.remote_branches_containing(intro.sha)]
            if not in_namespace([norm], rf):
                problems.append('reviewer_branch %s is not in the origin/%s/* namespace' % (rb, rf))
            elif norm not in br:
                problems.append('reviewer_branch %s does not contain the introducing commit %s' % (rb, intro.sha[:12]))
    problems.extend(hist)
    return finish('REV-01', problems, missing, 'reviewer %s (%s) differs from author %s; %s'
                  % (fam, review.get('reviewer_session'), af, '; '.join(notes)))


# --------------------------------------------------------------------------------------
# REV-02..05 (executed checks, findings, shortcut attempt, canary)
# --------------------------------------------------------------------------------------

def review_file(ctx: common.Ctx, entry: Any) -> Tuple[Optional[str], str]:
    """(delivery-relative path under review/, problem) for a script/output entry given delivery- or
    repo-relative."""
    if not s(entry):
        return None, 'path is not set'
    try:
        rel = common.norm_rel(entry)
    except InvalidInput as e:
        return None, str(e)
    d = ctx.delivery_rel
    if d and rel.startswith(d + '/'):
        rel = rel[len(d) + 1:]
    if not rel.startswith('review/'):
        return None, '%s is not under the delivery review/ dir' % entry
    if not ctx.path(rel).is_file():
        return None, '%s does not exist' % rel
    return rel, ''


def committed_sha(ctx: common.Ctx, rel: str) -> Optional[str]:
    data = ctx.git.file_at(ctx.git.head(), ctx.repo_rel_of(rel))
    return common.sha256_bytes(data) if data is not None else None


def executed_checks(review: dict) -> List[dict]:
    ec = review.get('executed_checks')
    return [e for e in ec if isinstance(e, dict)] if isinstance(ec, list) else []


@check('REV-02')
def check_rev_02(ctx: common.Ctx) -> Result:
    review = read_review(ctx)
    checks = executed_checks(review)
    problems: List[str] = []
    if not isinstance(review.get('executed_checks'), list) or not checks:
        return Result('REV-02', FAIL, 'executed_checks is empty: a finding without an executed check is UNVERIFIED')
    kinds = [e.get('kind') for e in checks]
    need = list(REQUIRED_KINDS) + (['oracle_upstream'] if ctx.lock.get('stage_chain') is True else [])
    for k in need:
        if k not in kinds:
            problems.append('no executed check of kind %s' % k)
    seen = set()
    for i, e in enumerate(checks):
        tag = 'executed_checks[%d] %s' % (i, e.get('id'))
        if not s(e.get('id')):
            problems.append('%s has no id' % tag)
        elif e['id'] in seen:
            problems.append('%s: duplicate id' % tag)
        seen.add(e.get('id'))
        if e.get('kind') not in CHECK_KINDS:
            problems.append('%s kind %r is not one of %s' % (tag, e.get('kind'), '|'.join(CHECK_KINDS)))
        for fkey, skey in (('script', 'script_sha256'), ('output', 'output_sha256')):
            rel, why = review_file(ctx, e.get(fkey))
            if rel is None:
                problems.append('%s %s: %s' % (tag, fkey, why))
                continue
            cur = common.sha256_file(ctx.path(rel))
            if strip_sha(e.get(skey)) != cur:
                problems.append('%s %s sha256 %s does not match %s (%s)' % (tag, fkey, str(e.get(skey))[:16], rel, cur[:16]))
            if committed_sha(ctx, rel) != cur:
                problems.append('%s %s %s is not committed as is' % (tag, fkey, rel))
        cmd = e.get('cmd')
        if not (s(cmd) or (isinstance(cmd, list) and cmd and all(isinstance(x, str) for x in cmd))):
            problems.append('%s has no cmd' % tag)
        cap = e.get('cpu_cap_s')
        if not is_num(cap) or cap <= 0:
            problems.append('%s cpu_cap_s %r is not a positive number' % (tag, cap))
        if e.get('result') is None or e.get('result') == admitcheck.FILL or e.get('result') == '':
            problems.append('%s records no result' % tag)
        if e.get('kind') in COMPARED_KINDS:
            if not s(e.get('compared_to')):
                problems.append('%s (%s) names nothing it was compared_to' % (tag, e.get('kind')))
            if not isinstance(e.get('match'), bool):
                problems.append('%s (%s) match %r is not a boolean' % (tag, e.get('kind'), e.get('match')))
        elif e.get('match') is not None and not isinstance(e.get('match'), bool):
            problems.append('%s (%s) match %r is neither null nor a boolean' % (tag, e.get('kind'), e.get('match')))
    return finish('REV-02', problems, [], '%d executed checks (%s) with committed scripts and outputs, matching sha256 '
                  'and CPU caps' % (len(checks), ','.join(sorted(set(str(k) for k in kinds)))))


def _finding_refs(f: dict) -> Set[str]:
    refs: Set[str] = set()
    for k in ('executed', 'executed_check', 'check_id', 'check', 'checks', 'evidence'):
        v = f.get(k)
        if isinstance(v, str):
            refs.add(v)
            refs.update(re.findall(r'[A-Za-z0-9_.:-]+', v))
        elif isinstance(v, list):
            for x in v:
                if isinstance(x, str):
                    refs.add(x)
                    refs.update(re.findall(r'[A-Za-z0-9_.:-]+', x))
    claim = f.get('claim')
    if isinstance(claim, str):
        refs.update(re.findall(r'[A-Za-z0-9_.:-]+', claim))
    return refs


def findings(review: dict) -> List[dict]:
    fs = review.get('findings')
    if fs is None:
        return []
    if not isinstance(fs, list):
        raise InvalidInput('REVIEW.findings must be a list')
    return [f for f in fs if isinstance(f, dict)]


@check('REV-03')
def check_rev_03(ctx: common.Ctx) -> Result:
    review = read_review(ctx)
    reruns = [e for e in executed_checks(review) if e.get('kind') == 'rerun_admission']
    if not reruns:
        return Result('REV-03', FAIL, 'no rerun_admission executed check (REV-02)')
    fs = findings(review)
    problems = []
    mism = 0
    for e in reruns:
        m = e.get('match')
        if not isinstance(m, bool):
            problems.append('rerun_admission %s match %r is not a boolean' % (e.get('id'), m))
            continue
        if m is False:
            mism += 1
            if not any(s(e.get('id')) and e['id'] in _finding_refs(f) for f in fs):
                problems.append('rerun_admission %s does not reproduce the admission (match false) and no finding '
                                'references it' % e.get('id'))
    return finish('REV-03', problems, [], '%d rerun_admission checks, %d mismatching, each with a finding'
                  % (len(reruns), mism))


@check('REV-04')
def check_rev_04(ctx: common.Ctx) -> Result:
    review = read_review(ctx)
    csa = review.get('cheapest_shortcut_attempt')
    problems = []
    if not isinstance(csa, dict):
        return Result('REV-04', FAIL, 'cheapest_shortcut_attempt is not recorded')
    if not s(csa.get('description')):
        problems.append('cheapest_shortcut_attempt.description is empty')
    rel, why = review_file(ctx, csa.get('script'))
    if rel is None:
        problems.append('cheapest_shortcut_attempt.script: %s' % why)
    plr = csa.get('passed_locked_rule')
    if not isinstance(plr, bool):
        problems.append('cheapest_shortcut_attempt.passed_locked_rule %r is not a boolean' % (plr,))
    if not any(e.get('kind') == 'shortcut_attempt' for e in executed_checks(review)):
        problems.append('no executed check of kind shortcut_attempt')
    if plr is True and review.get('decision') != 'BLOCK':
        problems.append('the cheapest shortcut passes the locked rule but decision is %r (must be BLOCK)'
                        % (review.get('decision'),))
    return finish('REV-04', problems, [], 'shortcut attempt recorded (%s); passed_locked_rule %s, decision %s'
                  % (csa.get('description'), plr, review.get('decision')))


def current_lock_commit(ctx: common.Ctx) -> Optional[common.Commit]:
    cur = ctx.path('ENDPOINT_LOCK.json').read_bytes()
    return ctx.git.first_commit_where(ctx.repo_rel_of('ENDPOINT_LOCK.json'), lambda b: b == cur)


@check('REV-05')
def check_rev_05(ctx: common.Ctx) -> Result:
    review = read_review(ctx)
    can = review.get('canary')
    if not isinstance(can, dict):
        return Result('REV-05', FAIL, 'REVIEW.canary is not recorded (run MFC canary <dir> --seed <hex> after the lock)')
    problems: List[str] = []
    missing: List[str] = []
    n, det = can.get('n'), can.get('detected')
    if not is_int(n) or n < 1:
        problems.append('canary.n %r is not a positive integer' % (n,))
    if not is_int(det):
        problems.append('canary.detected %r is not an integer' % (det,))
    if is_int(n) and is_int(det) and det != n:
        problems.append('canary detected %d of %d mutants: every applicable mutant must be caught (a survivor '
                        'becomes an incident)' % (det, n))
    seed = can.get('seed')
    if not s(seed) or not re.fullmatch(r'[0-9a-fA-F]+', seed.strip()):
        problems.append('canary.seed %r is not hex' % (seed,))
    rep_rel = can.get('report')
    rel = None
    if not s(rep_rel):
        problems.append('canary.report is not set')
    else:
        rel = common.norm_rel(rep_rel)
        d = ctx.delivery_rel
        if d and rel.startswith(d + '/'):
            rel = rel[len(d) + 1:]
        if not re.fullmatch(r'checks/CANARY_[^/]+\.json', rel):
            problems.append('canary.report %s is not checks/CANARY_<utc>.json' % rel)
            rel = None
        elif not ctx.path(rel).is_file():
            missing.append('canary report %s does not exist' % rel)
            rel = None
    if rel is not None:
        rep = ctx.read_json(rel)
        if not isinstance(rep, dict) or rep.get('schema') != SCHEMA_CANARY_REPORT:
            problems.append('%s is not a %s report' % (rel, SCHEMA_CANARY_REPORT))
        else:
            if str(rep.get('seed', '')).lower() != str(seed or '').strip().lower():
                problems.append('%s seed %r differs from REVIEW.canary.seed %r' % (rel, rep.get('seed'), seed))
            if rep.get('n_mutants') != n or rep.get('detected') != det:
                problems.append('%s reports %r/%r mutants detected, REVIEW says %r/%r'
                                % (rel, rep.get('detected'), rep.get('n_mutants'), det, n))
            if rep.get('exit_code') != 0 or rep.get('verdict') != 'PASS':
                problems.append('%s verdict %r (CAN-01..03 must all PASS)' % (rel, rep.get('verdict')))
            sts = {r.get('id'): r.get('status') for r in rep.get('results') or [] if isinstance(r, dict)}
            for cid in ('CAN-01', 'CAN-02', 'CAN-03'):
                if sts.get(cid) != PASS:
                    problems.append('%s %s is %r' % (rel, cid, sts.get(cid)))
            if rep.get('delivery') != ctx.delivery_rel:
                problems.append('%s names delivery %r' % (rel, rep.get('delivery')))
            if strip_sha(rep.get('endpoint_lock_sha256')) != ctx.endpoint_lock_sha256:
                problems.append('%s ran on another ENDPOINT_LOCK (%s)' % (rel, str(rep.get('endpoint_lock_sha256'))[:12]))
            if ctx.git_mode == 'real' and rep.get('git_mode') != 'real':
                problems.append('%s was written in %s mode' % (rel, rep.get('git_mode')))
            c_lock = current_lock_commit(ctx)
            c_rep = ctx.git.first_commit_adding(ctx.repo_rel_of(rel))
            if c_lock is None:
                missing.append('the current ENDPOINT_LOCK is not committed')
            elif c_rep is None:
                missing.append('canary report %s is not committed' % rel)
            elif not strictly_before(ctx, c_lock, c_rep):
                problems.append('canary report commit %s does not follow the ENDPOINT_LOCK commit %s (the seed must be '
                                'chosen after the lock)' % (c_rep.sha[:12], c_lock.sha[:12]))
            elif ctx.git_mode == 'real':
                want = common.checker_sha256_at(ctx.git, c_rep.sha)
                if want is None or rep.get('checker_sha256') != want:
                    problems.append('%s checker_sha256 %s is not the committed checker at %s (%s)'
                                    % (rel, str(rep.get('checker_sha256'))[:12], c_rep.sha[:12], str(want)[:12]))
    return finish('REV-05', problems, missing, 'canary seed %s detected %s/%s mutants; report %s committed after the lock'
                  % (seed, det, n, rel))


# --------------------------------------------------------------------------------------
# REV-06 (timing, decision, findings), REV-07 (padding)
# --------------------------------------------------------------------------------------

def norm_status(st: Any) -> Optional[str]:
    if not isinstance(st, str):
        return None
    t = st.strip().upper()
    t = LEGACY_STATUS.get(t, t)
    return t if t in STATUSES else None


def _reconciled(f: dict) -> bool:
    r = f.get('reconciled_by_author')
    return isinstance(r, dict) and s(r.get('note')) and common.normalize_utc(r.get('utc')) is not None


def _executed(f: dict, check_ids: Set[str]) -> bool:
    ex = f.get('executed')
    if ex is True:
        return True
    if isinstance(ex, str) and ex.strip() in check_ids:
        return True
    return False


@check('REV-06')
def check_rev_06(ctx: common.Ctx) -> Result:
    review = read_review(ctx)
    problems: List[str] = []
    missing: List[str] = []
    notes: List[str] = []
    if review.get('decision') != 'ADMIT':
        problems.append('decision is %r, not ADMIT' % (review.get('decision'),))
    ids = set(str(e.get('id')) for e in executed_checks(review) if s(e.get('id')))
    fs = findings(review)
    for i, f in enumerate(fs):
        tag = 'finding %s' % (f.get('id') or i)
        sev = f.get('severity')
        st = norm_status(f.get('status') if f.get('status') is not None else 'OPEN')
        if sev not in SEVERITIES:
            problems.append('%s severity %r is not one of %s' % (tag, sev, '|'.join(SEVERITIES)))
            continue
        if st is None:
            problems.append('%s status %r is not a known status (or V51 legacy status)' % (tag, f.get('status')))
            continue
        if sev == 'BLOCKER' and not _executed(f, ids):
            problems.append('%s is an UNVERIFIED BLOCKER (no executed check): it pauses the stage until executed' % tag)
        if sev in ('BLOCKER', 'MAJOR'):
            if st == 'OPEN':
                problems.append('%s (%s) is open' % (tag, sev))
            elif not _reconciled(f):
                problems.append('%s (%s, %s) is not reconciled by the author (reconciled_by_author {utc, note})'
                                % (tag, sev, st))
        if sev == 'BLOCKER' and st == 'REFUTED':
            ud = f.get('ruling_user_decision')
            ok_ud = s(ud) and ctx.kb.user_decision(ud.strip()) is not None
            if f.get('countersigned_by_reviewer') is not True and not ok_ud:
                problems.append('%s is a REFUTED BLOCKER without the reviewer countersign or an owner ruling '
                                '(ruling_user_decision in USER_DECISIONS)' % tag)
    rrel = ctx.repo_rel_of(REVIEW_REL)

    def admits(b: bytes) -> bool:
        try:
            o = common.loads_strict(b, what=REVIEW_REL)
        except InvalidInput:
            return False
        return isinstance(o, dict) and o.get('decision') == 'ADMIT'
    c_admit = ctx.git.first_commit_where(rrel, admits)
    c_dev = admitcheck.first_output_commit(ctx, 'dev')
    wt_dev = admitcheck.worktree_outputs(ctx, 'dev')
    if c_admit is None:
        if c_dev is not None or wt_dev:
            problems.append('no committed REVIEW.json with decision ADMIT precedes the dev outputs')
        else:
            missing.append('REVIEW.json with decision ADMIT is not committed')
    elif c_dev is not None and not strictly_before(ctx, c_admit, c_dev):
        problems.append('the ADMIT review commit %s is not a proper ancestor of the first dev output commit %s'
                        % (c_admit.sha[:12], c_dev.sha[:12]))
    else:
        notes.append('ADMIT committed at %s%s' % (c_admit.sha[:12], (' before dev output %s' % c_dev.sha[:12])
                                                  if c_dev else ' (no dev output committed yet)'))
    return finish('REV-06', problems, missing, 'decision ADMIT; %d findings, no open BLOCKER/MAJOR; %s'
                  % (len(fs), '; '.join(notes)))


def _other_review_files(ctx: common.Ctx) -> Tuple[List[Tuple[str, Any]], str]:
    """[(repo-relative path, Path)] of scanned files inside some review/ directory outside this delivery."""
    scan = ctx.repo_scan()
    me = ctx.delivery_rel
    out = []
    for p in scan.files:
        rel = scan.rel(p)
        if me and (rel == me or rel.startswith(me + '/')):
            continue
        parts = rel.split('/')
        if 'review' in parts[:-1]:
            out.append((rel, p))
    return out, scan.scope()


@check('REV-07')
def check_rev_07(ctx: common.Ctx) -> Result:
    review = read_review(ctx)
    outs: List[Tuple[str, str, str]] = []  # (check id, rel, sha)
    problems: List[str] = []
    missing: List[str] = []
    for e in executed_checks(review):
        rel, why = review_file(ctx, e.get('output'))
        if rel is None:
            problems.append('executed check %s output: %s' % (e.get('id'), why))
            continue
        if not rel.endswith(REV07_EXTS):
            missing.append('executed check %s output %s is outside the bounded scan types %s (SPEC 20.4): write it '
                           'as .json/.jsonl/.md so padding can be checked' % (e.get('id'), rel, ','.join(REV07_EXTS)))
            continue
        outs.append((str(e.get('id')), rel, common.sha256_file(ctx.path(rel))))
    by_sha: Dict[str, List[str]] = {}
    for cid, rel, sha in outs:
        by_sha.setdefault(sha, []).append('%s (%s)' % (rel, cid))
    for sha, lst in by_sha.items():
        if len(lst) > 1:
            problems.append('executed-check outputs %s are byte-identical (padding)' % ', '.join(lst))
    others, scope = _other_review_files(ctx)
    sizes: Dict[int, List[Tuple[str, str, str]]] = {}
    for o in outs:
        sizes.setdefault(ctx.path(o[1]).stat().st_size, []).append(o)
    n_cmp = 0
    for orel, p in others:
        try:
            sz = p.stat().st_size
        except OSError as e:
            raise MissingInput('repo scan could not stat %s: %s' % (orel, e))
        cands = sizes.get(sz)
        if not cands:
            continue
        n_cmp += 1
        sha = common.sha256_file(p)
        for cid, rel, osha in cands:
            if sha == osha:
                problems.append('executed check %s output %s is byte-identical to %s in another delivery (padding)'
                                % (cid, rel, orel))
    return finish('REV-07', problems, missing, '%d executed-check outputs unique against %d review/ files elsewhere '
                  '(%d size matches hashed); %s' % (len(outs), len(others), n_cmp, scope))


# --------------------------------------------------------------------------------------
# REV-08 (post-run replay) and the replay plan
# --------------------------------------------------------------------------------------

def row_digest(rows: Sequence[dict]) -> str:
    """sha256 over canonical rows (DIGEST_FIELDS; attempt_id excluded) sorted by (pool, checkpoint, item_id)."""
    h = hashlib.sha256()
    keyed = []
    for r in rows:
        sub = {k: r.get(k) for k in DIGEST_FIELDS}
        keyed.append(((str(sub['pool']), str(sub['checkpoint']), str(sub['item_id'])), common.canonical_json(sub)))
    for _k, b in sorted(keyed):
        h.update(b)
        h.update(b'\n')
    return h.hexdigest()


def official_pair_digests(ctx: common.Ctx) -> Dict[Tuple[str, str], dict]:
    """{(world, arm): {digest, n_rows}} of results/official/OUTPUTS rows (gz and bundles allowed)."""
    key = 'wp4_official_digests'
    if key in ctx._cache:
        return ctx._cache[key]
    groups: Dict[Tuple[str, str], List[dict]] = {}
    for r in ctx.iter_rows('results/official/OUTPUTS.jsonl'):
        groups.setdefault((str(r.get('world')), str(r.get('arm'))), []).append(r)
    out = {k: {'digest': row_digest(v), 'n_rows': len(v)} for k, v in groups.items()}
    ctx._cache[key] = out
    return out


def _seed_bytes(seed: Any) -> bytes:
    if not isinstance(seed, str) or not re.fullmatch(r'(?:[0-9a-fA-F]{2})+', seed.strip()):
        raise InvalidInput('seed %r must be an even-length hex string' % (seed,))
    return bytes.fromhex(seed.strip())


def replay_plan(ctx: common.Ctx, seed: str, n: int = MIN_REPLAY_PAIRS) -> dict:
    """Deterministic, seed-keyed choice of n (world, arm) pairs from the official outputs: the pairs with the
    smallest HMAC-SHA256(seed, '<world>|<arm>')."""
    key = _seed_bytes(seed)
    if not is_int(n) or n < 1:
        raise InvalidInput('--n must be a positive integer')
    dig_ = official_pair_digests(ctx)
    if not dig_:
        raise MissingInput('no official rows (results/official/OUTPUTS.jsonl): nothing to replay')
    ranked = sorted(dig_, key=lambda p: (hmac.new(key, ('%s|%s' % p).encode('utf-8'), hashlib.sha256).hexdigest(), p))
    chosen = ranked[:n]
    argv = None
    for a in common.read_attempts(ctx):
        if a.get('stage') == 'official' and isinstance(a.get('argv'), list):
            argv = list(a['argv'])
            break
    pairs = []
    for w, a in chosen:
        ent = {'world': w, 'arm': a, 'expected_digest': dig_[(w, a)]['digest'], 'n_rows': dig_[(w, a)]['n_rows']}
        if argv is not None:
            ent['replay_argv'] = argv + ['--only-world', w, '--only-arm', a]
        pairs.append(ent)
    return {'schema': SCHEMA_REPLAY_PLAN, 'delivery': ctx.delivery_rel, 'seed': seed.strip().lower(), 'n': len(pairs),
            'pairs_available': len(dig_), 'digest_fields': list(DIGEST_FIELDS), 'pairs': pairs,
            'endpoint_lock_sha256': ctx.endpoint_lock_sha256, 'source_lock_sha256': ctx.source_lock_sha256}


@check('REV-08')
def check_rev_08(ctx: common.Ctx) -> Result:
    if not (common.sealed_confirm_intended(ctx) or common.adoption_intended(ctx)):
        return Result('REV-08', NOT_APPLICABLE, 'neither SEALED_CONFIRM nor adoption is intended')
    review = read_review(ctx)
    post = review.get('postrun')
    if not isinstance(post, dict) or not isinstance(post.get('replay'), list) or not post.get('replay'):
        return Result('REV-08', FAIL, 'REVIEW.postrun.replay is empty: the reviewer must replay seeded official '
                      'histories (MFC review replay-plan)')
    dig_ = official_pair_digests(ctx)
    if not dig_:
        raise MissingInput('no official rows to compare the replay with')
    problems: List[str] = []
    seen = set()
    for i, e in enumerate(post['replay']):
        if not isinstance(e, dict):
            problems.append('postrun.replay[%d] is not an object' % i)
            continue
        pair = (str(e.get('world')), str(e.get('arm')))
        tag = 'replay %s/%s' % pair
        if pair in seen:
            problems.append('%s listed twice' % tag)
        seen.add(pair)
        ref = dig_.get(pair)
        if ref is None:
            problems.append('%s has no official rows' % tag)
            continue
        if strip_sha(e.get('expected_digest')) != ref['digest']:
            problems.append('%s expected_digest %s is not the canonical digest of the official rows %s'
                            % (tag, str(e.get('expected_digest'))[:16], ref['digest'][:16]))
        if strip_sha(e.get('observed_digest')) != ref['digest']:
            problems.append('%s observed_digest %s does not reproduce the official rows %s'
                            % (tag, str(e.get('observed_digest'))[:16], ref['digest'][:16]))
    need = min(MIN_REPLAY_PAIRS, len(dig_))
    if len(seen) < need:
        problems.append('%d distinct pairs replayed, at least %d required' % (len(seen), need))
    seed = post.get('seed')
    note = 'pairs chosen by the reviewer (no postrun.seed recorded)'
    if seed is not None:
        plan = replay_plan(ctx, seed, max(len(seen), 1))
        want = set((p['world'], p['arm']) for p in plan['pairs'])
        if want != seen:
            problems.append('replayed pairs %s are not the replay plan of seed %s %s' % (sorted(seen), seed, sorted(want)))
        note = 'pairs equal the replay plan of seed %s' % seed
    return finish('REV-08', problems, [], '%d replayed (world, arm) histories reproduce the official row digests; %s'
                  % (len(seen), note))


# --------------------------------------------------------------------------------------
# DEBT-01 / DEBT-02
# --------------------------------------------------------------------------------------

def _same_delivery(a: Any, b: Any) -> bool:
    if not isinstance(a, str) or not isinstance(b, str):
        return False
    a, b = a.strip().strip('/'), b.strip().strip('/')
    return a == b or a.rsplit('/', 1)[-1] == b.rsplit('/', 1)[-1]


def open_debts(ctx: common.Ctx) -> List[dict]:
    rows = ctx.kb.rows('REVIEW_DEBT')
    repaid = set(r.get('id') for r in rows if r.get('repaid_by'))
    return [r for r in rows if not r.get('repaid_by') and r.get('id') not in repaid]


def my_debts(ctx: common.Ctx) -> Tuple[List[dict], List[dict]]:
    """(open debts of this delivery, repaid debts of this delivery)."""
    rows = ctx.kb.rows('REVIEW_DEBT')
    me = [r for r in rows if _same_delivery(r.get('delivery'), ctx.delivery_rel)]
    repaid_ids = set(r.get('id') for r in me if r.get('repaid_by'))
    open_ = [r for r in me if not r.get('repaid_by') and r.get('id') not in repaid_ids]
    repaid = [r for r in me if r.get('repaid_by')]
    return open_, repaid


@check('DEBT-01')
def check_debt_01(ctx: common.Ctx) -> Result:
    mine, _rep = my_debts(ctx)
    problems: List[str] = []
    cap = ctx.kb.constraints().get('review_debt_max_open')
    if not is_int(cap):
        raise MissingInput('CONSTRAINTS.review_debt_max_open missing')
    if not mine:
        return Result('DEBT-01', FAIL, 'no valid REVIEW and no open REVIEW_DEBT row for %s: record debt (dev and pilot '
                      'only) or obtain the cross-family review' % ctx.delivery_rel)
    for r in mine:
        if not s(r.get('reason')) or not s(r.get('recorded_by')) or common.normalize_utc(r.get('utc')) is None:
            problems.append('REVIEW_DEBT %s lacks reason, recorded_by or utc' % r.get('id'))
    total = len(set(r.get('id') for r in open_debts(ctx)))
    if total >= cap:
        problems.append('%d open review debts program-wide (cap review_debt_max_open %d): no new T2 roster may start'
                        % (total, cap))
    role = dig(ctx.lock, 'worlds.official_role_intended')
    official = ctx.stage == 'official' or ctx.options.get('launch_stage') == 'official'
    if role == 'SEALED_CONFIRM' and official:
        problems.append('review debt cannot cover a SEALED_CONFIRM official run')
    elif official and role != 'PILOT' and not ctx.dry:
        problems.append('review debt covers dev and pilot only: the %s official run needs a real cross-family review'
                        % role)
    return finish('DEBT-01', problems, [], 'open REVIEW_DEBT %s (%d open program-wide < %d); outputs are labelled DEV '
                  'or PILOT' % (','.join(str(r.get('id')) for r in mine), total, cap))


@check('DEBT-02')
def check_debt_02(ctx: common.Ctx) -> Result:
    mine, repaid = my_debts(ctx)
    notes = []
    if repaid:
        # the author family is needed only to judge a repayment (a legacy pointer target has no lock and no debt)
        try:
            af = admitcheck.author_family(ctx.lock)
        except MissingInput:
            af = None
        ok = False
        try:
            rv = read_review(ctx)
            ok = family(rv.get('reviewer_agent')) not in (None, af)
        except (MissingInput, InvalidInput):
            ok = False
        if not ok:
            mine = mine + repaid
            notes.append('repaid debt %s has no cross-family review/REVIEW.json: still open'
                         % ','.join(str(r.get('id')) for r in repaid))
    if not mine:
        return Result('DEBT-02', PASS, 'no open review debt for %s' % ctx.delivery_rel)
    ids = ','.join(sorted(set(str(r.get('id')) for r in mine)))
    problems: List[str] = []
    if ctx.stage == 'pointer':
        problems.append('open review debt %s: the pointer stage fails until a cross-family review repays it' % ids)
    if common.sealed_confirm_intended(ctx):
        problems.append('open review debt %s: the delivery cannot be SEALED_CONFIRM' % ids)
    if common.adoption_intended(ctx):
        problems.append('open review debt %s: adoption is blocked' % ids)
    acts = []
    try:
        rows = ctx.kb.rows('FINDINGS')
    except MissingInput:
        rows = []
    mine_f = set(r.get('id') for r in rows if r.get('kind', 'finding') == 'finding'
                 and _same_delivery(r.get('delivery'), ctx.delivery_rel))
    for r in rows:
        if r.get('kind') == 'activation' and r.get('refers_to') in mine_f:
            acts.append(str(r.get('id')))
    if acts:
        problems.append('open review debt %s: findings of this delivery were activated (%s)' % (ids, ','.join(acts)))
    if problems:
        return Result('DEBT-02', FAIL, '; '.join(problems + notes), evidence=problems)
    return Result('DEBT-02', PASS, 'open review debt %s: not SEALED_CONFIRM, no adoption, no activated findings%s'
                  % (ids, ('; ' + '; '.join(notes)) if notes else ''))


# --------------------------------------------------------------------------------------
# MFC review
# --------------------------------------------------------------------------------------

REV_PRIMARY = ('REV-01', 'REV-02', 'REV-03', 'REV-04', 'REV-05', 'REV-06', 'REV-07')


def review_results(ctx: common.Ctx) -> List[Result]:
    """REV-01..07 with DEBT-01 as the declared alternative (as in the roster bundle), REV-08 when SEALED_CONFIRM or
    adoption is intended, and DEBT-02."""
    rules = ctx.rules or common.load_rules()
    idx = common.rule_index(rules)
    out: List[Result] = []
    prim: List[Result] = []
    # the group's primary ids come from RULES.json (roster bundle group review_or_debt), so this command and the
    # gate can never disagree; REV_PRIMARY is the fallback when the group is absent
    group = [e for e in ((rules.get('bundles') or {}).get('roster') or {}).get('checks') or []
             if isinstance(e, dict) and e.get('group') == 'review_or_debt']
    primary_ids = list(group[0].get('primary') or []) if group else list(REV_PRIMARY)
    for rid in primary_ids:
        prim.extend(common.run_check(ctx, idx[rid]))
    if all(r.status in common.NON_BLOCKING_STATUSES for r in prim):
        out.extend(prim)
        out.append(Result('DEBT-01', NOT_APPLICABLE, 'alternative review_or_debt satisfied by %s' % ','.join(primary_ids)))
    else:
        fb = common.run_check(ctx, idx['DEBT-01'])
        if all(r.status in common.NON_BLOCKING_STATUSES for r in fb) and any(r.status == PASS for r in fb):
            for r in prim:
                if r.status in (FAIL, MISSING):
                    r.detail = '[covered by DEBT-01; was %s] %s' % (r.status, r.detail)
                    r.status = WARN
        out.extend(prim)
        out.extend(fb)
    out.extend(common.run_check(ctx, idx['REV-08']))
    out.extend(common.run_check(ctx, idx['DEBT-02']))
    return out


def cmd_review(args: Any) -> int:
    action = getattr(args, 'action', None)
    ctx = common.make_ctx(args, delivery=args.dir, stage='roster', options={'invoked_by': 'review'})
    ok, why = common.v1_status(ctx.delivery)
    if not ok:
        sys.stderr.write('legacy delivery: use retro (%s)\n' % why)
        return common.EXIT_MISSING
    if action == 'replay-plan':
        if not getattr(args, 'seed', None):
            sys.stderr.write('review replay-plan needs --seed HEX (chosen by the reviewer)\n')
            return common.EXIT_ERROR
        try:
            _seed_bytes(args.seed)
        except InvalidInput as e:
            sys.stderr.write('%s\n' % e)
            return common.EXIT_ERROR
        plan = replay_plan(ctx, args.seed, int(getattr(args, 'n', MIN_REPLAY_PAIRS) or MIN_REPLAY_PAIRS))
        if getattr(args, 'json', False):
            print(json.dumps(plan, indent=2, sort_keys=True))
        else:
            print('replay plan for %s (seed %s): %d of %d official (world, arm) pairs'
                  % (plan['delivery'], plan['seed'], plan['n'], plan['pairs_available']))
            for p in plan['pairs']:
                print('  %-16s %-12s rows %-6d expected_digest %s%s' % (
                    p['world'], p['arm'], p['n_rows'], p['expected_digest'],
                    ('\n    replay: %s' % ' '.join(p['replay_argv'])) if p.get('replay_argv') else ''))
        return common.EXIT_PASS
    if action == 'check':
        try:
            ctx.tier, ctx.options['tier_reasons'] = common.compute_tier(ctx)
        except (MissingInput, InvalidInput) as e:
            ctx.tier = 2
            ctx.options['tier_reasons'] = ['tier computation failed (%s): treated as 2' % e]
        results = review_results(ctx)
        rc = common.exit_code(results)
        if getattr(args, 'json', False):
            print(json.dumps({'delivery': ctx.delivery_rel, 'tier': ctx.tier, 'results': [r.to_dict() for r in results],
                              'exit_code': rc, 'verdict': common.VERDICT_BY_EXIT[rc]}, indent=2))
        else:
            print(common.format_results(results))
            print('review check: %s (exit %d)' % (common.VERDICT_BY_EXIT[rc], rc))
        return rc
    sys.stderr.write('review: unknown action %r\n' % (action,))
    return common.EXIT_ERROR
