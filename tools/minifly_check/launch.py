"""MRK-1 launch wrapper and typed cycles (WP1).

    MFC launch --stage {testbed|technical|dev|rehearsal|official} <dir> -- <cmd ...>
    MFC cycle close <dir> --kind {testbed|technical|dev|rehearsal} --decision {continue|revise|stop|escalate} --reason TEXT

launch is the only legal way to produce counted results at T2 (SPEC 3.1, 3.4). It
1. runs the stage's gate bundle (testbed -> lock, technical -> technical, dev/rehearsal -> roster,
   official -> official) and refuses on any nonzero exit;
2. refuses when ENDPOINT_LOCK.json, SOURCE_LOCK.json or any closure file is uncommitted, modified
   or differs from its locked sha256;
3. refuses an attempt of another stage while a cycle is open (attempts no `cycle close` covers), technical, dev
   and rehearsal launches once closed plus open test cycles would exceed the cycle cap, a command line that
   names a lock world of another stage, and any official launch after an official attempt produced
   assessment-bearing rows;
4. runs the command (cwd = the delivery dir, stdio inherited; the env carries MINIFLY_ATTEMPT_ID,
   MINIFLY_STAGE, MINIFLY_DELIVERY, MINIFLY_ENDPOINT_LOCK_SHA256, MINIFLY_SOURCE_LOCK_SHA256,
   MINIFLY_NATIVE_CLOSURE_SHA and MINIFLY_PARAMS_SHA256 for the runner to copy into its rows),
   measures RUSAGE_CHILDREN deltas and wall time, and appends an ATTEMPTS.jsonl row listing the
   new or changed files under results/ and manifests/TAUGHT/ with their sha256 -- also when the command
   fails -- and exits 1 when the attempt wrote outside its stage (another stage's results directory, receipts
   or TAUGHT of another stage's worlds, learner-arm work from a testbed attempt), which CYC-01 then FAILs.

Checks: RUN-01, RUN-02, RUN-03, RUN-04, CYC-01, CYC-02, CYC-03. Never commits, never runs state-changing git.
"""
from __future__ import annotations

import gzip
import json
import os
import re
import secrets
import socket
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import common
from common import (FAIL, MISSING, NOT_APPLICABLE, PASS, WARN, InvalidInput, MissingInput, Result, check)

GATE_FOR_STAGE = {'testbed': 'lock', 'technical': 'technical', 'dev': 'roster', 'rehearsal': 'roster',
                  'official': 'official'}
LAUNCH_STAGES = tuple(GATE_FOR_STAGE)
CYCLE_KINDS = ('testbed', 'technical', 'dev', 'rehearsal')
TEST_CYCLE_KINDS = ('technical', 'dev', 'rehearsal')
DECISIONS = ('continue', 'revise', 'stop', 'escalate')
COVERED_RESULT_DIRS = ('results/admission/', 'results/dev/', 'results/rehearsal/', 'results/official/',
                       'results/receipts/', 'results/official_replay/')
# Written by `MFC score` (checker-owned), never by launch; verified by DEC-01 / claim regeneration.
CHECKER_WRITTEN = ('SCORES.json', 'VERDICT.json', 'TESTBED.json')
WORLD_FLAG_RE = re.compile(r'^--?(?:only[-_])?worlds?(?:[-_]ids?)?$')
OUTPUTS_GLOB = 'results/**/OUTPUTS*.jsonl'
# Stage scope (SPEC 3.1 stage map, 3.2 typed cycles): each stage writes only its own results directory and the
# receipts/TAUGHT manifests of its own worlds. The testbed (C0, not counted against the cap) runs evaluator-side
# CEILING_*/FLOOR_* grids on technical worlds only, so it can never carry dev, rehearsal or official outcomes.
ROLE_DIRS = ('admission', 'dev', 'rehearsal', 'official')
STAGE_RESULT_DIR = {'testbed': 'admission', 'technical': 'admission', 'dev': 'dev', 'rehearsal': 'rehearsal',
                    'official': 'official'}
STAGE_WORLD_ROLE = {'testbed': 'technical', 'technical': 'technical', 'dev': 'dev', 'rehearsal': 'rehearsal',
                    'official': 'official'}


def _err(msg: str) -> None:
    sys.stderr.write(msg.rstrip() + '\n')


def _is_checker_written(rel: str) -> bool:
    parts = rel.split('/')
    return len(parts) == 3 and parts[0] == 'results' and parts[2] in CHECKER_WRITTEN


def _is_empty_keep(ctx: common.Ctx, rel: str) -> bool:
    return rel.endswith('/.gitkeep') and ctx.path(rel).stat().st_size == 0


def _strip_sha(v: Any) -> Optional[str]:
    if not isinstance(v, str) or not v:
        return None
    return v.split(':', 1)[1] if v.startswith('sha256:') else v


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = common.canonical_json(row) + b'\n'
    with open(path, 'ab') as fh:
        fh.write(line)
        fh.flush()
        os.fsync(fh.fileno())


# --------------------------------------------------------------------------------------
# closure, snapshots, cycles
# --------------------------------------------------------------------------------------

def _resolve_entry(ctx: common.Ctx, p: str, repo_first: bool) -> Optional[str]:
    rel = common.norm_rel(p)
    order = ('repo', 'delivery') if repo_first else ('delivery', 'repo')
    for where in order:
        if where == 'delivery' and ctx.path(rel).is_file():
            return ctx.repo_rel_of(rel)
        if where == 'repo' and ctx.repo_path(rel).is_file():
            return rel
    return None


def closure_files(ctx: common.Ctx, stage: str) -> Tuple[Dict[str, Optional[str]], List[str]]:
    """Files a launch must find committed and unmodified: {repo-relative path: expected sha256 or None}
    plus a list of problems (missing files, sha mismatches)."""
    lock = ctx.lock
    entries: Dict[str, Optional[str]] = {ctx.repo_rel_of('ENDPOINT_LOCK.json'): None}
    problems: List[str] = []
    if ctx.path('SOURCE_LOCK.json').is_file():
        entries[ctx.repo_rel_of('SOURCE_LOCK.json')] = None
        sl = ctx.source_lock
        clo = sl.get('closure_sha256')
        if clo is None:
            clo = {}
        if not isinstance(clo, dict):
            problems.append('SOURCE_LOCK.closure_sha256 is not an object')
            clo = {}
        for k, v in sorted(clo.items()):
            rr = _resolve_entry(ctx, k, repo_first=True)
            if rr is None:
                problems.append('closure file missing: %s' % k)
            else:
                entries[rr] = _strip_sha(v)
    elif stage != 'testbed':
        raise MissingInput('SOURCE_LOCK.json is required before a %s launch (run MFC lock build)' % stage)
    files = lock.get('files') if isinstance(lock.get('files'), dict) else {}
    required_keys = ('admit', 'evaluator', 'generators') if stage == 'testbed' else (
        'admit', 'evaluator', 'generators', 'runner', 'native', 'shared', 'assets', 'params')
    for key in ('admit', 'evaluator', 'generators', 'runner', 'native', 'shared', 'assets', 'params'):
        v = files.get(key)
        for e in ([v] if isinstance(v, str) else (v or [])):
            if not isinstance(e, str) or not e:
                problems.append('files.%s has a non-path entry %r' % (key, e))
                continue
            rr = _resolve_entry(ctx, e, repo_first=False)
            if rr is None:
                if key in required_keys:
                    problems.append('files.%s entry missing: %s' % (key, e))
                continue
            entries.setdefault(rr, None)
    efs = lock.get('endpoint_files_sha256')
    if isinstance(efs, dict):
        for k, v in sorted(efs.items()):
            rr = _resolve_entry(ctx, k, repo_first=False)
            if rr is None:
                problems.append('endpoint file missing: %s' % k)
            else:
                exp = _strip_sha(v)
                if entries.get(rr) and exp and entries[rr] != exp:
                    problems.append('%s: SOURCE_LOCK and endpoint_files_sha256 disagree' % rr)
                entries[rr] = entries.get(rr) or exp
    elif efs is not None:
        problems.append('endpoint_files_sha256 is not an object')
    for extra in ('AUTHORIZATION.json', 'AMENDMENTS.jsonl'):
        if ctx.path(extra).is_file():
            entries.setdefault(ctx.repo_rel_of(extra), None)
    for rr, exp in sorted(entries.items()):
        if exp:
            got = common.sha256_file(ctx.repo_path(rr))
            if got != exp:
                problems.append('%s sha256 %s differs from locked %s' % (rr, got[:12], exp[:12]))
    return entries, problems


def closure_state(ctx: common.Ctx, stage: str) -> Tuple[bool, List[str]]:
    entries, problems = closure_files(ctx, stage)
    dirty = ctx.git.dirty_paths(sorted(entries))
    msgs = list(problems) + ['uncommitted or modified: %s' % d for d in dirty]
    return not msgs, msgs


# Run-written trees whose new or changed files an ATTEMPTS row records: results/** and the per-arm TAUGHT
# manifests the runner writes (SPEC 6.2 item 7; seal-derived worlds can only get them during the run).
SNAPSHOT_DIRS = ('results', 'manifests/TAUGHT')
# The runtime closure the runner of every counted run writes at its end (RUN-03; SPEC 20.7):
# runlog/RUNTIME_CLOSURE_<attempt_id>.json = {schema, attempt_id, files: [{path (repo-relative), sha256}]}.
SCHEMA_RUNTIME_CLOSURE = 'minifly.runtime_closure.v1'
RUNTIME_CLOSURE_STAGES = ('technical', 'dev', 'rehearsal', 'official')
RUNTIME_CLOSURE_RE = re.compile(r'^runlog/RUNTIME_CLOSURE_(.+)\.json$')


def runtime_closure_rel(attempt_id: Any) -> str:
    return 'runlog/RUNTIME_CLOSURE_%s.json' % attempt_id


def snapshot_results(delivery: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for top in SNAPSHOT_DIRS:
        root = Path(delivery) / top
        if not root.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in common.SKIP_DIR_NAMES)
            for fn in filenames:
                p = Path(dirpath, fn)
                out[p.relative_to(delivery).as_posix()] = common.sha256_file(p)
    rl = Path(delivery) / 'runlog'
    if rl.is_dir():
        for p in sorted(rl.iterdir()):
            rel = 'runlog/' + p.name
            if p.is_file() and RUNTIME_CLOSURE_RE.match(rel):
                out[rel] = common.sha256_file(p)
    return out


def closed_test_cycles(rows: Sequence[dict]) -> List[dict]:
    return [r for r in rows if r.get('kind') in TEST_CYCLE_KINDS]


RUNLOG_APPEND_ONLY = ('runlog/CYCLES.jsonl', 'runlog/ATTEMPTS.jsonl')


def cycle_accounting(ctx: common.Ctx) -> dict:
    """Cycles as ATTEMPTS and CYCLES together show them (SPEC 3.2). The cap cannot rest only on the rows the
    executor chose to write: every cycle ends with `cycle close`, so attempts that no CYCLES row covers form
    open cycles (one per stage), and a cycle row may cover only attempts of its own kind.

    Returns {rows, attempts, covered, uncovered (attempt rows of a cycle stage covered by no row), open_stages
    (stages of uncovered attempts, first appearance order), open_test (open_stages within technical/dev/
    rehearsal), closed (test-cycle rows), problems (inconsistencies between the two logs)}."""
    rows = common.read_cycles(ctx)
    attempts = common.read_attempts(ctx)
    by_id: Dict[Any, dict] = {}
    order: Dict[Any, int] = {}
    for i, a in enumerate(attempts):
        aid = a.get('attempt_id')
        if aid in by_id:
            continue
        by_id[aid] = a
        order[aid] = i
    covered: Dict[Any, int] = {}
    problems: List[str] = []
    for i, r in enumerate(rows, 1):
        k = r.get('k', i)
        comp = r.get('computed') if isinstance(r.get('computed'), dict) else None
        ids = comp.get('attempts') if comp is not None else None
        if not isinstance(ids, list):
            problems.append('cycle %s has no computed.attempts list (rows are written by `MFC cycle close`)' % k)
            continue
        for aid in ids:
            a = by_id.get(aid)
            if a is None:
                problems.append('cycle %s lists attempt %r, which runlog/ATTEMPTS.jsonl does not hold' % (k, aid))
                continue
            if aid in covered:
                problems.append('attempt %s is covered by cycles %s and %s' % (aid, covered[aid], k))
                continue
            covered[aid] = k
            st = a.get('stage')
            if st in CYCLE_KINDS and st != r.get('kind'):
                problems.append('cycle %s (kind %s) covers %s attempt %s: every cycle ends with its own `cycle '
                                'close` (SPEC 3.2)' % (k, r.get('kind'), st, aid))
    uncovered = [a for a in attempts if a.get('stage') in CYCLE_KINDS and a.get('attempt_id') not in covered]
    last_cov = max([order[aid] for aid in covered if aid in order] or [-1])
    early = [a.get('attempt_id') for a in uncovered if order.get(a.get('attempt_id'), -1) < last_cov]
    if early:
        problems.append('attempt(s) %s precede a later cycle close but no cycle covers them' % ', '.join(map(str, early[:5])))
    open_stages: List[str] = []
    for a in uncovered:
        if a.get('stage') not in open_stages:
            open_stages.append(a.get('stage'))
    return {'rows': rows, 'attempts': attempts, 'covered': covered, 'uncovered': uncovered,
            'open_stages': open_stages, 'open_test': [s for s in open_stages if s in TEST_CYCLE_KINDS],
            'closed': closed_test_cycles(rows), 'problems': problems}


def runlog_append_only_problems(ctx: common.Ctx) -> List[str]:
    """runlog/CYCLES.jsonl and runlog/ATTEMPTS.jsonl are written only by `cycle close` and `launch`, which only
    append: every committed version's rows must survive in every later version and in the worktree. A shallow
    clone raises MissingInput (SPEC 7.6)."""
    problems: List[str] = []
    git = ctx.git
    if git.is_shallow():
        raise MissingInput('shallow clone: runlog history cannot be verified append-only (git fetch --unshallow)')
    for rel in RUNLOG_APPEND_ONLY:
        repo_rel = ctx.repo_rel_of(rel)
        prev: Optional[set] = None
        prev_sha = None
        for c in git.log([repo_rel]):
            data = git.file_at(c.sha, repo_rel)
            cur = set(_raw_lines(data))
            if prev is not None:
                gone = prev - cur
                if gone:
                    problems.append('%s: commit %s %s %d row(s) committed at %s' % (
                        rel, c.sha[:12], 'deletes the file and its' if data is None else 'removes or edits',
                        len(gone), str(prev_sha)[:12]))
            prev, prev_sha = cur, c.sha
        head_data = git.file_at(git.head(), repo_rel)
        wt = ctx.path(rel)
        cur = set(_raw_lines(wt.read_bytes())) if wt.is_file() else set()
        gone = set(_raw_lines(head_data)) - cur
        if gone:
            problems.append('%s: the worktree %s %d committed row(s)' % (
                rel, 'deletes the file and its' if not wt.is_file() else 'removes or edits', len(gone)))
    return problems


def _raw_lines(data: Optional[bytes]) -> List[str]:
    if not data:
        return []
    return [ln.strip() for ln in data.decode('utf-8', 'replace').splitlines() if ln.strip()]


def _invalid_cycle_rows(rows: Sequence[dict]) -> List[str]:
    bad = []
    for i, r in enumerate(rows, 1):
        if r.get('kind') not in CYCLE_KINDS:
            bad.append('row %d kind %r' % (i, r.get('kind')))
        if r.get('decision') not in DECISIONS:
            bad.append('row %d decision %r' % (i, r.get('decision')))
    return bad


def world_allowed(stage: str, world: Any, worlds: Dict[str, List[str]]) -> bool:
    """A world an attempt of `stage` may run: one the lock lists for that stage's role, or, when the lock lists
    none for it, any world the lock does not give to another role (official ids may be seal-derived)."""
    role = STAGE_WORLD_ROLE.get(stage)
    if role is None or not isinstance(world, str):
        return False
    if worlds.get(role):
        return world in worlds[role]
    return not any(world in worlds.get(r, []) for r in worlds if r != role)


def _stem(name: str) -> str:
    for ext in ('.jsonl.gz', '.jsonl', '.json.gz', '.json'):
        if name.endswith(ext):
            return name[:-len(ext)]
    return name


def path_scope_problem(ctx: common.Ctx, stage: str, path: Any, sha: Any, worlds: Dict[str, List[str]],
                       cycle0: bool) -> Optional[str]:
    """Why an attempt of `stage` may not have produced `path` (None when it may). cycle0: a testbed attempt that
    precedes every learner attempt (C0 runs before learner code exists), which may generate the TAUGHT streams of
    any lock world."""
    if not isinstance(path, str) or not path:
        return 'records a produced path that is not a string: %r' % (path,)
    parts = path.split('/')
    name = parts[-1]
    if parts[0] == 'results':
        if len(parts) >= 3 and parts[1] == common.REPLAY_DIR.split('/', 1)[1]:
            if stage != 'official':
                return 'wrote %s (only an official resume attempt writes %s/)' % (path, common.REPLAY_DIR)
            return None
        if len(parts) >= 3 and parts[1] in ROLE_DIRS:
            want = STAGE_RESULT_DIR[stage]
            if parts[1] != want:
                return 'wrote %s (a %s attempt writes results/%s/ only)' % (path, stage, want)
            return None
        if len(parts) >= 4 and parts[1] == 'receipts':
            world, arm = parts[2], _stem(name)
            if stage == 'testbed' and not common.is_evaluator_arm(arm):
                return 'wrote learner-arm receipt %s (the testbed runs CEILING_*/FLOOR_* arms only)' % path
            if not world_allowed(stage, world, worlds):
                return 'wrote receipt %s of world %s, which is not a %s world' % (path, world, STAGE_WORLD_ROLE[stage])
            return None
        if common.RESULT_BUNDLE_RE.match(name):
            return bundle_scope_problem(ctx, stage, path, sha, worlds, cycle0)
        return None
    if path.startswith('manifests/TAUGHT/') and len(parts) >= 4:
        world, arm = parts[2], _stem(name)
        if stage == 'testbed':
            if cycle0 or (common.is_evaluator_arm(arm) and world_allowed(stage, world, worlds)):
                return None
            return ('wrote TAUGHT manifest %s after learner attempts exist (a later testbed attempt writes '
                    'CEILING_*/FLOOR_* manifests of technical worlds only)' % path)
        if not world_allowed(stage, world, worlds):
            return 'wrote TAUGHT manifest %s of world %s, which is not a %s world' % (
                path, world, STAGE_WORLD_ROLE[stage])
    return None


def bundle_scope_problem(ctx: common.Ctx, stage: str, path: str, sha: Any, worlds: Dict[str, List[str]],
                         cycle0: bool) -> Optional[str]:
    """A RESULT_BUNDLE*.zip outside results/<role>/ is judged by its members' logical paths (common.row_sources
    rules), when the file still holds the recorded bytes; otherwise RUN-01 judges the current file."""
    p = ctx.path(path)
    if not p.is_file() or (isinstance(sha, str) and common.sha256_file(p) != sha):
        return None
    zdir = path.rsplit('/', 1)[0] if '/' in path else ''
    try:
        with zipfile.ZipFile(str(p)) as zf:
            names = [i.filename for i in zf.infolist() if not i.is_dir()]
    except zipfile.BadZipFile as e:
        return 'produced %s, which is not a valid zip (%s)' % (path, e)
    for m in names:
        m = common.norm_rel(m)
        logical = m if (m.startswith('results/') or not zdir) else zdir + '/' + m
        if common.RESULT_BUNDLE_RE.match(logical.rsplit('/', 1)[-1]):
            continue
        why = path_scope_problem(ctx, stage, logical, None, worlds, cycle0)
        if why:
            return '%s (member of %s)' % (why, path)
    return None


def stage_scope_problems(ctx: common.Ctx, only: Optional[Any] = None, max_rows: int = 5) -> List[str]:
    """Attempts that wrote outside their stage (SPEC 3.1, 3.2). Two sources: produced_paths (results/<role>/,
    receipts and TAUGHT of the stage's worlds) and the rows themselves: an OUTPUTS row under results/<role>/ whose
    attempt_id names an ATTEMPTS row must name one of the stage that writes that directory, on a world of that
    stage, and a testbed attempt's rows are CEILING_*/FLOOR_* rows only. Rows that name no attempt are RUN-01's
    (T2 enforced, T1 WARN) and, for official rows, RUN-02's. `only` (an attempt id or a collection of ids)
    restricts the analysis to those attempts."""
    if isinstance(only, str):
        only = {only}
    elif only is not None:
        only = set(only)
        if not only:
            return []
    attempts = common.read_attempts(ctx)
    worlds = lock_world_ids(ctx.lock)
    problems: List[str] = []
    by_id: Dict[Any, dict] = {}
    seen_learner = False
    for i, a in enumerate(attempts, 1):
        st = a.get('stage')
        aid = a.get('attempt_id')
        by_id.setdefault(aid, a)
        if st not in STAGE_RESULT_DIR:
            problems.append('ATTEMPTS row %d has stage %r (launch writes %s)' % (i, st, '|'.join(LAUNCH_STAGES)))
            continue
        if only is None or aid in only:
            pp = a.get('produced_paths') or []
            if not isinstance(pp, list):
                problems.append('%s attempt %s: produced_paths is not a list' % (st, aid))
                pp = []
            for e in pp:
                if not isinstance(e, dict):
                    problems.append('%s attempt %s: a produced_paths entry is not an object' % (st, aid))
                    continue
                why = path_scope_problem(ctx, st, e.get('path'), e.get('sha256'), worlds,
                                         cycle0=(st == 'testbed' and not seen_learner))
                if why:
                    problems.append('%s attempt %s %s' % (st, aid, why))
        if st != 'testbed':
            seen_learner = True
    for logical, _kind, _ref, _gz in ctx.row_sources(OUTPUTS_GLOB):
        parts = logical.split('/')
        if len(parts) < 3 or parts[0] != 'results' or parts[1] not in ROLE_DIRS:
            continue
        role_dir = parts[1]
        counts: Dict[str, int] = {}
        first: Dict[str, str] = {}
        for label, lineno, row in ctx.iter_rows_with_source(logical):
            a = by_id.get(row.get('attempt_id'))
            if a is None or (only is not None and a.get('attempt_id') not in only):
                continue
            st = a.get('stage')
            if STAGE_RESULT_DIR.get(st) != role_dir:
                why = 'rows of %s attempt %s under results/%s/' % (st, a.get('attempt_id'), role_dir)
            elif st == 'testbed' and not common.is_evaluator_arm(row.get('arm')):
                why = 'learner-arm rows (arm %s) from testbed attempt %s' % (row.get('arm'), a.get('attempt_id'))
            elif not world_allowed(st, row.get('world'), worlds):
                why = 'rows of %s attempt %s on world %s, which is not a %s world' % (
                    st, a.get('attempt_id'), row.get('world'), STAGE_WORLD_ROLE[st])
            else:
                continue
            counts[why] = counts.get(why, 0) + 1
            first.setdefault(why, '%s:%d' % (label, lineno))
        for why in sorted(counts)[:max_rows]:
            problems.append('%s: %d %s (first %s)' % (logical, counts[why], why, first[why]))
    return problems


def unattributed_rows(ctx: common.Ctx, limit: int = 20) -> Tuple[int, List[str]]:
    """OUTPUTS rows under results/<role>/ whose attempt_id names no ATTEMPTS row: at T2 every counted row comes
    from `launch`, which hands the child MINIFLY_ATTEMPT_ID (SPEC 3.1, 3.4, 6.2 runner contract)."""
    ids = set(a.get('attempt_id') for a in common.read_attempts(ctx))
    n = 0
    ex: List[str] = []
    for logical, _kind, _ref, _gz in ctx.row_sources(OUTPUTS_GLOB):
        parts = logical.split('/')
        if len(parts) < 3 or parts[0] != 'results' or parts[1] not in ROLE_DIRS:
            continue
        for label, lineno, row in ctx.iter_rows_with_source(logical):
            if row.get('attempt_id') not in ids or row.get('attempt_id') is None:
                n += 1
                if len(ex) < limit:
                    ex.append('%s:%d attempt_id=%r' % (label, lineno, row.get('attempt_id')))
    return n, ex


def argv_world_problems(stage: str, argv: Sequence[str], lock: dict) -> List[str]:
    """Lock world ids named in a pending launch argv that belong to another stage's role."""
    worlds = lock_world_ids(lock)
    every = sorted(set(x for v in worlds.values() for x in v))
    named = _mentions(' '.join(str(x) for x in argv), every)
    return sorted(w for w in named if not world_allowed(stage, w, worlds))


def learner_outcomes_exist(ctx: common.Ctx) -> bool:
    for row in ctx.iter_rows(OUTPUTS_GLOB):
        if not common.is_evaluator_arm(row.get('arm')):
            return True
    for rel in ctx.delivery_files():
        if rel.startswith('results/receipts/') and rel.endswith('.json'):
            r = ctx.read_json(rel)
            if isinstance(r, dict) and not common.is_evaluator_arm(r.get('arm')):
                return True
    return False


def _gate_summary(rep: Optional[dict]) -> Optional[dict]:
    if not rep:
        return None
    return {'path': rep['_path'], 'stage': rep.get('stage'), 'dry': bool(rep.get('dry')),
            'verdict': rep.get('verdict'), 'utc': rep.get('utc'), 'source_lock_sha256': rep.get('source_lock_sha256'),
            'sha256': None}


def exit_criterion(ctx: common.Ctx, row: dict, gates: Optional[List[dict]] = None) -> Tuple[bool, str]:
    """SPEC 3.2 exit criterion of a closed cycle row (used for decision continue)."""
    kind = row.get('kind')
    row_utc = common.normalize_utc(row.get('utc')) or ''
    if kind == 'testbed':
        hits = [f for f in ctx.delivery_files(refresh=True)
                if f.startswith('results/') and f.rsplit('/', 1)[-1] == 'TESTBED.json']
        return bool(hits), ('TESTBED result %s' % hits[0]) if hits else 'no TESTBED result under results/'
    if kind == 'dev':
        ok = ctx.path('results/dev/SCORES.json').is_file()
        return ok, 'results/dev/SCORES.json %s' % ('exists' if ok else 'absent')
    if gates is None:
        gates = common.list_gate_reports(ctx)
    if kind == 'technical':
        want = (row.get('computed') or {}).get('source_lock_sha256') or ctx.source_lock_sha256
        ok = [g for g in gates if g.get('stage') == 'roster' and not g.get('dry') and g.get('verdict') == 'PASS'
              and want and g.get('source_lock_sha256') == want and (common.normalize_utc(g.get('utc')) or '~') <= row_utc]
        return bool(ok), (('PASS roster gate %s' % ok[-1]['_path']) if ok else
                          'no PASS roster GATE report at SOURCE_LOCK %s before %s' % ((want or 'none')[:12], row_utc))
    if kind == 'rehearsal':
        ok = [g for g in gates if g.get('stage') == 'official' and g.get('dry') and g.get('verdict') == 'PASS'
              and (common.normalize_utc(g.get('utc')) or '~') <= row_utc]
        return bool(ok), (('PASS official --dry report %s' % ok[-1]['_path']) if ok else
                          'no PASS official --dry report before %s' % row_utc)
    return False, 'unknown cycle kind %r' % kind


# --------------------------------------------------------------------------------------
# official attempts (RUN-02 and the launch guard)
# --------------------------------------------------------------------------------------

def official_attempt_analysis(ctx: common.Ctx) -> dict:
    attempts = common.read_attempts(ctx)
    official = [a for a in attempts if a.get('stage') == 'official']
    official_ids = [a.get('attempt_id') for a in official]
    counts: Dict[Any, int] = {}
    unattributed: List[str] = []
    for label, lineno, row in ctx.iter_rows_with_source(OUTPUTS_GLOB):
        is_official = (label.startswith('results/official/') or '!results/official/' in label
                       or row.get('world_role') == 'official')
        if not is_official:
            continue
        aid = row.get('attempt_id')
        if aid in official_ids:
            counts[aid] = counts.get(aid, 0) + 1
        else:
            unattributed.append('%s:%d attempt_id=%r' % (label, lineno, aid))
    bearing = []
    for a in official:
        pp = a.get('produced_paths') or []
        wrote_outputs = any(isinstance(e, dict) and isinstance(e.get('path'), str) and (
            (e['path'].startswith('results/official/') and e['path'].rsplit('/', 1)[-1].startswith('OUTPUTS'))
            or common.RESULT_BUNDLE_RE.match(e['path'].rsplit('/', 1)[-1])) for e in pp)
        if counts.get(a.get('attempt_id'), 0) > 0 or wrote_outputs:
            bearing.append(a)
    later = []
    if bearing:
        first = official.index(bearing[0])
        # a resume attempt (launch --resume, resume_of set) is judged by resume_analysis (SPEC 20.8); any other
        # official attempt after an assessment-bearing one is a rerun
        later = [a.get('attempt_id') for a in official[first + 1:] if not a.get('resume_of')]
    return {'official': official_ids, 'bearing': [a.get('attempt_id') for a in bearing], 'rows': counts,
            'unattributed': unattributed, 'after_first_bearing': later,
            'bearing_non_resume': [a.get('attempt_id') for a in bearing if not a.get('resume_of')]}


# --------------------------------------------------------------------------------------
# resumable official runs (SPEC 20.8): RUN-02 as amended, RUN-04
# --------------------------------------------------------------------------------------

OFFICIAL_OUTPUTS = 'results/official/OUTPUTS.jsonl'


def _unit_key(world: Any, arm: Any) -> str:
    return '%s|%s' % (world, arm)


def _row_digest(rows: Sequence[dict]) -> str:
    import review  # lazy (SPEC 7.2): the canonical row digest of REV-08 (SCHEMAS 13), attempt_id excluded
    return review.row_digest(rows)


def unit_rows(ctx: common.Ctx, pattern: str) -> Dict[Any, Dict[str, List[dict]]]:
    """{attempt_id: {'world|arm': rows}} of the rows matching `pattern` (gz and bundles allowed)."""
    out: Dict[Any, Dict[str, List[dict]]] = {}
    for r in ctx.iter_rows(pattern):
        out.setdefault(r.get('attempt_id'), {}).setdefault(_unit_key(r.get('world'), r.get('arm')), []).append(r)
    return out


def units_record(by_unit: Dict[str, List[dict]]) -> List[dict]:
    """The ATTEMPTS `units` field: [{world, arm, n_rows, digest}] sorted by (world, arm)."""
    out = []
    for key in sorted(by_unit):
        w, a = key.split('|', 1)
        out.append({'world': w, 'arm': a, 'n_rows': len(by_unit[key]), 'digest': _row_digest(by_unit[key])})
    return out


def attempt_interrupted(a: dict) -> bool:
    ec = a.get('exit_code')
    return bool(a.get('interrupted')) or bool(a.get('recovered')) or not (
        isinstance(ec, int) and not isinstance(ec, bool) and ec == 0)


def expected_official_units(ctx: common.Ctx, seen_worlds: Sequence[str]) -> Tuple[set, Optional[int]]:
    """(expected 'world|arm' units, official.n) from the lock: listed official ids (else the worlds seen in official
    rows) times every declared learner arm."""
    lock = ctx.lock
    arms = [a.get('id') for a in (lock.get('arms') or []) if isinstance(a, dict) and isinstance(a.get('id'), str)
            and not common.is_evaluator_arm(a.get('id'))]
    ids = lock_world_ids(lock)['official'] or sorted(set(seen_worlds))
    off = (lock.get('worlds') or {}).get('official') if isinstance(lock.get('worlds'), dict) else None
    n = off.get('n') if isinstance(off, dict) and isinstance(off.get('n'), int) and not isinstance(off.get('n'), bool) else None
    return set(_unit_key(w, a) for w in ids for a in arms), n


def _recorded_units(a: dict) -> Optional[Dict[str, dict]]:
    u = a.get('units')
    if not isinstance(u, list):
        return None
    out = {}
    for e in u:
        if isinstance(e, dict) and isinstance(e.get('world'), str) and isinstance(e.get('arm'), str):
            out[_unit_key(e['world'], e['arm'])] = e
    return out


def _is_row_file(path: str) -> bool:
    name = path.rsplit('/', 1)[-1]
    return name.startswith('OUTPUTS') or bool(common.RESULT_BUNDLE_RE.match(name))


def resume_analysis(ctx: common.Ctx, pending_resume: bool = False) -> dict:
    """Legality of resumed official attempts (SPEC 20.8, RUN-02 as amended). A resume attempt R (resume_of set by
    launch --resume) is legal only if (a) its ENDPOINT_LOCK and SOURCE_LOCK shas equal those of the attempt it
    resumes and of every earlier official attempt; (b) it resumes the latest earlier official attempt, whose
    ATTEMPTS row exists and which was interrupted (nonzero or killed exit, or an incomplete unit set); (c) it
    produced OUTPUTS rows and receipts only for (world, arm) units no earlier official attempt has rows for, left
    every earlier attempt's recorded unit rows unchanged (same canonical digest) and rewrote no other file an
    earlier official attempt produced. pending_resume: judge whether a new `launch --resume` may start now."""
    attempts = common.read_attempts(ctx)
    official = [a for a in attempts if a.get('stage') == 'official']
    rows_by = unit_rows(ctx, OFFICIAL_OUTPUTS)
    problems: List[str] = []
    resumes: List[str] = []
    completed: Dict[str, dict] = {}       # unit -> {'attempt', 'digest', 'n_rows'} (first attempt that completed it)
    produced_before: Dict[str, Any] = {}  # non-row path -> attempt that produced it
    seen_worlds = sorted(set(k.split('|', 1)[0] for d in rows_by.values() for k in d))
    expected, n_expected = expected_official_units(ctx, seen_worlds)
    first = official[0] if official else None

    def interrupted(k: int) -> Tuple[bool, str]:
        prev = official[k]
        if attempt_interrupted(prev):
            return True, 'exit %r%s' % (prev.get('exit_code'), ', interrupted' if prev.get('interrupted') else '')
        have = set(completed)
        miss = sorted(expected - have)
        worlds_have = set(u.split('|', 1)[0] for u in have)
        if miss or (n_expected is not None and len(worlds_have) < n_expected):
            return True, 'incomplete unit set (%d of %d units%s)' % (
                len(have & expected), len(expected), (', %d of %d worlds' % (len(worlds_have), n_expected))
                if n_expected is not None else '')
        return False, 'exit 0 with every unit complete'

    for k, a in enumerate(official):
        aid = a.get('attempt_id')
        mine = rows_by.get(aid, {})
        ro = a.get('resume_of')
        if ro is not None:
            resumes.append(aid)
            prev = official[k - 1] if k > 0 else None
            if prev is None or prev.get('attempt_id') != ro:
                problems.append('resume attempt %s names resume_of %r, but the latest earlier official attempt is %s'
                                % (aid, ro, prev.get('attempt_id') if prev else 'none'))
            else:
                ok, why = interrupted(k - 1)
                if not ok:
                    problems.append('resume attempt %s resumes %s, which was not interrupted (%s)' % (aid, ro, why))
            for key in ('endpoint_lock_sha256', 'source_lock_sha256'):
                want = [_strip_sha(o.get(key)) for o in official[:k]]
                if any(_strip_sha(a.get(key)) != w for w in want):
                    problems.append('resume attempt %s %s %s differs from the interrupted attempt(s) %s'
                                    % (aid, key, str(a.get(key))[:12], sorted(set(str(w)[:12] for w in want))))
            again = sorted(set(mine) & set(completed))
            if again:
                problems.append('resume attempt %s wrote rows for %d unit(s) an earlier official attempt completed (no '
                                'overwrite, no duplicates): %s' % (aid, len(again), ', '.join(again[:6])))
            for e in a.get('produced_paths') or []:
                path = e.get('path') if isinstance(e, dict) else None
                if isinstance(path, str) and not _is_row_file(path) and path in produced_before:
                    problems.append('resume attempt %s rewrote %s, produced by official attempt %s'
                                    % (aid, path, produced_before[path]))
            for path in a.get('removed_paths') or []:
                if isinstance(path, str) and path in produced_before:
                    problems.append('resume attempt %s removed %s, produced by official attempt %s'
                                    % (aid, path, produced_before[path]))
        rec = _recorded_units(a)
        units_now = {u: rows for u, rows in mine.items()}
        if rec is None and (k + 1 < len(official) and official[k + 1].get('resume_of') or pending_resume and
                            k == len(official) - 1):
            problems.append('official attempt %s records no units (launch writes them for every official attempt); a '
                            'resume cannot show that its rows stay unchanged' % aid)
        for u, rows in units_now.items():
            completed.setdefault(u, {'attempt': aid, 'digest': (rec or {}).get(u, {}).get('digest') or _row_digest(rows),
                                     'n_rows': len(rows)})
        for u, e in (rec or {}).items():
            if u not in units_now:
                problems.append('rows of unit %s recorded by official attempt %s are gone' % (u, aid))
            elif _row_digest(units_now[u]) != e.get('digest'):
                problems.append('rows of unit %s recorded by official attempt %s changed after it ended (canonical '
                                'digest differs)' % (u, aid))
            completed.setdefault(u, {'attempt': aid, 'digest': e.get('digest'), 'n_rows': e.get('n_rows')})
        for e in a.get('produced_paths') or []:
            path = e.get('path') if isinstance(e, dict) else None
            if isinstance(path, str) and not _is_row_file(path):
                produced_before.setdefault(path, aid)
    if resumes:
        # receipts belong to the attempt whose rows cover their unit
        owner = {u: v['attempt'] for u, v in completed.items()}
        for rel in ctx.delivery_files():
            parts = rel.split('/')
            if len(parts) == 4 and parts[:2] == ['results', 'receipts'] and rel.endswith('.json'):
                u = _unit_key(parts[2], parts[3][:-len('.json')])
                if u not in owner:
                    continue
                try:
                    rc = ctx.read_json(rel)
                except InvalidInput as e:
                    problems.append('%s does not parse: %s' % (rel, e))
                    continue
                if not isinstance(rc, dict) or rc.get('attempt_id') != owner[u]:
                    problems.append('%s names attempt %r, but the rows of unit %s come from %s'
                                    % (rel, rc.get('attempt_id') if isinstance(rc, dict) else None, u, owner[u]))
    pending: List[str] = []
    if pending_resume:
        if not official:
            pending.append('no official attempt to resume')
        else:
            last = official[-1]
            ok, why = interrupted(len(official) - 1)
            if not ok:
                pending.append('the latest official attempt %s was not interrupted (%s): nothing to resume'
                               % (last.get('attempt_id'), why))
            if not completed:
                pending.append('no official unit is complete yet: launch without --resume (an attempt with zero '
                               'rows never blocks a new official attempt)')
            for key, cur in (('endpoint_lock_sha256', ctx.endpoint_lock_sha256),
                             ('source_lock_sha256', ctx.source_lock_sha256)):
                if _strip_sha(last.get(key)) != _strip_sha(cur):
                    pending.append('current %s %s differs from the interrupted attempt %s (%s): a resume runs under '
                                   'the identical lock' % (key, str(cur)[:12], last.get('attempt_id'),
                                                           str(last.get(key))[:12]))
    return {'official': [a.get('attempt_id') for a in official], 'resumes': resumes, 'problems': problems,
            'completed': completed, 'expected': sorted(expected), 'pending_problems': pending,
            'first': first.get('attempt_id') if first else None}


def replay_analysis(ctx: common.Ctx) -> dict:
    """RUN-04 (SPEC 20.8): every resume attempt that wrote official rows re-ran at least one (world, arm) unit an
    earlier official attempt completed (every unit launch requested) into results/official_replay/, and each
    replayed unit's canonical row digest (attempt_id excluded) equals the original's."""
    attempts = common.read_attempts(ctx)
    official = [a for a in attempts if a.get('stage') == 'official']
    resume_ids = set(a.get('attempt_id') for a in official if a.get('resume_of'))
    replay = unit_rows(ctx, common.REPLAY_OUTPUTS)
    rows_by = unit_rows(ctx, OFFICIAL_OUTPUTS)
    problems: List[str] = []
    replayed: Dict[str, List[str]] = {}
    for aid in sorted(replay, key=str):
        if aid not in resume_ids:
            problems.append('%s rows of attempt %r: only a resume attempt (launch --resume) writes them'
                            % (common.REPLAY_OUTPUTS, aid))
    completed: Dict[str, str] = {}   # unit -> original digest
    owner: Dict[str, Any] = {}       # unit -> the official attempt whose rows completed it
    vouched: Set[str] = set()        # units replayed with the original digest
    for a in official:
        aid = a.get('attempt_id')
        if aid in resume_ids:
            reps = replay.get(aid, {})
            # a resume that wrote official rows must show they come from the same deterministic code; one that
            # crashed before writing any row contributed nothing to vouch for (a later resume replays instead)
            if rows_by.get(aid):
                if not reps:
                    problems.append('resume attempt %s wrote official rows but re-ran no completed unit into %s/ (a '
                                    'resume must show determinism)' % (aid, common.REPLAY_DIR))
                req = a.get('replay_requested')
                for r in (req if isinstance(req, list) else []):
                    key = _unit_key(*r) if isinstance(r, (list, tuple)) and len(r) == 2 else str(r)
                    if key not in reps:
                        problems.append('resume attempt %s did not replay the unit %s that launch chose' % (aid, key))
            for u, rows in sorted(reps.items()):
                if u not in completed:
                    problems.append('resume attempt %s replayed unit %s, which no earlier official attempt completed'
                                    % (aid, u))
                    continue
                got = _row_digest(rows)
                if got != completed[u]:
                    problems.append('resume attempt %s replay of unit %s has canonical digest %s, the original %s'
                                    % (aid, u, got[:12], completed[u][:12]))
                else:
                    vouched.add(u)
            replayed[aid] = sorted(reps)
        rec = _recorded_units(a) or {}
        for u, rows in rows_by.get(aid, {}).items():
            completed.setdefault(u, (rec.get(u) or {}).get('digest') or _row_digest(rows))
            owner.setdefault(u, aid)
    # SPEC 20.17 (round 1): a recovered in-flight official attempt (its launch process died: exit code and rusage are
    # lost) vouches for its rows only through a determinism replay: at least one unit it completed must be re-run
    # by a later resume attempt with the identical canonical digest
    recovered = [a.get('attempt_id') for a in official if attempt_recovered(a)]
    unvouched: Dict[str, List[str]] = {}
    for rid in recovered:
        mine = sorted(u for u, o in owner.items() if o == rid)
        if mine and not (set(mine) & vouched):
            unvouched[str(rid)] = mine
            problems.append('recovered official attempt %s (rusage lost) completed %d unit(s) but none was replayed with '
                            'an identical digest (run launch --stage official --resume, which replays one of them)'
                            % (rid, len(mine)))
    return {'resumes': sorted(resume_ids), 'replayed': replayed, 'problems': problems,
            'replay_attempts': sorted(str(x) for x in replay), 'recovered': [str(x) for x in recovered],
            'recovered_unvouched': unvouched}


def attempt_recovered(a: dict) -> bool:
    """An ATTEMPTS row that launch reconstructed from an in-flight marker (rusage_lost / recovered)."""
    return bool(a.get('rusage_lost')) or bool(a.get('recovered'))


# --------------------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------------------

def _wait(proc: subprocess.Popen) -> Tuple[int, int, bool]:
    interrupted = False
    while True:
        try:
            _pid, status, ru = os.wait4(proc.pid, 0)
            break
        except InterruptedError:
            continue
        except KeyboardInterrupt:
            interrupted = True
            try:
                proc.terminate()
            except OSError:
                pass
    rc = os.waitstatus_to_exitcode(status)
    proc.returncode = rc
    rss = int(ru.ru_maxrss)
    if sys.platform == 'darwin':
        rss //= 1024
    return rc, rss, interrupted


# In-flight marker (SPEC 20.8): launch writes runlog/INFLIGHT_<attempt_id>.json before it starts the command and
# deletes it once the ATTEMPTS row is appended. When the launch process itself dies (container restart), the next
# launch or cycle close turns a marker whose process is gone into an ATTEMPTS row marked interrupted and recovered.
SCHEMA_INFLIGHT = 'minifly.inflight.v1'
INFLIGHT_RE = re.compile(r'^INFLIGHT_(.+)\.json$')


def inflight_rel(attempt_id: Any) -> str:
    return 'runlog/INFLIGHT_%s.json' % attempt_id


def _boot_id() -> str:
    try:
        return Path('/proc/sys/kernel/random/boot_id').read_text(encoding='ascii').strip()
    except OSError:
        return ''


def _pid_alive(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (OSError, ValueError, TypeError, OverflowError):
        return False
    return True


def _write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'wb') as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(str(tmp), str(path))


def official_units_now(args: Any, delivery: Any, aid: str) -> Tuple[List[dict], List[dict], Optional[str]]:
    """(units, replay units, error) of attempt `aid` from the official and replay rows on disk now."""
    try:
        c = common.make_ctx(args, delivery=delivery, stage='official', options={'invoked_by': 'launch'})
        units = units_record(unit_rows(c, OFFICIAL_OUTPUTS).get(aid, {}))
        replay = units_record(unit_rows(c, common.REPLAY_OUTPUTS).get(aid, {}))
        return units, replay, None
    except Exception as e:  # noqa: BLE001 - recorded in the row; RUN-02/RUN-04 then fail closed
        return [], [], common.summarize_exception(e)


def recover_inflight(args: Any, delivery: Any) -> Tuple[List[str], List[str]]:
    """(recovered attempt ids, refusals): every in-flight marker whose attempt has no ATTEMPTS row and whose launch
    process is gone becomes an ATTEMPTS row (exit_code -1, interrupted, recovered; rusage lost), with produced_paths
    computed against the snapshot the marker recorded before the command started."""
    ctx = common.make_ctx(args, delivery=delivery, stage='launch', options={'invoked_by': 'launch'})
    rl = ctx.path('runlog')
    if not rl.is_dir():
        return [], []
    known = set(a.get('attempt_id') for a in common.read_attempts(ctx))
    recovered: List[str] = []
    refusals: List[str] = []
    for m in sorted(rl.iterdir()):
        if not (m.is_file() and INFLIGHT_RE.match(m.name)):
            continue
        try:
            doc = common.load_json_file(m)
        except InvalidInput as e:
            refusals.append('in-flight marker runlog/%s does not parse (%s): its attempt cannot be recorded' % (m.name, e))
            continue
        aid = doc.get('attempt_id') if isinstance(doc, dict) else None
        if not isinstance(doc, dict) or doc.get('schema') != SCHEMA_INFLIGHT or not isinstance(aid, str) or \
                m.name != 'INFLIGHT_%s.json' % aid:
            refusals.append('runlog/%s is not a %s marker of its attempt' % (m.name, SCHEMA_INFLIGHT))
            continue
        if aid in known:
            m.unlink()  # the row was written; the marker outlived its launch
            continue
        if doc.get('boot_id') == _boot_id() and doc.get('host') == socket.gethostname() and _pid_alive(doc.get('pid')):
            refusals.append('attempt %s (stage %s) is still running under launch pid %s' % (aid, doc.get('stage'),
                                                                                         doc.get('pid')))
            continue
        before = doc.get('before') if isinstance(doc.get('before'), dict) else {}
        after = snapshot_results(ctx.delivery)
        stage = doc.get('stage')
        try:
            clean_after, msgs_after = closure_state(ctx, stage)
        except Exception as e:  # noqa: BLE001
            clean_after, msgs_after = False, [common.summarize_exception(e)]
        now = common.utc_now()
        row = {'schema': common.SCHEMA_ATTEMPT}
        for k in ('attempt_id', 'stage', 'argv', 'cwd', 'utc_start', 'git_head', 'endpoint_lock_sha256',
                  'source_lock_sha256', 'native_closure_sha', 'params_sha256', 'gate_report', 'resume_of',
                  'replay_requested'):
            if k in doc:
                row[k] = doc[k]
        row.update({
            'utc_end': now, 'closure_clean': True, 'closure_clean_after': clean_after, 'exit_code': -1,
            'wall_s': 0.0, 'cpu_children_user_s': 0.0, 'cpu_children_sys_s': 0.0, 'max_rss_kb': 0,
            'produced_paths': [{'path': p, 'sha256': h} for p, h in sorted(after.items()) if before.get(p) != h],
            'removed_paths': sorted(p for p in before if p not in after), 'interrupted': True, 'rusage_lost': True,
            'recovered': {'utc': now, 'reason': 'the launch process ended before it wrote this row (in-flight marker '
                                               '%s); exit code and CPU are unknown' % inflight_rel(aid)}})
        if not clean_after:
            row['closure_problems_after'] = msgs_after[:20]
        if stage == 'official':
            units, replay, err = official_units_now(args, delivery, aid)
            row['units'] = units
            if row.get('resume_of'):
                row['replay_units'] = [[u['world'], u['arm']] for u in replay]
            if err:
                row['units_error'] = err
        append_jsonl(ctx.path('runlog/ATTEMPTS.jsonl'), row)
        m.unlink()
        recovered.append(aid)
    return recovered, refusals


def cmd_launch(args: Any) -> int:
    stage = args.stage
    if stage not in GATE_FOR_STAGE:
        _err('launch: unknown stage %r' % stage)
        return common.EXIT_ERROR
    cmd = [str(c) for c in (getattr(args, 'cmd', None) or [])]
    if cmd and cmd[0] == '--':
        cmd = cmd[1:]
    if not cmd:
        _err('launch: a command is required after --')
        return common.EXIT_MISSING
    if getattr(args, 'git', None):
        _err('launch runs real commands and is refused in --git timeline (selftest) mode')
        return common.EXIT_ERROR
    try:
        import resource
    except ImportError as e:
        _err('launch REFUSED: cannot measure child rusage on this platform (%s)' % e)
        return common.EXIT_ERROR
    resume = bool(getattr(args, 'resume', False))
    if resume and stage != 'official':
        _err('launch: --resume continues an interrupted official run only (--stage official)')
        return common.EXIT_ERROR
    bundle = GATE_FOR_STAGE[stage]
    ok, why = common.v1_status(common.make_ctx(args, delivery=args.dir, stage=bundle).delivery)
    if not ok:
        _err('launch REFUSED: legacy delivery: use retro (%s)' % why)
        return common.EXIT_MISSING
    recovered, refusals = recover_inflight(args, args.dir)
    for aid in recovered:
        _err('launch: recorded attempt %s, whose launch process ended before writing its ATTEMPTS row (interrupted, '
             'recovered)' % aid)
    if refusals:
        _err('launch REFUSED: %s' % '; '.join(refusals))
        return common.EXIT_FAIL
    opts = {'invoked_by': 'launch', 'launch_stage': stage, 'pending_argv': cmd}
    if resume:
        opts['resume'] = True
    ctx = common.make_ctx(args, delivery=args.dir, stage=bundle, options=opts)
    results = common.run_bundle(ctx, bundle)
    rc = common.exit_code(results)
    try:
        report = common.write_gate_report(ctx, bundle, results)
    except common.CheckerError as e:
        _err(common.format_results([r for r in results if r.status in common.BLOCKING_STATUSES]))
        _err('launch REFUSED: %s' % e)
        return common.EXIT_ERROR
    report_rel = report.relative_to(ctx.delivery).as_posix()
    if rc != 0:
        _err(common.format_results([r for r in results if r.status in common.BLOCKING_STATUSES]))
        _err('launch REFUSED: %s gate exit %d (report %s)' % (bundle, rc, report_rel))
        return rc
    try:
        clean, msgs = closure_state(ctx, stage)
    except MissingInput as e:
        _err('launch REFUSED: %s' % e)
        return common.EXIT_MISSING
    except InvalidInput as e:
        _err('launch REFUSED: %s' % e)
        return common.EXIT_FAIL
    if not clean:
        _err('launch REFUSED: closure, ENDPOINT_LOCK.json or SOURCE_LOCK.json not committed and clean:\n  '
             + '\n  '.join(msgs[:40]))
        return common.EXIT_FAIL
    acc = cycle_accounting(ctx)
    other_open = [st for st in acc['open_stages'] if st != stage]
    if other_open:
        _err('launch REFUSED: %s attempts since the last cycle close are not closed; run `MFC cycle close` for '
             'that cycle before a %s attempt (SPEC 3.2: every cycle ends with cycle close)'
             % ('/'.join(other_open), stage))
        return common.EXIT_FAIL
    named = argv_world_problems(stage, cmd, ctx.lock)
    if named:
        _err('launch REFUSED: the command names world(s) %s, which the lock gives to another stage; a %s attempt '
             'runs %s worlds only (SPEC 3.1, 3.2)' % (', '.join(named), stage, STAGE_WORLD_ROLE[stage]))
        return common.EXIT_FAIL
    if stage in TEST_CYCLE_KINDS:
        # SPEC 3.2: launch refuses dev and rehearsal attempts beyond the cap; a technical attempt beyond it is
        # refused too, because CYC-01 counts technical cycles against the same cap
        cap, basis = common.effective_cap(ctx, 'max_cycles')
        # this attempt joins the open cycle of its stage, or opens a new one
        in_use = len(acc['closed']) + len(set(acc['open_test']) | {stage})
        if in_use > cap:
            _err('launch REFUSED: %d closed test cycles%s reach the cycle cap %s (%s); no new %s attempts'
                 % (len(acc['closed']), ' plus the open %s cycle' % stage if stage in acc['open_test'] else '',
                    cap, basis, stage))
            return common.EXIT_FAIL
    resume_env: Dict[str, str] = {}
    resume_fields: Dict[str, Any] = {}
    if stage == 'official':
        an = official_attempt_analysis(ctx)
        if an['unattributed'] or (an['bearing'] and not resume):
            _err('launch REFUSED: official runs are never rerun (assessment-bearing official attempts %s; '
                 'unattributed official rows %d)%s' % (an['bearing'], len(an['unattributed']),
                                                       '' if resume else '; an interrupted official run continues only '
                                                       'with launch --stage official --resume'))
            return common.EXIT_FAIL
        if resume:
            ra = resume_analysis(ctx, pending_resume=True)
            if ra['problems'] or ra['pending_problems']:
                _err('launch REFUSED: --resume is not legal here (SPEC 20.8):\n  %s'
                     % '\n  '.join((ra['pending_problems'] + ra['problems'])[:20]))
                return common.EXIT_FAIL
            done = sorted(ra['completed'])
            # a recovered attempt's rows are vouched for only by a replay of one of its own units (SPEC 20.17): pick
            # among those while any recovered attempt is still unvouched
            unv = sorted(set(u for us in replay_analysis(ctx)['recovered_unvouched'].values() for u in us) & set(done))
            pick = secrets.SystemRandom().choice(unv or done)
            resume_fields = {'resume_of': ra['official'][-1], 'replay_requested': [pick.split('|', 1)]}
            resume_env = {'MINIFLY_RESUME_OF': ra['official'][-1],
                          'MINIFLY_COMPLETED_UNITS': json.dumps([u.split('|', 1) for u in done]),
                          'MINIFLY_REPLAY_UNITS': json.dumps([pick.split('|', 1)]),
                          'MINIFLY_REPLAY_DIR': common.REPLAY_DIR}
    sl = ctx.source_lock if ctx.path('SOURCE_LOCK.json').is_file() else {}
    el_sha, sl_sha = ctx.endpoint_lock_sha256, ctx.source_lock_sha256
    attempt_id = 'ATT-%s-%s' % (common.utc_now(), secrets.token_hex(3))
    git_head = ctx.git.head()
    before = snapshot_results(ctx.delivery)
    marker_doc = {'schema': SCHEMA_INFLIGHT, 'attempt_id': attempt_id, 'stage': stage, 'argv': cmd,
                  'cwd': ctx.delivery_rel, 'utc_start': common.utc_now(), 'git_head': git_head,
                  'endpoint_lock_sha256': el_sha, 'source_lock_sha256': sl_sha,
                  'native_closure_sha': sl.get('native_closure_sha'), 'params_sha256': sl.get('params_sha256'),
                  'gate_report': report_rel, 'pid': os.getpid(), 'boot_id': _boot_id(), 'host': socket.gethostname(),
                  'before': before}
    marker_doc.update(resume_fields)
    marker = ctx.path(inflight_rel(attempt_id))
    _write_atomic(marker, common.canonical_json(marker_doc) + b'\n')
    env = dict(os.environ)
    env.update({'MINIFLY_ATTEMPT_ID': attempt_id, 'MINIFLY_STAGE': stage, 'MINIFLY_DELIVERY': ctx.delivery_rel,
                'MINIFLY_ENDPOINT_LOCK_SHA256': el_sha or '', 'MINIFLY_SOURCE_LOCK_SHA256': sl_sha or '',
                'MINIFLY_NATIVE_CLOSURE_SHA': str(sl.get('native_closure_sha') or ''),
                'MINIFLY_PARAMS_SHA256': str(sl.get('params_sha256') or ''),
                'MINIFLY_REPO_ROOT': str(Path(ctx.repo_root).resolve()),
                'MINIFLY_RUNTIME_CLOSURE': runtime_closure_rel(attempt_id)})
    env.update(resume_env)
    utc_start = common.utc_now()
    child_rc: Optional[int] = None
    max_rss_kb = 0
    interrupted = False
    launch_error = None
    t0 = time.monotonic()
    ru0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        try:
            proc = subprocess.Popen(cmd, cwd=str(ctx.delivery), env=env)
        except OSError as e:
            child_rc, launch_error = 127, 'could not start command: %s' % e
        else:
            child_rc, max_rss_kb, interrupted = _wait(proc)
    finally:
        wall = time.monotonic() - t0
        ru1 = resource.getrusage(resource.RUSAGE_CHILDREN)
        utc_end = common.utc_now()
        snapshot_error = None
        try:
            after = snapshot_results(ctx.delivery)
        except Exception as e:  # noqa: BLE001 - the ATTEMPTS row must still be written
            after, snapshot_error = {}, 'results snapshot failed: %s' % common.summarize_exception(e)
        produced = [{'path': p, 'sha256': s} for p, s in sorted(after.items()) if before.get(p) != s]
        removed = sorted(p for p in before if p not in after) if snapshot_error is None else []
        try:
            clean_after, msgs_after = closure_state(ctx, stage)
        except Exception as e:  # noqa: BLE001
            clean_after, msgs_after = False, [common.summarize_exception(e)]
        row = {
            'schema': common.SCHEMA_ATTEMPT,
            'attempt_id': attempt_id,
            'stage': stage,
            'argv': cmd,
            'cwd': ctx.delivery_rel,
            'utc_start': utc_start,
            'utc_end': utc_end,
            'git_head': git_head,
            'closure_clean': True,
            'closure_clean_after': clean_after,
            'endpoint_lock_sha256': el_sha,
            'source_lock_sha256': sl_sha,
            'native_closure_sha': sl.get('native_closure_sha'),
            'params_sha256': sl.get('params_sha256'),
            'exit_code': child_rc if child_rc is not None else -1,
            'wall_s': round(wall, 3),
            'cpu_children_user_s': round(max(0.0, ru1.ru_utime - ru0.ru_utime), 6),
            'cpu_children_sys_s': round(max(0.0, ru1.ru_stime - ru0.ru_stime), 6),
            'max_rss_kb': int(max_rss_kb),
            'produced_paths': produced,
            'removed_paths': removed,
            'gate_report': report_rel,
        }
        if launch_error:
            row['launch_error'] = launch_error
        if snapshot_error:
            row['snapshot_error'] = snapshot_error
        if interrupted or child_rc is None:
            row['interrupted'] = True
        if not clean_after:
            row['closure_problems_after'] = msgs_after[:20]
        if stage == 'official':
            # the (world, arm) units this attempt wrote, with canonical digests: a later resume must leave them
            # unchanged and its determinism replay is compared with them (SPEC 20.8)
            units, replay, units_err = official_units_now(args, args.dir, attempt_id)
            row['units'] = units
            if units_err:
                row['units_error'] = units_err
            if resume_fields:
                row.update(resume_fields)
                row['replay_units'] = [[u['world'], u['arm']] for u in replay]
        append_jsonl(ctx.path('runlog/ATTEMPTS.jsonl'), row)
        try:
            marker.unlink()
        except OSError:
            pass
    print('launch %s: attempt %s exit %s, %d produced paths, %.1f s wall, %.1f s CPU'
          % (stage, attempt_id, child_rc, len(produced), wall,
             row['cpu_children_user_s'] + row['cpu_children_sys_s']))
    if not clean_after:
        _err('launch: closure changed during the attempt; RUN-01 will reject its outputs')
    try:
        scope = stage_scope_problems(common.make_ctx(args, delivery=args.dir, stage=bundle,
                                                     options={'invoked_by': 'launch'}), only=attempt_id)
    except Exception as e:  # noqa: BLE001 - the verdict is CYC-01's; report and fail the launch
        scope = ['stage scope could not be verified: %s' % common.summarize_exception(e)]
    if scope:
        _err('launch: attempt %s wrote outside the %s stage; CYC-01 FAILs on this attempt for good (ATTEMPTS is '
             'append-only):\n  %s' % (attempt_id, stage, '\n  '.join(scope[:20])))
    resume_bad: List[str] = []
    if resume_fields:
        try:
            c2 = common.make_ctx(args, delivery=args.dir, stage='official', options={'invoked_by': 'launch'})
            resume_bad = resume_analysis(c2)['problems'] + replay_analysis(c2)['problems']
        except Exception as e:  # noqa: BLE001 - RUN-02/RUN-04 give the verdict; fail the launch
            resume_bad = ['resume could not be verified: %s' % common.summarize_exception(e)]
        if resume_bad:
            _err('launch: resume attempt %s breaks SPEC 20.8 (RUN-02/RUN-04 FAIL for good):\n  %s'
                 % (attempt_id, '\n  '.join(resume_bad[:20])))
    return common.EXIT_PASS if child_rc == 0 and clean_after and not scope and not resume_bad else common.EXIT_FAIL


def cmd_cycle(args: Any) -> int:
    if getattr(args, 'action', 'close') != 'close':
        _err('cycle: only "close" is supported')
        return common.EXIT_ERROR
    kind, decision = args.kind, args.decision
    reason = (getattr(args, 'reason', None) or '').strip()
    if kind not in CYCLE_KINDS or decision not in DECISIONS:
        _err('cycle close: invalid --kind or --decision')
        return common.EXIT_ERROR
    if not reason:
        _err('cycle close: --reason "<one sentence>" is required')
        return common.EXIT_MISSING
    ctx = common.make_ctx(args, delivery=args.dir, stage='cycle', options={'invoked_by': 'cycle'})
    ok, why = common.v1_status(ctx.delivery)
    if not ok:
        _err('cycle close REFUSED: legacy delivery: use retro (%s)' % why)
        return common.EXIT_MISSING
    recovered, refusals = recover_inflight(args, args.dir)
    for aid in recovered:
        _err('cycle close: recorded attempt %s, whose launch process ended before writing its ATTEMPTS row' % aid)
    if refusals:
        _err('cycle close REFUSED: %s' % '; '.join(refusals))
        return common.EXIT_FAIL
    ctx = common.make_ctx(args, delivery=args.dir, stage='cycle', options={'invoked_by': 'cycle'})
    acc = cycle_accounting(ctx)
    rows = acc['rows']
    # the attempts of this cycle: every attempt of its stage that no earlier close covers (SPEC 3.2 'attempts
    # since the last close'); a cycle row never covers another stage's attempts (CYC-01)
    since = [a.get('attempt_id') for a in acc['uncovered'] if a.get('stage') == kind]
    others = [st for st in acc['open_stages'] if st != kind]
    if others and not since:
        _err('cycle close REFUSED: no %s attempt since the last close, but %s attempts are open: close that cycle '
             'with --kind %s' % (kind, '/'.join(others), others[0]))
        return common.EXIT_FAIL
    sl = ctx.source_lock if ctx.path('SOURCE_LOCK.json').is_file() else {}
    gates = common.list_gate_reports(ctx)
    k_designs, _pairs = common.designs_tried_k(ctx)
    row = {
        'schema': common.SCHEMA_CYCLE,
        'k': len(rows) + 1,
        'kind': kind,
        'decision': decision,
        'reason': reason,
        'utc': common.utc_now(),
        'computed': {
            'native_closure_sha': sl.get('native_closure_sha'),
            'params_sha256': sl.get('params_sha256'),
            'source_lock_sha256': ctx.source_lock_sha256,
            'attempts': since,
            'latest_gate': _gate_summary(gates[-1] if gates else None),
            'outcomes_readable': learner_outcomes_exist(ctx),
            'designs_tried_k': k_designs,
        },
    }
    if row['computed']['latest_gate']:
        row['computed']['latest_gate']['sha256'] = common.sha256_file(ctx.path(gates[-1]['_path']))
    scope = stage_scope_problems(ctx, only=since)
    if scope:
        _err('cycle close: attempts of this cycle wrote outside the %s stage; CYC-01 FAILs on them:\n  %s'
             % (kind, '\n  '.join(scope[:20])))
    if decision == 'continue':
        met, how = exit_criterion(ctx, row, gates)
        if not met:
            _err('cycle close REFUSED: decision continue but the %s exit criterion is not met (%s); '
                 'close with revise, stop or escalate instead' % (kind, how))
            return common.EXIT_FAIL
    append_jsonl(ctx.path('runlog/CYCLES.jsonl'), row)
    if kind in TEST_CYCLE_KINDS:
        cap, basis = common.effective_cap(ctx, 'max_cycles')
        n = len(closed_test_cycles(rows)) + 1
        if n > cap:
            _err('cycle close: %d closed test cycles exceed the cap %s (%s); CYC-01 will FAIL' % (n, cap, basis))
    print('cycle %d closed: %s %s (%d attempts, designs_tried_k=%d)' % (row['k'], kind, decision, len(since), k_designs))
    return common.EXIT_PASS


# --------------------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------------------

def _attempt_clean(a: dict) -> bool:
    return a.get('closure_clean') is True and a.get('closure_clean_after', True) is not False


@check('RUN-01')
def check_run_01(ctx: common.Ctx) -> Result:
    """Every file under results/{admission,dev,rehearsal,official}/ and results/receipts/ (and every
    RESULT_BUNDLE*.zip member) matches, by path and sha256, a produced_paths entry of an ATTEMPTS row
    that ran from a clean closure, and every OUTPUTS row there names an ATTEMPTS row (MINIFLY_ATTEMPT_ID)."""
    files = ctx.delivery_files(refresh=True)
    zips = [f for f in files if common.RESULT_BUNDLE_RE.match(f.rsplit('/', 1)[-1])]
    covered = [f for f in files if f.startswith(COVERED_RESULT_DIRS) and f not in zips
               and not _is_checker_written(f) and not _is_empty_keep(ctx, f)]
    attempts = common.read_attempts(ctx)
    known = set(a.get('attempt_id') for a in attempts)
    inflight = [f for f in files if f.startswith('runlog/') and INFLIGHT_RE.match(f.split('/', 1)[1])
                and INFLIGHT_RE.match(f.split('/', 1)[1]).group(1) not in known]
    if not covered and not zips and not inflight:
        return Result('RUN-01', PASS, 'no result files under %s' % ', '.join(COVERED_RESULT_DIRS))
    index: Dict[Tuple[str, str], List[dict]] = {}
    for i, a in enumerate(attempts, 1):
        pp = a.get('produced_paths')
        if pp is None:
            continue
        if not isinstance(pp, list):
            raise InvalidInput('ATTEMPTS row %d produced_paths is not a list' % i)
        for e in pp:
            if isinstance(e, dict) and isinstance(e.get('path'), str) and isinstance(e.get('sha256'), str):
                index.setdefault((common.norm_rel(e['path']), e['sha256']), []).append(a)
    uncovered: List[str] = []
    unclean: List[str] = []

    def judge(label: str, cands: List[Tuple[str, str]]) -> None:
        hits: List[dict] = []
        for c in cands:
            hits.extend(index.get(c, []))
        if not hits:
            uncovered.append(label)
        elif not any(_attempt_clean(a) for a in hits):
            unclean.append(label)

    for f in covered:
        p = ctx.path(f)
        cands = [(f, common.sha256_file(p))]
        if f.endswith('.gz'):
            try:
                with gzip.open(str(p), 'rb') as fh:
                    cands.append((f[:-3], common.sha256_bytes(fh.read())))
            except (OSError, EOFError):
                pass
        judge(f, cands)
    for z in zips:
        zdir = z.rsplit('/', 1)[0] if '/' in z else ''
        try:
            with zipfile.ZipFile(str(ctx.path(z))) as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    m = common.norm_rel(info.filename)
                    logical = m if (m.startswith('results/') or not zdir) else zdir + '/' + m
                    data = zf.read(info)
                    cands = [(logical, common.sha256_bytes(data))]
                    if logical.endswith('.gz'):
                        try:
                            cands.append((logical[:-3], common.sha256_bytes(gzip.decompress(data))))
                        except (OSError, EOFError):
                            pass
                    judge('%s!%s' % (z, info.filename), cands)
        except zipfile.BadZipFile as e:
            raise InvalidInput('%s is not a valid zip: %s' % (z, e))
    n = len(covered) + len(zips)
    n_rows, row_ex = unattributed_rows(ctx)
    if uncovered or unclean or n_rows or inflight:
        parts = []
        if inflight:
            parts.append('in-flight attempt(s) without an ATTEMPTS row (%s): a launch is running or died; the next '
                         'launch or cycle close records a dead one' % ', '.join(inflight[:4]))
        if uncovered or unclean:
            parts.append('%d of %d result files not launch-recorded with a matching sha256; %d from an attempt '
                         'without a clean closure' % (len(uncovered), n, len(unclean)))
        if n_rows:
            parts.append('%d OUTPUTS row(s) whose attempt_id names no ATTEMPTS row (the runner copies '
                         'MINIFLY_ATTEMPT_ID), e.g. %s' % (n_rows, '; '.join(row_ex[:3])))
        return Result('RUN-01', FAIL, '; '.join(parts), evidence=(inflight + uncovered + unclean + row_ex)[:40],
                      data={'uncovered': uncovered[:200], 'unclean': unclean[:200], 'unattributed_rows': n_rows})
    return Result('RUN-01', PASS, '%d result files (%d bundles) covered by %d ATTEMPTS rows'
                  % (n, len(zips), len(attempts)))


@check('RUN-02')
def check_run_02(ctx: common.Ctx) -> Result:
    """At most one official attempt produced assessment-bearing rows, except legal resume attempts (SPEC 20.8: launch
    --resume under the identical lock after an interrupted attempt, rows only for units no earlier attempt has, the
    earlier rows unchanged); any other official attempt after one that produced rows is a rerun (FAIL); official
    rows must be attributed to an official attempt."""
    an = official_attempt_analysis(ctx)
    resume_req = ctx.options.get('launch_stage') == 'official' and bool(ctx.options.get('resume'))
    ra = resume_analysis(ctx, pending_resume=resume_req)
    problems = []
    if len(an['bearing_non_resume']) > 1:
        problems.append('%d official attempts produced assessment-bearing rows without launch --resume: %s'
                        % (len(an['bearing_non_resume']), an['bearing_non_resume']))
    if an['after_first_bearing']:
        problems.append('official attempt(s) after an assessment-bearing one (rerun): %s' % an['after_first_bearing'])
    if an['unattributed']:
        problems.append('%d official rows not attributed to an official ATTEMPTS row' % len(an['unattributed']))
    problems.extend(ra['problems'])
    if resume_req:
        problems.extend(ra['pending_problems'])
    elif ctx.options.get('launch_stage') == 'official' and an['bearing']:
        problems.append('a new official launch was requested after attempt %s produced rows (only `launch --stage '
                        'official --resume` may continue an interrupted official run)' % an['bearing'][0])
    if problems:
        return Result('RUN-02', FAIL, '; '.join(problems[:8]), evidence=(problems + an['unattributed'])[:40], data=an)
    return Result('RUN-02', PASS, '%d official attempts, %d assessment-bearing%s' % (
        len(an['official']), len(an['bearing']),
        (', %d legal resume attempt(s) %s' % (len(ra['resumes']), ra['resumes'])) if ra['resumes'] else ''),
        data={'official': an['official'], 'bearing': an['bearing'], 'resumes': ra['resumes']})


@check('RUN-04')
def check_run_04(ctx: common.Ctx) -> Result:
    """Determinism replay of a resumed official run (SPEC 20.8): every resume attempt re-ran at least one (world,
    arm) unit completed by an earlier official attempt (every unit launch chose) into results/official_replay/, and
    each replayed unit's canonical row digest (attempt_id excluded) equals the original's."""
    ra = replay_analysis(ctx)
    if not ra['resumes'] and not ra['replay_attempts'] and not ra['recovered']:
        return Result('RUN-04', NOT_APPLICABLE, 'no resumed or recovered official attempt and no %s rows'
                      % common.REPLAY_OUTPUTS)
    if ra['problems']:
        return Result('RUN-04', FAIL, '; '.join(ra['problems'][:8]), evidence=ra['problems'][:40], data=ra)
    return Result('RUN-04', PASS, '%d resume attempt(s); replayed units %s reproduce the original rows bit for bit'
                  % (len(ra['resumes']), ', '.join('%s: %s' % (k, ','.join(v)) for k, v in sorted(ra['replayed'].items()))),
                  data=ra)


SHA256_HEX_RE = re.compile(r'^[0-9a-f]{64}$')


def source_lock_of_attempt(ctx: common.Ctx, a: dict) -> Optional[dict]:
    """The SOURCE_LOCK.json content an attempt ran under (by its recorded source_lock_sha256): the worktree file when
    it still has that sha, else the committed version with that sha (lock build runs before each cycle); None when
    no version with that sha exists."""
    want = _strip_sha(a.get('source_lock_sha256'))
    if not want:
        return None
    key = ('wp1_sl_by_sha', want)
    if key in ctx._cache:
        return ctx._cache[key]
    doc = None
    p = ctx.path('SOURCE_LOCK.json')
    if p.is_file() and common.sha256_file(p) == want:
        doc = common.load_json_file(p)
    else:
        rel = ctx.repo_rel_of('SOURCE_LOCK.json')
        c = ctx.git.first_commit_where(rel, lambda b: common.sha256_bytes(b) == want)
        if c is not None:
            data = ctx.git.file_at(c.sha, rel)
            doc = common.loads_strict(data) if data is not None else None
    ctx._cache[key] = doc if isinstance(doc, dict) else None
    return ctx._cache[key]


def _locked_map(ctx: common.Ctx, mapping: Any, repo_first: bool) -> Dict[str, Optional[str]]:
    """{repo-relative path: sha256 or None} of a {path: sha} object or a list of paths (resolved like launch's
    closure: repo-relative first for SOURCE_LOCK keys, delivery-relative first for lock file entries)."""
    out: Dict[str, Optional[str]] = {}
    items = mapping.items() if isinstance(mapping, dict) else [(e, None) for e in (mapping or [])]
    for k, v in items:
        if not isinstance(k, str) or not k:
            continue
        try:
            rr = _resolve_entry(ctx, k, repo_first=repo_first)
        except InvalidInput:
            rr = None
        if rr is None:
            try:
                rr = common.norm_rel(k) if repo_first else ctx.repo_rel_of(common.norm_rel(k))
            except InvalidInput:
                continue
        out[rr] = _strip_sha(v)
    return out


def runtime_closure_problems(ctx: common.Ctx, a: dict, produced: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """(problems, missing) of one counted attempt's runtime closure (RUN-03)."""
    aid = a.get('attempt_id')
    rel = runtime_closure_rel(aid)
    label = '%s attempt %s' % (a.get('stage'), aid)
    if not isinstance(aid, str) or not re.match(r'^[A-Za-z0-9._-]+$', aid):
        return ['%s: attempt_id is not a file-name-safe string' % label], []
    p = ctx.path(rel)
    if rel not in produced:
        return ['%s: %s is not among its produced_paths (the runner writes it at the end of every counted run; '
                'launch records it)%s' % (label, rel, '' if p.is_file() else ' and the file is absent')], []
    if not p.is_file():
        return ['%s: %s is absent' % (label, rel)], []
    if common.sha256_file(p) != produced[rel]:
        return ['%s: %s no longer holds the bytes the attempt recorded' % (label, rel)], []
    try:
        doc = common.load_json_file(p)
    except InvalidInput as e:
        return ['%s: %s does not parse: %s' % (label, rel, e)], []
    if not isinstance(doc, dict) or doc.get('schema') != SCHEMA_RUNTIME_CLOSURE:
        return ['%s: %s is not a %s object' % (label, rel, SCHEMA_RUNTIME_CLOSURE)], []
    if doc.get('attempt_id') != aid:
        return ['%s: %s names attempt %r' % (label, rel, doc.get('attempt_id'))], []
    files = doc.get('files')
    if not isinstance(files, list) or not files:
        return ['%s: %s lists no files (it holds every repo file in sys.modules at the end of the run)' % (label, rel)], []
    sl = source_lock_of_attempt(ctx, a)
    if sl is None:
        return [], ['%s: no SOURCE_LOCK.json with its source_lock_sha256 %s (worktree or history)'
                    % (label, str(a.get('source_lock_sha256'))[:12])]
    code = _locked_map(ctx, sl.get('closure_sha256') or {}, repo_first=True)
    lock = ctx.lock
    lf = lock.get('files') if isinstance(lock.get('files'), dict) else {}
    assets: Dict[str, Optional[str]] = dict(code)
    assets.update(_locked_map(ctx, lock.get('endpoint_files_sha256') or {}, repo_first=False))
    for k, v in _locked_map(ctx, lf.get('assets') or [], repo_first=False).items():
        assets.setdefault(k, v)
    if isinstance(lf.get('params'), str) and lf['params']:
        for k in _locked_map(ctx, [lf['params']], repo_first=False):
            assets[k] = _strip_sha(sl.get('params_sha256'))
    gap: List[str] = []
    bad: List[str] = []
    seen: set = set()
    for i, f in enumerate(files):
        path = f.get('path') if isinstance(f, dict) else None
        sha = f.get('sha256') if isinstance(f, dict) else None
        try:
            path = common.norm_rel(path) if isinstance(path, str) else None
        except InvalidInput:
            path = None
        if path is None or not isinstance(sha, str) or not SHA256_HEX_RE.match(sha):
            bad.append('files[%d] is not {path (repo-relative), sha256}: %r' % (i, f))
            continue
        if path in seen:
            bad.append('files[%d] repeats %s' % (i, path))
            continue
        seen.add(path)
        if path.endswith('.py'):
            if path not in code:
                gap.append(path)
            elif code[path] and code[path] != sha:
                bad.append('%s ran with sha256 %s, SOURCE_LOCK.closure_sha256 has %s' % (path, sha[:12], code[path][:12]))
            continue
        if path not in assets:
            bad.append('data asset %s is in no files.assets, endpoint_files_sha256 or SOURCE_LOCK entry' % path)
            continue
        want = assets[path]
        if want is None:
            data = ctx.git.file_at(a.get('git_head'), path) if isinstance(a.get('git_head'), str) else None
            want = common.sha256_bytes(data) if data is not None else None
            if want is None:
                bad.append('data asset %s is not committed at the attempt git_head %s' % (path, str(a.get('git_head'))[:12]))
                continue
        if want != sha:
            bad.append('data asset %s ran with sha256 %s, locked %s' % (path, sha[:12], want[:12]))
    out: List[str] = []
    if gap:
        out.append('%s: closure gap: %d repo .py file(s) present at runtime but absent from SOURCE_LOCK.closure_sha256: '
                   '%s' % (label, len(gap), ', '.join(gap[:8])))
    if bad:
        out.append('%s: %s' % (label, '; '.join(bad[:8])))
    return out, []


@check('RUN-03')
def check_run_03(ctx: common.Ctx) -> Result:
    """Every counted attempt (stage technical, dev, rehearsal or official) that produced any path produced its runtime closure
    runlog/RUNTIME_CLOSURE_<attempt_id>.json (listed in its produced_paths with the current sha256), and every repo
    .py file in it is in the SOURCE_LOCK.closure_sha256 the attempt ran under with the same sha256; data assets
    are declared (files.assets, files.params, endpoint_files_sha256 or SOURCE_LOCK) with the same sha256.
    T2 enforced, T1 WARN (RULES warn_below_tier). The checker never imports delivery code: the runner records
    what it imported (SPEC 20.7)."""
    attempts = common.read_attempts(ctx)
    counted = [a for a in attempts if a.get('stage') in RUNTIME_CLOSURE_STAGES]
    problems: List[str] = []
    missing: List[str] = []
    for a in attempts:
        for e in a.get('produced_paths') or []:
            path = e.get('path') if isinstance(e, dict) else None
            m = RUNTIME_CLOSURE_RE.match(path) if isinstance(path, str) else None
            if m and m.group(1) != a.get('attempt_id'):
                problems.append('%s attempt %s produced %s, the runtime closure of another attempt'
                                % (a.get('stage'), a.get('attempt_id'), path))
    exempt = 0
    for a in counted:
        produced: Dict[str, Any] = {}
        for e in a.get('produced_paths') or []:
            if isinstance(e, dict) and isinstance(e.get('path'), str):
                produced[e['path']] = e.get('sha256')
        if not [k for k in produced if k != runtime_closure_rel(a.get('attempt_id'))]:
            exempt += 1  # produced nothing that could count: there is nothing to attribute to its code
            continue
        pr, mi = runtime_closure_problems(ctx, a, produced)
        problems.extend(pr)
        missing.extend(mi)
    if problems:
        return Result('RUN-03', FAIL, '; '.join(problems[:6]), evidence=problems[:40])
    if missing:
        return Result('RUN-03', MISSING, '; '.join(missing[:6]), evidence=missing[:40])
    if not counted:
        return Result('RUN-03', PASS, 'no technical, dev, rehearsal or official attempt yet')
    return Result('RUN-03', PASS, '%d counted attempts: every runtime closure is launch-recorded and every repo .py '
                  'file and data asset it lists is locked with the same sha256%s'
                  % (len(counted) - exempt, ('; %d attempt(s) produced no paths' % exempt) if exempt else ''))


def _official_coverage_required(ctx: common.Ctx, attempts: Sequence[dict]) -> Optional[str]:
    """Why every test attempt must already sit in a closed cycle: the non-dry official stage (gate or launch),
    or an official attempt that exists. At `gate --stage official --dry` the rehearsal cycle is still open."""
    if ctx.options.get('launch_stage') == 'official':
        return 'at launch --stage official'
    if ctx.stage == 'official' and not ctx.dry:
        return 'at the official stage'
    if any(a.get('stage') == 'official' for a in attempts):
        return 'once an official attempt exists'
    return None


@check('CYC-01')
def check_cyc_01(ctx: common.Ctx) -> Result:
    """Test cycles (closed rows plus open cycles of attempts no row covers) are at most caps.max_cycles; the
    cycle and attempt logs are append-only and agree; every attempt wrote only within its stage (so a testbed
    attempt, not counted against the cap, never carries dev, rehearsal or official outcomes); before official
    every test attempt is in a closed cycle."""
    acc = cycle_accounting(ctx)
    rows = acc['rows']
    bad = _invalid_cycle_rows(rows)
    if bad:
        return Result('CYC-01', FAIL, 'invalid CYCLES rows: %s' % '; '.join(bad[:10]))
    problems = list(acc['problems'])
    problems.extend(runlog_append_only_problems(ctx))
    closed, open_test = acc['closed'], acc['open_test']
    cap, basis = common.effective_cap(ctx, 'max_cycles')
    n = len(closed) + len(open_test)
    if n > cap:
        problems.append('%d test cycles (%d closed%s) exceed caps.max_cycles %s (%s)' % (
            n, len(closed), ', open: %s' % '/'.join(open_test) if open_test else '', cap, basis))
    if len(acc['open_stages']) > 1:
        problems.append('attempts of stages %s follow the last cycle close: the %s cycle was not closed before '
                        'the next stage (SPEC 3.2: every cycle ends with `MFC cycle close`)'
                        % ('/'.join(acc['open_stages']), acc['open_stages'][0]))
    problems.extend(stage_scope_problems(ctx))
    why = _official_coverage_required(ctx, acc['attempts'])
    if why and acc['uncovered']:
        ids = ['%s (%s)' % (a.get('attempt_id'), a.get('stage')) for a in acc['uncovered']]
        problems.append('%s every testbed, technical, dev and rehearsal attempt must be in a closed cycle; %d are '
                        'in none: %s' % (why, len(ids), ', '.join(ids[:6])))
    if problems:
        return Result('CYC-01', FAIL, '; '.join(problems), evidence=problems[:20],
                      data={'closed': len(closed), 'open': open_test, 'cap': cap})
    return Result('CYC-01', PASS, '%d test cycles (%d closed%s) <= cap %s (%s); %d attempts in closed cycles' % (
        n, len(closed), ', open: %s' % '/'.join(open_test) if open_test else '', cap, basis, len(acc['covered'])),
        data={'closed': len(closed), 'open': open_test, 'cap': cap})


@check('CYC-04')
def check_cyc_04(ctx: common.Ctx) -> Result:
    """runlog/ATTEMPTS.jsonl and runlog/CYCLES.jsonl are append-only at publish too (SPEC 20.16, round 1): every row
    any commit recorded survives unchanged in every later commit and in the worktree (CYC-01's append-only part
    ran only up to the official gate, so an earlier ATTEMPTS row edited after the run reached publish unseen). A
    shallow clone gives MISSING."""
    problems = runlog_append_only_problems(ctx)
    if problems:
        return Result('CYC-04', FAIL, 'runlog is not append-only: %s' % '; '.join(problems[:6]), evidence=problems[:40])
    n = {rel: len(_raw_lines(ctx.path(rel).read_bytes())) if ctx.path(rel).is_file() else 0
         for rel in RUNLOG_APPEND_ONLY}
    return Result('CYC-04', PASS, 'append-only over the history and the worktree: %s'
                  % ', '.join('%s %d rows' % (k, v) for k, v in sorted(n.items())))


@check('CYC-02')
def check_cyc_02(ctx: common.Ctx) -> Result:
    rows = common.read_cycles(ctx)
    bad = _invalid_cycle_rows(rows)
    if bad:
        return Result('CYC-02', FAIL, 'invalid CYCLES rows: %s' % '; '.join(bad[:10]))
    gates = common.list_gate_reports(ctx)
    unmet, met = [], 0
    for i, r in enumerate(rows, 1):
        if r.get('decision') != 'continue':
            continue
        ok, how = exit_criterion(ctx, r, gates)
        if ok:
            met += 1
        else:
            unmet.append('cycle %s (%s): %s' % (r.get('k', i), r.get('kind'), how))
    if unmet:
        return Result('CYC-02', FAIL, 'continue without the exit criterion: %s' % '; '.join(unmet), evidence=unmet)
    return Result('CYC-02', PASS, '%d continue decisions met their exit criteria (%d cycles closed)' % (met, len(rows)))


def lock_world_ids(lock: dict) -> Dict[str, List[str]]:
    worlds = lock.get('worlds') if isinstance(lock.get('worlds'), dict) else {}

    def ids(v):
        out = []
        for e in (v or []):
            if isinstance(e, str):
                out.append(e)
            elif isinstance(e, dict) and isinstance(e.get('id'), str):
                out.append(e['id'])
        return out
    off = worlds.get('official') if isinstance(worlds.get('official'), dict) else {}
    return {'technical': ids(worlds.get('technical')), 'dev': ids(worlds.get('dev')),
            'rehearsal': ids(worlds.get('rehearsal')), 'official': ids(off.get('ids'))}


def _mentions(text: str, ids: Sequence[str]) -> List[str]:
    return [w for w in ids if re.search(r'(?<![A-Za-z0-9_])' + re.escape(w) + r'(?![A-Za-z0-9_])', text)]


def argv_template(argv: Sequence[str], world_ids: Sequence[str]) -> List[str]:
    """argv with world arguments removed: --world/--worlds/--only-world/--world-id(s) flags and their
    values, and tokens consisting only of known world ids."""
    known = set(world_ids)
    out: List[str] = []
    skip = False
    for tok in argv:
        t = str(tok)
        if skip:
            skip = False
            continue
        if t.startswith('-') and '=' in t and WORLD_FLAG_RE.match(t.split('=', 1)[0]):
            continue
        if WORLD_FLAG_RE.match(t):
            skip = True
            continue
        parts = [p for p in re.split(r'[,\s]+', t) if p]
        if parts and all(p in known for p in parts):
            continue
        out.append(t)
    return out


@check('CYC-03')
def check_cyc_03(ctx: common.Ctx) -> Result:
    """Rehearsal world ids appear in no dev or technical row, receipt or attempt argv; the rehearsal
    argv template equals the official one with world arguments removed."""
    lock = ctx.lock
    w = lock_world_ids(lock)
    reh = w['rehearsal']
    if not reh:
        return Result('CYC-03', FAIL, 'lock declares no rehearsal world (C3 rehearsal on a fresh world is required)')
    problems: List[str] = []
    overlap = sorted(set(reh) & (set(w['dev']) | set(w['technical'])))
    if overlap:
        problems.append('rehearsal world(s) also declared dev/technical: %s' % overlap)
    for label, lineno, row in ctx.iter_rows_with_source(OUTPUTS_GLOB):
        if row.get('world') in reh and (row.get('world_role') in ('dev', 'technical') or
                                       label.startswith(('results/dev/', 'results/admission/'))):
            problems.append('%s:%d uses rehearsal world %s as %s' % (label, lineno, row.get('world'), row.get('world_role')))
            if len(problems) > 50:
                break
    for rel in ctx.delivery_files():
        if not (rel.startswith('results/receipts/') and rel.endswith('.json')):
            continue
        r = ctx.read_json(rel)
        parts = rel.split('/')
        world = r.get('world') if isinstance(r, dict) else None
        if (world in reh or (len(parts) > 3 and parts[2] in reh)) and isinstance(r, dict) and \
                r.get('world_role') in ('dev', 'technical'):
            problems.append('%s: rehearsal world receipt with world_role %s' % (rel, r.get('world_role')))
    attempts = common.read_attempts(ctx)
    for a in attempts:
        if a.get('stage') in ('testbed', 'technical', 'dev'):
            hit = _mentions(' '.join(str(x) for x in (a.get('argv') or [])), reh)
            if hit:
                problems.append('%s attempt %s argv names rehearsal world(s) %s' % (a.get('stage'), a.get('attempt_id'), hit))
    reh_attempts = [a for a in attempts if a.get('stage') == 'rehearsal']
    if not reh_attempts:
        problems.append('no rehearsal attempt recorded in runlog/ATTEMPTS.jsonl')
    all_ids = w['technical'] + w['dev'] + w['rehearsal'] + w['official']
    official_argv = None
    source = ''
    if ctx.options.get('launch_stage') == 'official' and ctx.options.get('pending_argv'):
        official_argv, source = ctx.options['pending_argv'], 'pending official launch'
    else:
        offs = [a for a in attempts if a.get('stage') == 'official']
        if offs:
            official_argv, source = offs[0].get('argv') or [], 'official attempt %s' % offs[0].get('attempt_id')
    template_note = ''
    if official_argv is not None:
        ot = argv_template(official_argv, all_ids)
        for a in reh_attempts:
            rt = argv_template(a.get('argv') or [], all_ids)
            if rt != ot:
                problems.append('rehearsal attempt %s argv template %s != %s template %s'
                                % (a.get('attempt_id'), rt, source, ot))
    else:
        template_note = '; argv template comparison is enforced at launch --stage official (no official argv yet)'
    if problems:
        return Result('CYC-03', FAIL, '; '.join(problems[:12]), evidence=problems[:40])
    return Result('CYC-03', PASS, 'rehearsal world(s) %s fresh; %d rehearsal attempts%s'
                  % (reh, len(reh_attempts), template_note))
