"""MRK-1 governance (WP6): selftest, governance change classes, held-out canaries.

Commands (SPEC 7.1):
    MFC selftest [--mutate] [--only ID]
    MFC governance --base REV --head REV
    MFC heldout {seal|reveal|score} --batch B [--in PATH] [--key-file K] [--checker-sha S] [--out PATH] [--author ID]

Checks registered here (RULES.json module 'governance'): GOV-SELF, GOV-SEP, GOV-CLASS, GOV-REV, GOV-UD,
GOV-NUM, GOV-MONO, GOV-APPEND, GOV-SKILL, GOV-COI and HELD-01..03.

Python 3.9+, standard library only. Git is read-only through common.run_git; `git archive` is used only
here, into temporary directories. Kit code extracted from history (base/head checkers, the checker under
held-out test) runs only as a `python3 -I` subprocess on corpus fixtures in temporary copies; historical
delivery code is never imported or executed.
"""
from __future__ import annotations

import argparse
import ast
import concurrent.futures
import contextlib
import functools
import hashlib
import hmac
import importlib
import io
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import common
from common import (FAIL, MISSING, NOT_APPLICABLE, PASS, WARN, CheckerError, InvalidInput, MissingInput,  # noqa: F401
                    Result)

KIT_REL = common.KIT_REL_DEFAULT
SKILL_NAMES = ('minifly-research-loop', 'minifly-e3-design', 'minifly-claims-knowledge',
               'minifly-independent-review', 'minifly-governance')
SKILL_DESC_MAX = 1024
SKILL_BODY_MAX_LINES = 400  # body must be under 400 lines
AGENTS_POINTER_REL = 'PROJECT_DOCUMENTS/KB/AGENTS_POINTER.md'
ROOT_AGENTS_REL = 'AGENTS.md'
GOV_REVIEW_DIR = 'PROJECT_DOCUMENTS/KB/GOVERNANCE'
USER_DECISIONS_REL = 'PROJECT_DOCUMENTS/KB/USER_DECISIONS.jsonl'
CONSTRAINTS_REL = 'PROJECT_DOCUMENTS/KB/CONSTRAINTS.json'
CANARY_REL = 'PROJECT_DOCUMENTS/KB/CANARY_COMMITMENTS.jsonl'
CORPUS_SKIP = ('_build', 'heldout')
CANARY_SEED = 'c0ffee'
CANARY_FIXTURE = 'C00a'
SELFTEST_TARGET_S = 20.0
SELFTEST_SCHEMA = 'minifly.selftest.v1'
GOV_REPORT_SCHEMA = 'minifly.governance_report.v1'
HELDOUT_REPORT_SCHEMA = 'minifly.heldout_report.v1'
HELDOUT_BUNDLE_SCHEMA = 'minifly.heldout_bundle.v1'
FIXTURE_CMD_TIMEOUT_S = 300
SUBPROCESS_SELFTEST_TIMEOUT_S = 1200
GOV_DEPTH_ENV = 'MINIFLY_GOV_DEPTH'
GOV_DEPTH_MAX = 3
REVIEW_APPROVE = ('APPROVE', 'APPROVED', 'ACCEPT', 'ACCEPTED', 'ADMIT', 'PASS')
INFRA_TOP_DIRS = ('.claude', '.github', 'tools')
STATUS_ORDER = {'planned': 0, 'advisory': 1, 'enforced': 2}
# Removed lines containing one of these (case-insensitive) are removed rule lines (SPEC 13.2).
RULE_LINE_RE = re.compile(r'minifly_check|must|never', re.I)
TEST_LINE_RE = re.compile(r'^\s*(def\s+test|async\s+def\s+test|self\.assert|assert\b|with\s+self\.assert)')
DATE_DIR_RE = re.compile(r'(^|[_-])\d{8}$')
UD_ID_RE = re.compile(r'UD-[A-Za-z0-9][A-Za-z0-9_.-]*')
BATCH_ID_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$')
FM_RE = re.compile(r'^FM-\d\d$')

# Nesting state: SELF-01 increments common._SELFTEST_ACTIVE; this tracks our own in-process runs.
_STATE = {'selftest_depth': 0}


# ======================================================================================
# Small helpers
# ======================================================================================

def _family(identity: Any) -> Optional[str]:
    """'codex' / 'claude' from an Acting-Agent value or a reviewer/author field ('claude:<session>')."""
    if not isinstance(identity, str):
        return None
    fam = identity.strip().split(':', 1)[0].strip().lower()
    return fam if fam in ('codex', 'claude') else None


def _acting_values(commit: common.Commit) -> List[str]:
    return [v.strip() for v in commit.trailer_values('Acting-Agent') if v.strip()]


def _acting_families(commit: common.Commit) -> Set[str]:
    return set(f for f in (_family(v) for v in _acting_values(commit)) if f)


def _short(sha: Any) -> str:
    return str(sha)[:12]


def _numbers_of(value: Any) -> List[float]:
    """Numeric leaves of a JSON value (booleans excluded)."""
    out: List[float] = []
    if isinstance(value, bool) or value is None:
        return out
    if isinstance(value, (int, float)):
        out.append(float(value))
    elif isinstance(value, dict):
        for v in value.values():
            out.extend(_numbers_of(v))
    elif isinstance(value, (list, tuple)):
        for v in value:
            out.extend(_numbers_of(v))
    return out


def _fmt_num(x: float) -> str:
    return ('%d' % x) if float(x).is_integer() else repr(x)


def _parse_json_bytes(data: Optional[bytes], what: str) -> Tuple[Any, Optional[str]]:
    if data is None:
        return None, None
    try:
        return common.loads_strict(data, what=what), None
    except InvalidInput as e:
        return None, str(e)


def _parse_jsonl_bytes(data: Optional[bytes], what: str) -> Tuple[List[dict], Optional[str]]:
    if data is None:
        return [], None
    rows: List[dict] = []
    try:
        for i, line in enumerate(data.decode('utf-8').splitlines(), 1):
            if not line.strip():
                continue
            obj = common.loads_strict(line, what='%s:%d' % (what, i))
            if not isinstance(obj, dict):
                return rows, '%s:%d is not a JSON object' % (what, i)
            rows.append(obj)
    except (UnicodeDecodeError, InvalidInput) as e:
        return rows, str(e)
    return rows, None


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def json_pointer(obj: Any, ptr: str) -> Any:
    """RFC 6901 JSON pointer lookup; KeyError when it does not resolve."""
    if ptr == '':
        return obj
    if not isinstance(ptr, str) or not ptr.startswith('/'):
        raise KeyError(ptr)
    for raw in ptr[1:].split('/'):
        tok = raw.replace('~1', '/').replace('~0', '~')
        if isinstance(obj, dict):
            if tok not in obj:
                raise KeyError(ptr)
            obj = obj[tok]
        elif isinstance(obj, list):
            if not re.fullmatch(r'0|[1-9]\d*', tok) or int(tok) >= len(obj):
                raise KeyError(ptr)
            obj = obj[int(tok)]
        else:
            raise KeyError(ptr)
    return obj


def expect_matches(actual: Any, expected: Any) -> bool:
    """EXPECTED.json `expect` value semantics: a literal (numbers compared with 1e-9 relative tolerance,
    booleans never equal numbers), or {contains: s}, {approx: x, tol: t}, {in: [...]}."""
    if isinstance(expected, dict) and set(expected) == {'contains'}:
        needle = expected['contains']
        if isinstance(actual, str):
            return isinstance(needle, str) and needle in actual
        if isinstance(actual, list):
            return needle in actual
        return False
    if isinstance(expected, dict) and 'approx' in expected and set(expected) <= {'approx', 'tol'}:
        if not _is_number(actual) or not _is_number(expected['approx']):
            return False
        tol = expected.get('tol', 1e-9)
        return _is_number(tol) and abs(float(actual) - float(expected['approx'])) <= float(tol)
    if isinstance(expected, dict) and set(expected) == {'in'} and isinstance(expected['in'], list):
        return any(expect_matches(actual, e) for e in expected['in'])
    if isinstance(expected, bool) or isinstance(actual, bool):
        return isinstance(expected, bool) and isinstance(actual, bool) and expected == actual
    if _is_number(expected) and _is_number(actual):
        return abs(float(actual) - float(expected)) <= 1e-9 * max(1.0, abs(float(expected)))
    if isinstance(expected, list) and isinstance(actual, list):
        return len(expected) == len(actual) and all(expect_matches(a, e) for a, e in zip(actual, expected))
    if isinstance(expected, dict) and isinstance(actual, dict):
        return set(expected) == set(actual) and all(expect_matches(actual[k], expected[k]) for k in expected)
    return actual == expected


def _dir_digest(path: Path) -> str:
    h = hashlib.sha256()
    if not path.exists():
        return ''
    for dirpath, dirnames, filenames in os.walk(path):
        dirnames[:] = sorted(d for d in dirnames if d not in common.SKIP_DIR_NAMES)
        for fn in sorted(filenames):
            p = Path(dirpath, fn)
            rel = p.relative_to(path).as_posix()
            if common._is_cache_junk(rel):
                continue
            h.update(rel.encode('utf-8') + b'\0' + hashlib.sha256(p.read_bytes()).digest())
    return h.hexdigest()


def _inside(child: Any, parent: Any) -> bool:
    try:
        Path(child).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False


def _protected_rules(git: Optional[common.GitView], shas: Sequence[str], repo_root: Optional[Path] = None
                     ) -> Tuple[List[str], List[str]]:
    """Union of PROTECTED.json paths and append_only globs from the running kit, the repo's kit and the
    given commits (an edit can never unprotect itself)."""
    paths: Set[str] = set()
    append: Set[str] = set()
    blobs: List[Tuple[str, Optional[bytes]]] = []
    p = common.KIT_DIR / 'PROTECTED.json'
    blobs.append(('running kit', p.read_bytes() if p.is_file() else None))
    if repo_root is not None:
        rp = Path(repo_root) / KIT_REL / 'PROTECTED.json'
        blobs.append(('repo kit', rp.read_bytes() if rp.is_file() else None))
    for sha in shas:
        if git is not None:
            blobs.append((sha, git.file_at(sha, KIT_REL + '/PROTECTED.json')))
    for label, data in blobs:
        obj, err = _parse_json_bytes(data, 'PROTECTED.json (%s)' % label)
        if isinstance(obj, dict):
            paths.update(str(x) for x in (obj.get('paths') or []) if isinstance(x, str))
            append.update(str(x) for x in (obj.get('append_only') or []) if isinstance(x, str))
    if not paths:
        raise MissingInput('PROTECTED.json unavailable: no protected-path list')
    return sorted(paths), sorted(append)


def _matches_any(path: str, globs: Iterable[str]) -> bool:
    return any(common.glob_match(path, g) for g in globs)


def delivery_dir_of(path: str) -> Optional[str]:
    """The delivery directory a repo-relative path lies in, or None (strict: every top-level directory
    except tools/, .claude/ and .github/ is a delivery or legacy research dir; under studies/ and
    PROJECT_DOCUMENTS/ the dated or v1-named subdirectories are deliveries)."""
    parts = common.norm_rel(path).split('/')
    if len(parts) < 2:
        return None
    top = parts[0]
    if top in INFRA_TOP_DIRS:
        return None
    if top == 'PROJECT_DOCUMENTS':
        if len(parts) >= 3 and (DATE_DIR_RE.search(parts[1]) or common.V1_DIR_RE.match(parts[1])):
            return '/'.join(parts[:2])
        return None
    if top == 'studies':
        return '/'.join(parts[:2]) if len(parts) >= 3 else None
    return top


# ======================================================================================
# Git range helpers
# ======================================================================================

def resolve_rev(git: common.GitView, rev: str) -> str:
    if not isinstance(rev, str) or not rev.strip():
        raise MissingInput('empty revision')
    rev = rev.strip()
    if isinstance(git, common.TimelineGit):
        return git._commits[git._idx(rev)].sha
    if isinstance(git, common.RealGit):
        proc = common.run_git(git.root, ['rev-parse', '--verify', '--quiet', rev + '^{commit}'], ok=(0, 1, 128))
        out = proc.stdout.decode().strip()
        if proc.returncode != 0 or not out:
            raise MissingInput('unknown revision %s' % rev)
        return out
    c = git.commit(rev)
    if c is None:
        raise MissingInput('unknown revision %s' % rev)
    return c.sha


def range_commits(git: common.GitView, base: str, head: str) -> List[common.Commit]:
    """Commits in (base, head], oldest first. base must be an ancestor of head."""
    if isinstance(git, common.TimelineGit):
        bi, hi = git._idx(base), git._idx(head)
        if bi > hi:
            raise InvalidInput('base %s is not an ancestor of head %s' % (_short(base), _short(head)))
        return [git._commits[i] for i in range(bi + 1, hi + 1)]
    if isinstance(git, common.RealGit):
        proc = common.run_git(git.root, ['merge-base', '--is-ancestor', base, head], ok=(0, 1, 128))
        if proc.returncode == 128:
            raise MissingInput('cannot relate %s and %s' % (_short(base), _short(head)))
        if proc.returncode == 1:
            raise InvalidInput('base %s is not an ancestor of head %s' % (_short(base), _short(head)))
        proc = common.run_git(git.root, ['log', '--reverse', '--no-renames', '--name-only',
                                         '--format=%x1e%H%x1f%ct%x1f%an%x1f%B%x1f', '%s..%s' % (base, head)])
        out: List[common.Commit] = []
        for rec in proc.stdout.decode('utf-8', 'replace').split('\x1e'):
            if not rec.strip():
                continue
            parts = rec.split('\x1f', 4)
            if len(parts) < 5:
                raise MissingInput('unparseable git log record')
            sha, ct, author, body, rest = parts
            try:
                utc = common.utc_from_epoch(int(ct.strip()))
            except ValueError:
                utc = ''
            out.append(common.Commit(sha=sha.strip(), utc=utc, author=author,
                                     trailers=common.parse_trailers(body),
                                     paths=[ln.strip() for ln in rest.splitlines() if ln.strip()]))
        return out
    raise CheckerError('unsupported git view %r' % type(git).__name__)


def first_commit_adding_in(git: common.GitView, path: str, upto: str) -> Optional[common.Commit]:
    """First commit (in the history of `upto`) where `path` exists."""
    if isinstance(git, common.TimelineGit):
        hi = git._idx(upto)
        for i in range(hi + 1):
            c = git._commits[i]
            if path in c.paths and git.file_at(c.sha, path) is not None:
                return c
        return None
    if isinstance(git, common.RealGit):
        proc = common.run_git(git.root, ['log', '--reverse', '--no-renames', '--diff-filter=A', '--format=%H',
                                         upto, '--', path])
        shas = [ln.strip() for ln in proc.stdout.decode().splitlines() if ln.strip()]
        if not shas:
            return None
        rc = range_commits_single(git, shas[0])
        return rc
    return git.first_commit_adding(path)


def range_commits_single(git: common.RealGit, sha: str) -> common.Commit:
    proc = common.run_git(git.root, ['log', '-1', '--no-renames', '--name-only',
                                     '--format=%x1e%H%x1f%ct%x1f%an%x1f%B%x1f', sha])
    rec = proc.stdout.decode('utf-8', 'replace').split('\x1e')[-1]
    parts = rec.split('\x1f', 4)
    if len(parts) < 5:
        raise MissingInput('unparseable git log record for %s' % sha)
    s, ct, author, body, rest = parts
    try:
        utc = common.utc_from_epoch(int(ct.strip()))
    except ValueError:
        utc = ''
    return common.Commit(sha=s.strip(), utc=utc, author=author, trailers=common.parse_trailers(body),
                         paths=[ln.strip() for ln in rest.splitlines() if ln.strip()])


# ======================================================================================
# Change classification (SPEC 13.2)
# ======================================================================================

def _change(path: str, kind: str, cls: str, what: str, targets: Optional[Sequence[str]] = None,
            numbers: Optional[Iterable[float]] = None, **extra: Any) -> dict:
    d = {'path': path, 'kind': kind, 'class': cls, 'what': what,
         'targets': sorted(set(str(t) for t in (targets or []) if t)),
         'numbers': sorted(set(float(n) for n in (numbers or [])))}
    d.update(extra)
    return d


def _cmp_versions(a: Any, b: Any) -> int:
    va, vb = common.parse_version(a), common.parse_version(b)
    if va is None or vb is None:
        return 0 if a == b else 2  # 2 = incomparable
    return (va > vb) - (va < vb)


def _bundle_descriptors(rules: dict) -> Dict[str, Dict[str, tuple]]:
    """bundle -> {check id: descriptor} with extends followed, no validation (for diffs only)."""
    bundles = rules.get('bundles') if isinstance(rules.get('bundles'), dict) else {}
    out: Dict[str, Dict[str, tuple]] = {}

    def expand(name: str, stack: Tuple[str, ...]) -> Dict[str, tuple]:
        b = bundles.get(name)
        if not isinstance(b, dict) or name in stack:
            return {}
        d: Dict[str, tuple] = {}
        if isinstance(b.get('extends'), str):
            d.update(expand(b['extends'], stack + (name,)))
        for e in b.get('checks') or []:
            if isinstance(e, str):
                d[e] = ('plain', None, None, ())
            elif isinstance(e, dict) and 'group' in e:
                fb = tuple(sorted(str(x) for x in (e.get('fallback') or [])))
                for x in e.get('primary') or []:
                    d[str(x)] = ('primary', e.get('group'), e.get('when'), fb)
                for x in e.get('fallback') or []:
                    d[str(x)] = ('fallback', e.get('group'), e.get('when'), ())
            elif isinstance(e, dict) and 'id' in e:
                d[str(e['id'])] = ('plain', None, e.get('when'), ())
        return d

    for name in bundles:
        out[name] = expand(name, ())
    return out


def classify_rules(path: str, old_b: Optional[bytes], new_b: Optional[bytes]) -> List[dict]:
    old, eo = _parse_json_bytes(old_b, 'RULES.json (base)')
    new, en = _parse_json_bytes(new_b, 'RULES.json (head)')
    if new_b is None:
        return [_change(path, 'rules', 'loosen', 'RULES.json removed')]
    if eo or en or not isinstance(new, dict) or (old_b is not None and not isinstance(old, dict)):
        return [_change(path, 'rules', 'loosen', 'unclassifiable: RULES.json does not parse (%s)' % (eo or en))]
    old = old if isinstance(old, dict) else {'rules': [], 'bundles': {}}
    out: List[dict] = []

    def idx(obj: dict) -> Dict[str, dict]:
        return {str(r.get('id')): r for r in (obj.get('rules') or []) if isinstance(r, dict) and r.get('id')}
    oi, ni = idx(old), idx(new)
    for rid in sorted(set(ni) - set(oi)):
        out.append(_change(path, 'rules', 'tighten', 'new check id %s' % rid, [rid]))
    for rid in sorted(set(oi) - set(ni)):
        out.append(_change(path, 'rules', 'loosen', 'removed check id %s' % rid, [rid]))
    known = {'id', 'module', 'title', 'stages', 'tier_min', 'introduced_in', 'status', 'fm_ids', 'applies_when',
             'warn_below_tier', 'changed_in'}
    for rid in sorted(set(oi) & set(ni)):
        a, b = oi[rid], ni[rid]
        if a == b:
            continue
        sa, sb = STATUS_ORDER.get(a.get('status'), -1), STATUS_ORDER.get(b.get('status'), -1)
        if a.get('status') != b.get('status'):
            if sa < 0 or sb < 0:
                out.append(_change(path, 'rules', 'loosen', '%s status %r -> %r (unclassifiable)'
                                   % (rid, a.get('status'), b.get('status')), [rid]))
            else:
                out.append(_change(path, 'rules', 'tighten' if sb > sa else 'loosen', '%s status %s -> %s'
                                   % (rid, a.get('status'), b.get('status')), [rid]))
        st_a = set(a.get('stages') or []) if isinstance(a.get('stages'), list) else None
        st_b = set(b.get('stages') or []) if isinstance(b.get('stages'), list) else None
        if st_a is None or st_b is None:
            if a.get('stages') != b.get('stages'):
                out.append(_change(path, 'rules', 'loosen', '%s stages unclassifiable' % rid, [rid]))
        else:
            if st_a - st_b:
                out.append(_change(path, 'rules', 'loosen', '%s stage removed: %s' % (rid, ','.join(sorted(st_a - st_b))),
                                   [rid]))
            if st_b - st_a:
                out.append(_change(path, 'rules', 'tighten', '%s stage added: %s' % (rid, ','.join(sorted(st_b - st_a))),
                                   [rid]))
        ta, tb = a.get('tier_min'), b.get('tier_min')
        if ta != tb:
            if _is_number(ta) and _is_number(tb):
                out.append(_change(path, 'rules', 'loosen' if tb > ta else 'tighten', '%s tier_min %s -> %s'
                                   % (rid, ta, tb), [rid], [ta, tb]))
            else:
                out.append(_change(path, 'rules', 'loosen', '%s tier_min %r -> %r (unclassifiable)' % (rid, ta, tb),
                                   [rid], _numbers_of([ta, tb])))
        wa, wb = a.get('warn_below_tier'), b.get('warn_below_tier')
        if wa != wb:
            if wa is None and _is_number(wb):
                cls = 'loosen'
            elif wb is None and _is_number(wa):
                cls = 'tighten'
            elif _is_number(wa) and _is_number(wb):
                cls = 'loosen' if wb > wa else 'tighten'
            else:
                cls = 'loosen'
            out.append(_change(path, 'rules', cls, '%s warn_below_tier %r -> %r' % (rid, wa, wb), [rid],
                               _numbers_of([wa, wb])))
        if a.get('introduced_in') != b.get('introduced_in'):
            c = _cmp_versions(b.get('introduced_in'), a.get('introduced_in'))
            cls = 'loosen' if c in (1, 2) else 'tighten'
            out.append(_change(path, 'rules', cls, '%s introduced_in %r -> %r' % (rid, a.get('introduced_in'),
                                                                                 b.get('introduced_in')), [rid]))
        ca, cb = a.get('changed_in'), b.get('changed_in')
        if ca != cb:
            # changed_in records the release of a tightening (SPEC 20.10); held-out scoring reads it. Appending a
            # version is bookkeeping (the code change itself is classified on its own); dropping or rewriting a
            # recorded version would re-count a post-batch tightening, so it is a loosening.
            la = ca if isinstance(ca, list) else []
            lb = cb if isinstance(cb, list) else []
            appended = isinstance(cb, list) and lb[:len(la)] == la and (ca is None or isinstance(ca, list))
            # round-1 repair (verifier): an appended version is neutral only together with a change to the rule's
            # module in the same range (_analyze checks it); a version recorded for unchanged code would exclude the
            # check's held-out hits and false alarms as post_batch
            mod = b.get('module') if isinstance(b.get('module'), str) and b.get('module') else None
            extra_ci = {'changed_in_module': '%s/%s.py' % (path.rsplit('/', 1)[0] if '/' in path else '.', mod)} \
                if (appended and mod) else {}
            out.append(_change(path, 'rules', 'neutral' if appended else 'loosen', '%s changed_in %r -> %r'
                               % (rid, ca, cb), [rid], **extra_ci))
        aw, bw = a.get('applies_when'), b.get('applies_when')
        if aw != bw:
            a_always = isinstance(aw, str) and aw.strip().lower().startswith('always')
            b_always = isinstance(bw, str) and bw.strip().lower().startswith('always')
            cls = 'tighten' if (b_always and not a_always) else 'loosen'
            out.append(_change(path, 'rules', cls, '%s applies_when %r -> %r' % (rid, aw, bw), [rid]))
        if a.get('module') != b.get('module'):
            out.append(_change(path, 'rules', 'code_pending', '%s module %r -> %r' % (rid, a.get('module'),
                                                                                     b.get('module')), [rid],
                               module_change=True))
        if a.get('title') != b.get('title') or a.get('fm_ids') != b.get('fm_ids'):
            out.append(_change(path, 'rules', 'neutral', '%s title/fm_ids edited' % rid, [rid]))
        extra = sorted((set(a) | set(b)) - known)
        changed_extra = [k for k in extra if a.get(k) != b.get(k)]
        if changed_extra:
            out.append(_change(path, 'rules', 'loosen', '%s unclassifiable field change: %s'
                               % (rid, ','.join(changed_extra)), [rid]))
    ob, nb = _bundle_descriptors(old), _bundle_descriptors(new)
    for name in sorted(set(ob) - set(nb)):
        out.append(_change(path, 'rules', 'loosen', 'bundle %s removed' % name, ['bundle:' + name] + sorted(ob[name])))
    for name in sorted(set(nb) - set(ob)):
        out.append(_change(path, 'rules', 'tighten', 'bundle %s added' % name, ['bundle:' + name]))
    for name in sorted(set(ob) & set(nb)):
        a, b = ob[name], nb[name]
        for rid in sorted(set(a) - set(b)):
            out.append(_change(path, 'rules', 'loosen', '%s removed from bundle %s' % (rid, name),
                               [rid, 'bundle:' + name]))
        for rid in sorted(set(b) - set(a)):
            cls = 'loosen' if b[rid][0] == 'fallback' else 'tighten'
            out.append(_change(path, 'rules', cls, '%s added to bundle %s as %s' % (rid, name, b[rid][0]),
                               [rid, 'bundle:' + name]))
        for rid in sorted(set(a) & set(b)):
            da, db = a[rid], b[rid]
            if da == db:
                continue
            if da[0] == db[0] == 'primary' and da[1:3] == db[1:3]:
                fa, fb = set(da[3]), set(db[3])
                cls = 'tighten' if fb < fa else 'loosen'
                out.append(_change(path, 'rules', cls, '%s fallback set in bundle %s %s -> %s'
                                   % (rid, name, sorted(fa), sorted(fb)), [rid, 'bundle:' + name]))
            elif da[2] is not None and db[2] is None and da[0] == db[0]:
                out.append(_change(path, 'rules', 'tighten', '%s condition %s dropped in bundle %s' % (rid, da[2], name),
                                   [rid, 'bundle:' + name]))
            elif db[0] == 'plain' and da[0] == 'primary' and db[2] is None:
                out.append(_change(path, 'rules', 'tighten', '%s no longer has an alternative in bundle %s' % (rid, name),
                                   [rid, 'bundle:' + name]))
            else:
                out.append(_change(path, 'rules', 'loosen', '%s entry in bundle %s changed %r -> %r'
                                   % (rid, name, da, db), [rid, 'bundle:' + name]))
    for k in sorted((set(old) | set(new)) - {'rules', 'bundles'}):
        if old.get(k) != new.get(k):
            if k in ('schema', 'notes', 'checker_version'):
                out.append(_change(path, 'rules', 'neutral', 'RULES.json %s edited' % k))
            else:
                out.append(_change(path, 'rules', 'loosen', 'RULES.json unclassifiable top-level change %s' % k))
    if not out:
        out.append(_change(path, 'rules', 'neutral', 'RULES.json formatting-only change'))
    return out


def classify_expected(path: str, old_b: Optional[bytes], new_b: Optional[bytes]) -> List[dict]:
    old, eo = _parse_json_bytes(old_b, 'EXPECTED.json (base)')
    new, en = _parse_json_bytes(new_b, 'EXPECTED.json (head)')
    if new_b is None:
        return [_change(path, 'expected', 'loosen', 'EXPECTED.json removed')]
    if eo or en or not isinstance(new, dict) or (old_b is not None and not isinstance(old, dict)):
        return [_change(path, 'expected', 'loosen', 'unclassifiable: EXPECTED.json does not parse (%s)' % (eo or en))]
    old = old if isinstance(old, dict) else {}
    out: List[dict] = []
    for fid in sorted(set(new) - set(old)):
        out.append(_change(path, 'expected', 'tighten', 'new fixture %s' % fid, [fid]))
    for fid in sorted(set(old) - set(new)):
        out.append(_change(path, 'expected', 'loosen', 'fixture %s removed from EXPECTED' % fid, [fid]))
    for fid in sorted(set(old) & set(new)):
        a, b = old[fid], new[fid]
        if a == b:
            continue
        if not isinstance(a, dict) or not isinstance(b, dict):
            out.append(_change(path, 'expected', 'loosen', 'fixture %s entry unclassifiable' % fid, [fid]))
            continue
        for key in ('must_fire', 'must_not_fire', 'must_warn'):
            sa = set(a.get(key) or []) if isinstance(a.get(key), list) else None
            sb = set(b.get(key) or []) if isinstance(b.get(key), list) else None
            if sa is None or sb is None:
                if a.get(key) != b.get(key):
                    out.append(_change(path, 'expected', 'loosen', '%s %s unclassifiable' % (fid, key), [fid]))
                continue
            for rid in sorted(sa - sb):
                out.append(_change(path, 'expected', 'loosen', '%s id %s removed from %s' % (key, rid, fid), [fid, rid]))
            for rid in sorted(sb - sa):
                out.append(_change(path, 'expected', 'tighten', '%s id %s added to %s' % (key, rid, fid), [fid, rid]))
        ca, cb = a.get('commands'), b.get('commands')
        xa, xb = a.get('expected_exit'), b.get('expected_exit')
        if ca != cb or xa != xb:
            ext = (isinstance(ca, list) and isinstance(cb, list) and isinstance(xa, list) and isinstance(xb, list)
                   and len(cb) > len(ca) and cb[:len(ca)] == ca and xb[:len(xa)] == xa and len(xa) == len(ca))
            if ext:
                out.append(_change(path, 'expected', 'tighten', '%s commands extended' % fid, [fid]))
            else:
                out.append(_change(path, 'expected', 'loosen', '%s commands/expected_exit changed (unclassifiable)' % fid,
                                   [fid], _numbers_of([xa, xb])))
        ea = a.get('expect') if isinstance(a.get('expect'), dict) else {}
        eb = b.get('expect') if isinstance(b.get('expect'), dict) else {}
        for k in sorted(set(ea) - set(eb)):
            out.append(_change(path, 'expected', 'loosen', '%s expect %s removed' % (fid, k), [fid],
                               _numbers_of(ea[k])))
        for k in sorted(set(eb) - set(ea)):
            out.append(_change(path, 'expected', 'tighten', '%s expect %s added' % (fid, k), [fid]))
        for k in sorted(set(ea) & set(eb)):
            if ea[k] != eb[k]:
                out.append(_change(path, 'expected', 'loosen', '%s expect %s changed %r -> %r' % (fid, k, ea[k], eb[k]),
                                   [fid], _numbers_of([ea[k], eb[k]])))
        known = {'commands', 'expected_exit', 'must_fire', 'must_not_fire', 'must_warn', 'expect', 'source_incident',
                 'fm_ids'}
        if a.get('source_incident') != b.get('source_incident') or a.get('fm_ids') != b.get('fm_ids'):
            out.append(_change(path, 'expected', 'neutral', '%s metadata edited' % fid, [fid]))
        extra = [k for k in sorted((set(a) | set(b)) - known) if a.get(k) != b.get(k)]
        if extra:
            out.append(_change(path, 'expected', 'loosen', '%s unclassifiable field change: %s' % (fid, ','.join(extra)),
                               [fid]))
    if not out:
        out.append(_change(path, 'expected', 'neutral', 'EXPECTED.json formatting-only change'))
    return out


def classify_protected_json(path: str, old_b: Optional[bytes], new_b: Optional[bytes]) -> List[dict]:
    old, eo = _parse_json_bytes(old_b, 'PROTECTED.json (base)')
    new, en = _parse_json_bytes(new_b, 'PROTECTED.json (head)')
    if new_b is None or eo or en or not isinstance(new, dict):
        return [_change(path, 'protected', 'loosen', 'PROTECTED.json removed or unparseable')]
    old = old if isinstance(old, dict) else {}
    out = []
    for key in ('paths', 'append_only'):
        sa = set(old.get(key) or [])
        sb = set(new.get(key) or [])
        for g in sorted(sa - sb):
            out.append(_change(path, 'protected', 'loosen', '%s entry %s removed' % (key, g), [g]))
        for g in sorted(sb - sa):
            out.append(_change(path, 'protected', 'tighten', '%s entry %s added' % (key, g), [g]))
    for k in sorted((set(old) | set(new)) - {'paths', 'append_only'}):
        if old.get(k) != new.get(k):
            out.append(_change(path, 'protected', 'neutral' if k in ('schema', 'notes') else 'loosen',
                               'PROTECTED.json %s edited' % k))
    return out or [_change(path, 'protected', 'neutral', 'PROTECTED.json formatting-only change')]


def _flatten_constraints(old: Any, new: Any, prefix: str = '') -> List[Tuple[str, Any, Any]]:
    _ABSENT = '__absent__'
    out: List[Tuple[str, Any, Any]] = []
    if isinstance(old, dict) and isinstance(new, dict):
        for k in sorted(set(old) | set(new)):
            kp = (prefix + '.' + k) if prefix else str(k)
            if k not in old:
                out.append((kp, _ABSENT, new[k]))
            elif k not in new:
                out.append((kp, old[k], _ABSENT))
            elif old[k] != new[k]:
                out.extend(_flatten_constraints(old[k], new[k], kp))
        return out
    if old != new:
        out.append((prefix, old, new))
    return out


_LOWER_STRICTER_FLOORS = ('old_max_drop', 'alpha_max')
_LIST_MORE_STRICTER = ('native_forbidden', 'supplied_arithmetic_heuristics', 'required_arms_by_level',
                       'testbed_floor_grid', 'testbed_ceiling_grid')
_LIST_MORE_LOOSER = ('evaluator_only_exemptions',)
_NEUTRAL_KEYS = ('schema', 'owner', 'version', 'status', 'user_decision', 'notes')


def constraint_direction(keypath: str, old: Any, new: Any) -> str:
    """'tighten' | 'loosen' | 'neutral' | 'unknown' per the SPEC 13.2 direction map (unknown counts as
    loosening for GOV-NUM)."""
    absent = '__absent__'
    top = keypath.split('.')[0]
    leaf = keypath.split('.')[-1]
    if top in _NEUTRAL_KEYS:
        return 'neutral'
    if old == absent or new == absent:
        if isinstance(old, list) or isinstance(new, list):
            pass
        else:
            return 'unknown'
    if top in _LIST_MORE_STRICTER or top in _LIST_MORE_LOOSER:
        a = old if isinstance(old, list) else ([] if old == absent else None)
        b = new if isinstance(new, list) else ([] if new == absent else None)
        if a is None or b is None:
            return 'unknown'
        ka = set(json.dumps(x, sort_keys=True) for x in a)
        kb = set(json.dumps(x, sort_keys=True) for x in b)
        if ka == kb:
            return 'neutral'
        more = kb >= ka
        fewer = kb <= ka
        if top in _LIST_MORE_STRICTER:
            return 'tighten' if more else 'loosen'
        return 'tighten' if fewer else 'loosen'
    if top == 'threshold_floors' and leaf == 'shuf_band':
        if (isinstance(old, list) and isinstance(new, list) and len(old) == len(new) == 2
                and all(_is_number(x) for x in old + new)):
            if new[0] >= old[0] and new[1] <= old[1]:
                return 'tighten'
            return 'loosen'
        return 'unknown'
    if isinstance(old, bool) or isinstance(new, bool):
        if not (isinstance(old, bool) and isinstance(new, bool)):
            return 'unknown'
        if top == 'default_grant_caps':
            return 'loosen' if (new and not old) else 'tighten'
        if top == 'io':
            return 'loosen' if (old and not new) else 'tighten'
        if top == 'e3_cadence_counts_legacy':
            return 'tighten' if (new and not old) else 'loosen'
        return 'unknown'
    if _is_number(old) and _is_number(new):
        if top == 'threshold_floors':
            lower = leaf in _LOWER_STRICTER_FLOORS
        elif top in ('tier_thresholds', 'default_grant_caps', 'e3_cadence_max', 'review_debt_max_open'):
            lower = True
        elif leaf.startswith('min_') or leaf.endswith('_min'):
            lower = False
        elif leaf.startswith('max_') or leaf.endswith('_max'):
            lower = True
        else:
            return 'unknown'
        if new == old:
            return 'neutral'
        stricter = (new < old) if lower else (new > old)
        return 'tighten' if stricter else 'loosen'
    return 'unknown'


def classify_constraints(path: str, old_b: Optional[bytes], new_b: Optional[bytes]) -> List[dict]:
    old, eo = _parse_json_bytes(old_b, 'CONSTRAINTS.json (base)')
    new, en = _parse_json_bytes(new_b, 'CONSTRAINTS.json (head)')
    if eo or en or new_b is None or not isinstance(new, dict):
        return [_change(path, 'constraints', 'constraint', 'CONSTRAINTS.json removed or unparseable',
                        ['constraint:*'], direction='unknown', key='*')]
    old = old if isinstance(old, dict) else {}
    out = []
    for kp, a, b in _flatten_constraints(old, new):
        d = constraint_direction(kp, a, b)
        nums = _numbers_of([x for x in (a, b) if x != '__absent__'])
        out.append(_change(path, 'constraints', 'constraint', 'CONSTRAINTS %s %s -> %s (%s)'
                           % (kp, json.dumps(a)[:80], json.dumps(b)[:80], d), ['constraint:' + kp], nums,
                           direction=d, key=kp))
    return out or [_change(path, 'constraints', 'constraint', 'CONSTRAINTS.json formatting-only change',
                           ['constraint:*'], direction='neutral', key='*')]


def classify_text(path: str, kind: str, added: List[str], removed: List[str], existed_before: bool,
                  exists_after: bool, targets: Sequence[str] = ()) -> List[dict]:
    tg = [path] + list(targets)
    if not exists_after:
        return [_change(path, kind, 'loosen', 'file removed', tg, common.numbers_in_text('\n'.join(removed)))]
    if kind == 'gitattributes':
        rule_removed = [ln for ln in removed if ln.strip() and not ln.strip().startswith('#')]
    elif kind == 'tests':
        rule_removed = [ln for ln in removed if TEST_LINE_RE.search(ln)]
    else:
        rule_removed = [ln for ln in removed if RULE_LINE_RE.search(ln)]
    nums_r = common.numbers_in_text('\n'.join(removed))
    nums_a = common.numbers_in_text('\n'.join(added))
    changed_nums = _multiset_symdiff(nums_r, nums_a)
    if rule_removed:
        return [_change(path, kind, 'loosen', 'rule line removed: %s' % rule_removed[0].strip()[:120], tg,
                        changed_nums, removed_rule_lines=[ln.strip()[:200] for ln in rule_removed[:20]])]
    what = 'new file' if not existed_before else 'text edit with no rule line removed'
    return [_change(path, kind, 'neutral', what, tg, changed_nums)]


def _multiset_symdiff(a: List[float], b: List[float]) -> List[float]:
    from collections import Counter
    ca, cb = Counter(a), Counter(b)
    return sorted(set(list((ca - cb).elements()) + list((cb - ca).elements())))


def _path_kind(path: str) -> str:
    p = path
    if p == CONSTRAINTS_REL:
        return 'constraints'
    if p == USER_DECISIONS_REL:
        return 'user_decisions'
    if p == '.gitattributes':
        return 'gitattributes'
    if p in (AGENTS_POINTER_REL, ROOT_AGENTS_REL):
        return 'pointer'
    if p.startswith('.claude/skills/'):
        return 'skill' if p.endswith('.md') else 'other'
    if p.startswith(KIT_REL + '/'):
        rel = p[len(KIT_REL) + 1:]
        if rel == 'RULES.json':
            return 'rules'
        if rel == 'EXPECTED.json':
            return 'expected'
        if rel == 'PROTECTED.json':
            return 'protected'
        if rel.startswith('corpus/'):
            return 'corpus'
        if rel.startswith('tests/'):
            return 'tests' if rel.endswith('.py') else ('doc' if rel.endswith('.md') else 'other')
        if rel.startswith('templates/'):
            return 'template'
        if rel.endswith('.py'):
            return 'code'
        if rel.endswith('.md'):
            return 'doc'
        return 'other'
    return 'other'


def _corpus_fixture_of(path: str) -> Optional[str]:
    pre = KIT_REL + '/corpus/'
    if not path.startswith(pre):
        return None
    rest = path[len(pre):]
    return rest.split('/', 1)[0] if '/' in rest else None


def classify_paths(git: common.GitView, base: str, head: str, paths: Sequence[str]) -> List[dict]:
    """Classify every changed protected path (SPEC 13.2). Code-like edits get class 'code_pending',
    resolved later by the observed fired-set comparison (GOV-MONO)."""
    changes: List[dict] = []
    fixture_paths: Dict[str, List[str]] = {}
    for path in sorted(paths):
        kind = _path_kind(path)
        if kind == 'corpus':
            fid = _corpus_fixture_of(path)
            if fid is None:
                old_b, new_b = git.file_at(base, path), git.file_at(head, path)
                if path.endswith('.md'):
                    added, removed = git.diff_lines(base, head, path)
                    changes.extend(classify_text(path, 'doc', added, removed, old_b is not None, new_b is not None))
                else:
                    changes.append(_change(path, 'corpus', 'loosen', 'unclassifiable corpus file', [path]))
                continue
            fixture_paths.setdefault(fid, []).append(path)
            continue
        old_b, new_b = git.file_at(base, path), git.file_at(head, path)
        if kind == 'rules':
            changes.extend(classify_rules(path, old_b, new_b))
        elif kind == 'expected':
            changes.extend(classify_expected(path, old_b, new_b))
        elif kind == 'protected':
            changes.extend(classify_protected_json(path, old_b, new_b))
        elif kind == 'constraints':
            changes.extend(classify_constraints(path, old_b, new_b))
        elif kind == 'user_decisions':
            added, removed = git.diff_lines(base, head, path)
            if removed or new_b is None:
                changes.append(_change(path, kind, 'loosen', 'USER_DECISIONS.jsonl lines removed or file deleted',
                                       [path]))
            else:
                ids = []
                for ln in added:
                    obj, err = _parse_json_bytes(ln.encode('utf-8'), 'USER_DECISIONS row') if ln.strip() else (None, None)
                    if isinstance(obj, dict) and obj.get('id'):
                        ids.append(str(obj['id']))
                    elif ln.strip():
                        changes.append(_change(path, kind, 'loosen', 'unparseable USER_DECISIONS row added', [path]))
                changes.append(_change(path, kind, 'owner_decision', 'owner decision rows added: %s'
                                       % (', '.join(ids) or 'none'), [path], ud_ids=ids))
        elif kind in ('skill', 'pointer', 'gitattributes', 'doc', 'template', 'tests'):
            added, removed = git.diff_lines(base, head, path)
            tg = []
            if kind == 'skill':
                parts = path.split('/')
                if len(parts) > 2:
                    tg.append(parts[2])
            changes.extend(classify_text(path, kind, added, removed, old_b is not None, new_b is not None, tg))
        elif kind == 'code':
            if new_b is None:
                changes.append(_change(path, kind, 'code_pending', 'kit module removed', [path]))
            else:
                changes.append(_change(path, kind, 'code_pending', 'check code edit' if old_b is not None
                                       else 'new kit module', [path]))
        else:
            changes.append(_change(path, kind, 'loosen', 'unclassifiable protected change', [path]))
    for fid, fpaths in sorted(fixture_paths.items()):
        if fid in CORPUS_SKIP:
            for path in fpaths:
                old_b, new_b = git.file_at(base, path), git.file_at(head, path)
                if fid == 'heldout' and '/sealed/' in path:
                    if old_b is None and new_b is not None:
                        changes.append(_change(path, 'corpus', 'neutral', 'sealed held-out batch added', [path]))
                    else:
                        changes.append(_change(path, 'corpus', 'loosen', 'sealed held-out batch modified or removed',
                                               [path]))
                elif path.endswith('.md') or fid == '_build':
                    added, removed = git.diff_lines(base, head, path)
                    if fid == '_build' and not path.endswith('.md'):
                        changes.append(_change(path, 'corpus', 'neutral' if new_b is not None else 'loosen',
                                               'corpus builder edit (selftest never runs it)', [path]))
                    else:
                        changes.extend(classify_text(path, 'doc', added, removed, old_b is not None, new_b is not None))
                else:
                    changes.append(_change(path, 'corpus', 'loosen', 'unclassifiable held-out file', [path]))
            continue
        pre = '%s/corpus/%s' % (KIT_REL, fid)
        at_base = bool(git.list_files(base, pre))
        at_head = bool(git.list_files(head, pre))
        if at_head and not at_base:
            changes.append(_change(pre, 'corpus', 'tighten', 'new fixture %s' % fid, [fid], fixture=fid,
                                   fixture_change='added'))
        elif at_base and not at_head:
            changes.append(_change(pre, 'corpus', 'loosen', 'fixture %s removed' % fid, [fid], fixture=fid,
                                   fixture_change='removed'))
        else:
            changes.append(_change(pre, 'corpus', 'code_pending', 'fixture %s modified (%d files)' % (fid, len(fpaths)),
                                   [fid], fixture=fid, fixture_change='modified'))
    return changes


# ======================================================================================
# Corpus runs (shared by selftest, GOV-MONO and heldout score)
# ======================================================================================

def corpus_fixtures(corpus_dir: Path) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    if not corpus_dir.is_dir():
        return out
    for p in sorted(corpus_dir.iterdir()):
        if p.is_dir() and p.name not in CORPUS_SKIP and not p.name.startswith(('.', '__')) \
                and (p / 'timeline.json').is_file():
            out[p.name] = p
    return out


def fixture_is_clean(fid: str, fixture_dir: Optional[Path], entry: Optional[dict] = None) -> bool:
    kind = None
    if fixture_dir is not None:
        fj = fixture_dir / 'FIXTURE.json'
        if fj.is_file():
            try:
                obj = common.load_json_file(fj)
                kind = obj.get('kind') if isinstance(obj, dict) else None
            except (MissingInput, InvalidInput):
                kind = None
    if kind in ('clean', 'dirty'):
        return kind == 'clean'
    if isinstance(entry, dict) and 'clean' in (entry.get('fm_ids') or []):
        return True
    return bool(re.match(r'^C00[a-z]?$', fid))


def validate_expected(expected: Any, known_ids: Optional[Set[str]] = None) -> Dict[str, List[str]]:
    """Structural problems per fixture id ('' for file-level)."""
    probs: Dict[str, List[str]] = {}
    if not isinstance(expected, dict):
        return {'': ['EXPECTED.json must be an object {fixture_id: entry}']}
    for fid, e in expected.items():
        p: List[str] = []
        if not isinstance(e, dict):
            probs[fid] = ['entry is not an object']
            continue
        cmds = e.get('commands')
        if not isinstance(cmds, list) or not cmds or not all(
                isinstance(c, list) and c and all(isinstance(t, str) for t in c) for c in cmds):
            p.append('commands must be a non-empty list of argv lists of strings')
        else:
            for c in cmds:
                if any(t in ('--repo-root', '--git') or t.startswith(('--repo-root=', '--git=')) for t in c):
                    p.append('commands must not set --repo-root or --git (selftest sets them)')
        xe = e.get('expected_exit')
        if not isinstance(xe, list) or not all(isinstance(x, int) and not isinstance(x, bool) and 0 <= x <= 3 for x in xe):
            p.append('expected_exit must be a list of exit codes 0..3')
        elif isinstance(cmds, list) and len(xe) != len(cmds):
            p.append('expected_exit has %d entries for %d commands' % (len(xe), len(cmds)))
        for key in ('must_fire', 'must_not_fire', 'must_warn'):
            v = e.get(key, [])
            if not isinstance(v, list) or not all(isinstance(x, str) for x in v):
                p.append('%s must be a list of check ids' % key)
            elif known_ids is not None:
                unknown = [x for x in v if x not in known_ids]
                if unknown:
                    p.append('%s names ids absent from RULES.json: %s' % (key, ','.join(unknown)))
        if 'expect' in e and not isinstance(e.get('expect'), dict):
            p.append('expect must be an object')
        if p:
            probs[fid] = p
    return probs


def _collect_results(obj: Any) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    if isinstance(obj, dict) and isinstance(obj.get('results'), list):
        for r in obj['results']:
            if isinstance(r, dict) and isinstance(r.get('id'), str) and isinstance(r.get('status'), str):
                out.append((r['id'], r['status']))
    return out


def _snapshot(root: Path) -> Dict[str, Tuple[int, int, str]]:
    snap: Dict[str, Tuple[int, int, str]] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in common.SKIP_DIR_NAMES]
        for fn in filenames:
            p = Path(dirpath, fn)
            rel = p.relative_to(root).as_posix()
            if common._is_cache_junk(rel):
                continue
            try:
                st = p.stat()
                if st.st_size <= 8 * 1024 * 1024:
                    h = hashlib.sha256(p.read_bytes()).hexdigest()
                else:
                    h = 'large'
            except OSError:
                continue
            snap[rel] = (st.st_size, st.st_mtime_ns, h)
    return snap


def _is_report_rel(rel: str, obj: Any) -> bool:
    parts = rel.split('/')
    if 'checks' in parts[:-1]:
        return True
    return isinstance(obj, dict) and str(obj.get('schema', '')).startswith('minifly.gate_report')


class RunOutcome:
    """What one fixture's commands produced."""

    def __init__(self) -> None:
        self.exits: List[int] = []
        self.statuses: Dict[str, Set[str]] = {}
        self.per_command: List[List[Tuple[str, str]]] = []
        self.produced: Dict[str, Any] = {}      # repo-relative JSON artifacts written by the commands (not reports)
        self.final_json: Dict[str, Any] = {}    # every JSON file of the fixture repo after the last command
        self.errors: List[str] = []
        self.exit3_errors: List[Tuple[int, str]] = []   # (command index, stderr tail) of commands that exited 3
        self.runtime_s = 0.0

    @property
    def fired(self) -> Set[str]:
        return set(i for i, s in self.statuses.items() if s & {FAIL, MISSING})

    @property
    def warned(self) -> Set[str]:
        return set(i for i, s in self.statuses.items() if WARN in s)

    def to_dict(self) -> dict:
        return {'exits': list(self.exits), 'fired': sorted(self.fired), 'warn': sorted(self.warned),
                'statuses': {k: sorted(v) for k, v in sorted(self.statuses.items())},
                'errors': list(self.errors) + [m for _i, m in self.exit3_errors], 'runtime_s': round(self.runtime_s, 3)}


Invoker = Callable[[List[str], Path, Path], Tuple[int, str, str, List[Tuple[str, str]]]]


def _inprocess_invoker() -> Invoker:
    """Run `minifly_check.main(argv)` in this process with captured stdio; also captures the results of
    the outermost run_bundle call made for the fixture copy."""
    def invoke(argv: List[str], cwd: Path, copy_root: Path) -> Tuple[int, str, str, List[Tuple[str, str]]]:
        mc = importlib.import_module('minifly_check')
        captured: List[Tuple[str, str]] = []
        depth = {'n': 0}
        orig = common.run_bundle

        def wrapped(ctx, stage, *a, **kw):
            depth['n'] += 1
            try:
                res = orig(ctx, stage, *a, **kw)
            finally:
                depth['n'] -= 1
            # only bundles run on the fixture copy itself count (canary mutants run on temp copies elsewhere)
            dl = getattr(ctx, 'delivery', None)
            if depth['n'] == 0 and _inside(getattr(ctx, 'repo_root', '/nonexistent'), copy_root) \
                    and (dl is None or _inside(dl, copy_root)):
                captured.extend((r.id, r.status) for r in res if isinstance(r, Result))
            return res
        out_b, err_b = io.BytesIO(), io.BytesIO()
        out_t = io.TextIOWrapper(out_b, encoding='utf-8', errors='replace', write_through=True)
        err_t = io.TextIOWrapper(err_b, encoding='utf-8', errors='replace', write_through=True)
        old_cwd = os.getcwd()
        common.run_bundle = wrapped
        rc = common.EXIT_ERROR
        try:
            os.chdir(str(cwd))
            with contextlib.redirect_stdout(out_t), contextlib.redirect_stderr(err_t):
                try:
                    rc = mc.main(list(argv))
                except SystemExit as e:  # pragma: no cover - main handles SystemExit itself
                    rc = e.code if isinstance(e.code, int) else common.EXIT_ERROR
                except Exception:  # noqa: BLE001
                    traceback.print_exc(file=sys.stderr)
                    rc = common.EXIT_ERROR
        finally:
            common.run_bundle = orig
            os.chdir(old_cwd)
            out_t.flush()
            err_t.flush()
        return (rc if isinstance(rc, int) else common.EXIT_ERROR, out_b.getvalue().decode('utf-8', 'replace'),
                err_b.getvalue().decode('utf-8', 'replace'), captured)
    return invoke


def _subprocess_invoker(kit_dir: Path, env_extra: Optional[Dict[str, str]] = None,
                        timeout: float = FIXTURE_CMD_TIMEOUT_S) -> Invoker:
    entry = Path(kit_dir) / 'minifly_check.py'

    def invoke(argv: List[str], cwd: Path, copy_root: Path) -> Tuple[int, str, str, List[Tuple[str, str]]]:
        env = dict(os.environ)
        env.update(env_extra or {})
        env[GOV_DEPTH_ENV] = str(int(os.environ.get(GOV_DEPTH_ENV, '0') or 0) + 1)
        try:
            p = subprocess.run([sys.executable, '-I', str(entry)] + list(argv), cwd=str(cwd), stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, timeout=timeout, env=env)
        except subprocess.TimeoutExpired:
            return common.EXIT_ERROR, '', 'timed out after %ss' % timeout, []
        except OSError as e:
            return common.EXIT_ERROR, '', 'could not start checker: %s' % e, []
        return p.returncode, p.stdout.decode('utf-8', 'replace'), p.stderr.decode('utf-8', 'replace'), []
    return invoke


def run_fixture(fixture_dir: Path, commands: Sequence[Sequence[str]], invoke: Invoker) -> RunOutcome:
    """Copy the fixture to a temp dir and run each command there in order (gates write reports into the
    copy, never into the corpus). Fired/WARN ids come from report files written under the copy, the
    command's --json stdout and (in-process) the outermost run_bundle results."""
    oc = RunOutcome()
    t0 = time.perf_counter()
    tmp = Path(tempfile.mkdtemp(prefix='minifly_fx_'))
    try:
        copy = tmp / Path(fixture_dir).name
        shutil.copytree(str(fixture_dir), str(copy), symlinks=True)
        repo = copy / 'repo'
        if not repo.is_dir():
            oc.errors.append('fixture has no repo/ directory')
            return oc
        try:
            # committed content stays fixed while the commands rewrite the worktree, as in real git
            common.TimelineGit.freeze(copy)
        except (InvalidInput, MissingInput, OSError) as e:
            oc.errors.append('fixture timeline unusable: %s' % e)
            return oc
        for cmd in commands:
            before = _snapshot(repo)
            argv = ['--repo-root', str(repo), '--git', 'timeline:%s' % copy, '--json'] + list(cmd)
            rc, out, err, captured = invoke(argv, repo, copy)
            oc.exits.append(rc)
            after = _snapshot(repo)
            results: List[Tuple[str, str]] = list(captured)
            wrote_artifact = False
            for rel, sig in sorted(after.items()):
                if before.get(rel) == sig or not rel.endswith('.json'):
                    continue
                try:
                    obj = common.loads_strict((repo / rel).read_bytes(), what=rel)
                except (InvalidInput, OSError):
                    continue
                if _is_report_rel(rel, obj):
                    results.extend(_collect_results(obj))
                else:
                    oc.produced[rel] = obj
                    wrote_artifact = True
            try:
                sobj = json.loads(out) if out.strip().startswith('{') else None
            except ValueError:
                sobj = None
            results.extend(_collect_results(sobj))
            # a command whose product is only its --json stdout (e.g. `kb score` without --out) contributes
            # that object as an artifact; report-shaped stdout (results[]) never does
            if (not wrote_artifact and isinstance(sobj, dict) and not isinstance(sobj.get('results'), list)):
                oc.produced['<stdout of command %d>' % len(oc.exits)] = sobj
            if rc == common.EXIT_ERROR and err.strip():
                oc.exit3_errors.append((len(oc.exits) - 1, 'command %s exit 3: %s'
                                        % (' '.join(cmd), err.strip().splitlines()[-1][:300])))
            oc.per_command.append(sorted(set(results)))
            for rid, st in results:
                oc.statuses.setdefault(rid, set()).add(st)
        # explicit-path expectations ('path#/pointer') read files as they are at the end of the run
        oc.final_json = {rel: _safe_load(repo / rel) for rel in _snapshot(repo) if rel.endswith('.json')}
        return oc
    finally:
        oc.runtime_s = time.perf_counter() - t0
        shutil.rmtree(str(tmp), ignore_errors=True)


def _safe_load(p: Path) -> Any:
    try:
        if p.stat().st_size > 8 * 1024 * 1024:
            return None
        return common.loads_strict(p.read_bytes(), what=str(p))
    except (OSError, InvalidInput):
        return None


def evaluate_fixture(fid: str, entry: dict, oc: RunOutcome, clean: bool) -> List[str]:
    """Problems of one fixture against its EXPECTED entry (SPEC 7.7)."""
    probs: List[str] = list(oc.errors)
    xe = entry.get('expected_exit') or []
    cmds = entry.get('commands') or []
    # a CHECKER_ERROR is a problem unless the fixture expects exactly that exit for that command
    probs.extend(msg for i, msg in oc.exit3_errors if not (i < len(xe) and xe[i] == common.EXIT_ERROR))
    if len(oc.exits) != len(xe):
        probs.append('ran %d commands but expected_exit has %d entries' % (len(oc.exits), len(xe)))
    for i, (got, want) in enumerate(zip(oc.exits, xe)):
        if got != want:
            label = ' '.join(cmds[i]) if i < len(cmds) else '?'
            probs.append('command %d (%s) exited %d, expected %d' % (i + 1, label, got, want))
    fired, warned = oc.fired, oc.warned
    miss = [x for x in entry.get('must_fire') or [] if x not in fired]
    if miss:
        probs.append('must_fire not fired: %s' % ','.join(miss))
    bad = [x for x in entry.get('must_not_fire') or [] if x in fired]
    if bad:
        probs.append('must_not_fire fired: %s' % ','.join(bad))
    nw = [x for x in entry.get('must_warn') or [] if x not in warned]
    if nw:
        probs.append('must_warn not WARN: %s' % ','.join(nw))
    for key, want in sorted((entry.get('expect') or {}).items()):
        probs.extend(_check_expect(key, want, oc))
    if clean:
        if fired:
            probs.append('clean fixture fired: %s' % ','.join(sorted(fired)))
        if any(x != 0 for x in oc.exits):
            probs.append('clean fixture exit codes %s (all must be 0)' % oc.exits)
    return probs


def _check_expect(key: str, want: Any, oc: RunOutcome) -> List[str]:
    """A bare '/pointer' must resolve in at least one JSON artifact written by the fixture's commands
    (gate/canary reports under checks/ excluded) and hold in every one where it resolves; 'path#/pointer'
    reads that repo-relative file as it is after the last command."""
    lookup = oc.final_json
    if '#' in key:
        path, ptr = key.split('#', 1)
        path = path.strip().strip('/')
        cands = [r for r in lookup if r == path] or [r for r in lookup if r.endswith('/' + path)]
        if not cands:
            return ['expect %s: file not found in the fixture after the run' % key]
        if len(cands) > 1:
            return ['expect %s: ambiguous file (%s)' % (key, ', '.join(cands[:4]))]
        try:
            got = json_pointer(lookup[cands[0]], ptr)
        except KeyError:
            return ['expect %s: pointer does not resolve in %s' % (key, cands[0])]
        return [] if expect_matches(got, want) else ['expect %s: got %r, want %r' % (key, got, want)]
    resolved = []
    for rel, obj in sorted(oc.produced.items()):
        try:
            resolved.append((rel, json_pointer(obj, key)))
        except KeyError:
            continue
    if not resolved:
        return ['expect %s: no artifact produced by the commands resolves this pointer' % key]
    bad = [(rel, got) for rel, got in resolved if not expect_matches(got, want)]
    if bad:
        return ['expect %s: %s has %r, want %r' % (key, bad[0][0], bad[0][1], want)]
    return []


def load_expected_file(path: Path) -> dict:
    obj = common.load_json_file(path)
    if not isinstance(obj, dict):
        raise InvalidInput('%s must be an object' % path)
    return obj


# ======================================================================================
# Kit materialization (git archive / timeline) for base, head and held-out checkers
# ======================================================================================

def _safe_extract_tar(data: bytes, dest: Path) -> int:
    n = 0
    with tarfile.open(fileobj=io.BytesIO(data), mode='r:') as tf:
        for m in tf.getmembers():
            name = m.name
            if name.startswith('/') or '..' in name.split('/') or '\\' in name:
                raise InvalidInput('unsafe path in git archive: %r' % name)
            target = dest / name
            if m.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif m.isreg():
                target.parent.mkdir(parents=True, exist_ok=True)
                fh = tf.extractfile(m)
                target.write_bytes(fh.read() if fh else b'')
                n += 1
            # symlinks, hard links and devices are skipped (fail closed: the kit has none)
    return n


def materialize_kit(git: common.GitView, commit: str, dest_root: Path, other_commit: Optional[str] = None
                    ) -> Tuple[Path, str]:
    """Write the kit tree at `commit` to dest_root/tools/minifly_check. Real git: `git archive`.
    Timeline: from the fixture history; when that tree is not a runnable checker it is overlaid on a copy
    of the running kit without its corpus (fixture files win; files the fixture deletes are removed).
    Returns (kit_dir, note)."""
    kit = Path(dest_root) / KIT_REL
    if isinstance(git, common.RealGit):
        proc = common.run_git(git.root, ['archive', '--format=tar', commit, KIT_REL], allow_archive=True, timeout=120)
        _safe_extract_tar(proc.stdout, Path(dest_root))
        return kit, 'git archive %s %s' % (_short(commit), KIT_REL)
    files = [p for p in git.list_files(commit, KIT_REL) if not common._is_cache_junk(p)]
    blobs = git.files_at(commit, files)
    rels = {p[len(KIT_REL) + 1:]: blobs.get(p) for p in files}
    runnable = all(rels.get(x) is not None for x in ('minifly_check.py', 'common.py', 'RULES.json'))
    note = 'timeline kit tree at %s' % _short(commit)
    if not runnable:
        for rel, p in common.checker_files(common.KIT_DIR):
            if rel.startswith('corpus/'):
                continue
            t = kit / rel
            t.parent.mkdir(parents=True, exist_ok=True)
            t.write_bytes(p.read_bytes())
        note = 'running kit (without its corpus) overlaid with the timeline kit tree at %s' % _short(commit)
        if other_commit is not None:
            for p in git.list_files(other_commit, KIT_REL):
                rel = p[len(KIT_REL) + 1:]
                if rel not in rels and (kit / rel).is_file() and git.file_at(commit, p) is None:
                    (kit / rel).unlink()
    for rel, data in rels.items():
        if data is None:
            continue
        t = kit / rel
        t.parent.mkdir(parents=True, exist_ok=True)
        t.write_bytes(data)
    return kit, note


def kit_runnable(kit: Path) -> bool:
    return all((kit / x).is_file() for x in ('minifly_check.py', 'common.py', 'RULES.json'))


# ======================================================================================
# Observed monotonicity (GOV-MONO, code_pending resolution)
# ======================================================================================

def _run_corpus_jobs(jobs: List[Tuple[str, Path, List[List[str]], Invoker]]) -> Dict[str, RunOutcome]:
    out: Dict[str, RunOutcome] = {}
    workers = max(1, min(4, os.cpu_count() or 1, len(jobs)))
    if workers == 1:
        for key, fx, cmds, inv in jobs:
            out[key] = run_fixture(fx, cmds, inv)
        return out
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_fixture, fx, cmds, inv): key for key, fx, cmds, inv in jobs}
        for f in concurrent.futures.as_completed(futs):
            out[futs[f]] = f.result()
    return out


def observe_monotone(git: common.GitView, base: str, head: str, work: Path,
                     changed_fixtures: Optional[Set[str]] = None) -> dict:
    """Run the base and head checkers on the base and head corpora (temp copies, subprocess, python3 -I)
    and compare fired sets: for every base fixture fired(head) must contain fired(base), a fixture's head
    version must still fire what its base version fired, and clean fixtures stay clean."""
    res: dict = {'ok': False, 'problems': [], 'notes': [], 'runs': 0}
    depth = int(os.environ.get(GOV_DEPTH_ENV, '0') or 0)
    if depth >= GOV_DEPTH_MAX:
        res['problems'].append('nested governance depth %d reached: cannot run base/head checkers' % depth)
        return res
    kb_dir, kh_dir = work / 'kit_base', work / 'kit_head'
    try:
        kit_b, note_b = materialize_kit(git, base, kb_dir, other_commit=head)
        kit_h, note_h = materialize_kit(git, head, kh_dir, other_commit=base)
    except (MissingInput, InvalidInput, CheckerError, OSError) as e:
        res['problems'].append('cannot materialize base/head checkers: %s' % e)
        return res
    res['notes'] += ['base: ' + note_b, 'head: ' + note_h]
    for label, k in (('base', kit_b), ('head', kit_h)):
        if not kit_runnable(k):
            res['problems'].append('%s checker is not runnable (minifly_check.py, common.py or RULES.json missing)'
                                   % label)
    if res['problems']:
        return res

    def code_digest(k: Path) -> str:
        h = hashlib.sha256()
        for rel, p in common.checker_files(k):
            if rel.startswith('corpus/') or rel == 'EXPECTED.json':
                continue
            h.update(rel.encode() + b'\0' + hashlib.sha256(p.read_bytes()).digest())
        return h.hexdigest()
    same_code = code_digest(kit_b) == code_digest(kit_h)
    corp = {}
    for label, k in (('base', kit_b), ('head', kit_h)):
        exp_p = k / 'EXPECTED.json'
        try:
            exp = load_expected_file(exp_p) if exp_p.is_file() else {}
        except (MissingInput, InvalidInput) as e:
            res['problems'].append('%s EXPECTED.json unreadable: %s' % (label, e))
            return res
        fx = corpus_fixtures(k / 'corpus')
        corp[label] = (fx, exp)
    fx_b, exp_b = corp['base']
    fx_h, exp_h = corp['head']
    if not fx_b and not fx_h:
        res['problems'].append('no corpus fixtures at base or head: monotonicity cannot be established')
        return res
    inv_b = _subprocess_invoker(kit_b)
    inv_h = _subprocess_invoker(kit_h)
    jobs: List[Tuple[str, Path, List[List[str]], Invoker]] = []
    digests_b = {f: _dir_digest(p) for f, p in fx_b.items()}
    digests_h = {f: _dir_digest(p) for f, p in fx_h.items()}

    def cmds_for(exp: dict, f: str) -> Optional[List[List[str]]]:
        e = exp.get(f)
        if not isinstance(e, dict) or not isinstance(e.get('commands'), list):
            return None
        return [list(c) for c in e['commands'] if isinstance(c, list)]
    for f, p in fx_b.items():
        c = cmds_for(exp_b, f)
        if c is None:
            res['problems'].append('base fixture %s has no EXPECTED commands' % f)
            continue
        jobs.append(('head|base|' + f, p, c, inv_h))
        if not same_code:
            jobs.append(('base|base|' + f, p, c, inv_b))
    for f, p in fx_h.items():
        c = cmds_for(exp_h, f)
        if c is None:
            res['problems'].append('head fixture %s has no EXPECTED commands' % f)
            continue
        if f in fx_b and digests_b[f] == digests_h[f] and cmds_for(exp_b, f) == c:
            continue  # identical fixture and commands: the base-corpus runs cover it
        jobs.append(('head|head|' + f, p, c, inv_h))
        if not same_code:
            jobs.append(('base|head|' + f, p, c, inv_b))
    outs = _run_corpus_jobs(jobs)
    res['runs'] = sum(len(o.exits) for o in outs.values())

    def get(k: str, c: str, f: str) -> Optional[RunOutcome]:
        if same_code:
            k = 'head'  # identical checker code: the base runs equal the head runs
        if c == 'head' and f in fx_b and ('%s|head|%s' % (k, f)) not in outs:
            c = 'base'  # identical fixture at base and head
        return outs.get('%s|%s|%s' % (k, c, f))

    def compare(label: str, before: Optional[RunOutcome], after: Optional[RunOutcome]) -> None:
        if before is None or after is None:
            return
        lost = sorted(before.fired - after.fired)
        if lost:
            res['problems'].append('%s: no longer fires %s' % (label, ','.join(lost)))
        for i, (eb, eh) in enumerate(zip(before.exits, after.exits)):
            if eb != 0 and eh == 0:
                res['problems'].append('%s: command %d exit %d -> 0' % (label, i + 1, eb))
            elif eh == common.EXIT_ERROR and eb != common.EXIT_ERROR:
                res['problems'].append('%s: command %d exit %d -> 3 (head checker error)' % (label, i + 1, eb))
    for f in sorted(fx_b):
        compare('base fixture %s, base vs head checker' % f, get('base', 'base', f), get('head', 'base', f))
        hb = get('head', 'base', f)
        if hb is not None and fixture_is_clean(f, fx_b[f], exp_b.get(f)) and (hb.fired or any(hb.exits)):
            res['problems'].append('clean base fixture %s is not clean under the head checker (fired %s, exits %s)'
                                   % (f, ','.join(sorted(hb.fired)) or '-', hb.exits))
    for f in sorted(fx_h):
        compare('head fixture %s, base vs head checker' % f, get('base', 'head', f), get('head', 'head', f))
        if f in fx_b:
            compare('fixture %s, base version vs head version' % f, get('base', 'base', f), get('head', 'head', f))
        hh = get('head', 'head', f)
        if hh is not None and fixture_is_clean(f, fx_h[f], exp_h.get(f)) and (hh.fired or any(hh.exits)):
            res['problems'].append('clean head fixture %s is not clean under the head checker (fired %s, exits %s)'
                                   % (f, ','.join(sorted(hh.fired)) or '-', hh.exits))
    res['ok'] = not res['problems']
    res['same_code'] = same_code
    res['fixtures'] = {'base': sorted(fx_b), 'head': sorted(fx_h)}
    return res


def declared_monotone(git: common.GitView, base: str, head: str) -> List[str]:
    """EXPECTED-level monotonicity: every base fixture stays, keeps its must_fire ids, and clean fixtures
    stay clean (no must_fire, every expected exit 0)."""
    path = KIT_REL + '/EXPECTED.json'
    ob, nb = git.file_at(base, path), git.file_at(head, path)
    if ob == nb:
        return []
    old, eo = _parse_json_bytes(ob, 'EXPECTED.json (base)')
    new, en = _parse_json_bytes(nb, 'EXPECTED.json (head)')
    if eo or en:
        return ['EXPECTED.json does not parse (%s)' % (eo or en)]
    old = old if isinstance(old, dict) else {}
    if not isinstance(new, dict):
        return ['EXPECTED.json removed or not an object at head']
    probs = []
    for fid, a in sorted(old.items()):
        b = new.get(fid)
        if b is None:
            probs.append('base fixture %s removed from EXPECTED' % fid)
            continue
        if not isinstance(a, dict) or not isinstance(b, dict):
            continue
        lost = sorted(set(a.get('must_fire') or []) - set(b.get('must_fire') or []))
        if lost:
            probs.append('fixture %s must_fire lost %s' % (fid, ','.join(lost)))
        was_clean = not (a.get('must_fire') or []) and all(x == 0 for x in (a.get('expected_exit') or []))
        if was_clean and ((b.get('must_fire') or []) or any(x != 0 for x in (b.get('expected_exit') or []))):
            probs.append('clean fixture %s no longer expected clean' % fid)
    return probs


# ======================================================================================
# Governance analysis (shared by the GOV-* checks; computed once per Ctx)
# ======================================================================================

def _read_ud_rows(git: common.GitView, head: str) -> Tuple[Dict[str, dict], Optional[str]]:
    rows, err = _parse_jsonl_bytes(git.file_at(head, USER_DECISIONS_REL), USER_DECISIONS_REL)
    return {str(r.get('id')): r for r in rows if r.get('id')}, err


def analyze(ctx: common.Ctx) -> dict:
    if 'gov_analysis' in ctx._cache:
        a = ctx._cache['gov_analysis']
        if isinstance(a, Exception):
            raise a
        return a
    try:
        a = _analyze(ctx)
    except (MissingInput, InvalidInput, CheckerError) as e:
        ctx._cache['gov_analysis'] = e
        raise
    ctx._cache['gov_analysis'] = a
    return a


def cleanup(ctx: common.Ctx) -> None:
    """Remove the temporary work directory of a governance analysis (base/head checkers, corpora)."""
    td = ctx._cache.pop('gov_work', None)
    if td is not None:
        td.cleanup()


def _work_dir(ctx: common.Ctx) -> Path:
    if 'gov_work' not in ctx._cache:
        td = tempfile.TemporaryDirectory(prefix='minifly_gov_')
        ctx._cache['gov_work'] = td
    return Path(ctx._cache['gov_work'].name)


_LAZY_IMPORT_FUNCS = ('lazy', '_lazy', 'import_module', '__import__')


def kit_import_closure(read: Callable[[str], Optional[bytes]], module_path: str) -> Set[str]:
    """Repo paths of the kit modules a check module can run: the module and, transitively, every kit module it imports
    (an import statement at any depth, or lazy('x') / importlib.import_module('x') with a literal name)."""
    kit_dir = module_path.rsplit('/', 1)[0] if '/' in module_path else ''
    cache: Dict[str, Optional[bytes]] = {}

    def rd(name: str) -> Optional[bytes]:
        rel = '%s/%s.py' % (kit_dir, name) if kit_dir else '%s.py' % name
        if rel not in cache:
            cache[rel] = read(rel)
        return cache[rel]
    first = module_path.rsplit('/', 1)[-1][:-3] if module_path.endswith('.py') else module_path
    seen: Set[str] = set()
    todo = [first]
    while todo:
        m = todo.pop()
        if m in seen:
            continue
        seen.add(m)
        data = rd(m)
        if data is None:
            continue
        try:
            tree = ast.parse(data.decode('utf-8', 'replace'))
        except (SyntaxError, ValueError):
            continue
        for node in ast.walk(tree):
            names: List[str] = []
            if isinstance(node, ast.Import):
                names = [a.name.split('.')[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names = [node.module.split('.')[0]]
            elif isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.Constant) \
                    and isinstance(node.args[0].value, str):
                fn = node.func
                fname = fn.id if isinstance(fn, ast.Name) else (fn.attr if isinstance(fn, ast.Attribute) else '')
                if fname in _LAZY_IMPORT_FUNCS:
                    names = [node.args[0].value.split('.')[0]]
            for n in names:
                if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', n) and n not in seen and rd(n) is not None:
                    todo.append(n)
    return set(('%s/%s.py' % (kit_dir, m) if kit_dir else '%s.py' % m) for m in seen)


def require_module_change(changes: List[dict], changed: Sequence[str],
                          read: Optional[Callable[[str], Optional[bytes]]] = None) -> List[dict]:
    """An appended RULES changed_in version is neutral only when the same range also changes the code the check runs:
    its module or a kit module the module imports (kit_import_closure over the head tree, when `read` is given)
    (round-1 repair, verifier): a version recorded for unchanged code would exclude the check's held-out hits and false
    alarms as post_batch, so without such a change the append is a loosening."""
    changed_set = set(changed)
    for c in changes:
        mp = c.get('changed_in_module')
        if not mp or c['class'] != 'neutral':
            continue
        mp = common.norm_rel(mp)
        code = kit_import_closure(read, mp) if read is not None else {mp}
        hit = sorted(code & changed_set)
        if hit:
            c['what'] += ' (code changed in the range: %s)' % ', '.join(hit[:4])
        else:
            c['class'] = 'loosen'
            c['what'] += (' without a change to %s or a kit module it imports in the range (a version recorded for '
                          "unchanged code would exclude the check's held-out hits and false alarms as post_batch)" % mp)
    return changes


def _analyze(ctx: common.Ctx) -> dict:
    git = ctx.git
    b_rev, h_rev = ctx.options.get('base'), ctx.options.get('head')
    if not b_rev or not h_rev:
        raise MissingInput('governance needs --base and --head')
    base, head = resolve_rev(git, b_rev), resolve_rev(git, h_rev)
    commits = range_commits(git, base, head)
    changed = sorted(git.changed_paths(base, head))
    prot_globs, append_globs = _protected_rules(git, [base, head], ctx.repo_root)
    prot = [p for p in changed if _matches_any(p, prot_globs)]
    changes = require_module_change(classify_paths(git, base, head, prot), changed,
                                    lambda rel: git.file_at(head, rel))
    trailers_ud: List[str] = []
    trailers_rev: List[str] = []
    for c in commits:
        for v in c.trailer_values('User-Decision'):
            trailers_ud.extend(UD_ID_RE.findall(v))
        trailers_rev.extend(v.strip() for v in c.trailer_values('Governance-Review') if v.strip())
    observe = None
    needs = [c for c in changes if c['class'] == 'code_pending' or c.get('fixture_change') == 'added']
    if needs:
        observe = observe_monotone(git, base, head, _work_dir(ctx))
        for c in changes:
            if c['class'] == 'code_pending':
                if observe['ok']:
                    c['class'] = 'tighten_or_neutral'
                    c['what'] += ' (monotone on all fixtures)'
                else:
                    c['class'] = 'loosen'
                    c['what'] += ' (non-monotone or unverifiable: %s)' % '; '.join(observe['problems'][:2])
    return {'base': base, 'head': head, 'commits': commits, 'changed': changed, 'protected_globs': prot_globs,
            'append_only_globs': append_globs, 'protected_changed': prot, 'changes': changes,
            'ud_ids': sorted(set(trailers_ud)), 'review_refs': sorted(set(trailers_rev)), 'observe': observe}


def _classes(a: dict) -> Set[str]:
    return set(c['class'] for c in a['changes'])


def _needs_owner(c: dict) -> bool:
    return c['class'] in ('loosen', 'constraint')


def _needs_num(c: dict) -> bool:
    return c['class'] == 'loosen' or (c['class'] == 'constraint' and c.get('direction') in ('loosen', 'unknown'))


def _ud_covers(ud: dict, change: dict) -> bool:
    scopes = [str(s) for s in (ud.get('scope') or []) if isinstance(s, str)]
    if change['class'] == 'constraint':
        key = change.get('key') or '*'
        for s in scopes:
            if not s.startswith('constraint:'):
                continue
            k = s[len('constraint:'):].strip()
            if k and k != '*' and (key == k or key.startswith(k + '.')):
                return True
        return False
    if change['class'] == 'owner_decision':
        return True
    for s in scopes:
        kind, _, target = s.partition(':')
        if kind not in ('loosen', 'governance') or not target.strip():
            continue
        target = target.strip()
        for t in change['targets']:
            if t == target or (('/' in target) and (t == target or t.startswith(target.rstrip('/') + '/'))):
                return True
    return False


# ======================================================================================
# GOV-* checks
# ======================================================================================

def _change_lines(changes: Sequence[dict], limit: int = 12) -> str:
    return '; '.join('%s [%s] %s' % (c['path'], c['class'], c['what']) for c in changes[:limit]) + \
        (' (+%d more)' % (len(changes) - limit) if len(changes) > limit else '')


@common.check('GOV-SELF')
def check_gov_self(ctx: common.Ctx) -> Result:
    """selftest passes at head."""
    a = analyze(ctx)
    head = a['head']
    git = ctx.git
    if isinstance(git, common.TimelineGit):
        dirty = git.dirty_paths([KIT_REL])
        if dirty:
            return Result('GOV-SELF', FAIL, 'timeline mode: fixture kit paths are dirty, selftest cannot pass at head: %s'
                          % ', '.join(dirty[:8]), evidence=dirty[:20])
        return Result('GOV-SELF', PASS, 'timeline (corpus) mode: fixture kit paths clean at head %s; the head kit tree '
                      'of a fixture is not self-tested (the enclosing selftest verifies the running kit)' % _short(head),
                      data={'mode': 'timeline'})
    sha = common.checker_sha256_at(git, head)
    if sha is None:
        return Result('GOV-SELF', FAIL, 'no kit tree under %s at head %s' % (KIT_REL, _short(head)))
    ok, why = common.selftest_cache_valid(sha)
    if ok:
        return Result('GOV-SELF', PASS, 'selftest passed for head checker %s (%s)' % (sha[:12], why),
                      data={'checker_sha256': sha})
    work = _work_dir(ctx) / 'selftest_head'
    try:
        kit, note = materialize_kit(git, head, work)
    except (MissingInput, InvalidInput, CheckerError, OSError) as e:
        return Result('GOV-SELF', MISSING, 'cannot extract the head checker: %s' % e)
    if not kit_runnable(kit):
        return Result('GOV-SELF', FAIL, 'head checker at %s is not runnable' % _short(head))
    env = dict(os.environ)
    env[GOV_DEPTH_ENV] = str(int(os.environ.get(GOV_DEPTH_ENV, '0') or 0) + 1)
    try:
        p = subprocess.run([sys.executable, '-I', str(kit / 'minifly_check.py'), '--repo-root', str(work), '--json',
                            'selftest'], cwd=str(work), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           timeout=SUBPROCESS_SELFTEST_TIMEOUT_S, env=env)
    except subprocess.TimeoutExpired:
        return Result('GOV-SELF', FAIL, 'head selftest timed out after %ss' % SUBPROCESS_SELFTEST_TIMEOUT_S)
    summary = ''
    try:
        rep = json.loads(p.stdout.decode('utf-8', 'replace'))
        bad = [f for f, v in (rep.get('fixtures') or {}).items() if not v.get('ok')]
        summary = '%d fixtures, %d failing%s' % (len(rep.get('fixtures') or {}), len(bad),
                                                 (': ' + ','.join(sorted(bad)[:8])) if bad else '')
        if rep.get('preflight_problems'):
            summary += '; preflight: %s' % '; '.join(rep['preflight_problems'][:3])
    except (ValueError, AttributeError):
        summary = (p.stderr.decode('utf-8', 'replace').strip().splitlines() or ['no output'])[-1][:300]
    if p.returncode == 0:
        return Result('GOV-SELF', PASS, 'selftest of the head checker %s passed (%s; %s)' % (sha[:12], note, summary),
                      data={'checker_sha256': sha})
    return Result('GOV-SELF', FAIL, 'selftest of the head checker %s exited %d (%s)' % (sha[:12], p.returncode, summary),
                  data={'checker_sha256': sha})


@common.check('GOV-SEP')
def check_gov_sep(ctx: common.Ctx) -> Result:
    """A range touching protected paths touches no delivery directory."""
    a = analyze(ctx)
    if not a['protected_changed']:
        return Result('GOV-SEP', PASS, 'range %s..%s touches no protected path (%d paths changed)'
                      % (_short(a['base']), _short(a['head']), len(a['changed'])))
    dirs: Dict[str, List[str]] = {}
    for p in a['changed']:
        if _matches_any(p, a['protected_globs']):
            continue
        d = delivery_dir_of(p)
        if d:
            dirs.setdefault(d, []).append(p)
    if dirs:
        ev = [p for v in dirs.values() for p in v][:20]
        return Result('GOV-SEP', FAIL, 'range touches protected paths (%s) and delivery directories (%s): split the '
                      'governance change into its own range' % (', '.join(a['protected_changed'][:5]),
                                                                ', '.join(sorted(dirs)[:8])),
                      evidence=ev, data={'delivery_dirs': sorted(dirs)})
    return Result('GOV-SEP', PASS, 'protected-only range: %d protected paths, no delivery directory'
                  % len(a['protected_changed']))


@common.check('GOV-CLASS')
def check_gov_class(ctx: common.Ctx) -> Result:
    """Computed class for each change; a loosen or constraint class needs an owner decision cited."""
    a = analyze(ctx)
    ch = a['changes']
    counts: Dict[str, int] = {}
    for c in ch:
        counts[c['class']] = counts.get(c['class'], 0) + 1
    data = {'changes': ch, 'counts': counts}
    if not ch:
        return Result('GOV-CLASS', PASS, 'no protected-path change in range', data=data)
    owner = [c for c in ch if _needs_owner(c)]
    summary = ', '.join('%s=%d' % kv for kv in sorted(counts.items()))
    if owner and not a['ud_ids']:
        return Result('GOV-CLASS', FAIL, 'computed classes %s; %d change(s) are loosen/constraint but the range cites no '
                      'User-Decision: %s' % (summary, len(owner), _change_lines(owner)),
                      evidence=[c['path'] for c in owner][:20], data=data)
    return Result('GOV-CLASS', PASS, 'computed classes %s: %s' % (summary, _change_lines(ch, 6)), data=data)


def _load_review(git: common.GitView, head: str, ref: str) -> Tuple[Optional[str], Any, Optional[str]]:
    r = ref.strip()
    if not r.endswith('.json'):
        name = r if r.startswith('REVIEW_') else 'REVIEW_' + r
        r = '%s/%s.json' % (GOV_REVIEW_DIR, name)
    try:
        r = common.norm_rel(r)
    except InvalidInput as e:
        return None, None, str(e)
    if not r.startswith(GOV_REVIEW_DIR + '/'):
        return r, None, 'Governance-Review %s is not under %s/' % (ref, GOV_REVIEW_DIR)
    data = git.file_at(head, r)
    if data is None:
        return r, None, 'review record %s does not exist at head' % r
    obj, err = _parse_json_bytes(data, r)
    if err or not isinstance(obj, dict):
        return r, None, 'review record %s does not parse as an object (%s)' % (r, err)
    return r, obj, None


@common.check('GOV-REV')
def check_gov_rev(ctx: common.Ctx) -> Result:
    """tighten, new, neutral (and loosen) changes need a KB/GOVERNANCE review record by the other family,
    cited in a Governance-Review trailer."""
    a = analyze(ctx)
    need = [c for c in a['changes'] if c['class'] in ('tighten', 'tighten_or_neutral', 'neutral', 'loosen')]
    if not need:
        return Result('GOV-REV', NOT_APPLICABLE, 'no tighten, new, neutral or loosen change in range (classes: %s)'
                      % (', '.join(sorted(_classes(a))) or 'none'))
    git = ctx.git
    review_paths = set()
    authors: Set[str] = set()
    unknown: List[str] = []
    for c in a['commits']:
        prot = [p for p in c.paths if _matches_any(p, a['protected_globs'])]
        if not prot:
            continue
        # the author family of a protected-path commit: trailer AND namespace (common.commit_family, SPEC 20.15)
        fam, ist, iwhy = common.commit_family(git, c)
        if ist != common.IDENTITY_OK:
            unknown.append('%s (%s)' % (_short(c.sha), ist))
            continue
        authors.add(fam)
    if unknown:
        return Result('GOV-REV', FAIL, 'protected-path commits whose acting family cannot be derived from the '
                      'Acting-Agent trailer and the origin/<family>/* namespace (author family unknown): %s'
                      % ', '.join(unknown[:8]), evidence=unknown[:20])
    if not a['review_refs']:
        return Result('GOV-REV', FAIL, '%d change(s) need a cross-agent governance review, but no Governance-Review '
                      'trailer is in the range: %s' % (len(need), _change_lines(need, 6)))
    problems: List[str] = []
    good: List[str] = []
    for ref in a['review_refs']:
        path, obj, err = _load_review(git, a['head'], ref)
        if err:
            problems.append(err)
            continue
        review_paths.add(path)
        rfam = _family(obj.get('reviewer_agent'))
        if rfam is None:
            problems.append('%s: reviewer_agent %r is not codex or claude' % (path, obj.get('reviewer_agent')))
            continue
        if rfam in authors:
            problems.append('%s: reviewer family %s equals an author family of the change (%s)'
                            % (path, rfam, ','.join(sorted(authors))))
            continue
        dec = str(obj.get('decision') or '').strip().upper()
        if dec not in REVIEW_APPROVE:
            problems.append('%s: decision %r is not an approval' % (path, obj.get('decision')))
            continue
        # binding: the record names this range's base, a commit in the range, or every changed protected path
        in_range = set(c.sha for c in a['commits'])
        bound = False
        for k in ('base', 'reviewed_base'):
            v = obj.get(k)
            if isinstance(v, str) and (v == a['base'] or (len(v) >= 7 and a['base'].startswith(v))):
                bound = True
        for k in ('head', 'reviewed_head', 'reviewed_commit'):
            v = obj.get(k)
            if isinstance(v, str) and any(s == v or (len(v) >= 7 and s.startswith(v)) for s in in_range):
                bound = True
        rp = obj.get('paths')
        if isinstance(rp, list) and set(a['protected_changed']) - {path} <= set(str(x) for x in rp):
            bound = True
        declares = any(obj.get(k) is not None for k in ('base', 'reviewed_base', 'head', 'reviewed_head',
                                                         'reviewed_commit', 'paths'))
        if not bound and not declares:
            # a record that names no range but was first committed inside this very range was written for it (a
            # record from an earlier range can never be re-cited this way); a record naming another range never binds
            intro = first_commit_adding_in(git, path, a['head'])
            if intro is not None and intro.sha in in_range:
                bound = True
        if not bound:
            problems.append('%s: record does not bind to this range (needs base, a reviewed commit in the range, or '
                            'paths covering every changed protected path)' % path)
            continue
        introducing = first_commit_adding_in(git, path, a['head'])
        if introducing is None:
            problems.append('%s: no commit introduces the review record' % path)
            continue
        ok, why = common.commit_is_family(git, introducing, rfam)
        if not ok:
            # trailer AND origin/<family>/* namespace must both say the reviewer family (SPEC 20.15)
            ns = set(common.branch_namespace(b) for b in git.remote_branches_containing(introducing.sha))
            problems.append('%s: the introducing commit %s is not the reviewer family %s (%s)%s'
                            % (path, _short(introducing.sha), rfam, why,
                               '' if rfam in ns else '; no %s-namespace remote branch contains it' % rfam))
            continue
        good.append('%s by %s (%s)' % (path, rfam, why))
    if good and not problems:
        return Result('GOV-REV', PASS, 'cross-agent review: %s; covers %d change(s)' % ('; '.join(good), len(need)),
                      evidence=sorted(review_paths))
    return Result('GOV-REV', FAIL, 'governance review not valid: %s' % '; '.join(problems or ['no valid record']),
                  evidence=sorted(review_paths))


@common.check('GOV-UD')
def check_gov_ud(ctx: common.Ctx) -> Result:
    """loosen and constraint changes require a User-Decision trailer whose UD scope covers the change."""
    a = analyze(ctx)
    need = [c for c in a['changes'] if _needs_owner(c) or c['class'] == 'owner_decision']
    if not need:
        return Result('GOV-UD', NOT_APPLICABLE, 'no loosen, constraint or owner-decision change in range')
    uds, err = _read_ud_rows(ctx.git, a['head'])
    if err:
        return Result('GOV-UD', FAIL, 'USER_DECISIONS.jsonl at head does not parse: %s' % err)
    cited = a['ud_ids']
    problems = []
    missing_ud = [u for u in cited if u not in uds]
    if missing_ud:
        problems.append('cited UD(s) absent from USER_DECISIONS at head: %s' % ', '.join(missing_ud))
    cited_rows = [uds[u] for u in cited if u in uds]
    for c in need:
        if c['class'] == 'owner_decision':
            not_cited = [u for u in c.get('ud_ids') or [] if u not in cited]
            if not_cited:
                problems.append('added owner decision(s) %s are not cited in a User-Decision trailer'
                                % ', '.join(not_cited))
            continue
        if not any(_ud_covers(u, c) for u in cited_rows):
            want = ('constraint:%s' % c.get('key')) if c['class'] == 'constraint' else \
                ('loosen:<%s>' % '|'.join(c['targets'][:3]))
            problems.append('%s [%s] %s: no cited UD scope covers it (needs %s)' % (c['path'], c['class'], c['what'],
                                                                                   want))
    if problems:
        return Result('GOV-UD', FAIL, '; '.join(problems[:8]) + (' (+%d more)' % (len(problems) - 8)
                                                                  if len(problems) > 8 else ''),
                      evidence=[c['path'] for c in need][:20], data={'cited': cited})
    return Result('GOV-UD', PASS, '%d change(s) covered by %s' % (len(need), ', '.join(cited) or 'appended UD rows'),
                  data={'cited': cited})


@common.check('GOV-NUM')
def check_gov_num(ctx: common.Ctx) -> Result:
    """For a loosen UD, every changed numeric value (old and new) appears in the verbatim text."""
    a = analyze(ctx)
    need = [c for c in a['changes'] if _needs_num(c)]
    if not need or not a['ud_ids']:
        return Result('GOV-NUM', NOT_APPLICABLE, 'no loosen UD cited (%d loosening change(s), %d UD(s) cited)'
                      % (len(need), len(a['ud_ids'])))
    uds, err = _read_ud_rows(ctx.git, a['head'])
    if err:
        return Result('GOV-NUM', FAIL, 'USER_DECISIONS.jsonl at head does not parse: %s' % err)
    cited_rows = [uds[u] for u in a['ud_ids'] if u in uds]
    problems = []
    checked = 0
    for c in need:
        covering = [u for u in cited_rows if _ud_covers(u, c)] or cited_rows
        text = '\n'.join(str(u.get('verbatim') or '') for u in covering)
        for n in c['numbers']:
            checked += 1
            if not common.number_in_text(n, text):
                problems.append('%s: %s not in the verbatim text of %s' % (c['what'][:80], _fmt_num(n),
                                                                          ','.join(str(u.get('id')) for u in covering)))
    if not cited_rows:
        return Result('GOV-NUM', FAIL, 'cited UD(s) %s not found at head' % ', '.join(a['ud_ids']))
    if problems:
        return Result('GOV-NUM', FAIL, '; '.join(problems[:8]), evidence=[c['path'] for c in need][:20])
    return Result('GOV-NUM', PASS, '%d changed number(s) across %d loosening change(s) restated verbatim'
                  % (checked, len(need)))


@common.check('GOV-MONO')
def check_gov_mono(ctx: common.Ctx) -> Result:
    """For every base fixture fired(head) contains fired(base), and clean fixtures stay clean."""
    a = analyze(ctx)
    relevant = [c for c in a['changes'] if c['kind'] in ('code', 'corpus', 'expected') or c.get('module_change')]
    if not relevant:
        return Result('GOV-MONO', NOT_APPLICABLE, 'no check code, corpus or EXPECTED change in range')
    problems = declared_monotone(ctx.git, a['base'], a['head'])
    removed = [c for c in relevant if c.get('fixture_change') == 'removed']
    for c in removed:
        problems.append('fixture %s removed from the corpus' % c.get('fixture'))
    obs = a.get('observe')
    if obs is not None:
        problems.extend(obs['problems'])
    if problems:
        return Result('GOV-MONO', FAIL, '; '.join(problems[:10]) + (' (+%d more)' % (len(problems) - 10)
                                                                     if len(problems) > 10 else ''),
                      data={'observe': obs})
    how = 'declared EXPECTED sets monotone'
    if obs is not None:
        how += '; observed on base %s / head %s fixtures (%d command runs; %s)' % (
            len(obs.get('fixtures', {}).get('base', [])), len(obs.get('fixtures', {}).get('head', [])), obs['runs'],
            '; '.join(obs['notes']))
    return Result('GOV-MONO', PASS, how, data={'observe': obs})


@common.check('GOV-APPEND')
def check_gov_append(ctx: common.Ctx) -> Result:
    """Append-only files show only additions."""
    a = analyze(ctx)
    files = [p for p in a['changed'] if _matches_any(p, a['append_only_globs'])]
    if not files:
        return Result('GOV-APPEND', NOT_APPLICABLE, 'no append-only file changed in range')
    bad = []
    for p in files:
        if ctx.git.file_at(a['head'], p) is None:
            bad.append('%s deleted' % p)
            continue
        added, removed = ctx.git.diff_lines(a['base'], a['head'], p)
        if removed:
            bad.append('%s: %d line(s) removed or edited (first: %s)' % (p, len(removed), removed[0][:120]))
    if bad:
        return Result('GOV-APPEND', FAIL, '; '.join(bad[:8]), evidence=files[:20])
    return Result('GOV-APPEND', PASS, '%d append-only file(s) changed by additions only' % len(files), evidence=files[:20])


def parse_frontmatter(text: str) -> Tuple[Optional[dict], List[str], List[str]]:
    """Minimal YAML-frontmatter reader (stdlib): key: value scalars, quoted strings, folded/literal
    blocks and indented continuation lines. Returns (fields, body_lines, problems)."""
    lines = text.split('\n')
    if not lines or lines[0].strip() != '---':
        return None, lines, ['no YAML frontmatter (first line must be ---)']
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == '---':
            end = i
            break
    if end is None:
        return None, lines, ['frontmatter is not closed by ---']
    fields: Dict[str, str] = {}
    problems: List[str] = []
    i = 1
    while i < end:
        ln = lines[i]
        if not ln.strip() or ln.lstrip().startswith('#'):
            i += 1
            continue
        m = re.match(r'^([A-Za-z_][A-Za-z0-9_-]*):(?:\s+(.*))?$', ln)
        if not m or ln[:1].isspace():
            problems.append('unparseable frontmatter line %d: %r' % (i + 1, ln[:80]))
            i += 1
            continue
        key, val = m.group(1), (m.group(2) or '').rstrip()
        i += 1
        cont: List[str] = []
        while i < end and (lines[i][:1].isspace() or not lines[i].strip()):
            cont.append(lines[i])
            i += 1
        if val in ('|', '>', '|-', '>-', '|+', '>+'):
            body = [c.strip() for c in cont]
            while body and not body[-1]:
                body.pop()
            v = ('\n' if val.startswith('|') else ' ').join(body)
        elif val.startswith('"'):
            raw = ' '.join([val] + [c.strip() for c in cont if c.strip()])
            try:
                v = json.loads(raw)
                if not isinstance(v, str):
                    raise ValueError
            except ValueError:
                problems.append('bad double-quoted value for %s' % key)
                v = raw.strip('"')
        elif val.startswith("'"):
            raw = ' '.join([val] + [c.strip() for c in cont if c.strip()])
            if not (len(raw) >= 2 and raw.endswith("'")):
                problems.append('unterminated single-quoted value for %s' % key)
            v = raw[1:-1].replace("''", "'")
        else:
            v = ' '.join([val] + [c.strip() for c in cont if c.strip()]).strip()
        if key in fields:
            problems.append('duplicate frontmatter key %s' % key)
        fields[key] = v
    body = lines[end + 1:]
    if body and body[-1] == '':
        body = body[:-1]
    return fields, body, problems


def skill_problems(name: str, text: Optional[str]) -> List[str]:
    if text is None:
        return ['%s: SKILL.md missing' % name]
    fields, body, probs = parse_frontmatter(text)
    out = ['%s: %s' % (name, p) for p in probs]
    if fields is None:
        return out
    if fields.get('name') != name:
        out.append('%s: frontmatter name %r does not equal the skill directory' % (name, fields.get('name')))
    desc = fields.get('description')
    if not desc:
        out.append('%s: frontmatter description missing or empty' % name)
    elif len(desc) > SKILL_DESC_MAX:
        out.append('%s: description has %d characters (max %d)' % (name, len(desc), SKILL_DESC_MAX))
    if len(body) >= SKILL_BODY_MAX_LINES:
        out.append('%s: body has %d lines (must be under %d)' % (name, len(body), SKILL_BODY_MAX_LINES))
    return out


@common.check('GOV-SKILL')
def check_gov_skill(ctx: common.Ctx) -> Result:
    """Each SKILL.md has valid frontmatter (name, description <= 1024 chars) and a body under 400 lines;
    AGENTS_POINTER.md (and a tracked root AGENTS.md) lists all five skills."""
    a = analyze(ctx)
    git, head = ctx.git, a['head']
    problems: List[str] = []
    names = set(SKILL_NAMES)
    for p in git.list_files(head, '.claude/skills'):
        parts = p.split('/')
        if len(parts) >= 4 and parts[2].startswith('minifly-') and parts[3] == 'SKILL.md':
            names.add(parts[2])
    for n in sorted(names):
        data = git.file_at(head, '.claude/skills/%s/SKILL.md' % n)
        problems.extend(skill_problems(n, data.decode('utf-8', 'replace') if data is not None else None))
    for rel, required in ((AGENTS_POINTER_REL, True), (ROOT_AGENTS_REL, False)):
        data = git.file_at(head, rel)
        if data is None:
            if required:
                problems.append('%s missing at head' % rel)
            continue
        text = data.decode('utf-8', 'replace')
        absent = [s for s in SKILL_NAMES if s not in text]
        if absent:
            problems.append('%s does not list %s' % (rel, ', '.join(absent)))
    if problems:
        return Result('GOV-SKILL', FAIL, '; '.join(problems[:10]))
    return Result('GOV-SKILL', PASS, '%d skills valid at %s; pointer lists all five' % (len(names), _short(head)))


@common.check('GOV-COI')
def check_gov_coi(ctx: common.Ctx) -> Result:
    """A protected-path commit after this delivery's lock with the author's Acting-Agent fails, unless a UD
    (scope coi:<delivery>) names the delivery. A protected commit without an Acting-Agent counts as the
    author's (identity unknown)."""
    lock = ctx.lock
    rel = ctx.repo_rel_of('ENDPOINT_LOCK.json')
    c_lock = ctx.git.first_commit_adding(rel)
    if c_lock is None:
        raise MissingInput('ENDPOINT_LOCK.json is not committed: the lock commit is unknown')
    lf, lst, _lwhy = common.commit_family(ctx.git, c_lock)
    fams = set(f for f in [_family(lock.get('author_agent'))] if f) | (set([lf]) if lst == common.IDENTITY_OK else set())
    if not fams:
        return Result('GOV-COI', FAIL, 'author family unknown (lock.author_agent %r, lock commit %s has no Acting-Agent)'
                      % (lock.get('author_agent'), _short(c_lock.sha)))
    globs, _ = _protected_rules(None, [], ctx.repo_root)
    commits = ctx.git.log()
    idx = next((i for i, c in enumerate(commits) if c.sha == c_lock.sha), None)
    if idx is None:
        raise MissingInput('lock commit %s not in history' % _short(c_lock.sha))
    offenders = []
    for c in commits[idx + 1:]:
        prot = [p for p in c.paths if _matches_any(p, globs)]
        if not prot:
            continue
        cfam, cst, cwhy = common.commit_family(ctx.git, c)
        if cst != common.IDENTITY_OK:
            offenders.append((c, '%s (identity unknown: %s)' % (cst, cwhy), prot))
        elif cfam in fams:
            offenders.append((c, 'acting family %s (trailer and namespace)' % cfam, prot))
    if not offenders:
        return Result('GOV-COI', PASS, 'no protected-path commit by the author family (%s) after lock commit %s'
                      % (','.join(sorted(fams)), _short(c_lock.sha)))
    names = {ctx.delivery_name, ctx.delivery_rel}
    waiver = None
    for r in ctx.kb.rows('USER_DECISIONS'):
        for s in r.get('scope') or []:
            if isinstance(s, str) and s.startswith('coi:') and s[4:].strip() in names:
                waiver = r.get('id')
    detail = '; '.join('%s (%s): %s' % (_short(c.sha), why, ', '.join(p[:3])) for c, why, p in offenders[:6])
    if waiver:
        return Result('GOV-COI', PASS, 'protected-path commits by the author after the lock are covered by %s: %s'
                      % (waiver, detail))
    return Result('GOV-COI', FAIL, 'protected-path commit(s) by the author family %s after lock %s without a UD '
                  'scoped coi:%s: %s' % (','.join(sorted(fams)), _short(c_lock.sha), ctx.delivery_name, detail),
                  evidence=[c.sha for c, _, _ in offenders][:20])


def cmd_governance(args: Any) -> int:
    """MFC governance --base REV --head REV."""
    ctx = common.make_ctx(args, delivery=None, stage='governance',
                          options={'base': args.base, 'head': args.head, 'invoked_by': 'governance'})
    try:
        results = common.run_bundle(ctx, 'governance')
        rc = common.exit_code(results)
        try:
            a = analyze(ctx)
        except (MissingInput, InvalidInput, CheckerError):
            a = None
        report = {'schema': GOV_REPORT_SCHEMA, 'utc': common.utc_now(), 'checker_sha256': common.checker_sha256(),
                  'git_mode': ctx.git_mode, 'base': a['base'] if a else args.base, 'head': a['head'] if a else args.head,
                  'protected_changed': a['protected_changed'] if a else [],
                  'changes': a['changes'] if a else [], 'ud_ids': a['ud_ids'] if a else [],
                  'review_refs': a['review_refs'] if a else [],
                  'results': [r.to_dict() for r in results], 'exit_code': rc, 'verdict': common.VERDICT_BY_EXIT[rc]}
        if getattr(args, 'json', False):
            print(json.dumps(common._jsonable(report), indent=2, ensure_ascii=False))
        else:
            if a:
                for c in a['changes']:
                    print('%-19s %s: %s' % (c['class'], c['path'], c['what'][:160]))
            print(common.format_results(results))
            print('governance %s..%s: %s (exit %d)' % (_short(report['base']), _short(report['head']),
                                                      report['verdict'], rc))
        return rc
    finally:
        cleanup(ctx)


# ======================================================================================
# selftest
# ======================================================================================

def readme_planned_ids(text: str) -> Tuple[Set[str], List[str]]:
    """Ids listed under '## Not yet enforced' as '- `ID`: reason' bullets."""
    lines = text.split('\n')
    start = None
    for i, ln in enumerate(lines):
        if re.match(r'^##\s+Not yet enforced\s*$', ln.strip()):
            start = i + 1
            break
    if start is None:
        return set(), ["README has no '## Not yet enforced' section"]
    ids: Set[str] = set()
    probs: List[str] = []
    for ln in lines[start:]:
        if ln.startswith('## ') or ln.startswith('# '):
            break
        if not ln.startswith('- '):
            continue
        m = re.match(r'^-\s+`([A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+)`\s*:\s*\S', ln)
        if m:
            if m.group(1) in ids:
                probs.append('README lists %s twice' % m.group(1))
            ids.add(m.group(1))
        elif ln.startswith('- (none'):
            continue
        else:
            probs.append("README 'Not yet enforced' bullet not in the form - `ID`: reason: %r" % ln[:80])
    return ids, probs


def preflight(rules: dict, readme_text: Optional[str], checks: Optional[Dict[str, Callable]] = None,
              import_modules: bool = True) -> List[str]:
    """RULES/registry/README consistency (SPEC 7.4, 20.2; WP6 amendments)."""
    probs = list(common.validate_rules(rules))
    try:
        idx = common.rule_index(rules)
    except CheckerError as e:
        return probs + [str(e)]
    if import_modules:
        for mod in sorted(set(str(r.get('module')) for r in idx.values() if r.get('status') != 'planned')):
            try:
                importlib.import_module(mod)
            except Exception as e:  # noqa: BLE001
                probs.append('module %s failed to import: %s' % (mod, common.summarize_exception(e)))
    reg = common.CHECKS if checks is None else checks
    for rid, r in sorted(idx.items()):
        if r.get('status') == 'planned':
            continue
        fn = reg.get(rid)
        if fn is None:
            probs.append('RULES id %s (module %s) is neither registered nor planned' % (rid, r.get('module')))
            continue
        fmod = str(getattr(fn, '__module__', '')).split('.')[-1]
        if fmod != r.get('module'):
            probs.append('check %s is registered by module %s, RULES says %s' % (rid, fmod, r.get('module')))
    for rid in sorted(reg):
        if rid not in idx:
            probs.append('registered check %s is not in RULES.json' % rid)
    planned = set(rid for rid, r in idx.items() if r.get('status') == 'planned')
    for name, b in sorted((rules.get('bundles') or {}).items()):
        for e in (b.get('checks') or []) if isinstance(b, dict) else []:
            ids = ([e] if isinstance(e, str) else [e.get('id')] if isinstance(e, dict) and 'id' in e
                   else list((e or {}).get('primary', [])) + list((e or {}).get('fallback', [])))
            for i in ids:
                if i in planned:
                    probs.append('bundle %s contains planned id %s' % (name, i))
    if readme_text is None:
        probs.append('tools/minifly_check/README.md missing')
    else:
        listed, rp = readme_planned_ids(readme_text)
        probs.extend(rp)
        if listed != planned:
            if planned - listed:
                probs.append('README Not yet enforced omits planned id(s): %s' % ', '.join(sorted(planned - listed)))
            if listed - planned:
                probs.append('README Not yet enforced lists non-planned id(s): %s' % ', '.join(sorted(listed - planned)))
    return sorted(set(probs), key=probs.index)


def _fixture_mentions(entry: dict, rid: str) -> bool:
    return any(rid in (entry.get(k) or []) for k in ('must_fire', 'must_not_fire', 'must_warn'))


def _canary_delivery(fx_dir: Path, entry: Optional[dict]) -> Optional[str]:
    repo = fx_dir / 'repo'
    cands: List[str] = []
    for cmd in (entry or {}).get('commands') or []:
        for tok in cmd:
            if isinstance(tok, str) and not tok.startswith('-') and (repo / tok / 'ENDPOINT_LOCK.json').is_file():
                if tok not in cands:
                    cands.append(tok)
    if len(cands) == 1:
        return cands[0]
    if not cands and repo.is_dir():
        found = [p.name for p in sorted(repo.iterdir()) if p.is_dir() and common.is_v1_delivery(p)]
        if len(found) == 1:
            return found[0]
    return None


def run_canary_step(corpus: Dict[str, Path], expected: dict, invoke: Invoker,
                    outcomes: Optional[Dict[str, RunOutcome]] = None) -> Tuple[List[str], dict]:
    """SPEC 12.2: canary on C00a with seed c0ffee; every applicable operator must be caught (CAN-01..03).
    The canary runs on the delivery in the state C00a's own EXPECTED commands produce before it (score writes
    VERDICT.json, claim regenerates CLAIM.json): the committed fixture tree alone carries a CLAIM without its
    VERDICT, which no real delivery has. When C00a ran in this selftest its canary command's result is used;
    otherwise the command prefix up to the canary runs in a fresh copy (or the canary alone if C00a's
    commands do not include it)."""
    fx = corpus.get(CANARY_FIXTURE)
    if fx is None:
        return ['canary: fixture %s missing from the corpus' % CANARY_FIXTURE], {}
    entry = expected.get(CANARY_FIXTURE) or {}
    d = _canary_delivery(fx, entry)
    if d is None:
        return ['canary: cannot identify the single v1 delivery of %s' % CANARY_FIXTURE], {}
    want = ['canary', d, '--seed', CANARY_SEED]
    cmds = [list(c) for c in entry.get('commands') or []]
    idx = cmds.index(want) if want in cmds else None
    oc = (outcomes or {}).get(CANARY_FIXTURE)
    if idx is not None and oc is not None and len(oc.exits) > idx and len(oc.per_command) > idx:
        source = 'command %d of the %s run' % (idx + 1, CANARY_FIXTURE)
    else:
        run_cmds = cmds[:idx + 1] if idx is not None else [want]
        idx = len(run_cmds) - 1
        oc = run_fixture(fx, run_cmds, invoke)
        source = 'fresh copy, %d command(s)' % len(run_cmds)
    rc = oc.exits[idx] if len(oc.exits) > idx else None
    statuses: Dict[str, Set[str]] = {}
    for rid, st in (oc.per_command[idx] if len(oc.per_command) > idx else []):
        statuses.setdefault(rid, set()).add(st)
    probs = list(oc.errors) + [m for i, m in oc.exit3_errors if i == idx]
    if rc != 0:
        probs.append('canary on %s --seed %s exited %s (must be 0)' % (CANARY_FIXTURE, CANARY_SEED, rc))
    for cid in ('CAN-01', 'CAN-02', 'CAN-03'):
        st = statuses.get(cid)
        if not st:
            probs.append('canary produced no %s result' % cid)
        elif st != {PASS}:
            probs.append('canary %s status %s (must be PASS)' % (cid, ','.join(sorted(st))))
    info = {'exits': [rc], 'statuses': {k: sorted(v) for k, v in sorted(statuses.items())}, 'source': source,
            'delivery': d}
    return ['canary: ' + p if not p.startswith('canary') else p for p in probs], info


class SelftestConfig:
    """What selftest runs. The default is the running kit, its corpus, EXPECTED.json, RULES.json and README
    with the in-process dispatcher; only that genuine configuration writes the SELF-01 cache. Tests
    override pieces (and may pass their own cache_writer)."""

    def __init__(self, kit_dir: Path = common.KIT_DIR, corpus_dir: Optional[Path] = None,
                 expected_path: Optional[Path] = None, rules: Optional[dict] = None,
                 readme_text: Optional[str] = None, invoke: Optional[Invoker] = None,
                 checks: Optional[Dict[str, Callable]] = None, import_modules: bool = True,
                 run_canary: bool = True, cache_writer: Optional[Callable[[str, dict], Any]] = None) -> None:
        self.kit_dir = Path(kit_dir)
        self.corpus_dir = Path(corpus_dir) if corpus_dir else self.kit_dir / 'corpus'
        self.expected_path = Path(expected_path) if expected_path else self.kit_dir / 'EXPECTED.json'
        self.rules = rules
        self.readme_text = readme_text
        self.invoke = invoke
        self.checks = checks
        self.import_modules = import_modules
        self.run_canary = run_canary
        genuine = (corpus_dir is None and expected_path is None and rules is None and readme_text is None
                   and invoke is None and checks is None and import_modules and run_canary
                   and self.kit_dir.resolve() == common.KIT_DIR.resolve())
        if cache_writer is not None:
            self.cache_writer = cache_writer
        elif genuine:
            self.cache_writer = lambda sha, summary: common.write_selftest_cache(sha, summary=summary)
        else:
            self.cache_writer = None


def _select_fixtures(corpus: Dict[str, Path], expected: dict, only: Optional[str], rule_ids: Set[str]
                     ) -> Tuple[List[str], Optional[str]]:
    ids = sorted(set(corpus) | set(expected))
    if not only:
        return ids, None
    if only in ids:
        return [only], None
    if only in rule_ids:
        sel = [f for f in ids if (f in expected and _fixture_mentions(expected[f], only))
               or fixture_is_clean(f, corpus.get(f), expected.get(f))]
        return sel, only
    raise InvalidInput('--only %s is neither a fixture id nor a RULES.json check id' % only)


SELFTEST_JOBS_ENV = 'MINIFLY_SELFTEST_JOBS'


def selftest_jobs() -> int:
    """Worker processes for selftest fixture runs: MINIFLY_SELFTEST_JOBS, else min(4, CPUs)."""
    v = os.environ.get(SELFTEST_JOBS_ENV, '').strip()
    if v.isdigit():
        return max(1, int(v))
    return max(1, min(4, os.cpu_count() or 1))


def _fork_pool(jobs: int) -> Optional[concurrent.futures.Executor]:
    """A pool of forked worker processes (they inherit the imported kit and its registry), or None when only one
    job is wanted or fork is unavailable. Threads are not used: the in-process invoker patches module state."""
    if jobs <= 1 or not hasattr(os, 'fork'):
        return None
    try:
        import multiprocessing
        mp = multiprocessing.get_context('fork')
        return concurrent.futures.ProcessPoolExecutor(max_workers=jobs, mp_context=mp)
    except (ImportError, ValueError, OSError):
        return None


def _worker_run_fixture(fx: str, commands: List[List[str]], record: bool,
                        stub: Optional[Tuple[str, str, str]] = None) -> Tuple[RunOutcome, List[str]]:
    """Run one fixture in this (forked) worker with the in-process invoker; optionally record which checks were
    invoked and/or replace one check by an always-<status> stub (selftest --mutate). Process-local state only."""
    current: Dict[str, Set[str]] = {'set': set()}
    saved = dict(common.CHECKS)
    try:
        if stub is not None:
            rid, mod, status = stub
            common.CHECKS[rid] = _stub(rid, mod, status)
        if record:
            for rid, fn in list(common.CHECKS.items()):
                common.CHECKS[rid] = _recording_wrapper(rid, fn, current)
        oc = run_fixture(Path(fx), commands, _inprocess_invoker())
    finally:
        common.CHECKS.clear()
        common.CHECKS.update(saved)
    return oc, sorted(current['set'])


def _clean_noticed(rid: str, fid: str, entry: dict, oc: RunOutcome, baseline_warned: Set[str]) -> bool:
    """A clean fixture notices an always-FAIL replacement when its selftest assertions fail, or, for a check
    whose FAIL is reported as WARN (advisory status, warn_below_tier), when that id newly WARNs there."""
    return bool(evaluate_fixture(fid, entry, oc, True)) or (rid in oc.warned and rid not in baseline_warned)


def _worker_mutate_check(rid: str, mod: str, dirty: List[Tuple[str, str, dict]],
                         clean: List[Tuple[str, str, dict, List[str]]]) -> dict:
    """selftest --mutate for one check id in a worker: the first dirty fixture noticing an always-PASS stub and
    the first clean fixture noticing an always-FAIL stub (pools are (fid, fixture dir, EXPECTED entry[,
    baseline WARN ids]))."""
    res: Dict[str, Optional[str]] = {'exercised_by': None, 'clean_coverage_by': None}
    for fid, fx, entry in dirty:
        oc, _inv = _worker_run_fixture(fx, entry['commands'], False, (rid, mod, PASS))
        if evaluate_fixture(fid, entry, oc, False):
            res['exercised_by'] = fid
            break
    for fid, fx, entry, warned in clean:
        oc, _inv = _worker_run_fixture(fx, entry['commands'], False, (rid, mod, FAIL))
        if _clean_noticed(rid, fid, entry, oc, set(warned)):
            res['clean_coverage_by'] = fid
            break
    return res


def run_selftest(cfg: SelftestConfig, only: Optional[str] = None, mutate: bool = False,
                 log: Callable[[str], None] = lambda s: None) -> Tuple[int, dict]:
    t0 = time.perf_counter()
    sha_start = common.checker_sha256(cfg.kit_dir, refresh=True)
    rules = cfg.rules if cfg.rules is not None else common.load_rules(cfg.kit_dir / 'RULES.json')
    readme = cfg.readme_text
    if readme is None and (cfg.kit_dir / 'README.md').is_file():
        readme = (cfg.kit_dir / 'README.md').read_text(encoding='utf-8')
    report: dict = {'schema': SELFTEST_SCHEMA, 'checker_sha256': sha_start, 'checker_version': common.CHECKER_VERSION,
                    'utc': common.utc_now(), 'only': only, 'mutate': bool(mutate)}
    pre = preflight(rules, readme, cfg.checks, cfg.import_modules)
    report['preflight_problems'] = pre
    for p in pre:
        log('PREFLIGHT  %s' % p)
    if not cfg.corpus_dir.is_dir():
        raise MissingInput('corpus directory %s missing' % cfg.corpus_dir)
    if not cfg.expected_path.is_file():
        raise MissingInput('EXPECTED.json missing at %s' % cfg.expected_path)
    expected = load_expected_file(cfg.expected_path)
    corpus = corpus_fixtures(cfg.corpus_dir)
    rule_ids = set(common.rule_index(rules))
    vprobs = validate_expected(expected, rule_ids)
    sel, only_check = _select_fixtures(corpus, expected, only, rule_ids)
    invoke = cfg.invoke or _inprocess_invoker()
    fixtures: Dict[str, dict] = {}
    outcomes: Dict[str, RunOutcome] = {}
    invoked: Dict[str, Set[str]] = {}
    saved_checks = None
    if mutate:
        saved_checks = dict(common.CHECKS)
        current: Dict[str, Set[str]] = {'set': set()}
        for rid, fn in saved_checks.items():
            common.CHECKS[rid] = _recording_wrapper(rid, fn, current)
    _STATE['selftest_depth'] += 1
    pool = _fork_pool(selftest_jobs()) if cfg.invoke is None else None
    try:
        if pool is not None:
            # every runnable fixture in a forked worker (in-process invoker there); results are evaluated here
            if saved_checks is not None:  # workers record their own invocations
                common.CHECKS.clear()
                common.CHECKS.update(saved_checks)
            futs = {}
            for fid in sel:
                fx, entry = corpus.get(fid), expected.get(fid)
                if fx is not None and entry is not None and not vprobs.get(fid):
                    futs[fid] = pool.submit(_worker_run_fixture, str(fx), entry['commands'], bool(mutate))
            par: Dict[str, Tuple[Optional[RunOutcome], List[str], Optional[str]]] = {}
            for fid, fut in futs.items():
                try:
                    oc_, inv_ = fut.result()
                    par[fid] = (oc_, inv_, None)
                except Exception as e:  # noqa: BLE001 - a worker crash is a fixture problem, never a pass
                    par[fid] = (None, [], 'worker error: %s (rerun with %s=1 to run fixtures in this process)'
                                % (common.summarize_exception(e), SELFTEST_JOBS_ENV))
        for fid in sel:
            fx = corpus.get(fid)
            entry = expected.get(fid)
            probs: List[str] = []
            if fx is None:
                fixtures[fid] = {'ok': False, 'problems': ['EXPECTED entry without a fixture dir with timeline.json'],
                                 'clean': None}
                log('FAIL  %-6s EXPECTED entry without a fixture directory' % fid)
                continue
            if entry is None:
                fixtures[fid] = {'ok': False, 'problems': ['fixture dir without an EXPECTED entry'], 'clean': None}
                log('FAIL  %-6s fixture directory without an EXPECTED entry' % fid)
                continue
            if vprobs.get(fid):
                fixtures[fid] = {'ok': False, 'problems': ['EXPECTED: ' + p for p in vprobs[fid]], 'clean': None}
                log('FAIL  %-6s %s' % (fid, '; '.join(vprobs[fid])))
                continue
            clean = fixture_is_clean(fid, fx, entry)
            if pool is not None:
                oc, inv_ids, werr = par[fid]
                if oc is None:
                    fixtures[fid] = {'ok': False, 'problems': [werr], 'clean': clean}
                    log('FAIL  %-6s %s' % (fid, werr))
                    continue
                if mutate:
                    invoked[fid] = set(inv_ids)
            else:
                if mutate:
                    current['set'] = set()
                oc = run_fixture(fx, entry['commands'], invoke)
                if mutate:
                    invoked[fid] = set(current['set'])
            outcomes[fid] = oc
            probs = evaluate_fixture(fid, entry, oc, clean)
            fixtures[fid] = dict(oc.to_dict(), ok=not probs, problems=probs, clean=clean)
            log('%s  %-6s %s(%.2fs)%s' % ('ok  ' if not probs else 'FAIL', fid, 'clean ' if clean else '',
                                          oc.runtime_s, (': ' + '; '.join(probs)) if probs else ''))
        if vprobs.get(''):
            pre = pre + ['EXPECTED: ' + p for p in vprobs['']]
        if not fixtures:
            pre = pre + ['no fixtures selected or present']
        report['preflight_problems'] = pre
        canary_probs: List[str] = []
        if cfg.run_canary and not mutate and (not only or only == CANARY_FIXTURE):
            canary_probs, cinfo = run_canary_step(corpus, expected, invoke, outcomes)
            report['canary'] = dict(cinfo, problems=canary_probs, ok=not canary_probs)
            log('%s  canary %s --seed %s%s' % ('ok  ' if not canary_probs else 'FAIL', CANARY_FIXTURE, CANARY_SEED,
                                                (': ' + '; '.join(canary_probs)) if canary_probs else ''))
        report['fixtures'] = fixtures
        base_ok = not pre and all(v.get('ok') for v in fixtures.values()) and not canary_probs
        mut = None
        if mutate:
            if not base_ok:
                mut = {'skipped': 'baseline selftest failed; mutation testing needs a green baseline'}
            else:
                mut = _mutate(saved_checks or {}, rules, corpus, expected, sel, outcomes, invoked, invoke, only_check,
                              log, pool=pool)
            report['mutate_report'] = mut
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
        _STATE['selftest_depth'] -= 1
        if saved_checks is not None:
            common.CHECKS.clear()
            common.CHECKS.update(saved_checks)
    runtime = time.perf_counter() - t0
    report['runtime_s'] = round(runtime, 3)
    sha_end = common.checker_sha256(cfg.kit_dir, refresh=True)
    if sha_end != sha_start:
        report.setdefault('preflight_problems', []).append('kit files changed during selftest (%s -> %s)'
                                                           % (sha_start[:12], sha_end[:12]))
        base_ok = False
    ok = base_ok and (not mutate or (isinstance(mut, dict) and not mut.get('gaps') and 'skipped' not in mut))
    report['ok'] = bool(ok)
    report['baseline_ok'] = bool(base_ok)
    if runtime > SELFTEST_TARGET_S and not mutate:
        report['runtime_warning'] = 'runtime %.1fs exceeds the %.0fs target' % (runtime, SELFTEST_TARGET_S)
        log('WARN  runtime %.1fs exceeds the %.0fs target' % (runtime, SELFTEST_TARGET_S))
    # Only a complete, passing baseline run attests the checker sha (a partial --only run never does).
    if base_ok and not only and cfg.cache_writer is not None:
        p = cfg.cache_writer(sha_start, {'fixtures': len(fixtures), 'runtime_s': report['runtime_s']})
        report['cache'] = str(p)
    return (common.EXIT_PASS if ok else common.EXIT_FAIL), report


def _recording_wrapper(rid: str, fn: Callable, current: Dict[str, Set[str]]) -> Callable:
    @functools.wraps(fn)
    def w(ctx):
        current['set'].add(rid)
        return fn(ctx)
    return w


def _stub(rid: str, module: str, status: str) -> Callable:
    def stub(ctx):
        return Result(rid, status, 'selftest --mutate: always-%s stub' % status)
    stub.__module__ = module
    stub.__qualname__ = 'mutate_always_%s_%s' % (status, rid.replace('-', '_'))
    return stub


def _mutate(saved: Dict[str, Callable], rules: dict, corpus: Dict[str, Path], expected: dict, sel: List[str],
            outcomes: Dict[str, RunOutcome], invoked: Dict[str, Set[str]], invoke: Invoker, only_check: Optional[str],
            log: Callable[[str], None], pool: Optional[concurrent.futures.Executor] = None) -> dict:
    idx = common.rule_index(rules)
    ids = [only_check] if only_check else sorted(r for r in idx if idx[r].get('status') != 'planned')
    per: Dict[str, dict] = {}
    gaps: List[str] = []
    if pool is not None:
        return _mutate_parallel(saved, idx, ids, corpus, expected, sel, outcomes, invoked, log, pool)
    for rid in ids:
        if rid not in saved:
            per[rid] = {'status': 'unregistered'}
            gaps.append(rid)
            continue
        mod = str(idx[rid].get('module'))
        users = [f for f in sel if rid in invoked.get(f, set())]
        dirty = [f for f in users if not fixture_is_clean(f, corpus.get(f), expected.get(f))]
        clean = [f for f in users if fixture_is_clean(f, corpus.get(f), expected.get(f))]
        dirty.sort(key=lambda f: (0 if rid in (expected[f].get('must_fire') or []) else
                                  1 if rid in (expected[f].get('must_warn') or []) else 2, f))
        res = {'exercised_by': None, 'clean_coverage_by': None}
        for status, pool, key in ((PASS, dirty, 'exercised_by'), (FAIL, clean, 'clean_coverage_by')):
            common.CHECKS[rid] = _stub(rid, mod, status)
            try:
                for f in pool:
                    oc = run_fixture(corpus[f], expected[f]['commands'], invoke)
                    if status == PASS:
                        noticed = bool(evaluate_fixture(f, expected[f], oc, False))
                    else:
                        noticed = _clean_noticed(rid, f, expected[f], oc,
                                                 outcomes[f].warned if f in outcomes else set())
                    if noticed:
                        res[key] = f
                        break
            finally:
                common.CHECKS[rid] = _recording_wrapper(rid, saved[rid], {'set': set()})
        labels = []
        if res['exercised_by'] is None:
            labels.append('not exercised')
        if res['clean_coverage_by'] is None:
            labels.append('no clean coverage')
        res['status'] = ', '.join(labels) or 'ok'
        per[rid] = res
        if labels:
            gaps.append(rid)
        log('%s  mutate %-14s %s' % ('ok  ' if not labels else 'GAP ', rid, res['status']))
    return {'checks': per, 'gaps': gaps}


def _mutate_parallel(saved: Dict[str, Callable], idx: Dict[str, dict], ids: List[str], corpus: Dict[str, Path],
                     expected: dict, sel: List[str], outcomes: Dict[str, RunOutcome], invoked: Dict[str, Set[str]],
                     log: Callable[[str], None], pool: concurrent.futures.Executor) -> dict:
    """selftest --mutate with one task per check id in forked workers (same search order as the sequential path;
    within a pool, faster fixtures are tried first among equal priority)."""
    per: Dict[str, dict] = {}
    gaps: List[str] = []
    futs = {}
    for rid in ids:
        if rid not in saved:
            per[rid] = {'status': 'unregistered'}
            continue
        mod = str(idx[rid].get('module'))
        users = [f for f in sel if rid in invoked.get(f, set())]
        dirty = [f for f in users if not fixture_is_clean(f, corpus.get(f), expected.get(f))]
        clean = [f for f in users if fixture_is_clean(f, corpus.get(f), expected.get(f))]
        rt = {f: (outcomes[f].runtime_s if f in outcomes else 0.0) for f in users}
        dirty.sort(key=lambda f: (0 if rid in (expected[f].get('must_fire') or []) else
                                  1 if rid in (expected[f].get('must_warn') or []) else 2, rt[f], f))
        clean.sort(key=lambda f: (rt[f], f))
        futs[rid] = pool.submit(_worker_mutate_check, rid, mod,
                                [(f, str(corpus[f]), expected[f]) for f in dirty],
                                [(f, str(corpus[f]), expected[f],
                                  sorted(outcomes[f].warned) if f in outcomes else []) for f in clean])
    for rid in ids:
        if rid in per:
            gaps.append(rid)
            log('GAP   mutate %-14s unregistered' % rid)
            continue
        try:
            res = futs[rid].result()
        except Exception as e:  # noqa: BLE001
            res = {'exercised_by': None, 'clean_coverage_by': None,
                   'error': 'worker error: %s' % common.summarize_exception(e)}
        labels = []
        if res.get('exercised_by') is None:
            labels.append('not exercised')
        if res.get('clean_coverage_by') is None:
            labels.append('no clean coverage')
        res['status'] = ', '.join(labels) or 'ok'
        per[rid] = res
        if labels:
            gaps.append(rid)
        log('%s  mutate %-14s %s' % ('ok  ' if not labels else 'GAP ', rid, res['status']))
    return {'checks': per, 'gaps': gaps}


def cmd_selftest(args: Any) -> int:
    """MFC selftest [--mutate] [--only ID]."""
    if getattr(args, 'git', None):
        sys.stderr.write('selftest takes no --git: every fixture runs in its own timeline mode\n')
        return common.EXIT_ERROR
    nested = common._SELFTEST_ACTIVE.get('depth', 0) > 0  # called by SELF-01 inside a gate: keep stdout clean
    as_json = bool(getattr(args, 'json', False))
    stream = sys.stderr if (nested or as_json) else sys.stdout

    def log(s: str) -> None:
        if not as_json or nested:
            stream.write(s + '\n')
    rc, rep = run_selftest(SelftestConfig(), only=getattr(args, 'only', None), mutate=bool(getattr(args, 'mutate', False)),
                           log=log)
    if as_json and not nested:
        print(json.dumps(common._jsonable(rep), indent=2, ensure_ascii=False))
    else:
        n = len(rep.get('fixtures') or {})
        bad = [f for f, v in (rep.get('fixtures') or {}).items() if not v.get('ok')]
        for prob in (rep.get('preflight_problems') or [])[:20]:
            stream.write('PROBLEM %s\n' % prob)
        stream.write('selftest: %s (%d fixtures, %d failing, %d preflight problems, %.1fs)%s\n'
                     % ('PASS' if rc == 0 else 'FAIL', n, len(bad), len(rep.get('preflight_problems') or []),
                        rep.get('runtime_s', 0.0), (' cache %s' % rep['cache']) if rep.get('cache') else ''))
    return rc


# ======================================================================================
# Held-out canaries (SPEC 12.3)
# ======================================================================================

HB_MAGIC = b'MRKHB1\n'
HB_NONCE = 16
HB_TAG = 32
HB_MIN_KEY = 16


def _hb_keys(key: bytes, batch_id: str) -> Tuple[bytes, bytes]:
    enc = hmac.new(key, b'minifly-heldout|enc|' + batch_id.encode('utf-8'), hashlib.sha256).digest()
    mac = hmac.new(key, b'minifly-heldout|mac|' + batch_id.encode('utf-8'), hashlib.sha256).digest()
    return enc, mac


def _keystream_xor(enc_key: bytes, nonce: bytes, data: bytes) -> bytes:
    out = bytearray(len(data))
    for block, off in enumerate(range(0, len(data), 32)):
        ks = hmac.new(enc_key, nonce + block.to_bytes(8, 'big'), hashlib.sha256).digest()
        chunk = data[off:off + 32]
        x = int.from_bytes(chunk, 'big') ^ int.from_bytes(ks[:len(chunk)], 'big')
        out[off:off + len(chunk)] = x.to_bytes(len(chunk), 'big')
    return bytes(out)


def _batch_files(in_dir: Path) -> List[Tuple[str, bytes]]:
    files: List[Tuple[str, bytes]] = []
    for dirpath, dirnames, filenames in os.walk(in_dir):
        dirnames[:] = sorted(d for d in dirnames if d not in common.SKIP_DIR_NAMES)
        for d in dirnames:
            if Path(dirpath, d).is_symlink():
                raise InvalidInput('symlinked directory in batch: %s' % Path(dirpath, d))
        for fn in sorted(filenames):
            p = Path(dirpath, fn)
            rel = p.relative_to(in_dir).as_posix()
            if common._is_cache_junk(rel):
                continue
            if p.is_symlink():
                raise InvalidInput('symlink in batch: %s' % rel)
            files.append((rel, p.read_bytes()))
    return sorted(files)


def batch_key_entries(key_obj: Any) -> Dict[str, dict]:
    """KEY.json: {fixture_id: [expected classes] | {classes|expected: [...], commands?: [[...]]}}."""
    if not isinstance(key_obj, dict) or not key_obj:
        raise InvalidInput('KEY.json must be a non-empty object {fixture: expected fired classes}')
    out: Dict[str, dict] = {}
    for fid, v in key_obj.items():
        if isinstance(v, list):
            entry = {'classes': v}
        elif isinstance(v, dict):
            entry = {'classes': v.get('classes', v.get('expected')), 'commands': v.get('commands')}
        else:
            raise InvalidInput('KEY.json %s: expected a list or an object' % fid)
        if not isinstance(entry['classes'], list) or not all(isinstance(x, str) for x in entry['classes']):
            raise InvalidInput('KEY.json %s: classes must be a list of FM or check ids' % fid)
        cmds = entry.get('commands')
        if cmds is not None and not (isinstance(cmds, list) and cmds and all(
                isinstance(c, list) and c and all(isinstance(t, str) for t in c) for c in cmds)):
            raise InvalidInput('KEY.json %s: commands must be a list of argv lists' % fid)
        out[str(fid)] = entry
    return out


def validate_batch(in_dir: Path) -> dict:
    key_p = in_dir / 'KEY.json'
    if not key_p.is_file():
        raise MissingInput('batch %s has no KEY.json' % in_dir)
    entries = batch_key_entries(common.load_json_file(key_p))
    dirs = set(corpus_fixtures(in_dir))
    if set(entries) != dirs:
        raise InvalidInput('KEY.json fixtures %s do not match the fixture dirs %s'
                           % (sorted(entries), sorted(dirs)))
    n = len(entries)
    n_clean = sum(1 for e in entries.values() if not e['classes'])
    if not ((n >= 10 and n_clean >= 3) or n == 1):
        raise InvalidInput('a held-out batch needs at least 10 fixtures with at least 3 clean (or exactly 1 fixture '
                           'for a held-out incident); got n=%d, n_clean=%d' % (n, n_clean))
    fm = sorted(set(c for e in entries.values() for c in e['classes'] if FM_RE.match(c)))
    return {'n': n, 'n_clean': n_clean, 'fm_coverage': fm, 'entries': entries}


def seal_batch(batch_id: str, in_dir: Path, key: bytes) -> Tuple[bytes, dict]:
    if not BATCH_ID_RE.match(batch_id or ''):
        raise InvalidInput('batch id %r must match %s' % (batch_id, BATCH_ID_RE.pattern))
    if len(key) < HB_MIN_KEY:
        raise InvalidInput('key file must hold at least %d bytes' % HB_MIN_KEY)
    files = _batch_files(in_dir)
    manifest = {'schema': HELDOUT_BUNDLE_SCHEMA, 'batch_id': batch_id,
                'files': [{'path': rel, 'size': len(b), 'sha256': common.sha256_bytes(b)} for rel, b in files]}
    mbytes = common.canonical_json(manifest)
    plain = len(mbytes).to_bytes(8, 'big') + mbytes + b''.join(b for _, b in files)
    enc, mac = _hb_keys(key, batch_id)
    nonce = secrets.token_bytes(HB_NONCE)
    ct = _keystream_xor(enc, nonce, plain)
    body = HB_MAGIC + nonce + ct
    tag = hmac.new(mac, body, hashlib.sha256).digest()
    return body + tag, manifest


def open_batch(batch_id: str, blob: bytes, key: bytes) -> Tuple[dict, Dict[str, bytes]]:
    """Verify the MAC and every file hash, then return (manifest, files). Raises InvalidInput."""
    if len(blob) < len(HB_MAGIC) + HB_NONCE + 8 + HB_TAG or not blob.startswith(HB_MAGIC):
        raise InvalidInput('not a sealed held-out bundle')
    enc, mac = _hb_keys(key, batch_id)
    body, tag = blob[:-HB_TAG], blob[-HB_TAG:]
    if not hmac.compare_digest(hmac.new(mac, body, hashlib.sha256).digest(), tag):
        raise InvalidInput('bundle authentication failed (tampered bundle, wrong key or wrong batch id)')
    nonce = body[len(HB_MAGIC):len(HB_MAGIC) + HB_NONCE]
    plain = _keystream_xor(enc, nonce, body[len(HB_MAGIC) + HB_NONCE:])
    mlen = int.from_bytes(plain[:8], 'big')
    manifest = common.loads_strict(plain[8:8 + mlen], what='bundle manifest')
    if not isinstance(manifest, dict) or manifest.get('schema') != HELDOUT_BUNDLE_SCHEMA \
            or manifest.get('batch_id') != batch_id:
        raise InvalidInput('bundle manifest schema or batch id mismatch')
    pos = 8 + mlen
    files: Dict[str, bytes] = {}
    for f in manifest.get('files') or []:
        rel = common.norm_rel(f.get('path'))
        size = int(f.get('size'))
        data = plain[pos:pos + size]
        pos += size
        if len(data) != size or common.sha256_bytes(data) != f.get('sha256'):
            raise InvalidInput('bundle file %s does not match its manifest hash' % rel)
        files[rel] = data
    if pos != len(plain):
        raise InvalidInput('bundle has trailing bytes after the manifest files')
    return manifest, files


def _default_sealed_path(repo_root: Path, batch_id: str) -> Path:
    return Path(repo_root) / KIT_REL / 'corpus' / 'heldout' / 'sealed' / ('%s.bin' % batch_id)


def _extra_opt(args: Any, name: str) -> Optional[str]:
    extra = list(getattr(args, 'extra', None) or [])
    for i, tok in enumerate(extra):
        if tok == name and i + 1 < len(extra):
            return extra[i + 1]
        if tok.startswith(name + '='):
            return tok.split('=', 1)[1]
    return None


def _commitment_row(ctx: common.Ctx, batch_id: str) -> Optional[dict]:
    rows = [r for r in ctx.kb.rows('CANARY_COMMITMENTS') if r.get('batch_id') == batch_id and not r.get('revealed_utc')]
    if len(rows) > 1:
        raise InvalidInput('more than one commitment row for batch %s' % batch_id)
    return rows[0] if rows else None


def _kit_code_digest_at(git: common.GitView, commit: str) -> Optional[str]:
    """Digest of the kit tree at `commit` excluding corpus/heldout/ (sealing a batch is not a checker change)."""
    paths = [p for p in git.list_files(commit, KIT_REL) if not common._is_cache_junk(p)
             and not p.startswith(KIT_REL + '/corpus/heldout/')]
    if not paths:
        return None
    blobs = git.files_at(commit, paths)
    h = hashlib.sha256()
    for p in sorted(paths):
        data = blobs.get(p)
        if data is None:
            raise MissingInput('cannot read %s at %s' % (p, commit))
        h.update(p.encode() + b'\0' + hashlib.sha256(data).digest())
    return h.hexdigest()


def _checker_merge_commit(git: common.GitView, checker_commit: str) -> Optional[common.Commit]:
    target = _kit_code_digest_at(git, checker_commit)
    if target is None:
        return None
    for c in git.log([KIT_REL + '/**']):
        if not git.is_ancestor(c.sha, checker_commit):
            continue
        if _kit_code_digest_at(git, c.sha) == target:
            return c
    return None


def _held_ctx(ctx: common.Ctx) -> dict:
    h = ctx.options.get('heldout')
    if not isinstance(h, dict):
        raise MissingInput('HELD-* checks run only under MFC heldout reveal|score')
    return h


@common.check('HELD-01')
def check_held_01(ctx: common.Ctx) -> Result:
    """The commitment row is committed before the merge of the checker version under test."""
    h = _held_ctx(ctx)
    row = h.get('row')
    if not row:
        raise MissingInput('no CANARY_COMMITMENTS row for batch %s' % h.get('batch_id'))
    if not h.get('checker_commit'):
        raise MissingInput('no checker version under test (--checker-sha or the row checker_sha_under_test)')
    git = ctx.git
    s = resolve_rev(git, h['checker_commit'])
    bsha = row.get('bundle_sha256')

    def has_row(data: bytes) -> bool:
        rows, _ = _parse_jsonl_bytes(data, CANARY_REL)
        return any(r.get('batch_id') == row.get('batch_id') and r.get('bundle_sha256') == bsha for r in rows)
    c_row = git.first_commit_where(CANARY_REL, has_row)
    if c_row is None:
        return Result('HELD-01', FAIL, 'commitment row for %s is not committed' % row.get('batch_id'))
    merge = _checker_merge_commit(git, s)
    if merge is None:
        raise MissingInput('cannot locate the merge of the checker version at %s' % _short(s))
    if c_row.sha != merge.sha and git.is_ancestor(c_row.sha, merge.sha):
        return Result('HELD-01', PASS, 'commitment %s precedes the merge %s of the checker version under test (%s)'
                      % (_short(c_row.sha), _short(merge.sha), _short(s)))
    return Result('HELD-01', FAIL, 'commitment row commit %s does not strictly precede the merge %s of the checker '
                  'version under test' % (_short(c_row.sha), _short(merge.sha)))


@common.check('HELD-02')
def check_held_02(ctx: common.Ctx) -> Result:
    """The revealed bundle and key hashes match the commitment (and the bundle authenticates)."""
    h = _held_ctx(ctx)
    row = h.get('row')
    if not row:
        raise MissingInput('no CANARY_COMMITMENTS row for batch %s' % h.get('batch_id'))
    bp, kp = Path(h['bundle_path']), Path(h['key_path'])
    if not bp.is_file():
        raise MissingInput('sealed bundle %s missing' % bp)
    if not kp.is_file():
        raise MissingInput('key file %s missing' % kp)
    blob, key = bp.read_bytes(), kp.read_bytes()
    probs = []
    if common.sha256_bytes(blob) != row.get('bundle_sha256'):
        probs.append('bundle sha256 %s != committed %s' % (common.sha256_bytes(blob)[:12], str(row.get('bundle_sha256'))[:12]))
    if common.sha256_bytes(key) != row.get('key_sha256'):
        probs.append('key sha256 does not match the commitment')
    if not probs:
        try:
            manifest, files = open_batch(str(row.get('batch_id')), blob, key)
            h['opened'] = (manifest, files)
        except InvalidInput as e:
            probs.append(str(e))
    if probs:
        return Result('HELD-02', FAIL, '; '.join(probs))
    return Result('HELD-02', PASS, 'bundle and key hashes match the commitment; %d files authenticated'
                  % len(h['opened'][1]))


@common.check('HELD-03')
def check_held_03(ctx: common.Ctx) -> Result:
    """The batch author identity differs from the authors of tools/minifly_check commits in the interval
    (every kit-code commit up to the merge of the checker version under test)."""
    h = _held_ctx(ctx)
    row = h.get('row')
    if not row:
        raise MissingInput('no CANARY_COMMITMENTS row for batch %s' % h.get('batch_id'))
    author = str(row.get('author_identity') or '').strip()
    if not author:
        return Result('HELD-03', FAIL, 'commitment row has no author_identity')
    if not h.get('checker_commit'):
        raise MissingInput('no checker version under test (--checker-sha or the row checker_sha_under_test)')
    git = ctx.git
    s = resolve_rev(git, h['checker_commit'])
    merge = _checker_merge_commit(git, s) or git.commit(s)
    if merge is None:
        raise MissingInput('cannot locate the checker version under test %s' % _short(s))
    a_low = author.lower()
    a_fam = a_low if a_low in ('codex', 'claude') else None
    clashes, unknown = [], []
    for c in git.log([KIT_REL + '/**']):
        if not git.is_ancestor(c.sha, merge.sha):
            continue
        code = [p for p in c.paths if p.startswith(KIT_REL + '/') and not p.startswith(KIT_REL + '/corpus/heldout/')]
        if not code:
            continue
        vals = [v.lower() for v in _acting_values(c)]
        if not vals:
            unknown.append(_short(c.sha))
            continue
        cfam, cst, _cwhy = common.commit_family(git, c)
        if cst != common.IDENTITY_OK:
            unknown.append('%s (%s)' % (_short(c.sha), cst))
            continue
        if a_low in vals or (a_fam and cfam == a_fam) or a_low == c.author.strip().lower():
            clashes.append(_short(c.sha))
    if clashes:
        return Result('HELD-03', FAIL, 'batch author %r also authored kit commits %s' % (author, ', '.join(clashes[:8])))
    if unknown:
        return Result('HELD-03', FAIL, 'kit commits without an Acting-Agent, or whose trailer and origin/<family>/* '
                      'namespace disagree (author unknown): %s' % ', '.join(unknown[:8]))
    return Result('HELD-03', PASS, 'batch author %r differs from every kit-code author up to %s'
                  % (author, _short(merge.sha)))


def _refuse_out(repo_root: Path, out: Path) -> Optional[str]:
    """Reason to refuse writing to `out`, or None (outside every delivery dir, the kit and .git)."""
    if not _inside(out, repo_root):
        return None
    rel = Path(out).resolve().relative_to(Path(repo_root).resolve()).as_posix()
    rel = '' if rel == '.' else rel
    if not rel:
        return 'refusing to write into the repository root'
    if rel == '.git' or rel.startswith('.git/'):
        return 'refusing to write into .git'
    if rel == KIT_REL or rel.startswith(KIT_REL + '/') or rel.startswith('.claude/'):
        return 'refusing to write into a protected path (%s)' % rel
    d = delivery_dir_of(rel + '/x')
    if d:
        return '%s lies inside the delivery directory %s' % (rel, d)
    cur = Path(repo_root)
    for part in rel.split('/'):
        cur = cur / part
        if common.V1_DIR_RE.match(part) or any((cur / n).is_file() for n in ('ENDPOINT_LOCK.json', 'SOURCE_LOCK.json',
                                                                            'LOCK.json')):
            return '%s lies inside the delivery directory %s' % (rel, cur.relative_to(repo_root).as_posix())
    return None


def _score_fixtures(kit: Path, fixtures: Dict[str, Path], entries: Dict[str, dict]) -> Dict[str, RunOutcome]:
    inv = _subprocess_invoker(kit)
    jobs = []
    for fid, p in sorted(fixtures.items()):
        cmds = entries[fid].get('commands')
        if not cmds:
            cmds = []
            repo = p / 'repo'
            for d in sorted(x.name for x in repo.iterdir() if x.is_dir() and common.is_v1_delivery(x)):
                for st in ('lock', 'roster', 'official', 'publish'):
                    cmds.append(['gate', '--stage', st, d])
        if not cmds:
            cmds = [['gate', '--stage', 'lock', '.']]
        jobs.append((fid, p, cmds, inv))
    return _run_corpus_jobs(jobs)


def canary_checker_version(fixture_dir: Path) -> Optional[str]:
    """The checker_version a held-out canary was written against: the oldest parseable SOURCE_LOCK.checker_version in
    its final tree (repo/**/SOURCE_LOCK.json; SPEC 20.18). None when the canary has no SOURCE_LOCK."""
    found: List[str] = []
    repo = Path(fixture_dir) / 'repo'
    if repo.is_dir():
        for p in sorted(repo.rglob('SOURCE_LOCK.json')):
            try:
                v = common.load_json_file(p).get('checker_version')
            except (MissingInput, InvalidInput, AttributeError):
                continue
            if common.parse_version(v) is not None:
                found.append(v)
    return min(found, key=common.parse_version) if found else None


def heldout_score_report(rules: dict, entries: Dict[str, dict], outs: Dict[str, RunOutcome],
                         batch_versions: Optional[Dict[str, Optional[str]]] = None) -> dict:
    """Recall and false-alarm rate of a revealed batch. SPEC 20.18 (round 1; the pilot's RUN-03 drift): a fired id
    whose RULES introduced_in or changed_in is later than the checker_version in the canary's own SOURCE_LOCK
    (batch_versions[fid]) is 'post_batch': a check the batch could not have been written against. It is listed per
    fixture and never counts toward recall or a false alarm (a clean canary whose nonzero exits only post-batch ids
    explain is not an alarm)."""
    fm_of: Dict[str, Set[str]] = {}
    ridx: Dict[str, dict] = {}
    for r in rules.get('rules') or []:
        if isinstance(r, dict) and r.get('id'):
            fm_of[str(r['id'])] = set(str(x) for x in (r.get('fm_ids') or []))
            ridx[str(r['id'])] = r
    versions = dict(batch_versions or {})
    per: Dict[str, dict] = {}
    fm_tot: Dict[str, List[int]] = {}
    post_all: Dict[str, List[str]] = {}
    caught_n = dirty_n = clean_n = alarms = 0
    caught_raw = alarms_raw = 0
    for fid, e in sorted(entries.items()):
        oc = outs.get(fid)
        fired_all = sorted(oc.fired) if oc else []
        bv = versions.get(fid)
        post = sorted(rid for rid in fired_all if rid in ridx and common.rule_newer_than(ridx[rid], bv))
        fired = [rid for rid in fired_all if rid not in post]
        if post:
            post_all[fid] = post
        fired_fm = set()
        for rid in fired:
            fired_fm |= fm_of.get(rid, set())
        fired_fm_raw = set()
        for rid in fired_all:
            fired_fm_raw |= fm_of.get(rid, set())
        classes = list(e['classes'])
        if not classes:
            clean_n += 1
            bad_exit = any(x != 0 for x in (oc.exits if oc else [1]))
            alarm = bool(fired) or (oc is None) or (bad_exit and not post) or bool(oc and oc.exit3_errors)
            alarm_raw = bool(fired_all) or (oc is None) or bad_exit or bool(oc and oc.exit3_errors)
            alarms += int(alarm)
            alarms_raw += int(alarm_raw)
            per[fid] = {'clean': True, 'fired': fired, 'post_batch': post, 'batch_checker_version': bv,
                        'exits': oc.exits if oc else None, 'false_alarm': alarm, 'false_alarm_raw': alarm_raw}
            continue
        dirty_n += 1
        hit = {}
        for c in classes:
            ok = (c in fired_fm) if FM_RE.match(c) else (c in fired)
            hit[c] = ok
            fm_tot.setdefault(c, [0, 0])
            fm_tot[c][1] += 1
            fm_tot[c][0] += int(ok)
        caught = all(hit.values())
        caught_n += int(caught)
        c_raw = all(((c in fired_fm_raw) if FM_RE.match(c) else (c in fired_all)) for c in classes)
        caught_raw += int(c_raw)
        per[fid] = {'clean': False, 'expected': classes, 'fired': fired, 'fired_fm': sorted(fired_fm),
                    'post_batch': post, 'batch_checker_version': bv,
                    'exits': oc.exits if oc else None, 'caught': caught, 'caught_raw': c_raw, 'per_class': hit,
                    'errors': (oc.errors + [m for _i, m in oc.exit3_errors]) if oc else ['not run']}
    return {'fixtures': per,
            'recall_per_class': {k: {'caught': v[0], 'n': v[1], 'recall': v[0] / float(v[1])} for k, v in sorted(fm_tot.items())},
            'recall_fixtures': (caught_n / float(dirty_n)) if dirty_n else None, 'n_dirty': dirty_n,
            'false_alarm_rate_clean': (alarms / float(clean_n)) if clean_n else None, 'n_clean': clean_n,
            # round-1 repair (verifier): the raw numbers, post_batch ids counted, are always reported next to the
            # excluded ones, so a batch stamped with an old version cannot hide a later check's hits and alarms
            'recall_fixtures_raw': (caught_raw / float(dirty_n)) if dirty_n else None,
            'false_alarm_rate_clean_raw': (alarms_raw / float(clean_n)) if clean_n else None,
            'post_batch': post_all,
            'post_batch_rule': 'fired ids whose RULES introduced_in or changed_in is later than the checker_version in '
                               "the canary's own SOURCE_LOCK are post_batch: listed, never counted (SPEC 20.18)"}


def cmd_heldout(args: Any) -> int:
    """MFC heldout {seal|reveal|score} --batch B [--in PATH] [--key-file K] [--checker-sha S] [--out PATH]
    [--author ID]."""
    action = args.action
    batch = getattr(args, 'batch', None)
    if not batch or not BATCH_ID_RE.match(batch):
        sys.stderr.write('heldout %s needs --batch ID matching %s\n' % (action, BATCH_ID_RE.pattern))
        return common.EXIT_MISSING
    root, git = common.git_view_from_spec(getattr(args, 'git', None), getattr(args, 'repo_root', None))
    key_file = getattr(args, 'key_file', None)
    if not key_file:
        sys.stderr.write('heldout %s needs --key-file (held by the owner outside the repo)\n' % action)
        return common.EXIT_MISSING
    kp = Path(key_file).resolve()
    if _inside(kp, root):
        sys.stderr.write('refusing: the key file must lie outside the repository\n')
        return common.EXIT_FAIL
    if action == 'seal':
        in_dir = getattr(args, 'in_dir', None)
        if not in_dir or not Path(in_dir).is_dir():
            sys.stderr.write('heldout seal needs --in DIR (the plaintext batch)\n')
            return common.EXIT_MISSING
        if _inside(in_dir, root):
            sys.stderr.write('refusing: the plaintext batch must lie outside the repository\n')
            return common.EXIT_FAIL
        author = _extra_opt(args, '--author') or os.environ.get('MINIFLY_HELDOUT_AUTHOR')
        if not author:
            sys.stderr.write('heldout seal needs --author IDENTITY (or MINIFLY_HELDOUT_AUTHOR)\n')
            return common.EXIT_MISSING
        if not kp.is_file():
            sys.stderr.write('key file %s missing\n' % kp)
            return common.EXIT_MISSING
        info = validate_batch(Path(in_dir))
        blob, manifest = seal_batch(batch, Path(in_dir), kp.read_bytes())
        out = Path(getattr(args, 'out', None) or _default_sealed_path(root, batch))
        if out.exists():
            sys.stderr.write('refusing to overwrite %s (sealed batches are immutable)\n' % out)
            return common.EXIT_FAIL
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(blob)
        row = {'batch_id': batch, 'author_identity': author, 'utc': common.utc_now(),
               'bundle_sha256': common.sha256_bytes(blob), 'key_sha256': common.sha256_file(kp), 'n': info['n'],
               'n_clean': info['n_clean'], 'fm_coverage': info['fm_coverage'],
               'checker_sha_under_test': getattr(args, 'checker_sha', None), 'revealed_utc': None}
        sys.stderr.write('sealed %d files into %s; append this row with `MFC ledger append --ledger CANARY_COMMITMENTS` '
                         'and commit it before the checker version under test merges (HELD-01)\n'
                         % (len(manifest['files']), out))
        print(json.dumps(row, sort_keys=True))
        return common.EXIT_PASS
    if action not in ('reveal', 'score'):
        sys.stderr.write('unknown heldout action %r\n' % action)
        return common.EXIT_ERROR
    ctx = common.make_ctx(args, delivery=None, stage='heldout', options={'invoked_by': 'heldout'})
    row = _commitment_row(ctx, batch)
    bundle_path = Path(getattr(args, 'in_dir', None) or _default_sealed_path(root, batch))
    checker = getattr(args, 'checker_sha', None) or (row or {}).get('checker_sha_under_test')
    ctx.options['heldout'] = {'batch_id': batch, 'row': row, 'bundle_path': str(bundle_path), 'key_path': str(kp),
                              'checker_commit': checker}
    results = common.run_bundle(ctx, 'heldout')
    rc = common.exit_code(results)
    print(common.format_results(results))
    opened = ctx.options['heldout'].get('opened')
    held02 = [r for r in results if r.id == 'HELD-02']
    if not opened or not held02 or held02[0].status != PASS:
        sys.stderr.write('heldout %s: bundle not revealed (HELD-02 did not pass)\n' % action)
        return rc if rc != common.EXIT_PASS else common.EXIT_FAIL
    manifest, files = opened
    if action == 'reveal':
        out_dir = getattr(args, 'out', None)
        dest = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix='minifly_heldout_%s_' % batch))
        if _inside(dest, root):
            sys.stderr.write('refusing: revealed fixtures must not be written inside the repository\n')
            return common.EXIT_FAIL
        for rel, data in files.items():
            t = dest / rel
            t.parent.mkdir(parents=True, exist_ok=True)
            t.write_bytes(data)
        print(json.dumps(dict(row or {}, revealed_utc=common.utc_now()), sort_keys=True))
        sys.stderr.write('revealed %d files into %s\n' % (len(files), dest))
        return rc
    # score
    if not checker:
        sys.stderr.write('heldout score needs --checker-sha S (a commit)\n')
        return common.EXIT_MISSING
    out = Path(getattr(args, 'out', None) or (Path(root) / common.KB_REL / 'RETRO' /
                                              ('HELDOUT_%s.json' % common.utc_now()[:8])))
    why = _refuse_out(root, out.parent)
    if why:
        sys.stderr.write('refusing --out %s: %s\n' % (out, why))
        return common.EXIT_FAIL
    if out.exists():
        sys.stderr.write('refusing to overwrite %s\n' % out)
        return common.EXIT_FAIL
    with tempfile.TemporaryDirectory(prefix='minifly_heldout_score_') as td:
        tdp = Path(td)
        rev = tdp / 'revealed'
        for rel, data in files.items():
            t = rev / rel
            t.parent.mkdir(parents=True, exist_ok=True)
            t.write_bytes(data)
        entries = batch_key_entries(common.load_json_file(rev / 'KEY.json'))
        fixtures = corpus_fixtures(rev)
        s = resolve_rev(git, checker)
        kit, note = materialize_kit(git, s, tdp / 'checker')
        if not kit_runnable(kit):
            sys.stderr.write('checker at %s is not runnable\n' % _short(s))
            return common.EXIT_FAIL
        rules = common.load_json_file(kit / 'RULES.json')
        outs = _score_fixtures(kit, fixtures, entries)
        versions = {fid: canary_checker_version(p) for fid, p in fixtures.items()}
        rep = heldout_score_report(rules if isinstance(rules, dict) else {}, entries, outs, versions)
        valid = rc == common.EXIT_PASS
        report = {'schema': HELDOUT_REPORT_SCHEMA, 'batch_id': batch, 'utc': common.utc_now(), 'checker_commit': s,
                  'checker_sha256': common.checker_sha256(kit, refresh=True), 'checker_source': note,
                  'heldout_valid': valid, 'held_checks': [r.to_dict() for r in results],
                  'commitment': row, 'score': rep}
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(common.dumps_pretty(common._jsonable(report)), encoding='utf-8')
    print('heldout score %s: recall %s, false-alarm rate %s (raw, post_batch counted: recall %s, false-alarm rate %s)'
          '%s; report %s%s'
          % (batch, rep['recall_fixtures'], rep['false_alarm_rate_clean'], rep['recall_fixtures_raw'],
             rep['false_alarm_rate_clean_raw'],
             '; post_batch ids excluded: %s' % ', '.join(sorted(set(i for v in rep['post_batch'].values() for i in v)))
             if rep['post_batch'] else '', out,
             '' if valid else ' (NOT a valid held-out measurement: a HELD check did not pass)'))
    return rc
