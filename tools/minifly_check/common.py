"""MRK-1 common library (WP1): shared interfaces of SPEC Section 7.3.

Normative for every check module. Python 3.9+, standard library only.

Import convention: every kit module imports this file top-level as ``import common``
(``minifly_check.py`` and this module both put the kit directory at sys.path[0]).
Never use relative imports between kit modules.

Fail-closed rules (SPEC 7.6) implemented here:
- an exception inside a check gives FAIL with a traceback summary;
- MissingInput gives MISSING; InvalidInput gives FAIL with the reason;
- a missing module or check function gives CHECKER_ERROR (exit 3);
- a bundle that references a planned or unregistered id raises CheckerError (exit 3);
- NaN/Infinity in any JSON input is rejected (InvalidInput);
- git failure, timeout or a shallow clone (for history-dependent queries) gives MissingInput.
"""
from __future__ import annotations

import difflib
import gzip
import hashlib
import importlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

KIT_DIR = Path(__file__).resolve().parent
if str(KIT_DIR) not in sys.path:
    sys.path.insert(0, str(KIT_DIR))
if __name__ != 'common':
    # One registry per process: an import as tools.minifly_check.common aliases 'common'.
    sys.modules.setdefault('common', sys.modules[__name__])

CHECKER_VERSION = '1.0.1'
EXIT_PASS, EXIT_FAIL, EXIT_MISSING, EXIT_ERROR = 0, 1, 2, 3

PASS = 'PASS'
FAIL = 'FAIL'
MISSING = 'MISSING'
WARN = 'WARN'
ADVISORY_FAIL = 'ADVISORY_FAIL'
NOT_APPLICABLE = 'NOT_APPLICABLE'
CHECKER_ERROR = 'CHECKER_ERROR'
STATUSES = (PASS, FAIL, MISSING, WARN, ADVISORY_FAIL, NOT_APPLICABLE, CHECKER_ERROR)
BLOCKING_STATUSES = (FAIL, MISSING, CHECKER_ERROR)
NON_BLOCKING_STATUSES = (PASS, WARN, ADVISORY_FAIL, NOT_APPLICABLE)

RULE_STATUSES = ('enforced', 'advisory', 'planned')
KIT_INTEGRITY_IDS = ('SELF-01', 'SCHEMA-01', 'TIER-01')  # never downgraded

SCHEMA_ENDPOINT_LOCK = 'minifly.endpoint_lock.v1'
SCHEMA_SOURCE_LOCK = 'minifly.source_lock.v1'
SCHEMA_GATE_REPORT = 'minifly.gate_report.v1'
SCHEMA_ATTEMPT = 'minifly.attempt.v1'
SCHEMA_CYCLE = 'minifly.cycle.v1'

V1_DIR_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]*_(design|dev)_v[0-9]+_[0-9]{8}$')
RESULT_BUNDLE_RE = re.compile(r'^RESULT_BUNDLE[^/]*\.zip$')
EVALUATOR_ARM_PREFIXES = ('CEILING_', 'FLOOR_')
# Determinism replay of a resumed official run (SPEC 20.8, RUN-04): rows of completed (world, arm) units re-run by
# the resume attempt. Never scored; RUN-04 compares their canonical digest with the official rows.
REPLAY_DIR = 'results/official_replay'
REPLAY_OUTPUTS = REPLAY_DIR + '/OUTPUTS.jsonl'
KB_REL = 'PROJECT_DOCUMENTS/KB'
KIT_REL_DEFAULT = 'tools/minifly_check'

MAX_SCAN_BYTES = 64 * 1024 * 1024
DEFAULT_SCAN_PATTERNS = ('*.json', '*.jsonl', '*.jsonl.gz', '*.md', '*.py')
DEFAULT_SCAN_EXCLUDES = ('.git', 'tools/minifly_check/corpus', 'tools/minifly_check/templates')
SKIP_DIR_NAMES = ('.git', '__pycache__')

GIT_TIMEOUT_S = 30
_ALLOWED_GIT = ('rev-parse', 'log', 'show', 'cat-file', 'ls-files', 'diff', 'merge-base', 'branch', 'archive')
_EMPTY_TREE = {
    'sha1': '4b825dc642cb6eb9a060e54bf8d69288fbee4904',
    'sha256': '6ef19b41225c5369f1c104d45d8d85efa9b057b53b14b4b9b939dd74decc5321',
}
SPEC_DEFAULT_GRANT_CAPS = {'max_cpu_h': 2, 'max_records': 5000, 'max_cycles': 3}


# --------------------------------------------------------------------------------------
# Exceptions and results
# --------------------------------------------------------------------------------------

class MissingInput(Exception):
    """A required input is absent -> status MISSING (exit 2)."""


class InvalidInput(Exception):
    """An input is present but malformed -> status FAIL with the reason."""


class CheckerError(Exception):
    """The checker itself is inconsistent (RULES/bundle/registry) -> CHECKER_ERROR (exit 3)."""


@dataclass
class Result:
    id: str
    status: str
    detail: str = ''
    evidence: list = field(default_factory=list)
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = {'id': self.id, 'status': self.status, 'detail': self.detail,
               'evidence': [str(e) for e in (self.evidence or [])]}
        if self.data:
            out['data'] = _jsonable(self.data)
        return out


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        seq = sorted(obj, key=str) if isinstance(obj, set) else obj
        return [_jsonable(v) for v in seq]
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, float) and (obj != obj or obj in (float('inf'), float('-inf'))):
        return str(obj)
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    return str(obj)


CHECKS: Dict[str, Callable] = {}


def _def_key(fn: Callable) -> tuple:
    code = getattr(fn, '__code__', None)
    return (getattr(code, 'co_filename', None), getattr(code, 'co_firstlineno', None),
            getattr(fn, '__qualname__', None))


def check(check_id: str):
    """Register fn(ctx) -> Result | list[Result] under check_id. A duplicate id raises.

    Re-importing the same source definition (same file, line and qualname, e.g. a module
    imported under two names) is not a duplicate.
    """
    if not isinstance(check_id, str) or not check_id:
        raise CheckerError('check id must be a non-empty string')

    def deco(fn: Callable) -> Callable:
        prev = CHECKS.get(check_id)
        if prev is not None and _def_key(prev) != _def_key(fn):
            raise CheckerError('duplicate check id %s (%s vs %s)' % (check_id, _def_key(prev), _def_key(fn)))
        CHECKS[check_id] = fn
        return fn
    return deco


# --------------------------------------------------------------------------------------
# JSON, hashing, time
# --------------------------------------------------------------------------------------

def _reject_constant(name: str):
    raise InvalidInput('non-finite JSON constant %s is not allowed' % name)


def loads_strict(text: Any, what: str = 'JSON') -> Any:
    """json.loads that rejects NaN/Infinity (InvalidInput) and reports decode errors as InvalidInput."""
    if isinstance(text, (bytes, bytearray)):
        try:
            text = bytes(text).decode('utf-8')
        except UnicodeDecodeError as e:
            raise InvalidInput('%s is not UTF-8: %s' % (what, e))
    try:
        return json.loads(text, parse_constant=_reject_constant)
    except InvalidInput:
        raise
    except ValueError as e:
        raise InvalidInput('%s does not parse: %s' % (what, e))


def load_json_file(path: Any) -> Any:
    p = Path(path)
    if not p.is_file():
        raise MissingInput('missing file %s' % p)
    try:
        data = p.read_bytes()
    except OSError as e:
        raise MissingInput('unreadable file %s: %s' % (p, e))
    return loads_strict(data, what=str(p))


def load_jsonl_file(path: Any) -> List[dict]:
    p = Path(path)
    if not p.is_file():
        raise MissingInput('missing file %s' % p)
    out = []
    with open(p, 'rb') as fh:
        for _lineno, row in _iter_jsonl_stream(fh, str(p)):
            out.append(row)
    return out


def canonical_json(obj: Any) -> bytes:
    """sort_keys, separators (',', ':'), UTF-8, no NaN/Infinity (InvalidInput)."""
    try:
        return json.dumps(obj, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                          allow_nan=False).encode('utf-8')
    except ValueError as e:
        raise InvalidInput('canonical_json: %s' % e)
    except TypeError as e:
        raise InvalidInput('canonical_json: unsupported type: %s' % e)


def dumps_pretty(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
    except ValueError as e:
        raise InvalidInput('dumps_pretty: %s' % e)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Any) -> str:
    p = Path(path)
    h = hashlib.sha256()
    try:
        with open(p, 'rb') as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b''):
                h.update(chunk)
    except FileNotFoundError:
        raise MissingInput('missing file %s' % p)
    except OSError as e:
        raise MissingInput('unreadable file %s: %s' % (p, e))
    return h.hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj))


# The split manifests the lock fixes by hash (SPEC 6.2 "The held set is fixed in the lock (split.held_set_sha256)
# before any stream exists"; SCHEMAS.md section 1): (lock split key, pool).
SPLIT_MANIFESTS = (('held_manifest', 'HELD'), ('old_manifest', 'OLD'), ('new_manifest', 'NEW'))


def split_hashes(blobs: Dict[str, Optional[bytes]]) -> Tuple[Optional[str], Optional[str]]:
    """(held_set_sha256, split_sha256) of the split manifests, given {split key: file bytes or None}.
    held_set_sha256 = sha256(held manifest bytes); split_sha256 = sha256(canonical_json({held_manifest,
    old_manifest, new_manifest: sha256 of each file's bytes})). A missing manifest gives None."""
    shas = {k: (sha256_bytes(blobs[k]) if blobs.get(k) is not None else None) for k, _p in SPLIT_MANIFESTS}
    held = shas['held_manifest']
    split = sha256_json(shas) if all(v is not None for v in shas.values()) else None
    return held, split


def sha8(text: str) -> str:
    return sha256_bytes(text.encode('utf-8'))[:8]


def utc_now() -> str:
    """'YYYYMMDDTHHMMSSZ'."""
    return time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())


def utc_from_epoch(ts: float) -> str:
    return time.strftime('%Y%m%dT%H%M%SZ', time.gmtime(ts))


_UTC_FORMS = (
    (re.compile(r'^\d{8}T\d{6}Z$'), '%Y%m%dT%H%M%SZ'),
    (re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$'), '%Y-%m-%dT%H:%M:%SZ'),
    (re.compile(r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}Z$'), '%Y-%m-%dT%H:%MZ'),
)


def normalize_utc(value: Any) -> Optional[str]:
    """Return 'YYYYMMDDTHHMMSSZ' for the accepted UTC forms, else None."""
    if not isinstance(value, str):
        return None
    s = value.strip()
    if s.endswith('+00:00'):
        s = s[:-6] + 'Z'
    for rx, fmt in _UTC_FORMS:
        if rx.match(s):
            try:
                return time.strftime('%Y%m%dT%H%M%SZ', time.strptime(s, fmt))
            except ValueError:
                return None
    return None


def parse_version(v: Any) -> Optional[Tuple[int, int, int]]:
    if not isinstance(v, str):
        return None
    m = re.fullmatch(r'(\d+)\.(\d+)\.(\d+)', v.strip())
    if not m:
        return None
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)))


def summarize_exception(exc: BaseException, frames: int = 3) -> str:
    tb = [f for f in traceback.extract_tb(exc.__traceback__) if not f.filename.startswith('<frozen')][-frames:]
    where = '; '.join('%s:%d in %s' % (os.path.basename(f.filename), f.lineno, f.name) for f in tb)
    msg = ''.join(traceback.format_exception_only(type(exc), exc)).strip()
    return '%s [%s]' % (msg, where) if where else msg


# --------------------------------------------------------------------------------------
# Paths and globbing
# --------------------------------------------------------------------------------------

def norm_rel(path: Any) -> str:
    """Normalize a relative POSIX path; reject absolute paths and '..' escapes (InvalidInput)."""
    s = str(path).replace('\\', '/')
    if s.startswith('/') or re.match(r'^[A-Za-z]:', s):
        raise InvalidInput('absolute path not allowed: %s' % s)
    parts = []
    for part in s.split('/'):
        if part in ('', '.'):
            continue
        if part == '..':
            raise InvalidInput('path escapes its base: %s' % s)
        parts.append(part)
    return '/'.join(parts)


_GLOB_CACHE: Dict[str, Any] = {}


def glob_to_regex(pattern: str):
    rx = _GLOB_CACHE.get(pattern)
    if rx is not None:
        return rx
    i, n, out = 0, len(pattern), []
    while i < n:
        c = pattern[i]
        if c == '*':
            if pattern.startswith('**/', i):
                out.append('(?:.*/)?')
                i += 3
                continue
            if pattern.startswith('**', i):
                out.append('.*')
                i += 2
                continue
            out.append('[^/]*')
        elif c == '?':
            out.append('[^/]')
        elif c == '[':
            j = pattern.find(']', i + 1)
            if j == -1:
                out.append(re.escape(c))
            else:
                body = pattern[i + 1:j]
                if body.startswith('!'):
                    body = '^' + body[1:]
                out.append('[' + body.replace('\\', '\\\\') + ']')
                i = j + 1
                continue
        else:
            out.append(re.escape(c))
        i += 1
    rx = re.compile('^' + ''.join(out) + '$')
    _GLOB_CACHE[pattern] = rx
    return rx


def has_glob(pattern: str) -> bool:
    return any(ch in pattern for ch in '*?[')


def glob_match(path: str, pattern: str) -> bool:
    """Match a repo- or delivery-relative POSIX path.

    '*' and '?' never cross '/', '**' does, '**/' matches zero or more directories.
    A pattern without wildcards matches the path itself or anything below it as a directory.
    """
    path = str(path).replace('\\', '/').strip('/')
    pattern = str(pattern).replace('\\', '/').strip('/')
    if not has_glob(pattern):
        return path == pattern or path.startswith(pattern + '/')
    return bool(glob_to_regex(pattern).match(path))


def _is_cache_junk(rel: str) -> bool:
    return '__pycache__/' in rel or rel.startswith('__pycache__') or rel.endswith('.pyc')


# --------------------------------------------------------------------------------------
# Git views
# --------------------------------------------------------------------------------------

@dataclass
class Commit:
    sha: str
    utc: str
    author: str
    trailers: Dict[str, List[str]]
    paths: List[str]

    def trailer_values(self, key: str) -> List[str]:
        """Case-insensitive trailer lookup."""
        k = key.lower()
        out: List[str] = []
        for name, vals in (self.trailers or {}).items():
            if name.lower() == k:
                out.extend(vals)
        return out


_TRAILER_RE = re.compile(r'^([A-Za-z][A-Za-z0-9-]*):[ \t]+(.*\S)\s*$')


def parse_trailers(message: str) -> Dict[str, List[str]]:
    """Trailers of the last paragraph of a commit message (every non-blank line must be one)."""
    paras = [p for p in re.split(r'\n\s*\n', (message or '').strip()) if p.strip()]
    if len(paras) < 2:
        return {}  # a subject line alone never carries trailers
    last = paras[-1].splitlines()
    out: Dict[str, List[str]] = {}
    current = None
    for line in last:
        if not line.strip():
            continue
        if line[:1] in (' ', '\t') and current is not None:
            out[current][-1] = out[current][-1] + ' ' + line.strip()
            continue
        m = _TRAILER_RE.match(line)
        if not m:
            return {}
        current = m.group(1)
        out.setdefault(current, []).append(m.group(2))
    return out


AGENT_FAMILIES = ('codex', 'claude')
_NS_RE = re.compile(r'^(?:refs/remotes/|remotes/)?origin/(codex|claude)/.+')


def agent_family(agent: Any) -> Optional[str]:
    """'codex' or 'claude' for an identity string ('claude', 'claude:<session>', 'codex-cli', ...); None when it
    names neither family."""
    if not isinstance(agent, str):
        return None
    a = agent.strip().lower()
    for fam in AGENT_FAMILIES:
        if a == fam or (a.startswith(fam) and a[len(fam):len(fam) + 1] in (':', '-', '_', ' ', '/', '@')):
            return fam
    return None


def branch_namespace(branch: Any) -> Optional[str]:
    """The agent namespace of a remote branch: 'claude' for origin/claude/*, 'codex' for origin/codex/*, else None
    (another remote, or a branch outside both namespaces, never counts)."""
    m = _NS_RE.match(str(branch or '').strip())
    return m.group(1) if m else None


IDENTITY_OK, IDENTITY_MISMATCH, IDENTITY_UNPUSHED, IDENTITY_NO_COMMIT = 'ok', 'identity_mismatch', 'unpushed', \
    'no_commit'


def commit_family(git: 'GitView', commit: Optional['Commit']) -> Tuple[Optional[str], str, str]:
    """The acting family of a commit (SPEC 5, SPEC 20.15; round 1, INC-HB1-C3): (family, status, why).

    The family is derived from BOTH the commit's `Acting-Agent:` trailer and the remote branch namespaces that
    contain it (origin/claude/* vs origin/codex/*); git author names and row fields never count. status:
      ok                 the trailer names exactly one family and an origin/<family>/* branch contains the commit;
      identity_mismatch  no trailer, a trailer naming no or both families, or a trailer whose namespace contains
                         nothing: the commit is contained only in the other family's namespace (each agent pushes
                         only to its own namespace, so the trailer is forged or the commit was never pushed by it);
      unpushed           no origin/claude/* or origin/codex/* branch contains the commit (identity not derivable);
      no_commit          commit is None.
    A commit contained in both namespaces is shared history (the reviewer branches from the author's work, the
    author merges the review): it is attributed to its trailer family, which must be one of them. Every caller
    treats anything but ok as FAIL (unpushed: MISSING where the check allows it)."""
    if commit is None:
        return None, IDENTITY_NO_COMMIT, 'no commit'
    vals = [v.strip() for v in commit.trailer_values('Acting-Agent') if v.strip()]
    short = commit.sha[:12]
    if not vals:
        return None, IDENTITY_MISMATCH, 'commit %s has no Acting-Agent trailer' % short
    fams = [agent_family(v) for v in vals]
    if any(f is None for f in fams):
        return None, IDENTITY_MISMATCH, 'commit %s Acting-Agent %s names no agent family' % (short, vals)
    if len(set(fams)) != 1:
        return None, IDENTITY_MISMATCH, 'commit %s Acting-Agent %s names both families' % (short, vals)
    t = fams[0]
    branches = [str(b) for b in (git.remote_branches_containing(commit.sha) or [])]
    ns = sorted(set(f for f in (branch_namespace(b) for b in branches) if f))
    if not ns:
        return None, IDENTITY_UNPUSHED, ('commit %s (Acting-Agent %s) is in no origin/%s/* branch, nor in any '
                                         'origin/claude/* or origin/codex/* branch (branches: %s)'
                                         % (short, t, t, ', '.join(branches[:4]) or 'none'))
    if t not in ns:
        return None, IDENTITY_MISMATCH, ('commit %s says Acting-Agent %s but is in no origin/%s/* branch, only in '
                                         'origin/%s/* (branches: %s): trailer and namespace disagree'
                                         % (short, t, t, ns[0], ', '.join(branches[:4])))
    own = [b for b in branches if branch_namespace(b) == t]
    return t, IDENTITY_OK, '%s by %s (%s%s)' % (short, t, own[0], ', shared history' if len(ns) > 1 else '')


def commit_is_family(git: 'GitView', commit: Optional['Commit'], fam: Optional[str]) -> Tuple[bool, str]:
    """(ok, why): commit_family(commit) is `fam`."""
    got, st, why = commit_family(git, commit)
    if st != IDENTITY_OK:
        return False, '%s: %s' % (st, why)
    if got != fam:
        return False, 'commit %s acts as %s, not %s' % (commit.sha[:12] if commit else '-', got, fam)
    return True, why


class GitView:
    """Read-only view of history. Implemented by RealGit(repo_root) and TimelineGit(fixture_dir)."""
    mode = 'abstract'

    def head(self) -> str:
        raise NotImplementedError

    def is_shallow(self) -> bool:
        raise NotImplementedError

    def log(self, paths: Optional[Sequence[str]] = None) -> List[Commit]:
        raise NotImplementedError

    def first_commit_adding(self, glob: str) -> Optional[Commit]:
        raise NotImplementedError

    def first_commit_where(self, path: str, pred_bytes: Callable[[bytes], Any]) -> Optional[Commit]:
        raise NotImplementedError

    def is_ancestor(self, a: str, b: str) -> bool:
        raise NotImplementedError

    def file_at(self, sha: str, path: str) -> Optional[bytes]:
        raise NotImplementedError

    def remote_branches_containing(self, sha: str) -> List[str]:
        raise NotImplementedError

    def diff_lines(self, base: str, head: str, path: str) -> Tuple[List[str], List[str]]:
        raise NotImplementedError

    def changed_paths(self, base: str, head: str) -> List[str]:
        raise NotImplementedError

    def dirty_paths(self, paths: Sequence[str]) -> List[str]:
        """Paths under `paths` that are modified, deleted, staged-new or untracked in the worktree."""
        raise NotImplementedError

    def worktree_clean(self, paths: Sequence[str]) -> bool:
        return not self.dirty_paths(paths)

    def list_files(self, sha: str, prefix: str = '') -> List[str]:
        """Repo-relative paths of files present at commit `sha` under `prefix`."""
        raise NotImplementedError

    def files_at(self, sha: str, paths: Sequence[str]) -> Dict[str, Optional[bytes]]:
        return {p: self.file_at(sha, p) for p in paths}

    def commit(self, sha: str) -> Optional[Commit]:
        for c in self.log():
            if c.sha == sha or (len(sha) >= 4 and c.sha.startswith(sha)):
                return c
        return None


def _git_env() -> Dict[str, str]:
    env = dict(os.environ)
    env.update({'GIT_OPTIONAL_LOCKS': '0', 'GIT_LITERAL_PATHSPECS': '1', 'GIT_TERMINAL_PROMPT': '0',
                'GIT_PAGER': 'cat', 'PAGER': 'cat', 'LC_ALL': 'C', 'LANG': 'C'})
    for k in ('GIT_DIR', 'GIT_WORK_TREE', 'GIT_INDEX_FILE'):
        env.pop(k, None)
    return env


def run_git(root: Any, args: Sequence[str], input_bytes: Optional[bytes] = None,
            ok: Sequence[int] = (0,), allow_archive: bool = False,
            timeout: float = GIT_TIMEOUT_S) -> subprocess.CompletedProcess:
    """Run a read-only git command with fixed argv and a timeout. Failures raise MissingInput."""
    args = list(args)
    if not args or args[0] not in _ALLOWED_GIT:
        raise CheckerError('git subcommand not allowed: %r' % (args[:1],))
    if args[0] == 'merge-base' and '--is-ancestor' not in args:
        raise CheckerError('only merge-base --is-ancestor is allowed')
    if args[0] == 'branch' and not ('-r' in args and '--contains' in args):
        raise CheckerError('only branch -r --contains is allowed')
    if args[0] == 'archive' and not allow_archive:
        raise CheckerError('git archive is reserved for governance')
    argv = ['git', '-c', 'core.quotePath=false', '-c', 'core.fsmonitor=false', '-c', 'color.ui=never'] + args
    try:
        proc = subprocess.run(argv, cwd=str(root), input=input_bytes, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, timeout=timeout, env=_git_env())
    except subprocess.TimeoutExpired:
        raise MissingInput('git %s timed out after %ss' % (args[0], timeout))
    except OSError as e:
        raise MissingInput('git unavailable: %s' % e)
    if proc.returncode not in ok:
        raise MissingInput('git %s failed (%d): %s' % (' '.join(args[:3]), proc.returncode,
                                                       proc.stderr.decode('utf-8', 'replace').strip()[:300]))
    return proc


def git_toplevel(path: Any) -> Path:
    p = Path(path)
    cwd = p if p.is_dir() else p.parent
    proc = run_git(cwd, ['rev-parse', '--show-toplevel'])
    return Path(proc.stdout.decode('utf-8').strip()).resolve()


def _lines(b: bytes) -> List[str]:
    return [ln for ln in b.decode('utf-8', 'replace').splitlines() if ln.strip()]


class RealGit(GitView):
    mode = 'real'

    def __init__(self, repo_root: Any):
        self.root = Path(repo_root).resolve()
        self._log: Optional[List[Commit]] = None
        self._by_sha: Dict[str, Commit] = {}
        self._added: Optional[List[Tuple[str, List[str]]]] = None
        self._shallow: Optional[bool] = None
        self._head: Optional[str] = None
        self._fmt: Optional[str] = None

    def _git(self, args, **kw) -> subprocess.CompletedProcess:
        return run_git(self.root, args, **kw)

    def head(self) -> str:
        if self._head is None:
            self._head = self._git(['rev-parse', 'HEAD']).stdout.decode().strip()
        return self._head

    def is_shallow(self) -> bool:
        if self._shallow is None:
            out = self._git(['rev-parse', '--is-shallow-repository']).stdout.decode().strip()
            if out in ('true', 'false'):
                self._shallow = out == 'true'
            else:
                gd = self._git(['rev-parse', '--git-dir']).stdout.decode().strip()
                gdp = Path(gd) if os.path.isabs(gd) else self.root / gd
                self._shallow = (gdp / 'shallow').exists()
        return self._shallow

    def _require_full_history(self) -> None:
        if self.is_shallow():
            raise MissingInput('shallow clone: history-dependent git query unavailable (git fetch --unshallow)')

    def _load_log(self) -> List[Commit]:
        if self._log is None:
            self._require_full_history()
            proc = self._git(['log', '--reverse', '--no-renames', '--name-only',
                              '--format=%x1e%H%x1f%ct%x1f%an%x1f%B%x1f', 'HEAD'])
            text = proc.stdout.decode('utf-8', 'replace')
            commits: List[Commit] = []
            for rec in text.split('\x1e'):
                if not rec.strip():
                    continue
                parts = rec.split('\x1f', 4)
                if len(parts) < 5:
                    raise MissingInput('unparseable git log record')
                sha, ct, author, body, rest = parts
                try:
                    utc = utc_from_epoch(int(ct.strip()))
                except ValueError:
                    utc = ''
                paths = [ln.strip() for ln in rest.splitlines() if ln.strip()]
                c = Commit(sha=sha.strip(), utc=utc, author=author, trailers=parse_trailers(body), paths=paths)
                commits.append(c)
                self._by_sha[c.sha] = c
            self._log = commits
        return self._log

    def log(self, paths: Optional[Sequence[str]] = None) -> List[Commit]:
        commits = self._load_log()
        if paths is None:
            return list(commits)
        if isinstance(paths, str):
            paths = [paths]
        return [c for c in commits if any(glob_match(p, pat) for p in c.paths for pat in paths)]

    def commit(self, sha: str) -> Optional[Commit]:
        self._load_log()
        if sha in self._by_sha:
            return self._by_sha[sha]
        return GitView.commit(self, sha)

    def first_commit_adding(self, glob: str) -> Optional[Commit]:
        if self._added is None:
            self._require_full_history()
            proc = self._git(['log', '--reverse', '--no-renames', '--diff-filter=A', '--name-only',
                              '--format=%x1e%H', 'HEAD'])
            added = []
            for rec in proc.stdout.decode('utf-8', 'replace').split('\x1e'):
                lines = [ln.strip() for ln in rec.splitlines() if ln.strip()]
                if lines:
                    added.append((lines[0], lines[1:]))
            self._added = added
        for sha, paths in self._added:
            if any(glob_match(p, glob) for p in paths):
                return self.commit(sha)
        return None

    def first_commit_where(self, path: str, pred_bytes: Callable[[bytes], Any]) -> Optional[Commit]:
        path = norm_rel(path)
        for c in self.log([path]):
            data = self.file_at(c.sha, path)
            if data is not None and pred_bytes(data):
                return c
        return None

    def is_ancestor(self, a: str, b: str) -> bool:
        self._require_full_history()
        proc = self._git(['merge-base', '--is-ancestor', a, b], ok=(0, 1, 128))
        if proc.returncode == 128:
            raise MissingInput('unknown commit in is_ancestor(%s, %s)' % (a, b))
        return proc.returncode == 0

    def _commit_exists(self, sha: str) -> bool:
        proc = self._git(['cat-file', '-e', sha + '^{commit}'], ok=(0, 1, 128))
        return proc.returncode == 0

    def file_at(self, sha: str, path: str) -> Optional[bytes]:
        path = norm_rel(path)
        proc = self._git(['cat-file', 'blob', '%s:%s' % (sha, path)], ok=(0, 1, 128))
        if proc.returncode == 0:
            return proc.stdout
        if not self._commit_exists(sha):
            raise MissingInput('unknown commit %s' % sha)
        return None

    def files_at(self, sha: str, paths: Sequence[str]) -> Dict[str, Optional[bytes]]:
        paths = [norm_rel(p) for p in paths]
        if not paths:
            return {}
        if any('\n' in p for p in paths):
            return GitView.files_at(self, sha, paths)
        req = ''.join('%s:%s\n' % (sha, p) for p in paths).encode('utf-8')
        proc = self._git(['cat-file', '--batch'], input_bytes=req, timeout=max(GIT_TIMEOUT_S, 30))
        out, pos, res = proc.stdout, 0, {}
        for p in paths:
            nl = out.index(b'\n', pos)
            header = out[pos:nl].decode('utf-8', 'replace')
            pos = nl + 1
            if header.endswith(' missing') or header.endswith(' ambiguous'):
                res[p] = None
                continue
            fields = header.split()
            if len(fields) != 3:
                raise MissingInput('unexpected cat-file header %r' % header)
            size = int(fields[2])
            res[p] = out[pos:pos + size] if fields[1] == 'blob' else None
            pos += size + 1
        return res

    def remote_branches_containing(self, sha: str) -> List[str]:
        proc = self._git(['branch', '-r', '--contains', sha])
        out = []
        for ln in _lines(proc.stdout):
            name = ln.strip().lstrip('* ').strip()
            if '->' in name or not name:
                continue
            out.append(name)
        return out

    def diff_lines(self, base: str, head: str, path: str) -> Tuple[List[str], List[str]]:
        path = norm_rel(path)
        proc = self._git(['diff', '--no-renames', '--no-ext-diff', '--no-textconv', '-U0', base, head, '--', path])
        added: List[str] = []
        removed: List[str] = []
        in_hunk = False
        for raw in proc.stdout.decode('utf-8', 'replace').split('\n'):
            if raw.startswith('diff --git '):
                in_hunk = False
                continue
            if raw.startswith('@@'):
                in_hunk = True
                continue
            if not in_hunk:
                continue
            if raw.startswith('+'):
                added.append(raw[1:])
            elif raw.startswith('-'):
                removed.append(raw[1:])
        return added, removed

    def changed_paths(self, base: str, head: str) -> List[str]:
        proc = self._git(['diff', '--name-only', '--no-renames', base, head])
        return [ln.strip() for ln in _lines(proc.stdout)]

    def ls_files(self, paths: Sequence[str]) -> List[str]:
        rels = [norm_rel(p) or '.' for p in paths]
        proc = self._git(['ls-files', '--'] + rels)
        return [ln.strip() for ln in _lines(proc.stdout)]

    def dirty_paths(self, paths: Sequence[str]) -> List[str]:
        rels = [norm_rel(p) or '.' for p in paths]  # '' means the whole work tree
        if not rels:
            return []
        diff = self._git(['diff', '--name-only', '--no-renames', '--no-ext-diff', 'HEAD', '--'] + rels)
        others = self._git(['ls-files', '--others', '--'] + rels)
        out = set(ln.strip() for ln in _lines(diff.stdout))
        out.update(ln.strip() for ln in _lines(others.stdout) if not _is_cache_junk(ln.strip()))
        return sorted(out)

    def _object_format(self) -> str:
        if self._fmt is None:
            proc = self._git(['rev-parse', '--show-object-format'], ok=(0, 128, 129))
            out = proc.stdout.decode().strip()
            self._fmt = out if out in _EMPTY_TREE else 'sha1'
        return self._fmt

    def list_files(self, sha: str, prefix: str = '') -> List[str]:
        args = ['diff', '--name-only', '--no-renames', _EMPTY_TREE[self._object_format()], sha]
        if prefix:
            args += ['--', norm_rel(prefix)]
        proc = self._git(args)
        return sorted(ln.strip() for ln in _lines(proc.stdout))


class TimelineGit(GitView):
    """Fixture-backed history (SPEC 7.3).

    timeline.json = {head, commits: [{sha, utc, author, trailers, paths, branches}],
                     optional dirty: [repo paths modified/untracked in the worktree],
                     optional shallow: bool}.
    repo/ holds the final tree (the worktree); history/<sha>/<path> holds the content of a path at
    the commit that last touched it whenever that differs from repo/. History is linear.
    """
    mode = 'timeline'

    def __init__(self, fixture_dir: Any):
        self.dir = Path(fixture_dir).resolve()
        tl = load_json_file(self.dir / 'timeline.json')
        if not isinstance(tl, dict) or not isinstance(tl.get('commits'), list):
            raise InvalidInput('timeline.json must be an object with a commits list')
        self.repo = self.dir / 'repo'
        self.hist = self.dir / 'history'
        self._commits: List[Commit] = []
        self._branches: Dict[str, List[str]] = {}
        self._index: Dict[str, int] = {}
        self._touch: Dict[str, List[int]] = {}
        for i, raw in enumerate(tl['commits']):
            if not isinstance(raw, dict) or not isinstance(raw.get('sha'), str) or not raw['sha']:
                raise InvalidInput('timeline commit %d lacks a sha' % i)
            sha = raw['sha']
            if sha in self._index:
                raise InvalidInput('duplicate timeline sha %s' % sha)
            trailers: Dict[str, List[str]] = {}
            for k, v in (raw.get('trailers') or {}).items():
                trailers[str(k)] = [str(x) for x in v] if isinstance(v, list) else [str(v)]
            paths = [norm_rel(p) for p in (raw.get('paths') or [])]
            c = Commit(sha=sha, utc=str(raw.get('utc') or ''), author=str(raw.get('author') or ''),
                       trailers=trailers, paths=paths)
            self._index[sha] = i
            self._commits.append(c)
            self._branches[sha] = [str(b) for b in (raw.get('branches') or [])]
            for p in paths:
                self._touch.setdefault(p, []).append(i)
        if not self._commits:
            raise InvalidInput('timeline has no commits')
        head = tl.get('head') or self._commits[-1].sha
        self._head_idx = self._idx(head)
        self._dirty = [norm_rel(p) for p in (tl.get('dirty') or [])]
        self._shallow = bool(tl.get('shallow', False))

    @staticmethod
    def freeze(fixture_dir: Any) -> int:
        """Make the committed content of a fixture copy immutable before commands run in it.

        A fixture stores the committed version of a path in repo/ (the worktree) whenever the two agree, so a
        command that rewrites a committed file in place (`claim` rewriting CLAIM.json, a deleted runlog file)
        would otherwise rewrite history too, and checks that compare the worktree with HEAD (ERR-01, KB-02,
        CYC-01, dirty paths) could not see the edit. Real git never behaves that way. For every path whose
        last committing commit (at or before head) has no history/<sha>/<path> entry and whose repo/ file
        exists, the repo/ bytes are copied to that entry. Returns the number of entries written."""
        d = Path(fixture_dir).resolve()
        tl = load_json_file(d / 'timeline.json')
        commits = tl.get('commits') if isinstance(tl, dict) else None
        if not isinstance(commits, list) or not commits:
            raise InvalidInput('timeline.json must be an object with a commits list')
        shas = [str(c.get('sha')) for c in commits if isinstance(c, dict)]
        head = tl.get('head') or shas[-1]
        try:
            head_idx = shas.index(head)
        except ValueError:
            hits = [i for i, s_ in enumerate(shas) if s_.startswith(str(head))]
            if len(hits) != 1:
                raise InvalidInput('timeline head %r is not a commit' % (head,))
            head_idx = hits[0]
        last: Dict[str, int] = {}
        for i, c in enumerate(commits[:head_idx + 1]):
            for p_ in (c.get('paths') or []):
                last[norm_rel(p_)] = i
        n = 0
        for rel, i in sorted(last.items()):
            h = d / 'history' / shas[i] / rel
            r = d / 'repo' / rel
            if h.is_file() or not r.is_file():
                continue
            h.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(str(r), str(h))
            n += 1
        return n

    def _idx(self, sha: str) -> int:
        if not isinstance(sha, str) or not sha:
            raise MissingInput('unknown commit %r' % (sha,))
        if sha in self._index:
            return self._index[sha]
        if len(sha) >= 4:
            hits = [i for s, i in self._index.items() if s.startswith(sha)]
            if len(hits) == 1:
                return hits[0]
        raise MissingInput('unknown commit %s' % sha)

    def head(self) -> str:
        return self._commits[self._head_idx].sha

    def is_shallow(self) -> bool:
        return self._shallow

    def _require_full_history(self) -> None:
        if self._shallow:
            raise MissingInput('shallow clone: history-dependent git query unavailable')

    def log(self, paths: Optional[Sequence[str]] = None) -> List[Commit]:
        self._require_full_history()
        commits = self._commits[:self._head_idx + 1]
        if paths is None:
            return list(commits)
        if isinstance(paths, str):
            paths = [paths]
        return [c for c in commits if any(glob_match(p, pat) for p in c.paths for pat in paths)]

    def commit(self, sha: str) -> Optional[Commit]:
        try:
            return self._commits[self._idx(sha)]
        except MissingInput:
            return None

    def _final_toucher(self, path: str) -> Optional[int]:
        idxs = [i for i in self._touch.get(path, []) if i <= self._head_idx]
        return max(idxs) if idxs else None

    def file_at(self, sha: str, path: str) -> Optional[bytes]:
        ci = self._idx(sha)
        path = norm_rel(path)
        idxs = [i for i in self._touch.get(path, []) if i <= ci]
        if not idxs:
            return None
        t = max(idxs)
        h = self.hist / self._commits[t].sha / path
        if h.is_file():
            return h.read_bytes()
        if t == self._final_toucher(path):
            rp = self.repo / path
            return rp.read_bytes() if rp.is_file() else None
        return None

    def first_commit_adding(self, glob: str) -> Optional[Commit]:
        self._require_full_history()
        for i in range(self._head_idx + 1):
            c = self._commits[i]
            for p in c.paths:
                if not glob_match(p, glob):
                    continue
                prev = self.file_at(self._commits[i - 1].sha, p) if i > 0 else None
                if prev is None and self.file_at(c.sha, p) is not None:
                    return c
        return None

    def first_commit_where(self, path: str, pred_bytes: Callable[[bytes], Any]) -> Optional[Commit]:
        self._require_full_history()
        path = norm_rel(path)
        for i in self._touch.get(path, []):
            if i > self._head_idx:
                continue
            data = self.file_at(self._commits[i].sha, path)
            if data is not None and pred_bytes(data):
                return self._commits[i]
        return None

    def is_ancestor(self, a: str, b: str) -> bool:
        self._require_full_history()
        return self._idx(a) <= self._idx(b)

    def remote_branches_containing(self, sha: str) -> List[str]:
        return list(self._branches[self._commits[self._idx(sha)].sha])

    def diff_lines(self, base: str, head: str, path: str) -> Tuple[List[str], List[str]]:
        a = (self.file_at(base, path) or b'').decode('utf-8', 'replace').splitlines()
        b = (self.file_at(head, path) or b'').decode('utf-8', 'replace').splitlines()
        added: List[str] = []
        removed: List[str] = []
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
            if tag in ('replace', 'delete'):
                removed.extend(a[i1:i2])
            if tag in ('replace', 'insert'):
                added.extend(b[j1:j2])
        return added, removed

    def changed_paths(self, base: str, head: str) -> List[str]:
        bi, hi = self._idx(base), self._idx(head)
        lo, up = min(bi, hi), max(bi, hi)
        cand = set()
        for i in range(lo + 1, up + 1):
            cand.update(self._commits[i].paths)
        return sorted(p for p in cand if self.file_at(base, p) != self.file_at(head, p))

    def _repo_files_under(self, rel: str) -> List[str]:
        base = self.repo / rel if rel else self.repo
        if base.is_file():
            return [rel]
        out = []
        if base.is_dir():
            for dirpath, dirnames, filenames in os.walk(base):
                dirnames[:] = [d for d in dirnames if d not in SKIP_DIR_NAMES]
                for fn in filenames:
                    r = Path(dirpath, fn).relative_to(self.repo).as_posix()
                    if not _is_cache_junk(r):
                        out.append(r)
        return out

    def dirty_paths(self, paths: Sequence[str]) -> List[str]:
        res = set()
        for p in paths:
            rel = norm_rel(p)
            for d in self._dirty:
                if not rel or d == rel or d.startswith(rel + '/'):
                    res.add(d)
            for f in self._repo_files_under(rel):
                t = self._final_toucher(f)
                if t is None:
                    res.add(f)  # untracked
                    continue
                h = self.hist / self._commits[t].sha / f
                if h.is_file() and h.read_bytes() != (self.repo / f).read_bytes():
                    res.add(f)  # modified relative to head
            for tp, idxs in self._touch.items():
                if not (not rel or tp == rel or tp.startswith(rel + '/')):
                    continue
                t = self._final_toucher(tp)
                if t is None:
                    continue
                h = self.hist / self._commits[t].sha / tp
                if h.is_file() and not (self.repo / tp).is_file():
                    res.add(tp)  # deleted in the worktree
        return sorted(res)

    def list_files(self, sha: str, prefix: str = '') -> List[str]:
        ci = self._idx(sha)
        pre = norm_rel(prefix) if prefix else ''
        out = []
        for p, idxs in self._touch.items():
            if pre and not (p == pre or p.startswith(pre + '/')):
                continue
            if any(i <= ci for i in idxs) and self.file_at(self._commits[ci].sha, p) is not None:
                out.append(p)
        return sorted(out)


def git_view_from_spec(spec: Optional[str], repo_root: Optional[Any]) -> Tuple[Path, GitView]:
    """Resolve (--git, --repo-root) into (repo_root, GitView)."""
    if spec:
        if not spec.startswith('timeline:'):
            raise CheckerError('unsupported --git value %r (only timeline:<fixture_dir>)' % spec)
        g = TimelineGit(spec[len('timeline:'):])
        root = Path(repo_root).resolve() if repo_root else g.repo.resolve()
        try:
            root.relative_to(g.dir)
        except ValueError:
            # timeline mode is for corpus fixtures only: it never gates a real work tree
            raise CheckerError('--git timeline: requires --repo-root inside the fixture dir %s' % g.dir)
        return root, g
    root = Path(repo_root).resolve() if repo_root else git_toplevel(Path.cwd())
    return root, RealGit(root)


# --------------------------------------------------------------------------------------
# KB view
# --------------------------------------------------------------------------------------

def _k(row: dict, *names: str):
    if row.get(names[0]) in (None, ''):
        return None
    return tuple(row.get(n) for n in names) if len(names) > 1 else row.get(names[0])


LEDGER_KEYS: Dict[str, Callable[[dict], Any]] = {
    'FINDINGS': lambda r: _k(r, 'id'),
    'USER_DECISIONS': lambda r: _k(r, 'id'),
    # A closure row reuses the incident id with kind 'closure'.
    'INCIDENTS': lambda r: _k(r, 'id', 'kind'),
    # Repayment is appended as a new row with the same id and repaid_by set.
    'REVIEW_DEBT': lambda r: None if r.get('id') in (None, '') else (r.get('id'), 'repaid' if r.get('repaid_by') else 'open'),
    # Reveal is appended as a new row with the same batch_id and revealed_utc set.
    'CANARY_COMMITMENTS': lambda r: None if r.get('batch_id') in (None, '') else (r.get('batch_id'), 'revealed' if r.get('revealed_utc') else 'committed'),
    'CONFIRM_LEDGER': lambda r: _k(r, 'family_id', 'attempt_index'),
    'DELIVERY_LEDGER': lambda r: None,  # one row per `claim` run; no unique key
}

FINDING_STATUS_RANK = {'proposed': 0, 'provisional': 0, 'active': 1, 'unverifiable_in_repo': 2,
                       'narrowed': 3, 'superseded': 4, 'retracted': 5}
FINDING_KIND_STATUS = {'activation': 'active', 'narrowing': 'narrowed', 'supersession': 'superseded',
                       'retraction': 'retracted'}


class KB:
    """View of PROJECT_DOCUMENTS/KB under repo_root."""

    def __init__(self, repo_root: Any):
        self.repo_root = Path(repo_root)
        self.dir = self.repo_root / KB_REL
        self._cache: Dict[str, Any] = {}

    def path(self, name: str) -> Path:
        return self.dir / name

    def constraints(self) -> dict:
        if 'constraints' not in self._cache:
            obj = load_json_file(self.dir / 'CONSTRAINTS.json')
            if not isinstance(obj, dict):
                raise InvalidInput('CONSTRAINTS.json must be an object')
            self._cache['constraints'] = obj
        return self._cache['constraints']

    def constraints_sha256(self) -> str:
        return sha256_file(self.dir / 'CONSTRAINTS.json')

    @staticmethod
    def _ledger_name(name: str) -> str:
        n = os.path.basename(str(name))
        return n[:-6] if n.endswith('.jsonl') else n

    def rows(self, name: str) -> List[dict]:
        """Rows of KB/<name>.jsonl. Identical lines (union merge) are deduplicated; duplicate ids
        with different content raise InvalidInput. A missing ledger raises MissingInput."""
        lname = self._ledger_name(name)
        key = 'rows:' + lname
        if key in self._cache:
            return list(self._cache[key])
        p = self.dir / (lname + '.jsonl')
        keyfn = LEDGER_KEYS.get(lname, lambda r: _k(r, 'id'))
        seen_lines = set()
        seen_keys: Dict[Any, bytes] = {}
        out: List[dict] = []
        for row in load_jsonl_file(p):
            canon = canonical_json(row)
            if canon in seen_lines:
                continue
            seen_lines.add(canon)
            k = keyfn(row)
            if k is not None:
                hk = canonical_json(list(k)) if isinstance(k, tuple) else canonical_json(k)
                if hk in seen_keys:
                    raise InvalidInput('%s: duplicate id %r with different content' % (lname, k))
                seen_keys[hk] = canon
            out.append(row)
        self._cache[key] = out
        return list(out)

    def failure_classes(self) -> dict:
        if 'fc' not in self._cache:
            obj = load_json_file(self.dir / 'FAILURE_CLASSES.json')
            if not isinstance(obj, dict):
                raise InvalidInput('FAILURE_CLASSES.json must be an object')
            self._cache['fc'] = obj
        return self._cache['fc']

    def user_decision(self, ud_id: str) -> Optional[dict]:
        for r in self.rows('USER_DECISIONS'):
            if r.get('id') == ud_id:
                return r
        return None

    def finding_status(self, finding_id: str) -> Optional[str]:
        """Order-independent (union-merge safe) fold: the most restrictive status among the finding
        row and the rows referring to it, ranked proposed/provisional < active < unverifiable_in_repo
        (and unknown values) < narrowed < superseded < retracted. None if the finding row is absent.
        kb.py (WP5) owns the full fold with R-levels; this is the minimal shared view."""
        best = None
        found = False
        for r in self.rows('FINDINGS'):
            kind = r.get('kind', 'finding')
            if r.get('id') == finding_id and kind == 'finding':
                found = True
                cand = r.get('status') or 'proposed'
            elif r.get('refers_to') == finding_id:
                cand = FINDING_KIND_STATUS.get(kind) or (r.get('status') if kind == 'reconciliation' else None)
                if not cand:
                    continue
            else:
                continue
            if best is None or FINDING_STATUS_RANK.get(cand, 2) > FINDING_STATUS_RANK.get(best, 2):
                best = cand
        return best if found else None


# --------------------------------------------------------------------------------------
# Repo-wide scans (SPEC 20.4)
# --------------------------------------------------------------------------------------

@dataclass
class RepoScan:
    root: Path
    patterns: Tuple[str, ...]
    max_bytes: int
    excludes: Tuple[str, ...]
    files: List[Path] = field(default_factory=list)
    skipped_large: List[str] = field(default_factory=list)

    def __iter__(self) -> Iterator[Path]:
        return iter(self.files)

    def rel(self, p: Path) -> str:
        return p.relative_to(self.root).as_posix()

    def scope(self) -> str:
        return ('scan scope: %d files matching %s under the repo root, excluding %s; %d files over %d bytes '
                'skipped; archive members not read' % (len(self.files), ','.join(self.patterns),
                                                        ','.join(self.excludes), len(self.skipped_large),
                                                        self.max_bytes))


def iter_repo_files(repo_root: Any, patterns: Sequence[str] = DEFAULT_SCAN_PATTERNS,
                    max_bytes: int = MAX_SCAN_BYTES,
                    excludes: Sequence[str] = DEFAULT_SCAN_EXCLUDES) -> RepoScan:
    """Bounded repo-wide scan: files whose basename matches one of `patterns`, outside .git and the
    excluded repo-relative prefixes, skipping files larger than max_bytes. Archive members are never
    opened. Any walk error raises MissingInput (a scan that cannot complete never PASSes)."""
    root = Path(repo_root).resolve()
    scan = RepoScan(root=root, patterns=tuple(patterns), max_bytes=int(max_bytes), excludes=tuple(excludes))
    rxs = [glob_to_regex(p) for p in patterns]
    excl = [norm_rel(e) for e in excludes if norm_rel(e)]
    errors: List[str] = []

    def onerror(err):
        errors.append(str(err))

    if not root.is_dir():
        raise MissingInput('repo root %s is not a directory' % root)
    for dirpath, dirnames, filenames in os.walk(root, onerror=onerror):
        reld = Path(dirpath).relative_to(root).as_posix()
        reld = '' if reld == '.' else reld
        keep = []
        for d in sorted(dirnames):
            if d in SKIP_DIR_NAMES:
                continue
            rd = (reld + '/' + d) if reld else d
            if any(rd == e or rd.startswith(e + '/') for e in excl):
                continue
            keep.append(d)
        dirnames[:] = keep
        for fn in sorted(filenames):
            if not any(rx.match(fn) for rx in rxs):
                continue
            fp = Path(dirpath, fn)
            try:
                size = fp.stat().st_size
            except OSError as e:
                errors.append('%s: %s' % (fp, e))
                continue
            if size > max_bytes:
                scan.skipped_large.append(fp.relative_to(root).as_posix())
                continue
            scan.files.append(fp)
    if errors:
        raise MissingInput('repo scan could not complete: %s' % '; '.join(errors[:3]))
    return scan


# --------------------------------------------------------------------------------------
# Deliveries and context
# --------------------------------------------------------------------------------------

def v1_status(path: Any) -> Tuple[bool, str]:
    p = Path(path)
    if not p.is_dir():
        return False, 'not a directory: %s' % p
    lock = p / 'ENDPOINT_LOCK.json'
    if not lock.is_file():
        return False, 'no ENDPOINT_LOCK.json'
    try:
        obj = load_json_file(lock)
    except (MissingInput, InvalidInput) as e:
        return False, 'ENDPOINT_LOCK.json unreadable: %s' % e
    if not isinstance(obj, dict) or obj.get('schema') != SCHEMA_ENDPOINT_LOCK:
        return False, 'ENDPOINT_LOCK schema is %r, not %s' % (obj.get('schema') if isinstance(obj, dict) else None,
                                                              SCHEMA_ENDPOINT_LOCK)
    if not V1_DIR_RE.match(p.name):
        return False, 'directory name %r does not match <name>_{design|dev}_vN_YYYYMMDD' % p.name
    return True, 'v1 delivery'


def is_v1_delivery(path: Any) -> bool:
    return v1_status(path)[0]


def is_evaluator_arm(arm: Any) -> bool:
    return isinstance(arm, str) and arm.startswith(EVALUATOR_ARM_PREFIXES)


def _iter_jsonl_stream(stream, label: str) -> Iterator[Tuple[int, dict]]:
    for lineno, raw in enumerate(stream, 1):
        if isinstance(raw, bytes):
            try:
                raw = raw.decode('utf-8')
            except UnicodeDecodeError as e:
                raise InvalidInput('%s:%d is not UTF-8: %s' % (label, lineno, e))
        line = raw.strip()
        if not line:
            continue
        obj = loads_strict(line, what='%s:%d' % (label, lineno))
        if not isinstance(obj, dict):
            raise InvalidInput('%s:%d is not a JSON object' % (label, lineno))
        yield lineno, obj


@dataclass
class Ctx:
    repo_root: Path
    delivery: Optional[Path]
    git: GitView
    kb: KB
    rules: dict
    stage: Optional[str] = None
    tier: Optional[int] = None
    dry: bool = False
    options: dict = field(default_factory=dict)
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    # ---- identity
    @property
    def git_mode(self) -> str:
        return 'timeline' if isinstance(self.git, TimelineGit) else 'real'

    @property
    def delivery_rel(self) -> str:
        if self.delivery is None:
            raise MissingInput('no delivery directory given')
        try:
            rel = Path(self.delivery).resolve().relative_to(Path(self.repo_root).resolve()).as_posix()
            return '' if rel == '.' else rel
        except ValueError:
            raise InvalidInput('delivery %s is outside the repo root %s' % (self.delivery, self.repo_root))

    @property
    def delivery_name(self) -> str:
        if self.delivery is None:
            raise MissingInput('no delivery directory given')
        return Path(self.delivery).name

    def path(self, rel: str) -> Path:
        if self.delivery is None:
            raise MissingInput('no delivery directory given')
        return Path(self.delivery) / norm_rel(rel)

    def repo_path(self, rel: str) -> Path:
        return Path(self.repo_root) / norm_rel(rel)

    def repo_rel_of(self, rel: str) -> str:
        """Repo-relative path of a delivery-relative path."""
        d = self.delivery_rel
        r = norm_rel(rel)
        return (d + '/' + r) if d else r

    # ---- locks
    @property
    def lock(self) -> dict:
        if 'lock' not in self._cache:
            obj = load_json_file(self.path('ENDPOINT_LOCK.json'))
            if not isinstance(obj, dict):
                raise InvalidInput('ENDPOINT_LOCK.json must be an object')
            self._cache['lock'] = obj
        return self._cache['lock']

    @property
    def source_lock(self) -> dict:
        if 'source_lock' not in self._cache:
            obj = load_json_file(self.path('SOURCE_LOCK.json'))
            if not isinstance(obj, dict):
                raise InvalidInput('SOURCE_LOCK.json must be an object')
            self._cache['source_lock'] = obj
        return self._cache['source_lock']

    @property
    def endpoint_lock_sha256(self) -> Optional[str]:
        p = self.path('ENDPOINT_LOCK.json')
        return sha256_file(p) if p.is_file() else None

    @property
    def source_lock_sha256(self) -> Optional[str]:
        p = self.path('SOURCE_LOCK.json')
        return sha256_file(p) if p.is_file() else None

    # ---- files
    def read_json(self, rel: str) -> Any:
        return load_json_file(self.path(rel))

    def read_jsonl(self, rel: str) -> List[dict]:
        return load_jsonl_file(self.path(rel))

    def read_jsonl_optional(self, rel: str) -> List[dict]:
        p = self.path(rel)
        return load_jsonl_file(p) if p.is_file() else []

    def delivery_files(self, refresh: bool = False) -> List[str]:
        """Sorted delivery-relative POSIX paths of all files (excluding .git and __pycache__)."""
        if refresh or 'delivery_files' not in self._cache:
            if self.delivery is None:
                raise MissingInput('no delivery directory given')
            root = Path(self.delivery)
            out = []
            if root.is_dir():
                for dirpath, dirnames, filenames in os.walk(root):
                    dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIR_NAMES)
                    for fn in filenames:
                        rel = Path(dirpath, fn).relative_to(root).as_posix()
                        if not _is_cache_junk(rel):
                            out.append(rel)
            self._cache['delivery_files'] = sorted(out)
        return list(self._cache['delivery_files'])

    def row_sources(self, pattern: str) -> List[Tuple[str, str, Any, bool]]:
        """Sources of rows matching a delivery-relative glob: (logical_path, kind, ref, gz).

        Only *.jsonl logical names are row sources. Matches <pattern>, <pattern>.gz and members of RESULT_BUNDLE*.zip archives anywhere in the
        delivery. A member's logical path is its name when it starts with 'results/' (or the zip sits
        at the delivery root), else <zip dir>/<name>. The same logical path from two sources raises
        InvalidInput (rows are never double counted)."""
        base = norm_rel(pattern[:-3] if pattern.endswith('.gz') else pattern)
        sources: List[Tuple[str, str, Any, bool]] = []
        zips = []
        for rel in self.delivery_files():
            name = rel.rsplit('/', 1)[-1]
            if rel.endswith('.jsonl.gz'):
                if glob_match(rel[:-3], base):
                    sources.append((rel[:-3], 'file', rel, True))
            elif rel.endswith('.jsonl'):
                if glob_match(rel, base):
                    sources.append((rel, 'file', rel, False))
            if RESULT_BUNDLE_RE.match(name):
                zips.append(rel)
        for z in zips:
            zdir = z.rsplit('/', 1)[0] if '/' in z else ''
            try:
                with zipfile.ZipFile(str(self.path(z))) as zf:
                    for info in zf.infolist():
                        if info.is_dir():
                            continue
                        m = norm_rel(info.filename)
                        logical = m if (m.startswith('results/') or not zdir) else zdir + '/' + m
                        gz = logical.endswith('.jsonl.gz')
                        lg = logical[:-3] if gz else logical
                        if lg.endswith('.jsonl') and glob_match(lg, base):
                            sources.append((lg, 'zip', (z, info.filename), gz))
            except zipfile.BadZipFile as e:
                raise InvalidInput('%s is not a valid zip: %s' % (z, e))
        seen: Dict[str, str] = {}
        for logical, kind, ref, _gz in sources:
            label = ref if kind == 'file' else '%s!%s' % ref
            if logical in seen:
                raise InvalidInput('rows for %s found in two sources (%s, %s)' % (logical, seen[logical], label))
            seen[logical] = label
        return sorted(sources, key=lambda s: s[0])

    def iter_rows_with_source(self, pattern: str) -> Iterator[Tuple[str, int, dict]]:
        for logical, kind, ref, gz in self.row_sources(pattern):
            if kind == 'file':
                label = ref
                fp = self.path(ref)
                fh = gzip.open(str(fp), 'rb') if gz else open(fp, 'rb')
                try:
                    for lineno, row in _iter_jsonl_stream(fh, label):
                        yield label, lineno, row
                except (OSError, EOFError) as e:
                    raise InvalidInput('%s unreadable: %s' % (label, e))
                finally:
                    fh.close()
            else:
                z, member = ref
                label = '%s!%s' % (z, member)
                with zipfile.ZipFile(str(self.path(z))) as zf:
                    raw = zf.open(member)
                    fh = gzip.GzipFile(fileobj=raw) if gz else raw
                    try:
                        for lineno, row in _iter_jsonl_stream(fh, label):
                            yield label, lineno, row
                    except (OSError, EOFError, zipfile.BadZipFile) as e:
                        raise InvalidInput('%s unreadable: %s' % (label, e))
                    finally:
                        fh.close()

    def iter_rows(self, pattern: str) -> Iterator[dict]:
        """Rows from *.jsonl, *.jsonl.gz and RESULT_BUNDLE*.zip members matching `pattern`."""
        for _label, _lineno, row in self.iter_rows_with_source(pattern):
            yield row

    def repo_scan(self, patterns: Sequence[str] = DEFAULT_SCAN_PATTERNS, max_bytes: int = MAX_SCAN_BYTES,
                  excludes: Sequence[str] = DEFAULT_SCAN_EXCLUDES) -> RepoScan:
        key = ('scan', tuple(patterns), int(max_bytes), tuple(excludes))
        if key not in self._cache:
            self._cache[key] = iter_repo_files(self.repo_root, patterns, max_bytes, excludes)
        return self._cache[key]


def resolve_delivery(repo_root: Path, d: Optional[Any]) -> Optional[Path]:
    if d is None:
        return None
    p = Path(d)
    if p.is_absolute():
        cand = p.resolve()
    else:
        cands = [(Path(repo_root) / p).resolve(), (Path.cwd() / p).resolve()]
        cand = next((c for c in cands if c.is_dir()), cands[0])
    try:
        cand.relative_to(Path(repo_root).resolve())
    except ValueError:
        raise InvalidInput('delivery %s is outside the repo root %s' % (cand, repo_root))
    return cand


def make_ctx(args: Any, delivery: Optional[Any] = None, stage: Optional[str] = None, dry: bool = False,
             options: Optional[dict] = None) -> Ctx:
    """Build a Ctx from parsed CLI args (attributes repo_root, git)."""
    root, git = git_view_from_spec(getattr(args, 'git', None), getattr(args, 'repo_root', None))
    return Ctx(repo_root=root, delivery=resolve_delivery(root, delivery), git=git, kb=KB(root),
               rules=load_rules(), stage=stage, tier=None, dry=bool(dry), options=dict(options or {}))


# --------------------------------------------------------------------------------------
# RULES.json
# --------------------------------------------------------------------------------------

_RULES_CACHE: Dict[str, Any] = {}


def load_rules(path: Optional[Any] = None) -> dict:
    p = Path(path) if path else KIT_DIR / 'RULES.json'
    key = str(p)
    if key not in _RULES_CACHE:
        try:
            obj = load_json_file(p)
        except (MissingInput, InvalidInput) as e:
            raise CheckerError('RULES.json unavailable: %s' % e)
        if not isinstance(obj, dict) or not isinstance(obj.get('rules'), list) or not isinstance(obj.get('bundles'), dict):
            raise CheckerError('RULES.json must hold a rules list and a bundles object')
        _RULES_CACHE[key] = obj
    return _RULES_CACHE[key]


def rule_index(rules: dict) -> Dict[str, dict]:
    idx: Dict[str, dict] = {}
    for r in rules.get('rules', []):
        rid = r.get('id') if isinstance(r, dict) else None
        if not rid:
            raise CheckerError('RULES entry without id: %r' % (r,))
        if rid in idx:
            raise CheckerError('duplicate RULES id %s' % rid)
        idx[rid] = r
    return idx


BUNDLE_CONDITIONS = ('t2', 'sealed_confirm', 'sealed_or_adoption', 'oc_intended', 'udi_applicable')


def _norm_entry(e: Any) -> dict:
    if isinstance(e, str):
        return {'id': e}
    if isinstance(e, dict) and ('id' in e or 'group' in e):
        return dict(e)
    raise CheckerError('malformed bundle entry %r' % (e,))


def expand_bundle(rules: dict, stage: str, _stack: Optional[List[str]] = None) -> List[dict]:
    """Bundle entries in order, following 'extends'. Raises CheckerError for unknown bundles,
    cycles, duplicate ids, unregistered ids and planned ids."""
    stack = list(_stack or [])
    bundles = rules.get('bundles') or {}
    if stage not in bundles:
        raise CheckerError('RULES.json has no bundle %r' % stage)
    if stage in stack:
        raise CheckerError('bundle extends cycle: %s' % ' -> '.join(stack + [stage]))
    b = bundles[stage]
    entries: List[dict] = []
    if b.get('extends'):
        entries.extend(expand_bundle(rules, b['extends'], stack + [stage]))
    for e in b.get('checks', []):
        entries.append(_norm_entry(e))
    if _stack is None:
        idx = rule_index(rules)
        ids: List[str] = []
        for e in entries:
            if e.get('when') is not None and e['when'] not in BUNDLE_CONDITIONS:
                raise CheckerError('bundle %s: unknown condition %r' % (stage, e['when']))
            if 'group' in e:
                ids.extend(list(e.get('primary', [])) + list(e.get('fallback', [])))
            else:
                ids.append(e['id'])
        seen = set()
        for i in ids:
            if i in seen:
                raise CheckerError('bundle %s lists %s twice' % (stage, i))
            seen.add(i)
            if i not in idx:
                raise CheckerError('bundle %s references unregistered id %s' % (stage, i))
            if idx[i].get('status') == 'planned':
                raise CheckerError('bundle %s references planned id %s' % (stage, i))
    return entries


def bundle_ids(rules: dict, stage: str) -> List[str]:
    out = []
    for e in expand_bundle(rules, stage):
        if 'group' in e:
            out.extend(list(e.get('primary', [])) + list(e.get('fallback', [])))
        else:
            out.append(e['id'])
    return out


def validate_rules(rules: dict) -> List[str]:
    """Structural problems in RULES.json (empty list when consistent). Used by selftest."""
    problems: List[str] = []
    try:
        idx = rule_index(rules)
    except CheckerError as e:
        return [str(e)]
    for rid, r in idx.items():
        for f in ('module', 'stages', 'tier_min', 'introduced_in', 'status', 'fm_ids', 'applies_when'):
            if f not in r:
                problems.append('%s lacks %s' % (rid, f))
        if r.get('status') not in RULE_STATUSES:
            problems.append('%s has status %r' % (rid, r.get('status')))
        iv = parse_version(r.get('introduced_in'))
        if iv is None:
            problems.append('%s has introduced_in %r' % (rid, r.get('introduced_in')))
        elif parse_version(CHECKER_VERSION) is not None and iv > parse_version(CHECKER_VERSION):
            problems.append('%s introduced_in %s is later than CHECKER_VERSION %s' % (rid, r.get('introduced_in'),
                                                                                      CHECKER_VERSION))
        problems.extend(changed_in_problems(rid, r))
        if r.get('tier_min') not in (0, 1, 2):
            problems.append('%s has tier_min %r' % (rid, r.get('tier_min')))
        if not isinstance(r.get('stages'), list) or not isinstance(r.get('fm_ids'), list):
            problems.append('%s stages/fm_ids must be lists' % rid)
        for fm in r.get('fm_ids') or []:
            if not re.fullmatch(r'FM-\d\d', str(fm)):
                problems.append('%s has fm id %r' % (rid, fm))
        if not isinstance(r.get('applies_when'), str) or not r.get('applies_when'):
            problems.append('%s applies_when must be a non-empty string' % rid)
    for stage in (rules.get('bundles') or {}):
        try:
            ids = bundle_ids(rules, stage)
        except CheckerError as e:
            problems.append(str(e))
            continue
        for i in ids:
            if stage not in (idx[i].get('stages') or []):
                problems.append('%s is in bundle %s but its stages omit it' % (i, stage))
    return problems


def changed_in_problems(rid: str, r: dict) -> List[str]:
    """RULES `changed_in` (optional): the checker versions whose release tightened this check's behaviour after
    introduced_in (SPEC 20.10). A list of distinct versions, each later than introduced_in and not later than
    CHECKER_VERSION, in increasing order."""
    if 'changed_in' not in r:
        return []
    ci = r.get('changed_in')
    if not isinstance(ci, list) or not ci:
        return ['%s changed_in must be a non-empty list of versions' % rid]
    iv, cv = parse_version(r.get('introduced_in')), parse_version(CHECKER_VERSION)
    out: List[str] = []
    prev = iv
    for v in ci:
        pv = parse_version(v)
        if pv is None:
            out.append('%s changed_in has %r' % (rid, v))
            continue
        if prev is not None and pv <= prev:
            out.append('%s changed_in %s is not later than %s' % (rid, v, 'introduced_in' if prev == iv else
                                                                   'the previous entry'))
        if cv is not None and pv > cv:
            out.append('%s changed_in %s is later than CHECKER_VERSION %s' % (rid, v, CHECKER_VERSION))
        prev = pv
    return out


def rule_versions(rule: dict) -> List[str]:
    """introduced_in followed by every changed_in version of a RULES entry (SPEC 20.10)."""
    out = [rule.get('introduced_in')] if isinstance(rule.get('introduced_in'), str) else []
    ci = rule.get('changed_in')
    if isinstance(ci, list):
        out.extend(v for v in ci if isinstance(v, str))
    return out


def rule_newer_than(rule: dict, version: Any) -> Optional[str]:
    """The earliest of introduced_in / changed_in that is later than `version` (the version whose behaviour a
    held-out batch or a delivery was written against), or None when the check behaves as it did then. An
    unparseable `version` gives None (nothing can be called newer)."""
    base = parse_version(version)
    if base is None:
        return None
    later = [v for v in rule_versions(rule) if parse_version(v) is not None and parse_version(v) > base]
    return min(later, key=parse_version) if later else None


def not_applicable_declared(rule: dict) -> bool:
    """NOT_APPLICABLE is allowed only when RULES.json declares a condition: applies_when is a
    non-empty string that does not start with 'always'."""
    aw = rule.get('applies_when')
    return isinstance(aw, str) and bool(aw.strip()) and not aw.strip().lower().startswith('always')


# --------------------------------------------------------------------------------------
# Attempts, cycles, caps, gate reports (shared by launch and other modules)
# --------------------------------------------------------------------------------------

def admission_worlds(ctx: Ctx) -> Tuple[List[str], str]:
    """(worlds, basis): the technical worlds that carry the native technical (admission) run, i.e. where
    receipts, the candidate's TAUGHT/prefix rows and SHUF manifests are required (SPEC 3.1 C1: one technical
    record per arm; SPEC 8 ADMISSION.worlds). ADMISSION.json `worlds` when it is a non-empty list of strings
    (ADM-01 checks that they are lock technical worlds and that receipts cover exactly them); otherwise
    every lock.worlds.technical world (fail closed). Testbed-only technical worlds (evaluator CEILING_/
    FLOOR_ rows only) carry no learner run and are therefore not admission worlds."""
    W = ctx.lock.get('worlds') if isinstance(ctx.lock.get('worlds'), dict) else {}
    tech = [str(w) for w in (W.get('technical') or []) if isinstance(w, (str, int))]
    try:
        adm = ctx.read_json('ADMISSION.json') if ctx.path('ADMISSION.json').is_file() else None
    except InvalidInput:
        adm = None
    ws = adm.get('worlds') if isinstance(adm, dict) else None
    if isinstance(ws, list) and ws and all(isinstance(w, str) and w for w in ws):
        return sorted(set(ws)), 'ADMISSION.worlds'
    return tech, 'lock.worlds.technical (no usable ADMISSION.worlds)'


def read_attempts(ctx: Ctx) -> List[dict]:
    return ctx.read_jsonl_optional('runlog/ATTEMPTS.jsonl')


def read_cycles(ctx: Ctx) -> List[dict]:
    return ctx.read_jsonl_optional('runlog/CYCLES.jsonl')


def _num(v: Any, what: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or v in (float('inf'), float('-inf')):
        raise InvalidInput('%s must be a finite number, got %r' % (what, v))
    return float(v)


def attempts_cpu_s(rows: Iterable[dict], label: str = 'ATTEMPTS') -> float:
    total = 0.0
    for i, a in enumerate(rows):
        total += _num(a.get('cpu_children_user_s'), '%s[%d].cpu_children_user_s' % (label, i))
        total += _num(a.get('cpu_children_sys_s'), '%s[%d].cpu_children_sys_s' % (label, i))
    return total


_NUM_TOKEN_RE = re.compile(r'(?<![\w.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?!\d)')


def numbers_in_text(text: Any) -> List[float]:
    out = []
    for tok in _NUM_TOKEN_RE.findall(str(text or '')):
        t = tok.replace(',', '')
        try:
            out.append(float(t))
        except ValueError:
            continue
    return out


def number_in_text(value: Any, text: Any) -> bool:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    return any(abs(v - x) <= 1e-12 * max(1.0, abs(v)) for x in numbers_in_text(text))


def default_grant_caps(ctx: Ctx) -> dict:
    caps = dict(SPEC_DEFAULT_GRANT_CAPS)
    try:
        c = ctx.kb.constraints().get('default_grant_caps')
        if isinstance(c, dict):
            for k in caps:
                if isinstance(c.get(k), (int, float)) and not isinstance(c.get(k), bool):
                    caps[k] = c[k]
    except (MissingInput, InvalidInput):
        pass
    return caps


def effective_cap(ctx: Ctx, key: str) -> Tuple[float, str]:
    """AUTHORIZATION cap `key` when its number appears literally in the verbatim text; otherwise
    the default grant cap (or the recorded cap if it is lower). Returns (value, basis)."""
    default = default_grant_caps(ctx).get(key)
    try:
        auth = ctx.read_json('AUTHORIZATION.json')
    except MissingInput:
        return default, 'default (no AUTHORIZATION.json)'
    if not isinstance(auth, dict):
        raise InvalidInput('AUTHORIZATION.json must be an object')
    v = (auth.get('caps') or {}).get(key) if isinstance(auth.get('caps'), dict) else None
    if v is None or isinstance(v, bool) or not isinstance(v, (int, float)):
        return default, 'default (caps.%s not set)' % key
    if number_in_text(v, auth.get('verbatim')):
        return v, 'caps.%s=%s appears in the verbatim text' % (key, v)
    if default is not None and v > default:
        return default, 'default %s (caps.%s=%s not in the verbatim text)' % (default, key, v)
    return v, 'caps.%s=%s (not in verbatim; lower than the default)' % (key, v)


def designs_tried_k(ctx: Ctx) -> Tuple[int, List[List[Any]]]:
    """Distinct (native_closure_sha, params_sha256) pairs over learner-arm outcome rows (OUTPUTS and
    receipts) and outcome-bearing attempts. Evaluator arms (CEILING_*/FLOOR_*) and testbed attempts
    do not count."""
    pairs = set()
    for row in ctx.iter_rows('results/**/OUTPUTS.jsonl'):
        if not is_evaluator_arm(row.get('arm')):
            pairs.add((row.get('native_closure_sha'), row.get('params_sha256')))
    for rel in ctx.delivery_files():
        if rel.startswith('results/receipts/') and rel.endswith('.json'):
            r = ctx.read_json(rel)
            if isinstance(r, dict) and not is_evaluator_arm(r.get('arm')):
                pairs.add((r.get('native_closure_sha'), r.get('params_sha256')))
    for a in read_attempts(ctx):
        if a.get('stage') in ('technical', 'dev', 'rehearsal', 'official') and a.get('produced_paths'):
            p = (a.get('native_closure_sha'), a.get('params_sha256'))
            if p != (None, None):
                pairs.add(p)
    lst = sorted([list(p) for p in pairs], key=lambda x: (str(x[0]), str(x[1])))
    return len(lst), lst


def _report_consistent(rep: dict) -> Optional[str]:
    if rep.get('schema') != SCHEMA_GATE_REPORT:
        return 'schema'
    res = rep.get('results')
    if not isinstance(res, list) or not res:
        return 'results'
    try:
        rs = [Result(id=str(r['id']), status=str(r['status'])) for r in res]
    except (KeyError, TypeError):
        return 'result rows'
    if any(r.status not in STATUSES for r in rs):
        return 'status values'
    rc = exit_code(rs)
    if rep.get('exit_code') != rc or rep.get('verdict') != VERDICT_BY_EXIT[rc]:
        return 'verdict does not match its results'
    return None


def list_gate_reports(ctx: Ctx) -> List[dict]:
    """Parsed, internally consistent gate reports in <delivery>/checks/ (GATE_* and DRY_*), oldest first.
    Each dict gains '_path'. Reports naming another delivery are ignored, and in real-git mode so are
    reports written in timeline (corpus) mode. Gate reports are tamper-evident only (no signature)."""
    out = []
    for rel in ctx.delivery_files(refresh=True):
        if not rel.startswith('checks/'):
            continue
        name = rel.rsplit('/', 1)[-1]
        if not (name.startswith('GATE_') or name.startswith('DRY_')) or not name.endswith('.json'):
            continue
        try:
            rep = ctx.read_json(rel)
        except (MissingInput, InvalidInput):
            continue
        if not isinstance(rep, dict) or _report_consistent(rep):
            continue
        if ctx.git_mode == 'real' and rep.get('git_mode') != 'real':
            continue
        if rep.get('delivery') != ctx.delivery_rel:
            continue  # a report copied from another delivery never counts
        rep = dict(rep)
        rep['_path'] = rel
        out.append(rep)
    out.sort(key=lambda r: (normalize_utc(r.get('utc')) or '', r['_path']))
    return out


# --------------------------------------------------------------------------------------
# Tier and family (SPEC 4)
# --------------------------------------------------------------------------------------

def _tier_value(v: Any) -> int:
    if isinstance(v, bool):
        raise InvalidInput('tier_declared must be 0, 1 or 2')
    if isinstance(v, int) and v in (0, 1, 2):
        return v
    if isinstance(v, str) and re.fullmatch(r'[Tt]?[012]', v.strip()):
        return int(v.strip()[-1])
    raise InvalidInput('tier_declared must be 0, 1 or 2 (or T0..T2), got %r' % (v,))


def _official_planned(lock: dict) -> bool:
    worlds = lock.get('worlds') if isinstance(lock.get('worlds'), dict) else {}
    off = worlds.get('official') if isinstance(worlds.get('official'), dict) else {}
    ids = off.get('ids') or []
    n = off.get('n')
    return bool(ids) or (isinstance(n, (int, float)) and not isinstance(n, bool) and n > 0)


def adoption_intended(ctx: Ctx) -> bool:
    if ctx.options.get('adoption'):
        return True
    if ctx.lock.get('intent') == 'promotion':
        return True
    try:
        auth = ctx.read_json('AUTHORIZATION.json')
        st = auth.get('stages') if isinstance(auth, dict) else None
        if isinstance(st, dict) and (st.get('adoption') is True or (isinstance(st.get('adoption'), dict)
                                                                     and st['adoption'].get('granted'))):
            return True
    except MissingInput:
        pass
    return False


def sealed_confirm_intended(ctx: Ctx) -> bool:
    worlds = ctx.lock.get('worlds') if isinstance(ctx.lock.get('worlds'), dict) else {}
    if worlds.get('official_role_intended') == 'SEALED_CONFIRM':
        return True
    p = ctx.path('CLAIM.json')
    if p.is_file():
        claim = load_json_file(p)
        if isinstance(claim, dict) and claim.get('worlds_role') == 'SEALED_CONFIRM':
            return True
    return False


def _receipt_records(ctx: Ctx) -> float:
    total = 0.0
    for rel in ctx.delivery_files():
        if rel.startswith('results/receipts/') and rel.endswith('.json'):
            r = ctx.read_json(rel)
            if not isinstance(r, dict):
                raise InvalidInput('%s must be an object' % rel)
            if r.get('records') is not None:
                total += _num(r.get('records'), '%s.records' % rel)
    return total


def compute_tier(ctx: Ctx) -> Tuple[int, List[str]]:
    """tier = max(declared, tier(lock.budget), tier(measured ATTEMPTS and receipts of this delivery),
    tier(family cumulative CPU), stage floor). Official, SEALED_CONFIRM, adoption, pointer and
    finding activation are always tier 2. Thresholds come from KB CONSTRAINTS.tier_thresholds."""
    if 'tier' in ctx._cache:
        return ctx._cache['tier']
    th = ctx.kb.constraints().get('tier_thresholds')
    if not isinstance(th, dict):
        raise MissingInput('CONSTRAINTS.tier_thresholds missing')
    t0_cpu = _num(th.get('t0_max_cpu_h'), 'tier_thresholds.t0_max_cpu_h')
    t1_cpu = _num(th.get('t1_max_cpu_h'), 'tier_thresholds.t1_max_cpu_h')
    t1_rec = _num(th.get('t1_max_records'), 'tier_thresholds.t1_max_records')
    fam_cpu_max = _num(th.get('family_t1_max_cpu_h'), 'tier_thresholds.family_t1_max_cpu_h')
    lock = ctx.lock
    parts: List[Tuple[int, str]] = []

    if lock.get('tier_declared') is None:
        parts.append((0, 'declared: missing (0)'))
    else:
        parts.append((_tier_value(lock.get('tier_declared')), 'declared: %s' % lock.get('tier_declared')))

    budget = lock.get('budget') if isinstance(lock.get('budget'), dict) else {}
    b_cpu = budget.get('max_cpu_h')
    b_rec = budget.get('max_records')

    def _is_num(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and abs(v) != float('inf')
    bt = 0
    if not (_is_num(b_cpu) and _is_num(b_rec)):
        bt = 2  # an unknown budget never lowers the tier (LOCK-01 reports the field itself)
    else:
        if b_cpu > t1_cpu or b_rec > t1_rec:
            bt = 2
        elif b_cpu > t0_cpu or b_rec > 0:
            bt = 1
    parts.append((bt, 'lock.budget: max_cpu_h=%s max_records=%s -> %d%s'
                  % (b_cpu, b_rec, bt, '' if bt < 2 or (_is_num(b_cpu) and _is_num(b_rec)) else ' (not numeric)')))

    attempts = read_attempts(ctx)
    cpu_h = attempts_cpu_s(attempts) / 3600.0
    records = _receipt_records(ctx)
    mt = 0
    if attempts or cpu_h > t0_cpu or records > 0:
        mt = 1
    if cpu_h > t1_cpu or records > t1_rec:
        mt = 2
    if any(a.get('stage') == 'official' for a in attempts):
        mt = 2
    parts.append((mt, 'measured: %d attempts, %.4f CPU-h, %d receipt records -> %d'
                  % (len(attempts), cpu_h, int(records), mt)))

    fam_id, members = family_of(ctx)
    fam_cpu_h = family_cpu_h(ctx, members)
    ft = 2 if fam_cpu_h > fam_cpu_max else 0
    parts.append((ft, 'family %s (%d members): %.4f CPU-h vs family_t1_max_cpu_h %s -> %d'
                  % (fam_id, len(members), fam_cpu_h, th.get('family_t1_max_cpu_h'), ft)))

    floor = 0
    floor_why = []
    if ctx.stage in ('official', 'pointer'):
        floor = 2
        floor_why.append('stage %s' % ctx.stage)
    if _official_planned(lock):
        floor = 2
        floor_why.append('lock plans an official run')
    if sealed_confirm_intended(ctx):
        floor = 2
        floor_why.append('SEALED_CONFIRM')
    if adoption_intended(ctx):
        floor = 2
        floor_why.append('adoption/promotion intended')
    if ctx.options.get('activation'):
        floor = 2
        floor_why.append('finding activation')
    for i, am in enumerate(ctx.read_jsonl_optional('AMENDMENTS.jsonl')):
        fields_changed = am.get('fields_changed') or []
        if am.get('results_seen') and any(str(f).split('.')[0].split('[')[0] in ('primary', 'guards')
                                          for f in fields_changed):
            floor = 2
            floor_why.append('post-outcome amendment of primary/guards (AMENDMENTS row %d)' % (i + 1))
            break
    if any(str(p).startswith('results/official/') for p in ctx.delivery_files()):
        floor = 2
        floor_why.append('official results present')
    parts.append((floor, 'stage floor: %s -> %d' % (', '.join(floor_why) or 'none', floor)))

    tier = max(p[0] for p in parts)
    reasons = [p[1] for p in parts]
    ctx._cache['tier'] = (tier, reasons)
    return tier, reasons


def native_closure_hashes(repo_root: Path, delivery: Path) -> set:
    """Content-hash set of a v1 delivery's native closure from its SOURCE_LOCK.json.

    Uses SOURCE_LOCK.native_closure_sha256 {path: sha} when present; otherwise closure_sha256
    minus the lock's files.admit/evaluator/generators. Empty set when there is no SOURCE_LOCK."""
    sl_path = Path(delivery) / 'SOURCE_LOCK.json'
    if not sl_path.is_file():
        return set()
    sl = load_json_file(sl_path)
    if not isinstance(sl, dict):
        raise InvalidInput('%s must be an object' % sl_path)
    nat = sl.get('native_closure_sha256')
    if isinstance(nat, dict):
        return set(str(v) for v in nat.values())
    clo = sl.get('closure_sha256')
    if clo is None:
        return set()
    if not isinstance(clo, dict):
        raise InvalidInput('%s closure_sha256 must be an object' % sl_path)
    excluded = set()
    lock_p = Path(delivery) / 'ENDPOINT_LOCK.json'
    if lock_p.is_file():
        lock = load_json_file(lock_p)
        files = lock.get('files') if isinstance(lock, dict) and isinstance(lock.get('files'), dict) else {}
        drel = Path(delivery).resolve().relative_to(Path(repo_root).resolve()).as_posix()
        for key in ('admit', 'evaluator', 'generators'):
            v = files.get(key)
            for e in ([v] if isinstance(v, str) else (v or [])):
                if isinstance(e, str):
                    excluded.add(e)
                    excluded.add((drel + '/' + e) if drel else e)
    return set(str(v) for p, v in clo.items() if p not in excluded)


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / float(len(a | b))


def v1_deliveries(ctx: Ctx) -> List[str]:
    """Repo-relative paths of v1 delivery dirs found by a bounded repo scan."""
    key = 'v1_deliveries'
    if key not in ctx._cache:
        scan = ctx.repo_scan(patterns=('ENDPOINT_LOCK.json',))
        out = []
        for f in scan.files:
            if is_v1_delivery(f.parent):
                out.append(f.parent.relative_to(scan.root).as_posix())
        ctx._cache[key] = sorted(set(out))
        ctx._cache['v1_scan_scope'] = scan.scope()
    return list(ctx._cache[key])


def family_graph(ctx: Ctx) -> dict:
    if 'family_graph' in ctx._cache:
        return ctx._cache['family_graph']
    root = Path(ctx.repo_root).resolve()
    members = v1_deliveries(ctx)
    me = ctx.delivery_rel
    if me not in members:
        members = sorted(set(members) | {me})
    sets = {m: native_closure_hashes(root, root / m) for m in members}
    parent = {m: m for m in members}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    edges = []
    for i, a in enumerate(members):
        for b in members[i + 1:]:
            j = jaccard(sets[a], sets[b])
            if j >= 0.5:
                edges.append((a, b, j))
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
    comp = sorted(m for m in members if find(m) == find(me))
    g = {'members': members, 'sets': sets, 'edges': edges, 'family': comp,
         'family_id': 'FAM-' + sha8(min(comp)), 'scope': ctx._cache.get('v1_scan_scope', '')}
    ctx._cache['family_graph'] = g
    return g


def family_of(ctx: Ctx) -> Tuple[str, List[str]]:
    """(family_id, members): connected component of this delivery under native-closure Jaccard >= 0.5
    over v1 SOURCE_LOCK files across the repo; id FAM-<sha8 of the smallest member name>."""
    g = family_graph(ctx)
    return g['family_id'], list(g['family'])


def closure_jaccard_neighbors(ctx: Ctx) -> List[Tuple[str, float]]:
    """Other v1 deliveries with their native-closure Jaccard against this delivery (all pairs)."""
    g = family_graph(ctx)
    me = ctx.delivery_rel
    return sorted(((m, jaccard(g['sets'][me], g['sets'][m])) for m in g['members'] if m != me),
                  key=lambda x: (-x[1], x[0]))


def family_cpu_h(ctx: Ctx, members: Sequence[str]) -> float:
    total = 0.0
    root = Path(ctx.repo_root)
    for m in members:
        p = root / m / 'runlog' / 'ATTEMPTS.jsonl'
        if p.is_file():
            total += attempts_cpu_s(load_jsonl_file(p), label=m + '/runlog/ATTEMPTS.jsonl')
    return total / 3600.0


# --------------------------------------------------------------------------------------
# Bundles, gate reports, exit codes
# --------------------------------------------------------------------------------------

VERDICT_BY_EXIT = {EXIT_PASS: 'PASS', EXIT_FAIL: 'FAIL', EXIT_MISSING: 'MISSING', EXIT_ERROR: 'CHECKER_ERROR'}


def exit_code(results: Sequence[Result]) -> int:
    """3 for CHECKER_ERROR, a non-PASS SELF-01 or no results; else 1 for any FAIL; else 2 for any
    MISSING; else 0 (WARN, ADVISORY_FAIL and NOT_APPLICABLE do not block)."""
    results = list(results or [])
    if not results:
        return EXIT_ERROR
    for r in results:
        st = getattr(r, 'status', None)
        if st not in STATUSES or st == CHECKER_ERROR:
            return EXIT_ERROR
        if getattr(r, 'id', None) == 'SELF-01' and st not in (PASS,):
            return EXIT_ERROR
    if any(r.status == FAIL for r in results):
        return EXIT_FAIL
    if any(r.status == MISSING for r in results):
        return EXIT_MISSING
    return EXIT_PASS


def first_source_lock_version(ctx: Ctx) -> Optional[str]:
    """checker_version of the first committed SOURCE_LOCK.json of this delivery, or None."""
    try:
        rel = ctx.repo_rel_of('SOURCE_LOCK.json')
        c = ctx.git.first_commit_adding(rel)
        if c is None:
            return None
        data = ctx.git.file_at(c.sha, rel)
        if data is None:
            return None
        obj = loads_strict(data)
        v = obj.get('checker_version') if isinstance(obj, dict) else None
        return v if isinstance(v, str) else None
    except (MissingInput, InvalidInput):
        return None


def _condition(ctx: Ctx, name: str) -> Tuple[bool, str]:
    try:
        if name == 't2':
            return (ctx.tier or 0) >= 2, 'tier %s' % ctx.tier
        if name == 'sealed_confirm':
            return sealed_confirm_intended(ctx), 'SEALED_CONFIRM intended or claimed'
        if name in ('sealed_or_adoption', 'oc_intended'):
            return (sealed_confirm_intended(ctx) or adoption_intended(ctx)), 'SEALED_CONFIRM or adoption intended'
        if name == 'udi_applicable':
            udi = ctx.lock.get('udi')
            return (udi not in (None, {}, []) or adoption_intended(ctx)), 'udi block declared or E3 promotion intended'
    except (MissingInput, InvalidInput) as e:
        return True, 'condition %s undeterminable (%s): check runs' % (name, e)
    raise CheckerError('unknown bundle condition %r' % name)


def resolve_check(rule: dict) -> Tuple[Optional[Callable], str]:
    mod = rule.get('module')
    if not isinstance(mod, str) or not mod:
        return None, 'RULES entry %s has no module' % rule.get('id')
    try:
        importlib.import_module(mod)
    except Exception as e:  # noqa: BLE001 - any import failure is a checker error
        return None, 'module %s failed to import: %s' % (mod, summarize_exception(e))
    fn = CHECKS.get(rule['id'])
    if fn is None:
        return None, 'check %s is not registered by module %s' % (rule['id'], mod)
    fmod = str(getattr(fn, '__module__', '')).split('.')[-1]
    if fmod != mod:
        return None, 'check %s is registered by module %s, RULES says %s' % (rule['id'], fmod, mod)
    return fn, ''


def run_check(ctx: Ctx, rule: dict) -> List[Result]:
    """Run one registered check with the fail-closed mapping of SPEC 7.6."""
    rid = rule['id']
    fn, err = resolve_check(rule)
    if fn is None:
        return [Result(rid, CHECKER_ERROR, err)]
    try:
        out = fn(ctx)
    except MissingInput as e:
        out = [Result(rid, MISSING, str(e))]
    except InvalidInput as e:
        out = [Result(rid, FAIL, 'invalid input: %s' % e)]
    except CheckerError as e:
        out = [Result(rid, CHECKER_ERROR, str(e))]
    except Exception as e:  # noqa: BLE001 - fail closed
        out = [Result(rid, FAIL, 'exception in check: %s' % summarize_exception(e))]
    if isinstance(out, Result):
        out = [out]
    if not isinstance(out, (list, tuple)) or not out:
        return [Result(rid, FAIL, 'check returned no result (never silently skipped)')]
    res: List[Result] = []
    for r in out:
        if not isinstance(r, Result):
            res.append(Result(rid, FAIL, 'check returned a non-Result value %r' % (r,)))
            continue
        if r.id != rid:
            res.append(Result(rid, FAIL, 'check returned a result for foreign id %r' % (r.id,)))
            continue
        if r.status not in STATUSES:
            res.append(Result(rid, FAIL, 'unknown status %r: %s' % (r.status, r.detail)))
            continue
        if r.status == NOT_APPLICABLE and not not_applicable_declared(rule):
            res.append(Result(rid, FAIL, 'NOT_APPLICABLE returned but RULES.json declares no condition: %s' % r.detail))
            continue
        r.detail = str(r.detail or '')
        r.evidence = list(r.evidence or [])
        r.data = dict(r.data or {})
        res.append(r)
    return res


def _downgrade(ctx: Ctx, rule: dict, r: Result, enforce_all: bool, first_version: Optional[str]) -> Result:
    if rule['id'] in KIT_INTEGRITY_IDS or r.status not in (FAIL, MISSING):
        return r
    if rule.get('status') == 'advisory':
        r.detail = '[advisory rule; was %s] %s' % (r.status, r.detail)
        r.status = WARN
        return r
    wbt = rule.get('warn_below_tier')
    if isinstance(wbt, int) and (ctx.tier or 0) < wbt:
        r.detail = '[tier %s < %d: WARN; was %s] %s' % (ctx.tier, wbt, r.status, r.detail)
        r.status = WARN
        return r
    if not enforce_all and first_version is not None:
        fv, iv, cv = parse_version(first_version), parse_version(rule.get('introduced_in')), parse_version(CHECKER_VERSION)
        if fv and iv and cv and (1, 0, 0) <= fv <= cv and iv > fv:
            r.detail = '[introduced_in %s > delivery checker_version %s: advisory window; was %s] %s' % (
                rule.get('introduced_in'), first_version, r.status, r.detail)
            r.status = ADVISORY_FAIL
    return r


def run_bundle(ctx: Ctx, stage: str) -> List[Result]:
    """Run RULES.json bundle `stage` (SPEC 7.5) and return the results in bundle order."""
    rules = ctx.rules or load_rules()
    idx = rule_index(rules)
    entries = expand_bundle(rules, stage)
    if ctx.stage is None:
        ctx.stage = stage
    try:
        tier, reasons = compute_tier(ctx)
        ctx.options['tier_error'] = None
    except (MissingInput, InvalidInput, CheckerError) as e:
        tier, reasons = 2, ['tier computation failed (%s): bundle resolved at tier 2' % e]
        ctx.options['tier_error'] = str(e)
    ctx.tier = tier
    ctx.options['tier_reasons'] = reasons
    enforce_all = tier >= 2 and stage in ('official', 'publish', 'pointer')
    first_version = None if enforce_all else first_source_lock_version(ctx)
    ctx.options['advisory_basis'] = {'enforce_all': enforce_all, 'first_source_lock_checker_version': first_version}
    results: List[Result] = []

    def run_entry(rid: str, when: Optional[str]) -> List[Result]:
        rule = idx[rid]
        t0 = time.perf_counter()
        if when is not None:
            ok, why = _condition(ctx, when)
            if not ok:
                return [Result(rid, NOT_APPLICABLE, 'bundle condition %s is false (%s)' % (when, why),
                               data={'runtime_s': 0.0})]
        if int(rule.get('tier_min', 0)) > tier:
            return [Result(rid, NOT_APPLICABLE, 'tier_min %s > computed tier %d' % (rule.get('tier_min'), tier),
                           data={'runtime_s': 0.0})]
        out = [_downgrade(ctx, rule, r, enforce_all, first_version) for r in run_check(ctx, rule)]
        dt = round(time.perf_counter() - t0, 4)
        for r in out:
            r.data.setdefault('runtime_s', dt)
        return out

    for e in entries:
        if 'group' in e:
            when = e.get('when')
            prim_ids = list(e.get('primary', []))
            fb_ids = list(e.get('fallback', []))
            if when is not None:
                ok, why = _condition(ctx, when)
                if not ok:
                    for rid in prim_ids + fb_ids:
                        results.append(Result(rid, NOT_APPLICABLE, 'bundle condition %s is false (%s)' % (when, why)))
                    continue
            prim: List[Result] = []
            for rid in prim_ids:
                prim.extend(run_entry(rid, None))
            if all(r.status in NON_BLOCKING_STATUSES for r in prim):
                results.extend(prim)
                for rid in fb_ids:
                    results.append(Result(rid, NOT_APPLICABLE, 'alternative %s satisfied by %s'
                                          % (e['group'], ','.join(prim_ids))))
                continue
            fb: List[Result] = []
            for rid in fb_ids:
                fb.extend(run_entry(rid, None))
            if fb and all(r.status in NON_BLOCKING_STATUSES for r in fb) and any(r.status == PASS for r in fb):
                for r in prim:
                    if r.status in (FAIL, MISSING):
                        r.detail = '[covered by %s; was %s] %s' % (','.join(fb_ids), r.status, r.detail)
                        r.status = WARN
            results.extend(prim)
            results.extend(fb)
            continue
        rid = e['id']
        if rid == 'SCHEMA-01':
            out = run_entry(rid, e.get('when'))
            results.extend(out)
            if any(r.status != PASS for r in out):
                ctx.options['legacy_stop'] = True
                return results  # gate never runs checks on non-v1 directories
            continue
        results.extend(run_entry(rid, e.get('when')))
    return results


def checker_files(kit_dir: Path = KIT_DIR) -> List[Tuple[str, Path]]:
    out = []
    for dirpath, dirnames, filenames in os.walk(kit_dir):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIR_NAMES)
        for fn in filenames:
            p = Path(dirpath, fn)
            rel = p.relative_to(kit_dir).as_posix()
            if not _is_cache_junk(rel):
                out.append((rel, p))
    return sorted(out)


def _checker_digest(entries: Iterable[Tuple[str, str]]) -> str:
    return sha256_bytes(''.join('%s  %s\n' % (h, rel) for rel, h in sorted(entries)).encode('utf-8'))


# checker_sha256 of an empty kit listing (== sha256(b'')): the committed checker of a corpus fixture commit
# whose tree holds no tools/minifly_check (timeline mode only; a real commit without a kit never counts)
EMPTY_KIT_SHA256 = _checker_digest([])

_CHECKER_SHA: Dict[str, str] = {}


def checker_sha256(kit_dir: Path = KIT_DIR, refresh: bool = False) -> str:
    """sha256 over '<sha256(file)>  <kit-relative path>\\n' lines of every kit file (sorted;
    __pycache__ and *.pyc excluded). Equals checker_sha256_at() of a commit whose kit tree matches."""
    key = str(kit_dir)
    if refresh or key not in _CHECKER_SHA:
        _CHECKER_SHA[key] = _checker_digest((rel, sha256_file(p)) for rel, p in checker_files(kit_dir))
    return _CHECKER_SHA[key]


def checker_sha256_at(git: GitView, commit: str, kit_rel: str = KIT_REL_DEFAULT) -> Optional[str]:
    """checker_sha256 of the kit tree committed at `commit` (None if the kit is absent there)."""
    kit_rel = norm_rel(kit_rel)
    paths = [p for p in git.list_files(commit, kit_rel) if not _is_cache_junk(p)]
    if not paths:
        return None
    blobs = git.files_at(commit, paths)
    entries = []
    for p in paths:
        data = blobs.get(p)
        if data is None:
            raise MissingInput('cannot read %s at %s' % (p, commit))
        entries.append((p[len(kit_rel) + 1:], sha256_bytes(data)))
    return _checker_digest(entries)


def write_gate_report(ctx: Ctx, stage: str, results: Sequence[Result]) -> Path:
    """Write <delivery>/checks/GATE_<stage>_<utc>.json (DRY_<stage>_<utc>.json when ctx.dry).
    Refuses (CheckerError) to write into a non-v1 directory, and when SELF-01 is in the results but did
    not PASS: a locally modified or unverified checker never produces a gate report (SPEC 13.2)."""
    if ctx.delivery is None or not is_v1_delivery(ctx.delivery):
        raise CheckerError('refusing to write a gate report into a non-v1 directory')
    bad_self = [r for r in results if r.id == 'SELF-01' and r.status != PASS]
    if bad_self:
        raise CheckerError('SELF-01 %s: no gate report is written (%s)' % (bad_self[0].status, bad_self[0].detail))
    rc = exit_code(results)
    try:
        fam_id, fam_members = family_of(ctx)
    except (MissingInput, InvalidInput, CheckerError) as e:
        fam_id, fam_members = None, ['family unavailable: %s' % e]
    try:
        head = ctx.git.head()
    except MissingInput as e:
        head = 'unavailable: %s' % e
    counts: Dict[str, int] = {}
    for r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    utc = utc_now()
    report = {
        'schema': SCHEMA_GATE_REPORT,
        'stage': stage,
        'dry': bool(ctx.dry),
        'delivery': ctx.delivery_rel,
        'utc': utc,
        'checker_sha256': checker_sha256(),
        'checker_version': CHECKER_VERSION,
        'git_mode': ctx.git_mode,
        'git_head': head,
        'endpoint_lock_sha256': ctx.endpoint_lock_sha256,
        'source_lock_sha256': ctx.source_lock_sha256,
        'tier': ctx.tier,
        'tier_reasons': list(ctx.options.get('tier_reasons') or []),
        'family_id': fam_id,
        'family_members': fam_members,
        'advisory_basis': ctx.options.get('advisory_basis'),
        'invoked_by': ctx.options.get('invoked_by', 'gate'),
        'runtime_s': round(sum(float(r.data.get('runtime_s', 0.0) or 0.0) for r in results), 4),
        'counts': counts,
        'results': [r.to_dict() for r in results],
        'exit_code': rc,
        'verdict': VERDICT_BY_EXIT[rc],
    }
    checks_dir = ctx.path('checks')
    checks_dir.mkdir(parents=True, exist_ok=True)
    prefix = 'DRY' if ctx.dry else 'GATE'
    path = checks_dir / ('%s_%s_%s.json' % (prefix, stage, utc))
    n = 1
    while path.exists():
        path = checks_dir / ('%s_%s_%s_%d.json' % (prefix, stage, utc, n))
        n += 1
    path.write_text(dumps_pretty(report), encoding='utf-8')
    ctx._cache.pop('delivery_files', None)
    return path


def format_results(results: Sequence[Result]) -> str:
    lines = []
    for r in results:
        d = r.detail.replace('\n', ' ')
        lines.append('%-15s %-14s %s' % (r.status, r.id, d[:240]))
    return '\n'.join(lines)


def cmd_gate(args: Any) -> int:
    """MFC gate --stage {lock|roster|official|publish|pointer} [--dry] [--pointer PATH] <dir>."""
    stage = args.stage
    if stage == 'pointer' and not getattr(args, 'pointer', None):
        sys.stderr.write('gate --stage pointer requires --pointer PATH\n')
        return EXIT_MISSING
    ctx = make_ctx(args, delivery=args.dir, stage=stage, dry=bool(getattr(args, 'dry', False)),
                   options={'pointer': getattr(args, 'pointer', None), 'invoked_by': 'gate'})
    ok, why = v1_status(ctx.delivery)
    if not ok and stage != 'pointer':
        msg = 'legacy delivery: use retro (%s)' % why
        # SCHEMA-01 is the check that refuses: its own (registered) result is reported. The gate refuses with
        # exit 2 and writes nothing into the legacy directory whatever it returns (SPEC 7.6).
        rule = rule_index(ctx.rules).get('SCHEMA-01') if isinstance(ctx.rules, dict) else None
        try:
            sres = run_check(ctx, rule) if rule else [Result('SCHEMA-01', MISSING, msg)]
        except Exception as e:  # noqa: BLE001 - the refusal never depends on the check succeeding
            sres = [Result('SCHEMA-01', CHECKER_ERROR, summarize_exception(e))]
        if getattr(args, 'json', False):
            print(json.dumps({'stage': stage, 'delivery': str(args.dir), 'exit_code': EXIT_MISSING,
                              'verdict': 'MISSING', 'detail': msg, 'results': [r.to_dict() for r in sres]}))
        else:
            sys.stderr.write(msg + '\n')
        return EXIT_MISSING
    results = run_bundle(ctx, stage)
    rc = exit_code(results)
    path = None
    if ok:
        try:
            path = write_gate_report(ctx, stage, results)
        except CheckerError as e:
            sys.stderr.write('%s\n' % e)
            rc = EXIT_ERROR
    if getattr(args, 'json', False):
        rep = load_json_file(path) if path else {'stage': stage, 'results': [r.to_dict() for r in results],
                                                 'exit_code': rc, 'verdict': VERDICT_BY_EXIT[rc]}
        print(json.dumps(rep, indent=2, ensure_ascii=False))
    else:
        print(format_results(results))
        print('gate %s: %s (exit %d)%s' % (stage, VERDICT_BY_EXIT[rc], rc,
                                          (' report %s' % path.relative_to(ctx.repo_root).as_posix()) if path else ''))
    return rc


# --------------------------------------------------------------------------------------
# Selftest cache and the kit-integrity checks SELF-01, SCHEMA-01, TIER-01
# --------------------------------------------------------------------------------------

_SELFTEST_ACTIVE = {'depth': 0}


def selftest_cache_path(sha: Optional[str] = None) -> Path:
    return Path(tempfile.gettempdir()) / ('minifly_check_selftest_%s.ok' % (sha or checker_sha256()))


def write_selftest_cache(sha: Optional[str] = None, summary: Optional[dict] = None) -> Path:
    """Called by governance.cmd_selftest after a successful run."""
    s = sha or checker_sha256(refresh=True)
    p = selftest_cache_path(s)
    body = {'checker_sha256': s, 'checker_version': CHECKER_VERSION, 'utc': utc_now(), 'summary': summary or {}}
    tmp = p.with_name(p.name + '.%d.tmp' % os.getpid())
    tmp.write_text(json.dumps(body), encoding='utf-8')
    os.replace(str(tmp), str(p))
    return p


def selftest_cache_valid(sha: str) -> Tuple[bool, str]:
    p = selftest_cache_path(sha)
    if not p.is_file():
        return False, 'no selftest cache %s' % p
    try:
        st = p.stat()
        if hasattr(os, 'getuid') and st.st_uid != os.getuid():
            return False, 'selftest cache %s is not owned by this user' % p
        if st.st_mode & 0o002:
            return False, 'selftest cache %s is world-writable' % p
        body = loads_strict(p.read_bytes())
    except (OSError, InvalidInput) as e:
        return False, 'selftest cache unreadable: %s' % e
    if not isinstance(body, dict) or body.get('checker_sha256') != sha:
        return False, 'selftest cache does not name checker sha %s' % sha
    return True, 'selftest cache %s' % p


def _run_selftest_for_self01(kit_root: Path) -> Tuple[int, str]:
    import argparse
    try:
        gov = importlib.import_module('governance')
        fn = getattr(gov, 'cmd_selftest')
    except Exception as e:  # noqa: BLE001
        return EXIT_ERROR, 'governance.cmd_selftest unavailable: %s' % summarize_exception(e)
    ns = argparse.Namespace(command='selftest', mutate=False, only=None, repo_root=str(kit_root), git=None,
                            json=False)
    _SELFTEST_ACTIVE['depth'] += 1
    try:
        rc = fn(ns)
    except SystemExit as e:
        rc = e.code if isinstance(e.code, int) else EXIT_ERROR
    except Exception as e:  # noqa: BLE001
        return EXIT_ERROR, 'selftest raised: %s' % summarize_exception(e)
    finally:
        _SELFTEST_ACTIVE['depth'] -= 1
    if not isinstance(rc, int):
        return EXIT_ERROR, 'selftest returned %r, not an exit code' % (rc,)
    return rc, 'selftest exit %d' % rc


@check('SELF-01')
def check_self_01(ctx: Ctx) -> Result:
    """Every tracked kit file equals its HEAD blob (no untracked kit files either), and selftest has
    passed for this checker sha (cached in the system temp dir; run first when absent)."""
    if isinstance(ctx.git, TimelineGit):
        kit_rel = KIT_REL_DEFAULT
        dirty = ctx.git.dirty_paths([kit_rel])
        if dirty:
            return Result('SELF-01', FAIL, 'checker files differ from HEAD in this fixture: %s' % ', '.join(dirty[:10]),
                          evidence=dirty[:20])
        return Result('SELF-01', PASS, 'timeline (corpus) mode: fixture kit paths clean; the real-kit worktree and '
                      'selftest cache are verified by the outer selftest, not inside it', data={'mode': 'timeline'})
    try:
        kit_root = git_toplevel(KIT_DIR)
    except MissingInput as e:
        return Result('SELF-01', FAIL, 'checker is not inside a git work tree: %s' % e)
    g = ctx.git if isinstance(ctx.git, RealGit) and Path(ctx.git.root) == kit_root else RealGit(kit_root)
    kit_rel = KIT_DIR.relative_to(kit_root).as_posix()
    tracked = g.ls_files([kit_rel])
    if not tracked:
        return Result('SELF-01', FAIL, 'checker files under %s are not committed' % kit_rel)
    dirty = g.dirty_paths([kit_rel])
    if dirty:
        return Result('SELF-01', FAIL, 'checker worktree differs from HEAD (commit or revert through governance): %s'
                      % ', '.join(dirty[:10]), evidence=dirty[:50])
    sha = checker_sha256(refresh=True)
    ok, why = selftest_cache_valid(sha)
    if not ok:
        if _SELFTEST_ACTIVE['depth'] > 0:
            return Result('SELF-01', FAIL, 'selftest cache absent while selftest is running (%s)' % why)
        rc, how = _run_selftest_for_self01(kit_root)
        ok, why = selftest_cache_valid(sha)
        if rc != 0 or not ok:
            return Result('SELF-01', FAIL, 'selftest did not pass for checker %s (%s; %s)' % (sha[:12], how, why))
    return Result('SELF-01', PASS, 'checker %s clean at HEAD; %s' % (sha[:12], why),
                  data={'checker_sha256': sha, 'tracked_files': len(tracked)})


@check('SCHEMA-01')
def check_schema_01(ctx: Ctx) -> Result:
    if ctx.delivery is None:
        raise MissingInput('no delivery directory given')
    ok, why = v1_status(ctx.delivery)
    if ok:
        return Result('SCHEMA-01', PASS, '%s: %s' % (ctx.delivery_rel, why))
    return Result('SCHEMA-01', MISSING, 'legacy delivery: use retro (%s)' % why)


@check('TIER-01')
def check_tier_01(ctx: Ctx) -> Result:
    tier, reasons = compute_tier(ctx)
    return Result('TIER-01', PASS, 'tier %d: %s' % (tier, '; '.join(reasons)),
                  data={'tier': tier, 'reasons': reasons})
