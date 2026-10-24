"""MRK-1 lockcheck (WP2): `init`, `lock build`, LOCK-01..13, AMD-01..05 and WIT-01 (SPEC 7.4, 8).

    MFC init <dir> [--parent <dir>] [--claim-level E1|E2|E3] [--task-family F] [--agent codex|claude]
    MFC lock build <dir>                                   # writes SOURCE_LOCK.json

Delivery code is never imported or executed: modules are parsed with `ast` only. Installed
packages are located with importlib.machinery.PathFinder and versioned with importlib.metadata,
neither of which imports them. Every check follows the fail-closed rules of SPEC 7.6: a missing
input is MISSING, a malformed one FAIL, and git history problems (shallow clone) MISSING.
"""
from __future__ import annotations

import ast
import gzip
import importlib
import importlib.machinery
import io
import os
import platform
import posixpath
import re
import sys
import sysconfig
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import common
from common import FAIL, NOT_APPLICABLE, PASS, WARN, InvalidInput, MissingInput, Result, check

FILL = '__FILL__'
AGENTS = ('codex', 'claude')
INTENTS = ('diagnostic', 'promotion')
LEVELS = ('E0', 'E1', 'E2', 'E3', 'E4')
TASK_FAMILIES = ('addition_multi_digit', 'addition_single_digit', 'addition_component', 'other_arithmetic',
                 'non_arithmetic')
ADDITION_FAMILIES = ('addition_multi_digit', 'addition_single_digit', 'addition_component')
FORMAT_KINDS = ('addition', 'other')
OUTPUT_ENUM = ('own_bytes_own_eos', 'own_bytes_evaluator_length', 'classification_label', 'margin_or_rank')
HELD_ENUM = ('unseen_composition', 'changed_presentation', 'taught_only')
FEEDBACK_ENUM = ('gradual_warmup', 'other')
WITNESS_ENUM = ('pair_rectangle', 'column_closure')
ARM_ROLES = ('candidate', 'no_prior_teaching', 'lesion', 'shuf', 'frozen', 'oracle_plumbing', 'positive_control',
             'other')
POOLS = ('TAUGHT', 'OLD', 'NEW', 'HELD', 'PRESENTATION')
STRATA = ('S0', 'S1', 'S2', 'S3', 'S4')
E3_STRATA = ('S2', 'S3', 'S4')
AGGREGATORS = ('paired_lcb', 'k_of_n', 'all_worlds')
RULE_TYPES = ('declarative', 'custom')
OFFICIAL_MODES = ('listed', 'sealed')
OFFICIAL_ROLES = ('MAIN', 'SEALED_CONFIRM', 'PILOT')
GUARD_KINDS = ('old_drop', 'old_acq')
AMD_REASONS = ('ceiling', 'testbed_invalid', 'measurement_defect', 'operational')
OBJECTIVE_METRICS = ('exact_own_eos', 'balanced_exact_own_eos')
OBJECTIVE_OUTPUT = 'own_bytes_own_eos'
SNR_CHANGES = ('frontend', 'read', 'init')
REVIEW_REL = 'review/REVIEW.json'

SHA_RE = re.compile(r'^[0-9a-f]{64}$')
HEX_RE = re.compile(r'^(?:[0-9a-f]{2})+$')
PARENT_RE = re.compile(r'^(?P<dir>[^@\s]+)@(?P<sha>[0-9a-f]{7,64})$')
PROTOCOL_RE = re.compile(r'^(?P<base>.+)-p(?P<n>[0-9]+)$')

# LOCK-08: names or keys that look like integrity booleans (case-insensitive; SPEC 3.4).
INTEGRITY_SUFFIX_RE = re.compile(r'(_ok|_passed|_passes|overlap|_verified|_independent)$', re.IGNORECASE)
INTEGRITY_PREFIX_RE = re.compile(r'^(no_|all_)', re.IGNORECASE)

# AMD-02: lock sections whose post-outcome change needs a UD, a co-primary and a protocol bump.
AMD02_SECTIONS = ('primary', 'guards', 'gates', 'arms', 'causal_control_arm', 'positive_control_arm', 'units',
                  'split', 'task', 'checkpoints')
# AMD-03: administrative lock fields an amendment changes by construction.
AMD_ADMIN_FIELDS = ('protocol_id', 'coprimary', 'user_decisions')
# AMD-02: objective_distance axes, ranked from the objective (0) outward. Unknown values rank highest.
DISTANCE_RANKS = {
    'task_family': {'addition_multi_digit': 0, 'addition_single_digit': 1, 'addition_component': 2,
                    'other_arithmetic': 3, 'non_arithmetic': 4},
    'output': {'own_bytes_own_eos': 0, 'own_bytes_evaluator_length': 1, 'classification_label': 2,
               'margin_or_rank': 3},
    'held': {'unseen_composition': 0, 'changed_presentation': 1, 'taught_only': 2},
    'native_only_path': {True: 0, False: 1},
    'feedback': {'gradual_warmup': 0, 'other': 1},
    'old_protection_measured': {True: 0, False: 1},
}
# LOCK-10: governed lock blocks and the CONSTRAINTS key each must equal.
GOVERNED_BLOCKS = (('io', 'io'), ('warmup', 'warmup'), ('native', 'native_forbidden'),
                   ('native_forbidden', 'native_forbidden'))

IMPORT_MODULE_FUNCS = ('importlib.import_module', '__import__', 'importlib.__import__', 'builtins.__import__',
                       'runpy.run_module')
# function -> index of the positional path argument (or keyword 'location'/'path_name'/'path')
FILE_LOAD_FUNCS = {'importlib.util.spec_from_file_location': 1, 'importlib.machinery.SourceFileLoader': 1,
                   'importlib.machinery.SourcelessFileLoader': 1, 'runpy.run_path': 0, 'imp.load_source': 1,
                   'imp.load_compiled': 1}
EXEC_FUNCS = ('exec', 'eval', 'builtins.exec', 'builtins.eval')
OPTIONAL_IMPORT_EXC = ('ImportError', 'ModuleNotFoundError', 'Exception', 'BaseException')
EXT_SUFFIXES = tuple(importlib.machinery.EXTENSION_SUFFIXES)


# --------------------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------------------

def strip_sha(v: Any) -> Optional[str]:
    """Lowercase 64-hex sha (an optional 'sha256:' prefix is stripped), else None."""
    if not isinstance(v, str):
        return None
    s = v.strip().lower()
    if s.startswith('sha256:'):
        s = s[7:]
    return s if SHA_RE.match(s) else None


def is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and v == v and v not in (float('inf'),
                                                                                            float('-inf'))


def is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def dig(obj: Any, dotted: str, default: Any = None) -> Any:
    cur = obj
    for part in dotted.split('.'):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def fill_paths(obj: Any, prefix: str = '') -> List[str]:
    """Paths of every string (value or key) that contains '__FILL__'."""
    out: List[str] = []
    if isinstance(obj, str):
        if FILL in obj:
            out.append(prefix or '<root>')
    elif isinstance(obj, dict):
        for k, v in obj.items():
            p = ('%s.%s' % (prefix, k)) if prefix else str(k)
            if FILL in str(k):
                out.append(p)
            out.extend(fill_paths(v, p))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(fill_paths(v, '%s[%d]' % (prefix, i)))
    return out


def leaf_diff(a: Any, b: Any, prefix: str = '') -> List[str]:
    """Dotted leaf paths whose canonical JSON differs between a and b."""
    if isinstance(a, dict) and isinstance(b, dict):
        out: List[str] = []
        for k in sorted(set(a) | set(b), key=str):
            p = ('%s.%s' % (prefix, k)) if prefix else str(k)
            if k not in a or k not in b:
                out.append(p)
            else:
                out.extend(leaf_diff(a[k], b[k], p))
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out.extend(leaf_diff(x, y, '%s[%d]' % (prefix, i)))
        return out
    try:
        same = common.canonical_json(a) == common.canonical_json(b)
    except InvalidInput:
        same = False
    return [] if same else [prefix or '<root>']


def path_top(p: str) -> str:
    return re.split(r'[.\[]', p, maxsplit=1)[0]


def path_covered(leaf: str, declared: Iterable[str]) -> bool:
    for d in declared:
        d = str(d).strip()
        if not d:
            continue
        if leaf == d or leaf.startswith(d + '.') or leaf.startswith(d + '['):
            return True
    return False


def delivery_ids(ctx: common.Ctx) -> List[str]:
    """Names that identify this delivery in UD scopes and ledgers (dir name and repo-relative path)."""
    out = [ctx.delivery_name]
    try:
        rel = ctx.delivery_rel
        if rel and rel not in out:
            out.append(rel)
    except (MissingInput, InvalidInput):
        pass
    return out


def floors(ctx: common.Ctx) -> dict:
    f = ctx.kb.constraints().get('threshold_floors')
    if not isinstance(f, dict):
        raise MissingInput('CONSTRAINTS.threshold_floors missing')
    return f


def author_family(lock: dict) -> Optional[str]:
    a = lock.get('author_agent')
    if not isinstance(a, str):
        return None
    fam = a.split(':', 1)[0].strip().lower()
    return fam if fam in AGENTS else None


def reviewer_family(lock: dict) -> Optional[str]:
    fam = author_family(lock)
    if fam is None:
        return None
    return 'claude' if fam == 'codex' else 'codex'


def in_namespace(branches: Iterable[str], family: str) -> List[str]:
    rx = re.compile(r'^(?:remotes/)?origin/%s/' % re.escape(family))
    return [b for b in branches if rx.match(str(b))]


def ud_with_scope(ctx: common.Ctx, ud_ids: Iterable[Any], scopes: Sequence[str]) -> Optional[str]:
    """First cited UD id present in USER_DECISIONS whose scope contains one of `scopes`."""
    for uid in ud_ids:
        if not isinstance(uid, str) or not uid.strip():
            continue
        ud = ctx.kb.user_decision(uid.strip())
        if not isinstance(ud, dict):
            continue
        sc = ud.get('scope') or []
        if isinstance(sc, str):
            sc = [sc]
        if any(s in sc for s in scopes):
            return uid.strip()
    return None


def lock_ud_ids(lock: dict) -> List[str]:
    out = []
    uds = lock.get('user_decisions')
    if isinstance(uds, list):
        out.extend(u for u in uds if isinstance(u, str))
    return out


def current_tier(ctx: common.Ctx) -> int:
    if isinstance(ctx.tier, int):
        return ctx.tier
    try:
        return common.compute_tier(ctx)[0]
    except (MissingInput, InvalidInput, common.CheckerError):
        return 2  # an undeterminable tier is treated as the strictest


def read_review(ctx: common.Ctx) -> Optional[dict]:
    p = ctx.path(REVIEW_REL)
    if not p.is_file():
        return None
    obj = ctx.read_json(REVIEW_REL)
    if not isinstance(obj, dict):
        raise InvalidInput('%s must be an object' % REVIEW_REL)
    return obj


def resolve_entry(ctx: common.Ctx, entry: str) -> Tuple[Optional[str], str]:
    """(repo-relative path, how) of a delivery-relative (preferred) or repo-relative files.* entry."""
    rel = common.norm_rel(entry)
    if ctx.path(rel).is_file():
        return ctx.repo_rel_of(rel), 'delivery'
    if ctx.repo_path(rel).is_file():
        return rel, 'repo'
    return None, 'missing'


def files_entries(lock: dict, key: str) -> List[str]:
    files = lock.get('files') if isinstance(lock.get('files'), dict) else {}
    v = files.get(key)
    if isinstance(v, str):
        return [v] if v else []
    if isinstance(v, list):
        return [e for e in v if isinstance(e, str) and e]
    return []


def expand_entries(ctx: common.Ctx, entries: Sequence[str]) -> Tuple[List[str], List[str]]:
    """Repo-relative files for files.* entries (globs allowed, delivery-relative first) and missing ones."""
    found: List[str] = []
    missing: List[str] = []
    for e in entries:
        if common.has_glob(e):
            hits = [ctx.repo_rel_of(f) for f in ctx.delivery_files() if common.glob_match(f, e)]
            if not hits:
                missing.append(e)
            found.extend(h for h in hits if h not in found)
            continue
        rr, _how = resolve_entry(ctx, e)
        if rr is None:
            missing.append(e)
        elif rr not in found:
            found.append(rr)
    return found, missing


# --------------------------------------------------------------------------------------
# Git helpers (strict ordering, content commits, outcome rows at commits)
# --------------------------------------------------------------------------------------

def log_index(ctx: common.Ctx) -> Dict[str, int]:
    key = 'wp2_log_index'
    if key not in ctx._cache:
        ctx._cache[key] = {c.sha: i for i, c in enumerate(ctx.git.log())}
    return ctx._cache[key]


def strictly_before(ctx: common.Ctx, a: Optional[common.Commit], b: Optional[common.Commit]) -> bool:
    """a is a proper ancestor of b (same commit does not count)."""
    if a is None or b is None or a.sha == b.sha:
        return False
    return ctx.git.is_ancestor(a.sha, b.sha)


def earliest(ctx: common.Ctx, commits: Iterable[Optional[common.Commit]]) -> Optional[common.Commit]:
    idx = log_index(ctx)
    cs = [c for c in commits if c is not None]
    if not cs:
        return None
    return min(cs, key=lambda c: idx.get(c.sha, 1 << 30))


def commits_before(ctx: common.Ctx, c: Optional[common.Commit]) -> List[common.Commit]:
    """Commits strictly before c in log order (all commits when c is None)."""
    commits = ctx.git.log()
    if c is None:
        return commits
    out = []
    for x in commits:
        if x.sha == c.sha:
            break
        out.append(x)
    return out


def is_empty_keep(path: str, data: Optional[bytes]) -> bool:
    return path.rsplit('/', 1)[-1] == '.gitkeep' and data is not None and len(data) == 0


def first_commit_adding_under(ctx: common.Ctx, prefixes: Sequence[str],
                              skip: Optional[Callable[[str], bool]] = None) -> Optional[common.Commit]:
    """Earliest commit in which a file under one of the repo-relative prefixes (globs allowed)
    exists, ignoring empty .gitkeep files and paths for which skip(path) is true."""
    prefixes = [p for p in prefixes if p]
    if not prefixes:
        return None
    for c in ctx.git.log(list(prefixes)):
        for p in c.paths:
            if not any(common.glob_match(p, pre) for pre in prefixes):
                continue
            if skip and skip(p):
                continue
            data = ctx.git.file_at(c.sha, p)
            if data is None or is_empty_keep(p, data):
                continue
            return c
    return None


def content_commit(ctx: common.Ctx, rel: str, pred: Callable[[Any], bool], parse_json: bool = True
                   ) -> Optional[common.Commit]:
    """First commit where `rel` (repo-relative) satisfies pred (on parsed JSON, or raw bytes)."""
    def p(b: bytes) -> bool:
        if not parse_json:
            return bool(pred(b))
        try:
            obj = common.loads_strict(b, what=rel)
        except InvalidInput:
            return False
        return bool(pred(obj))
    return ctx.git.first_commit_where(rel, p)


def jsonl_rows_from_bytes(data: bytes, label: str) -> List[dict]:
    out = []
    for lineno, raw in enumerate(data.splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        obj = common.loads_strict(line, what='%s:%d' % (label, lineno))
        if not isinstance(obj, dict):
            raise InvalidInput('%s:%d is not a JSON object' % (label, lineno))
        out.append(obj)
    return out


def is_outputs_name(name: str) -> bool:
    base = name.rsplit('/', 1)[-1]
    return base.startswith('OUTPUTS') and (base.endswith('.jsonl') or base.endswith('.jsonl.gz'))


def is_receipt_rel(drel: str) -> bool:
    return drel.startswith('results/receipts/') and drel.endswith('.json')


def outcome_rows_from_bytes(drel: str, data: bytes) -> List[dict]:
    """Outcome rows in one version of a delivery-relative file: OUTPUTS*.jsonl[.gz], members of
    RESULT_BUNDLE*.zip (OUTPUTS rows and receipts) or a receipt JSON. Other files give []."""
    name = drel.rsplit('/', 1)[-1]
    try:
        if common.RESULT_BUNDLE_RE.match(name):
            rows: List[dict] = []
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    m = info.filename
                    b = zf.read(info)
                    if m.endswith('.gz'):
                        b = gzip.decompress(b)
                        m = m[:-3]
                    label = '%s!%s' % (drel, info.filename)
                    if is_outputs_name(m):
                        rows.extend(jsonl_rows_from_bytes(b, label))
                    elif ('/receipts/' in '/' + m) and m.endswith('.json'):
                        obj = common.loads_strict(b, what=label)
                        if isinstance(obj, dict):
                            rows.append(obj)
            return rows
        if name.endswith('.jsonl.gz') and is_outputs_name(name):
            return jsonl_rows_from_bytes(gzip.decompress(data), drel)
        if name.endswith('.jsonl') and is_outputs_name(name):
            return jsonl_rows_from_bytes(data, drel)
        if is_receipt_rel(drel):
            obj = common.loads_strict(data, what=drel)
            if not isinstance(obj, dict):
                raise InvalidInput('%s must be an object' % drel)
            return [obj]
    except (zipfile.BadZipFile, OSError, EOFError) as e:
        raise InvalidInput('%s unreadable: %s' % (drel, e))
    return []


def current_outcome_rows(ctx: common.Ctx) -> List[Tuple[str, dict]]:
    """(source label, row) for every OUTPUTS row and receipt in the delivery worktree."""
    out: List[Tuple[str, dict]] = []
    for label, _ln, row in ctx.iter_rows_with_source('results/**/OUTPUTS*.jsonl'):
        out.append((label, row))
    for rel in ctx.delivery_files():
        if is_receipt_rel(rel):
            obj = ctx.read_json(rel)
            if not isinstance(obj, dict):
                raise InvalidInput('%s must be an object' % rel)
            out.append((rel, obj))
    return out


def delivery_prefix(ctx: common.Ctx) -> str:
    d = ctx.delivery_rel
    return (d + '/') if d else ''


def committed_results_files(ctx: common.Ctx, commits: Sequence[common.Commit]) -> List[Tuple[common.Commit, str, str]]:
    """(commit, repo path, delivery path) of every results/** version present in `commits`."""
    pre = delivery_prefix(ctx) + 'results/'
    out = []
    for c in commits:
        for p in c.paths:
            if p.startswith(pre):
                out.append((c, p, p[len(delivery_prefix(ctx)):]))
    return out


def learner_outcome_before(ctx: common.Ctx, commit: Optional[common.Commit], include_worktree: bool
                           ) -> Optional[str]:
    """Evidence of a committed learner-arm outcome (a receipt or OUTPUTS row whose arm is not
    CEILING_*/FLOOR_*) in a commit strictly before `commit` (every commit when None), plus the
    worktree when include_worktree."""
    for c, p, drel in committed_results_files(ctx, commits_before(ctx, commit)):
        if not (is_outputs_name(drel) or is_receipt_rel(drel) or common.RESULT_BUNDLE_RE.match(drel.rsplit('/', 1)[-1])):
            continue
        data = ctx.git.file_at(c.sha, p)
        if data is None:
            continue
        for row in outcome_rows_from_bytes(drel, data):
            if not common.is_evaluator_arm(row.get('arm')):
                return '%s@%s (arm %r)' % (drel, c.sha[:12], row.get('arm'))
    if include_worktree:
        for label, row in current_outcome_rows(ctx):
            if not common.is_evaluator_arm(row.get('arm')):
                return '%s (worktree; arm %r)' % (label, row.get('arm'))
    return None


def is_checker_written(drel: str) -> bool:
    parts = drel.split('/')
    return len(parts) == 3 and parts[0] == 'results' and parts[2] in ('SCORES.json', 'VERDICT.json', 'TESTBED.json')


def outcome_files_before(ctx: common.Ctx, commit: Optional[common.Commit], include_worktree: bool,
                         checker_written: bool = False) -> List[str]:
    """Delivery-relative outcome files (results/**, except checker-written SCORES/VERDICT/TESTBED and
    empty .gitkeep) committed strictly before `commit` (all commits when None) [+ worktree]. With
    checker_written=True, return instead the checker-written score files present before `commit`."""
    seen: Set[str] = set()
    for c, p, drel in committed_results_files(ctx, commits_before(ctx, commit)):
        if is_checker_written(drel) != checker_written or drel in seen:
            continue
        data = ctx.git.file_at(c.sha, p)
        if data is None or is_empty_keep(p, data):
            continue
        seen.add(drel)
    if include_worktree:
        for rel in ctx.delivery_files():
            if rel.startswith('results/') and is_checker_written(rel) == checker_written:
                if rel.endswith('/.gitkeep') and ctx.path(rel).stat().st_size == 0:
                    continue
                seen.add(rel)
    return sorted(seen)


# --------------------------------------------------------------------------------------
# Lock history and amendments
# --------------------------------------------------------------------------------------

def lock_rel(ctx: common.Ctx) -> str:
    return ctx.repo_rel_of('ENDPOINT_LOCK.json')


def parse_lock_bytes(data: Optional[bytes], label: str) -> Optional[dict]:
    if data is None:
        return None
    obj = common.loads_strict(data, what=label)
    if not isinstance(obj, dict):
        raise InvalidInput('%s must be an object' % label)
    return obj


def lock_versions(ctx: common.Ctx) -> List[Tuple[common.Commit, Optional[bytes]]]:
    """(commit, ENDPOINT_LOCK bytes or None when deleted) for each commit touching the lock, oldest first."""
    key = 'wp2_lock_versions'
    if key not in ctx._cache:
        rel = lock_rel(ctx)
        ctx._cache[key] = [(c, ctx.git.file_at(c.sha, rel)) for c in ctx.git.log([rel])]
    return list(ctx._cache[key])


def lock_content_by_sha(ctx: common.Ctx, sha: Optional[str]) -> Optional[dict]:
    """Parsed ENDPOINT_LOCK content whose sha256 is `sha` (committed or worktree), else None."""
    if not sha:
        return None
    p = ctx.path('ENDPOINT_LOCK.json')
    if p.is_file() and common.sha256_file(p) == sha:
        return parse_lock_bytes(p.read_bytes(), 'ENDPOINT_LOCK.json')
    try:
        versions = lock_versions(ctx)
    except MissingInput:
        return None
    for _c, data in versions:
        if data is not None and common.sha256_bytes(data) == sha:
            return parse_lock_bytes(data, 'ENDPOINT_LOCK.json@history')
    return None


def amendments(ctx: common.Ctx) -> List[dict]:
    return ctx.read_jsonl_optional('AMENDMENTS.jsonl')


def amendment_commit(ctx: common.Ctx, row: dict) -> Optional[common.Commit]:
    """Commit at which the amendment took effect: the first commit holding its new lock content, else
    the first commit holding the row. None means it is still pending in the worktree."""
    new = strip_sha(row.get('new_lock_sha256'))
    if new:
        c = content_commit(ctx, lock_rel(ctx), lambda b: common.sha256_bytes(b) == new, parse_json=False)
        if c is not None:
            return c
    rid = row.get('id')
    if rid:
        rel = ctx.repo_rel_of('AMENDMENTS.jsonl')

        def has_row(b: bytes) -> bool:
            try:
                return any(r.get('id') == rid for r in jsonl_rows_from_bytes(b, rel))
            except InvalidInput:
                return False
        return content_commit(ctx, rel, has_row, parse_json=False)
    return None


# --------------------------------------------------------------------------------------
# Static import closure (lock build, LOCK-05, LOCK-07)
# --------------------------------------------------------------------------------------

class _PathVal:
    """A pathlib.Path value inside the literal evaluator."""
    __slots__ = ('p',)

    def __init__(self, p: str):
        self.p = p


def _ospath_func(dotted: Optional[str]) -> Optional[str]:
    if not dotted:
        return None
    for pre in ('os.path.', 'posixpath.'):
        if dotted.startswith(pre):
            f = dotted[len(pre):]
            if f in ('join', 'dirname', 'abspath', 'realpath', 'normpath', 'basename'):
                return f
    return None


def collect_aliases(tree: ast.AST) -> Dict[str, str]:
    """Local name -> dotted origin for imports and simple attribute aliases (x = np.linalg)."""
    al: Dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.asname:
                    al[a.asname] = a.name
                else:
                    top = a.name.split('.')[0]
                    al.setdefault(top, top)
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            for a in node.names:
                if a.name != '*':
                    al[a.asname or a.name] = '%s.%s' % (node.module, a.name)
    for node in ast.walk(tree):
        if (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, (ast.Attribute, ast.Name))):
            d = dotted_name(node.value, al)
            if d and '.' in d and node.targets[0].id not in al:
                al[node.targets[0].id] = d
    return al


def dotted_name(node: ast.AST, aliases: Dict[str, str]) -> Optional[str]:
    if isinstance(node, ast.Name):
        return aliases.get(node.id, node.id)
    if isinstance(node, ast.Attribute):
        base = dotted_name(node.value, aliases)
        return ('%s.%s' % (base, node.attr)) if base else None
    return None


class _Lit:
    """Evaluate literal path compositions: string constants, __file__, f-strings of literals, '+',
    os.path.join/dirname/abspath/realpath/normpath, str(), pathlib.Path(...) with '/', .parent,
    .parents[n], .resolve(), .absolute(), .joinpath(), .as_posix(), and module names bound to such
    values. Anything else is non-literal (None)."""

    def __init__(self, file_abs: str, cwd_abs: str, aliases: Dict[str, str]):
        self.file_abs = file_abs
        self.cwd = cwd_abs
        self.aliases = aliases
        self.env: Dict[str, Any] = {}

    def abspath(self, s: str) -> str:
        return posixpath.normpath(s if s.startswith('/') else posixpath.join(self.cwd, s))

    @staticmethod
    def as_str(x: Any) -> Optional[str]:
        if isinstance(x, _PathVal):
            return x.p
        return x if isinstance(x, str) else None

    def ev(self, node: ast.AST) -> Any:
        try:
            return self._ev(node)
        except Exception:  # noqa: BLE001 - anything odd is simply non-literal
            return None

    def _ev(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            return node.value if isinstance(node.value, str) else None
        if isinstance(node, ast.Name):
            if node.id == '__file__':
                return self.file_abs
            return self.env.get(node.id)
        if isinstance(node, ast.JoinedStr):
            parts = []
            for v in node.values:
                if isinstance(v, ast.Constant):
                    parts.append(str(v.value))
                elif isinstance(v, ast.FormattedValue) and v.format_spec is None and v.conversion in (-1, 115):
                    s = self.as_str(self._ev(v.value))
                    if s is None:
                        return None
                    parts.append(s)
                else:
                    return None
            return ''.join(parts)
        if isinstance(node, ast.BinOp):
            left, right = self._ev(node.left), self._ev(node.right)
            if isinstance(node.op, ast.Add) and isinstance(left, str) and isinstance(right, str):
                return left + right
            if isinstance(node.op, ast.Div) and isinstance(left, _PathVal):
                rs = self.as_str(right)
                return _PathVal(posixpath.join(left.p, rs)) if rs is not None else None
            return None
        if isinstance(node, ast.Attribute):
            base = self._ev(node.value)
            if isinstance(base, _PathVal):
                if node.attr == 'parent':
                    return _PathVal(posixpath.dirname(base.p))
                if node.attr == 'name':
                    return posixpath.basename(base.p)
            return None
        if isinstance(node, ast.Subscript):
            if isinstance(node.value, ast.Attribute) and node.value.attr == 'parents':
                base = self._ev(node.value.value)
                sl = node.slice
                sl = getattr(sl, 'value', sl) if not isinstance(sl, ast.Constant) else sl
                if isinstance(base, _PathVal) and isinstance(sl, ast.Constant) and is_int(sl.value) and sl.value >= 0:
                    p = base.p
                    for _ in range(sl.value + 1):
                        p = posixpath.dirname(p)
                    return _PathVal(p)
            return None
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                base = self._ev(node.func.value)
                if isinstance(base, _PathVal):
                    a = node.func.attr
                    if a in ('resolve', 'absolute') and not node.args:
                        return _PathVal(self.abspath(base.p))
                    if a == 'joinpath':
                        ss = [self.as_str(self._ev(x)) for x in node.args]
                        if any(s is None for s in ss):
                            return None
                        return _PathVal(posixpath.join(base.p, *ss))
                    if a == 'as_posix' and not node.args:
                        return base.p
                    return None
            if node.keywords:
                return None
            fn = dotted_name(node.func, self.aliases)
            args = [self._ev(a) for a in node.args]
            sargs = [self.as_str(a) for a in args]
            f = _ospath_func(fn)
            if f:
                if not sargs or any(s is None for s in sargs):
                    return None
                if f == 'join':
                    return posixpath.join(*sargs)
                if len(sargs) != 1:
                    return None
                s = sargs[0]
                if f == 'dirname':
                    return posixpath.dirname(s)
                if f == 'basename':
                    return posixpath.basename(s)
                if f == 'normpath':
                    return posixpath.normpath(s)
                return self.abspath(s)  # abspath / realpath
            if fn in ('str', 'os.fspath') and len(args) == 1:
                return self.as_str(args[0])
            if fn in ('pathlib.Path', 'pathlib.PurePath', 'pathlib.PosixPath', 'pathlib.PurePosixPath'):
                if any(s is None for s in sargs):
                    return None
                return _PathVal(posixpath.join(*sargs) if sargs else '.')
            return None
        return None


class _ModInfo:
    def __init__(self, abs_path: str, rel: str):
        self.abs = abs_path
        self.rel = rel
        self.tree: Optional[ast.AST] = None
        self.src: Optional[str] = None
        self.error: Optional[str] = None
        self.aliases: Dict[str, str] = {}
        self.imports: List[Tuple[str, Tuple[str, ...], int, int, bool]] = []
        self.ins: List[str] = []
        self.app: List[str] = []
        self.nonliteral: List[str] = []
        self.dynamic: List[str] = []
        self.loads: List[Tuple[str, int]] = []


def _catches_import(st: ast.AST) -> bool:
    for h in getattr(st, 'handlers', []) or []:
        if h.type is None:
            return True
        elts = h.type.elts if isinstance(h.type, ast.Tuple) else [h.type]
        for e in elts:
            n = e.id if isinstance(e, ast.Name) else (e.attr if isinstance(e, ast.Attribute) else None)
            if n in OPTIONAL_IMPORT_EXC:
                return True
    return False


class _ModuleScanner:
    """Collect imports, literal sys.path targets, dynamic imports and file loads of one module, in
    statement order (so module-level path variables are known when sys.path is modified)."""

    def __init__(self, info: _ModInfo, cwd_abs: str):
        self.info = info
        self.lit = _Lit(info.abs, cwd_abs, info.aliases)

    def site(self, node: ast.AST) -> str:
        # the literal source text (not ast.unparse, whose output varies across Python versions), so a
        # SOURCE_LOCK built on one interpreter compares equal on another
        src = None
        if self.info.src is not None:
            try:
                src = ast.get_source_segment(self.info.src, node)
            except Exception:  # noqa: BLE001
                src = None
        if not src:
            try:
                src = ast.unparse(node)
            except Exception:  # noqa: BLE001
                src = type(node).__name__
        src = ' '.join(src.split())
        return '%s:%d: %s' % (self.info.rel, getattr(node, 'lineno', 0), src[:160])

    def run(self, tree: ast.Module) -> None:
        self.block(tree.body, False)

    def block(self, stmts: Sequence[ast.stmt], optional: bool) -> None:
        for st in stmts:
            self.stmt(st, optional)

    def stmt(self, st: ast.stmt, optional: bool) -> None:
        info = self.info
        if isinstance(st, ast.Import):
            for a in st.names:
                info.imports.append((a.name, (), 0, st.lineno, optional))
            return
        if isinstance(st, ast.ImportFrom):
            info.imports.append((st.module or '', tuple(a.name for a in st.names), st.level or 0, st.lineno, optional))
            return
        try_types = tuple(t for t in (ast.Try, getattr(ast, 'TryStar', None)) if t is not None)
        if isinstance(st, try_types):
            opt = optional or _catches_import(st)
            self.block(st.body, opt)
            for h in st.handlers:
                if h.type is not None:
                    self.scan_expr(h.type, optional)
                self.block(h.body, optional)
            self.block(st.orelse, optional)
            self.block(st.finalbody, optional)
            return
        has_body = False
        for _name, value in ast.iter_fields(st):
            if isinstance(value, list) and value and all(isinstance(v, ast.stmt) for v in value):
                has_body = True
                self.block(value, optional)
            elif isinstance(value, list):
                for v in value:
                    sub_body = getattr(v, 'body', None)
                    if isinstance(v, ast.AST) and isinstance(sub_body, list) and any(
                            isinstance(b, ast.stmt) for b in sub_body):
                        # match_case (3.10+) and similar nodes that carry their own statement body
                        has_body = True
                        for _n2, v2 in ast.iter_fields(v):
                            if v2 is sub_body:
                                continue
                            for x in (v2 if isinstance(v2, list) else [v2]):
                                if isinstance(x, ast.AST):
                                    self.scan_expr(x, optional)
                        self.block(sub_body, optional)
                    elif isinstance(v, ast.AST) and not isinstance(v, ast.stmt):
                        self.scan_expr(v, optional)
            elif isinstance(value, ast.AST):
                self.scan_expr(value, optional)
        if not has_body:
            self.assignment(st)

    def scan_expr(self, node: ast.AST, optional: bool) -> None:
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                self.call(sub, optional)

    def target(self, node: ast.AST) -> Optional[str]:
        s = self.lit.as_str(self.lit.ev(node))
        return self.lit.abspath(s) if s is not None else None

    def add_targets(self, elts: Sequence[ast.AST], where: List[str], node: ast.AST) -> None:
        for e in elts:
            t = self.target(e)
            if t is None:
                self.info.nonliteral.append(self.site(node))
            else:
                where.append(t)

    def is_sys_path(self, node: ast.AST) -> bool:
        return dotted_name(node, self.info.aliases) == 'sys.path'

    def assignment(self, st: ast.stmt) -> None:
        info = self.info
        if isinstance(st, ast.Assign):
            for t in st.targets:
                if self.is_sys_path(t) or (isinstance(t, ast.Subscript) and self.is_sys_path(t.value)):
                    v = st.value
                    if isinstance(v, (ast.List, ast.Tuple)):
                        self.add_targets(v.elts, info.ins, st)
                    elif (isinstance(v, ast.BinOp) and isinstance(v.op, ast.Add)
                          and isinstance(v.left, (ast.List, ast.Tuple)) and self.is_sys_path(v.right)):
                        self.add_targets(v.left.elts, info.ins, st)
                    elif (isinstance(v, ast.BinOp) and isinstance(v.op, ast.Add)
                          and self.is_sys_path(v.left) and isinstance(v.right, (ast.List, ast.Tuple))):
                        self.add_targets(v.right.elts, info.app, st)
                    else:
                        info.nonliteral.append(self.site(st))
            if len(st.targets) == 1 and isinstance(st.targets[0], ast.Name):
                val = self.lit.ev(st.value)
                if val is None:
                    self.lit.env.pop(st.targets[0].id, None)
                else:
                    self.lit.env[st.targets[0].id] = val
        elif isinstance(st, ast.AnnAssign) and isinstance(st.target, ast.Name) and st.value is not None:
            val = self.lit.ev(st.value)
            if val is None:
                self.lit.env.pop(st.target.id, None)
            else:
                self.lit.env[st.target.id] = val
        elif isinstance(st, ast.AugAssign) and self.is_sys_path(st.target):
            if isinstance(st.value, (ast.List, ast.Tuple)):
                self.add_targets(st.value.elts, info.app, st)
            else:
                info.nonliteral.append(self.site(st))

    def call(self, node: ast.Call, optional: bool) -> None:
        info = self.info
        func = node.func
        if isinstance(func, ast.Attribute) and self.is_sys_path(func.value):
            a = func.attr
            if a == 'insert':
                if len(node.args) == 2:
                    self.add_targets([node.args[1]], info.ins, node)
                else:
                    info.nonliteral.append(self.site(node))
            elif a == 'append':
                if len(node.args) == 1:
                    self.add_targets([node.args[0]], info.app, node)
                else:
                    info.nonliteral.append(self.site(node))
            elif a == 'extend':
                if len(node.args) == 1 and isinstance(node.args[0], (ast.List, ast.Tuple)):
                    self.add_targets(node.args[0].elts, info.app, node)
                else:
                    info.nonliteral.append(self.site(node))
            return
        fn = dotted_name(func, info.aliases)
        if fn == 'site.addsitedir':
            if node.args:
                self.add_targets([node.args[0]], info.app, node)
            else:
                info.nonliteral.append(self.site(node))
            return
        if fn in IMPORT_MODULE_FUNCS:
            arg = node.args[0] if node.args else None
            if arg is None:
                for kw in node.keywords:
                    if kw.arg in ('name', 'mod_name'):
                        arg = kw.value
            name = arg.value if isinstance(arg, ast.Constant) and isinstance(arg.value, str) else None
            if not name:
                info.dynamic.append(self.site(node))
                return
            level = len(name) - len(name.lstrip('.'))
            names: Tuple[str, ...] = ()
            if fn != 'runpy.run_module' and fn != 'importlib.import_module' and len(node.args) >= 4:
                fl = node.args[3]
                if isinstance(fl, (ast.List, ast.Tuple)):
                    names = tuple(e.value for e in fl.elts if isinstance(e, ast.Constant) and isinstance(e.value, str))
            if level and fn == 'importlib.import_module':
                pkg = node.args[1] if len(node.args) > 1 else None
                for kw in node.keywords:
                    if kw.arg == 'package':
                        pkg = kw.value
                if isinstance(pkg, ast.Constant) and isinstance(pkg.value, str) and pkg.value:
                    parts = pkg.value.split('.')
                    keep = len(parts) - (level - 1)
                    if keep <= 0:
                        info.dynamic.append(self.site(node))
                        return
                    rest = name.lstrip('.')
                    info.imports.append(('.'.join(parts[:keep] + ([rest] if rest else [])), names, 0, node.lineno,
                                         optional))
                    return
            info.imports.append((name.lstrip('.'), names, level, node.lineno, optional))
            return
        if fn in FILE_LOAD_FUNCS:
            idx = FILE_LOAD_FUNCS[fn]
            arg = node.args[idx] if len(node.args) > idx else None
            if arg is None:
                for kw in node.keywords:
                    if kw.arg in ('location', 'path_name', 'path', 'pathname'):
                        arg = kw.value
            t = self.target(arg) if arg is not None else None
            if t is None:
                info.dynamic.append(self.site(node))
            else:
                info.loads.append((t, node.lineno))
            return
        if fn in EXEC_FUNCS:
            info.dynamic.append(self.site(node))


class ClosureBuilder:
    """Static AST import closure of a delivery (SPEC 7.4 LOCK-05; Section 8 SOURCE_LOCK).

    Module search order for absolute imports: literal sys.path.insert targets, the delivery dir and
    the directories of the root files (regular modules only), then the stdlib and installed packages
    (never hashed; versions recorded), then literal sys.path.append targets and the repo root, then
    namespace-package portions. Relative imports resolve against the importing file's package."""

    def __init__(self, ctx: common.Ctx):
        self.ctx = ctx
        self.repo = Path(ctx.repo_root).resolve().as_posix()
        self.delivery = Path(ctx.delivery).resolve().as_posix()
        self.lock = ctx.lock
        self.shared = files_entries(self.lock, 'shared')
        self._infos: Dict[str, _ModInfo] = {}
        self._ext: Dict[str, str] = {}
        kit = common.KIT_DIR.resolve().as_posix()
        cwd = os.path.realpath(os.getcwd())
        self._ext_path = []
        for p in sys.path:
            if not p or p in ('.',):
                continue
            rp = os.path.realpath(p)
            if rp == kit or rp == cwd or rp == self.repo or rp.startswith(self.repo + '/'):
                continue
            self._ext_path.append(p)

    # ---- paths
    def rel(self, abs_path: str) -> Optional[str]:
        if abs_path == self.repo:
            return ''
        if abs_path.startswith(self.repo + '/'):
            return abs_path[len(self.repo) + 1:]
        return None

    def allowed(self, rel: str) -> bool:
        d = self.ctx.delivery_rel
        if rel == d or rel.startswith(d + '/'):
            return True
        if rel.startswith('tools/'):
            return True
        for e in self.shared:
            try:
                if common.glob_match(rel, common.norm_rel(e)) or common.glob_match(rel, self.ctx.repo_rel_of(e)):
                    return True
            except InvalidInput:
                continue
        return False

    # ---- parsing
    def info(self, abs_path: str) -> _ModInfo:
        if abs_path in self._infos:
            return self._infos[abs_path]
        rel = self.rel(abs_path) or abs_path
        info = _ModInfo(abs_path, rel)
        self._infos[abs_path] = info
        if not abs_path.endswith('.py'):
            return info
        try:
            data = Path(abs_path).read_bytes()
            tree = ast.parse(data, filename=rel)
        except (SyntaxError, ValueError) as e:
            info.error = '%s: does not parse (%s)' % (rel, e)
            return info
        except OSError as e:
            info.error = '%s: unreadable (%s)' % (rel, e)
            return info
        info.tree = tree
        info.src = data.decode('utf-8', 'replace')
        info.aliases = collect_aliases(tree)
        _ModuleScanner(info, self.delivery).run(tree)
        return info

    # ---- resolution
    @staticmethod
    def _regular(d: str, name: str) -> Optional[Tuple[List[str], Optional[str]]]:
        pkg = posixpath.join(d, name)
        init = posixpath.join(pkg, '__init__.py')
        if os.path.isfile(init):
            return [init], pkg
        if os.path.isfile(pkg + '.py'):
            return [pkg + '.py'], None
        for s in EXT_SUFFIXES:
            if os.path.isfile(pkg + s):
                return [pkg + s], None
        return None

    def _descend(self, pkg_dirs: List[str], parts: Sequence[str]) -> Optional[Tuple[List[str], List[str]]]:
        files: List[str] = []
        dirs = list(pkg_dirs)
        for part in parts:
            if not dirs:
                return None  # a submodule of a plain module
            hit = None
            for d in dirs:
                hit = self._regular(d, part)
                if hit:
                    break
            if hit:
                files.extend(hit[0])
                dirs = [hit[1]] if hit[1] else []
                continue
            ns = [posixpath.join(d, part) for d in dirs if os.path.isdir(posixpath.join(d, part))]
            if not ns:
                return None
            dirs = ns
        return files, dirs

    def _find_regular(self, search: Sequence[str], parts: Sequence[str]) -> Optional[Tuple[List[str], List[str]]]:
        for d in search:
            hit = self._regular(d, parts[0])
            if hit:
                rest = self._descend([hit[1]] if hit[1] else [], parts[1:]) if len(parts) > 1 else ([], [hit[1]] if hit[1] else [])
                if rest is None:
                    return None
                return hit[0] + rest[0], rest[1]
        return None

    def _find_namespace(self, search: Sequence[str], parts: Sequence[str]) -> Optional[Tuple[List[str], List[str]]]:
        ns = [posixpath.join(d, parts[0]) for d in search if os.path.isdir(posixpath.join(d, parts[0]))]
        if not ns:
            return None
        if len(parts) == 1:
            return [], ns
        return self._descend(ns, parts[1:])

    def classify_external(self, top: str) -> str:
        if top in self._ext:
            return self._ext[top]
        cls = 'unresolved'
        if top in sys.builtin_module_names:
            cls = 'stdlib'
        elif top in (getattr(sys, 'stdlib_module_names', None) or ()):
            cls = 'stdlib'
        else:
            try:
                spec = importlib.machinery.PathFinder.find_spec(top, self._ext_path)
            except Exception:  # noqa: BLE001
                spec = None
            if spec is not None:
                origin = spec.origin if isinstance(spec.origin, str) else ''
                locs = list(spec.submodule_search_locations or [])
                where = origin or (locs[0] if locs else '')
                parts = where.replace('\\', '/').split('/')
                if 'site-packages' in parts or 'dist-packages' in parts:
                    cls = 'third_party'
                else:
                    std = [sysconfig.get_paths().get(k) for k in ('stdlib', 'platstdlib')]
                    std = [os.path.realpath(s) for s in std if s]
                    rw = os.path.realpath(where) if where else ''
                    cls = 'stdlib' if any(rw == s or rw.startswith(s + os.sep) for s in std) else 'third_party'
        self._ext[top] = cls
        return cls

    def resolve(self, imp: Tuple[str, Tuple[str, ...], int, int, bool], importer: _ModInfo,
                pre: Sequence[str], post: Sequence[str]) -> Tuple[str, List[str], str]:
        """(status, files, detail): status repo | external | unresolved | outside."""
        module, names, level, _lineno, _opt = imp
        if level:
            base = posixpath.dirname(importer.abs)
            for _ in range(level - 1):
                base = posixpath.dirname(base)
            if self.rel(base) is None:
                return 'outside', [], 'relative import climbs out of the repo'
            files: List[str] = []
            init = posixpath.join(base, '__init__.py')
            if os.path.isfile(init):
                files.append(init)
            dirs = [base]
            if module:
                r = self._descend([base], module.split('.'))
                if r is None:
                    return 'unresolved', [], 'relative module %s%s not found' % ('.' * level, module)
                files.extend(r[0])
                dirs = r[1]
            for n in names:
                if n != '*':
                    sub = self._descend(dirs, [n])
                    if sub:
                        files.extend(sub[0])
            return 'repo', files, ''
        if not module:
            return 'unresolved', [], 'empty module name'
        parts = module.split('.')
        r = self._find_regular(pre, parts)
        if r is None:
            cls = self.classify_external(parts[0])
            if cls in ('stdlib', 'third_party'):
                return 'external', [], cls
            r = self._find_regular(post, parts)
            if r is None:
                r = self._find_namespace(list(pre) + list(post), parts)
            if r is None:
                return 'unresolved', [], 'not in the repo search path, the stdlib or installed packages'
        files, dirs = list(r[0]), r[1]
        for n in names:
            if n != '*':
                sub = self._descend(dirs, [n])
                if sub:
                    files.extend(sub[0])
        return 'repo', files, ''

    # ---- closure
    def roots_for(self, key: str) -> Tuple[List[str], List[str]]:
        entries = files_entries(self.lock, key)
        found, missing = expand_entries(self.ctx, entries)
        return [posixpath.join(self.repo, f) for f in found], missing

    def closure(self, roots: Sequence[str]) -> dict:
        root_dirs: List[str] = []
        for r in roots:
            d = posixpath.dirname(r)
            if d != self.delivery and d not in root_dirs:
                root_dirs.append(d)
        ins: List[str] = []
        app: List[str] = []
        result: dict = {}
        for _round in range(12):
            pre = ins + [self.delivery] + root_dirs
            post = app + [self.repo]
            files: Dict[str, _ModInfo] = {}
            queue = list(roots)
            new_ins: List[str] = []
            new_app: List[str] = []
            externals: Dict[str, str] = {}
            unresolved: List[str] = []
            optional_missing: List[str] = []
            outside: List[str] = []
            edges: List[Tuple[str, str, int]] = []
            while queue:
                f = queue.pop(0)
                if f in files:
                    continue
                info = self.info(f)
                files[f] = info
                for t in info.ins:
                    if t not in new_ins:
                        new_ins.append(t)
                for t in info.app:
                    if t not in new_app:
                        new_app.append(t)
                for imp in info.imports:
                    status, rfiles, detail = self.resolve(imp, info, pre, post)
                    label = '%s:%d: import %s%s%s' % (info.rel, imp[3], '.' * imp[2], imp[0],
                                                      (' (%s)' % ','.join(imp[1])) if imp[1] else '')
                    if status == 'external':
                        externals[imp[0].split('.')[0]] = detail
                    elif status == 'unresolved':
                        (optional_missing if imp[4] else unresolved).append('%s: %s' % (label, detail))
                    elif status == 'outside':
                        outside.append('%s: %s' % (label, detail))
                    for rf in rfiles:
                        edges.append((info.rel, self.rel(rf) or rf, imp[3]))
                        if rf not in files:
                            queue.append(rf)
                for (t, lineno) in info.loads:
                    if os.path.isfile(t):
                        edges.append((info.rel, self.rel(t) or t, lineno))
                        if t not in files:
                            queue.append(t)
                    else:
                        unresolved.append('%s:%d: file load %s: no such file' % (info.rel, lineno, self.rel(t) or t))
            result = {'files': files, 'externals': externals, 'unresolved': unresolved,
                      'optional_missing': optional_missing, 'outside': outside, 'edges': edges,
                      'ins': new_ins, 'app': new_app, 'converged': True}
            if new_ins == ins and new_app == app:
                return result
            ins, app = new_ins, new_app
        result['converged'] = False
        return result

    def build(self) -> dict:
        """Everything `lock build` writes and LOCK-05/LOCK-07 compare."""
        problems: List[str] = []
        kinds: Dict[str, dict] = {}
        all_files: Dict[str, _ModInfo] = {}
        externals: Dict[str, str] = {}
        optional_missing: List[str] = []
        for key in ('native', 'runner', 'admit'):
            roots, missing = self.roots_for(key)
            for m in missing:
                problems.append('files.%s entry %s does not exist' % (key, m))
            if not files_entries(self.lock, key):
                problems.append('files.%s is empty' % key)
            res = self.closure(roots)
            if not res['converged']:
                problems.append('files.%s: literal sys.path targets did not converge' % key)
            fmap: Dict[str, str] = {}
            for f, info in res['files'].items():
                rel = self.rel(f)
                if rel is None:
                    problems.append('%s closure reaches %s outside the repo (collision)' % (key, f))
                    continue
                if not self.allowed(rel):
                    problems.append('collision: %s closure resolves %s, outside the delivery dir, tools/ and '
                                    'files.shared' % (key, rel))
                if info.error:
                    problems.append(info.error)
                fmap[rel] = common.sha256_file(f)
                all_files[rel] = info
            for u in res['unresolved']:
                problems.append('unresolved %s' % u)
            for o in res['outside']:
                problems.append('collision: %s' % o)
            for info in res['files'].values():
                for s in info.nonliteral:
                    problems.append('non-literal sys.path change %s' % s)
            externals.update(res['externals'])
            optional_missing.extend(res['optional_missing'])
            kinds[key] = {'map': fmap, 'edges': res['edges']}
        closure: Dict[str, str] = {}
        for k in kinds.values():
            closure.update(k['map'])
        dynamic = sorted(set(s for info in all_files.values() for s in info.dynamic))
        params_sha = None
        pe = files_entries(self.lock, 'params')
        if len(pe) != 1:
            problems.append('files.params must name exactly one file')
        else:
            rr, _how = resolve_entry(self.ctx, pe[0])
            if rr is None:
                problems.append('files.params %s does not exist' % pe[0])
            else:
                params_sha = common.sha256_file(self.ctx.repo_path(rr))
        return {
            'closure_sha256': dict(sorted(closure.items())),
            'native_closure_sha256': dict(sorted(kinds['native']['map'].items())),
            'runner_closure_sha256': dict(sorted(kinds['runner']['map'].items())),
            'admit_closure_sha256': dict(sorted(kinds['admit']['map'].items())),
            'native_closure_sha': common.sha256_json(dict(sorted(kinds['native']['map'].items()))),
            'runner_closure_sha': common.sha256_json(dict(sorted(kinds['runner']['map'].items()))),
            'params_sha256': params_sha,
            'dynamic_imports': dynamic,
            'externals': dict(sorted(externals.items())),
            'optional_missing': sorted(set(optional_missing)),
            'problems': sorted(set(problems)),
            'infos': all_files,
            'kinds': kinds,
        }


def build_closure(ctx: common.Ctx) -> dict:
    key = 'wp2_closure'
    if key not in ctx._cache:
        ctx._cache[key] = ClosureBuilder(ctx).build()
    return ctx._cache[key]


def _packages_distributions() -> Dict[str, List[str]]:
    try:
        import importlib.metadata as md
    except ImportError:  # pragma: no cover - Python < 3.8
        return {}
    fn = getattr(md, 'packages_distributions', None)
    if fn is not None:
        try:
            return {k: list(v) for k, v in fn().items()}
        except Exception:  # noqa: BLE001
            pass
    out: Dict[str, List[str]] = {}
    try:
        for dist in md.distributions():
            name = dist.metadata.get('Name') if dist.metadata else None
            if not name:
                continue
            tops = (dist.read_text('top_level.txt') or '').split()
            if not tops:
                for f in (dist.files or []):
                    first = str(f).split('/')[0]
                    if first.endswith('.py'):
                        first = first[:-3]
                    if first and not first.endswith(('.dist-info', '.egg-info')) and first != '..':
                        tops.append(first)
            for t in set(tops):
                out.setdefault(t, []).append(name)
    except Exception:  # noqa: BLE001
        return out
    return out


def package_versions(externals: Dict[str, str]) -> Dict[str, str]:
    """{distribution: version} for third-party top-level modules, via importlib.metadata (no import)."""
    try:
        import importlib.metadata as md
    except ImportError:  # pragma: no cover
        return {}
    tops = sorted(t for t, cls in externals.items() if cls == 'third_party')
    if not tops:
        return {}
    pd = _packages_distributions()
    out: Dict[str, str] = {}
    for t in tops:
        dists = pd.get(t) or []
        if not dists:
            out[t] = 'unknown (no distribution metadata)'
            continue
        for d in dists:
            try:
                out[d] = md.version(d)
            except Exception:  # noqa: BLE001
                out[d] = 'unknown'
    return dict(sorted(out.items()))


SOURCE_LOCK_COMPARED = ('schema', 'delivery', 'endpoint_lock_sha256', 'checker_version', 'closure_sha256',
                        'native_closure_sha256', 'dynamic_imports', 'native_closure_sha', 'runner_closure_sha',
                        'params_sha256')


def source_lock_from_build(ctx: common.Ctx, b: dict) -> dict:
    return {
        'schema': common.SCHEMA_SOURCE_LOCK,
        'delivery': ctx.delivery_name,
        'endpoint_lock_sha256': ctx.endpoint_lock_sha256,
        'built_utc': common.utc_now(),
        'checker_version': common.CHECKER_VERSION,
        'closure_sha256': b['closure_sha256'],
        'native_closure_sha256': b['native_closure_sha256'],
        'dynamic_imports': b['dynamic_imports'],
        'env': {'python': platform.python_version(), 'implementation': platform.python_implementation(),
                'stdlib_modules': sorted(t for t, c in b['externals'].items() if c == 'stdlib'),
                'packages': package_versions(b['externals'])},
        'native_closure_sha': b['native_closure_sha'],
        'runner_closure_sha': b['runner_closure_sha'],
        'params_sha256': b['params_sha256'],
    }


# --------------------------------------------------------------------------------------
# init and lock build
# --------------------------------------------------------------------------------------

def _out(msg: str) -> None:
    sys.stdout.write(msg.rstrip('\n') + '\n')


def _err(msg: str) -> None:
    sys.stderr.write(msg.rstrip('\n') + '\n')


def laplace_base_rate(kb: common.KB, family: Optional[str]) -> Tuple[Any, str]:
    """Laplace base rate (k+1)/(n+2) of primary_gate PASS over v1 DELIVERY_LEDGER rows of the family."""
    if not family:
        return FILL, FILL
    try:
        rows = kb.rows('DELIVERY_LEDGER')
    except (MissingInput, InvalidInput) as e:
        return FILL, 'DELIVERY_LEDGER unavailable (%s): fill in by hand' % e
    # the same population as `MFC kb score` (kb.latest_delivery_rows): the latest v1 row per delivery
    latest: Dict[str, Tuple[Tuple[str, int], dict]] = {}
    for i, r in enumerate(rows):
        d = r.get('delivery')
        if not isinstance(d, str) or not d or r.get('legacy') is not False:
            continue
        key = (str(common.normalize_utc(r.get('utc')) or ''), i)
        if d not in latest or key >= latest[d][0]:
            latest[d] = (key, r)
    latest = {d: v[1] for d, v in latest.items()}
    # one row per delivery (`claim` appends a row per run; the latest one counts), scored outcomes only
    fam = [r for r in latest.values() if r.get('task_family') == family and r.get('primary_gate') in ('PASS', 'FAIL')]
    k = sum(1 for r in fam if r.get('primary_gate') == 'PASS')
    n = len(fam)
    value = round((k + 1.0) / (n + 2.0), 4)
    return value, ('laplace over the latest DELIVERY_LEDGER v1 row per delivery with task_family=%s and primary_gate '
                   'PASS|FAIL: %d PASS of %d -> (%d+1)/(%d+2)' % (family, k, n, k, n))


def cmd_init(args: Any) -> int:
    """MFC init <dir> [--parent <dir>] [--claim-level E1|E2|E3] [--task-family F] [--agent A]."""
    ctx = common.make_ctx(args, delivery=args.dir, stage='lock', options={'invoked_by': 'init'})
    d = Path(ctx.delivery)
    lock_path = d / 'ENDPOINT_LOCK.json'
    if lock_path.exists():
        _err('init REFUSED: %s already exists (never overwritten; change a lock only through AMENDMENTS)'
             % lock_path.relative_to(ctx.repo_root).as_posix())
        return common.EXIT_FAIL
    if not common.V1_DIR_RE.match(d.name):
        _err('init REFUSED: directory name %r does not match <name>_{design|dev}_vN_YYYYMMDD (schema v1)' % d.name)
        return common.EXIT_MISSING
    if (d / 'results').exists() and any(p.is_file() for p in (d / 'results').rglob('*')):
        _err('init REFUSED: %s already holds results/ files; the lock must precede every result (LOCK-06)' % d.name)
        return common.EXIT_FAIL
    tpl = common.load_json_file(common.KIT_DIR / 'templates' / 'ENDPOINT_LOCK.template.json')
    cons = ctx.kb.constraints()
    fl = cons.get('threshold_floors')
    if not isinstance(fl, dict):
        raise MissingInput('CONSTRAINTS.threshold_floors missing')
    lock = tpl
    notes: List[str] = []
    lock['delivery'] = d.name
    lock['protocol_id'] = '%s-p1' % d.name
    lock['created_utc'] = common.utc_now()
    agent = getattr(args, 'agent', None) or os.environ.get('MINIFLY_AGENT')
    if agent and agent.split(':', 1)[0].strip().lower() in AGENTS:
        lock['author_agent'] = agent.split(':', 1)[0].strip().lower()
    else:
        if agent:
            notes.append('MINIFLY_AGENT=%r is not codex|claude: author_agent left __FILL__' % agent)
        lock['author_agent'] = FILL
    lock['constraints_sha256'] = ctx.kb.constraints_sha256()

    # thresholds at the CONSTRAINTS floors
    prim = lock['primary']
    for key, fkey in (('margin_abs', 'margin_abs'), ('per_world_floor', 'per_world_floor'),
                      ('min_worlds', 'min_worlds_nonpilot')):
        if fkey in fl:
            prim[key] = fl[fkey]
    if isinstance(prim.get('aggregator'), dict) and prim['aggregator'].get('type') == 'paired_lcb' and 'alpha_max' in fl:
        prim['aggregator']['alpha'] = fl['alpha_max']
    for g in lock.get('guards') or []:
        if g.get('kind') == 'old_drop' and 'old_max_drop' in fl:
            g['max_drop'] = fl['old_max_drop']
        if g.get('kind') == 'old_acq' and 'old_acq_margin' in fl:
            g['margin'] = fl['old_acq_margin']
    at = lock['admission_thresholds']
    for key in ('input_contrast_min', 'prompt_id_margin_min', 'taught_acq_margin_min', 'shuf_band'):
        if key in fl:
            at[key] = fl[key]
    if 'min_realized_dose' in fl:
        for a in lock['arms']:
            if a.get('role') in ('no_prior_teaching', 'lesion'):
                a['min_realized_dose'] = fl['min_realized_dose']

    # claim level and required arms
    level = getattr(args, 'claim_level', None)
    lock['claim_level_target'] = level or FILL
    if level:
        req = (cons.get('required_arms_by_level') or {}).get(level)
        if not isinstance(req, list):
            notes.append('CONSTRAINTS.required_arms_by_level has no %s list: arms kept from the template' % level)
        else:
            keep = []
            for a in lock['arms']:
                if a.get('role') in req or a.get('id') == lock.get('causal_control_arm'):
                    keep.append(a)
            for role in req:
                if not any(a.get('role') == role for a in keep):
                    keep.append({'id': {'positive_control': 'POS', 'lesion': 'LES', 'frozen': 'FROZEN'}.get(
                        role, role.upper()), 'role': role, 'zero_channels': [FILL] if role == 'lesion' else [],
                        'declared_changed': [FILL] if role in ('positive_control', 'oracle_plumbing') else [],
                        'teaching_withheld': [], 'min_realized_dose': None})
            lock['arms'] = keep
    pos = [a['id'] for a in lock['arms'] if a.get('role') == 'positive_control']
    lock['positive_control_arm'] = pos[0] if pos else FILL
    if level == 'E3':
        lock['objective_distance']['held'] = 'unseen_composition'
        prim['pool'] = 'HELD'
        prim['strata_counted'] = list(E3_STRATA)

    # task family, format defaults and objective distance
    fam = getattr(args, 'task_family', None)
    lock['task']['family'] = fam or FILL
    lock['objective_distance']['task_family'] = fam or FILL
    if fam and fam not in ADDITION_FAMILIES:
        lock['task']['format'] = {'kind': 'other', 'sep_hex': FILL, 'terminator_hex': FILL, 'eos_hex': FILL,
                                  'gen_cap_bytes': FILL, 'normalize_leading_zeros': FILL}
    else:
        fmt = lock['task']['format']
        fmt.update({'kind': 'addition', 'sep_hex': '2b', 'terminator_hex': '3d', 'eos_hex': '0a',
                    'normalize_leading_zeros': True})

    # testbed grids
    for key, ckey in (('ceiling_grid', 'testbed_ceiling_grid'), ('floor_grid', 'testbed_floor_grid')):
        if isinstance(cons.get(ckey), list):
            lock['testbed'][key] = cons[ckey]
        else:
            notes.append('CONSTRAINTS.%s missing: testbed.%s left as in the template' % (ckey, key))

    # lineage
    parent = getattr(args, 'parent', None)
    if parent:
        pdir = common.resolve_delivery(ctx.repo_root, parent)
        if pdir is None or not pdir.is_dir():
            _err('init REFUSED: --parent %s is not a directory in the repo' % parent)
            return common.EXIT_MISSING
        prel = pdir.relative_to(Path(ctx.repo_root).resolve()).as_posix()
        try:
            commits = ctx.git.log([prel])
            psha = commits[-1].sha if commits else None
        except MissingInput as e:
            psha = None
            notes.append('parent commit unavailable (%s): lineage.parent sha left __FILL__' % e)
        lock['lineage']['parent'] = '%s@%s' % (prel, psha or FILL)
    else:
        lock['lineage']['parent'] = None

    # prediction skeletons with Laplace base rates
    value, source = laplace_base_rate(ctx.kb, fam)
    for p in lock.get('predictions') or []:
        bf = p.setdefault('base_rate_forecast', {})
        if p.get('target') == 'primary':
            bf['value'], bf['source'] = value, source
        else:
            bf['value'] = 0.5
            bf['source'] = 'laplace with no recorded guard outcomes in DELIVERY_LEDGER: (0+1)/(0+2)'

    # endpoint files already present
    efs = {}
    for e in files_entries(lock, 'admit') + files_entries(lock, 'evaluator') + files_entries(lock, 'generators'):
        p = d / e
        efs[e] = common.sha256_file(p) if p.is_file() else FILL
    lock['endpoint_files_sha256'] = efs

    # split hashes of manifests already present (SPEC 6.2: the held set is fixed in the lock; AMD-05 recomputes)
    split = lock.get('split') if isinstance(lock.get('split'), dict) else {}
    blobs = {}
    for key, _pool in common.SPLIT_MANIFESTS:
        rel = split.get(key)
        p = d / rel if isinstance(rel, str) and rel else None
        blobs[key] = p.read_bytes() if p is not None and p.is_file() else None
    held_sha, split_sha = common.split_hashes(blobs)
    if held_sha:
        split['held_set_sha256'] = held_sha
    if split_sha:
        split['split_sha256'] = split_sha
    else:
        notes.append('split manifests (%s) not all present: split.split_sha256%s left __FILL__ (formulas: '
                     'SCHEMAS.md section 1; AMD-05 recomputes both at every stage)'
                     % (', '.join(str(split.get(k)) for k, _p in common.SPLIT_MANIFESTS),
                        '' if held_sha else ' and held_set_sha256'))

    # E3 cadence
    try:
        kbmod = importlib.import_module('kb')
        fn = getattr(kbmod, 'deliveries_since_last_e3')
        due = fn(ctx.kb)
        cmax = cons.get('e3_cadence_max')
        if is_num(due) and is_num(cmax) and due >= cmax:
            notes.append('E3 cadence DUE: %s deliveries since the last held-E3 opening (e3_cadence_max %s). This lock '
                         'must open held E3 (pool HELD, strata S2..S4, claim_level_target E3, official MAIN or '
                         'SEALED_CONFIRM) or cite an owner waiver UD with scope e3_waiver:%s in '
                         'e3_cadence.waiver_user_decision (LOCK-09).' % (due, cmax, d.name))
        else:
            notes.append('E3 cadence: %s deliveries since the last held-E3 opening (max %s)' % (due, cmax))
    except Exception as e:  # noqa: BLE001 - informational only; LOCK-09 enforces
        notes.append('E3 cadence unknown (kb.deliveries_since_last_e3 unavailable: %s); LOCK-09 will report it'
                     % common.summarize_exception(e, 1))

    d.mkdir(parents=True, exist_ok=True)
    tmp = lock_path.with_name(lock_path.name + '.tmp')
    tmp.write_text(common.dumps_pretty(lock), encoding='utf-8')
    if lock_path.exists():  # raced
        tmp.unlink()
        _err('init REFUSED: %s appeared while writing' % lock_path)
        return common.EXIT_FAIL
    os.replace(str(tmp), str(lock_path))
    fills = fill_paths(lock)
    rel = lock_path.relative_to(Path(ctx.repo_root).resolve()).as_posix()
    if getattr(args, 'json', False):
        print(common.dumps_pretty({'path': rel, 'fill_remaining': fills, 'notes': notes}), end='')
    else:
        _out('init: wrote %s (%d __FILL__ values left; LOCK-01 fails until none remain)' % (rel, len(fills)))
        for n in notes:
            _out('  note: %s' % n)
    return common.EXIT_PASS


def cmd_lock_build(args: Any) -> int:
    """MFC lock build <dir>: write SOURCE_LOCK.json from the static AST closure."""
    if getattr(args, 'action', 'build') != 'build':
        _err('lock: only "build" is supported')
        return common.EXIT_ERROR
    ctx = common.make_ctx(args, delivery=args.dir, stage='technical', options={'invoked_by': 'lock build'})
    ok, why = common.v1_status(ctx.delivery)
    if not ok:
        _err('lock build REFUSED: legacy delivery: use retro (%s)' % why)
        return common.EXIT_MISSING
    b = build_closure(ctx)
    if b['problems']:
        _err('lock build REFUSED: the static closure is not complete and clean (SOURCE_LOCK.json not written):')
        for p in b['problems']:
            _err('  %s' % p)
        return common.EXIT_FAIL
    sl = source_lock_from_build(ctx, b)
    p = ctx.path('SOURCE_LOCK.json')
    if p.is_file():
        try:
            old = common.load_json_file(p)
        except (MissingInput, InvalidInput):
            old = None
        if isinstance(old, dict) and all(canonical_eq(old.get(k), sl.get(k)) for k in SOURCE_LOCK_COMPARED):
            # rewriting would only change built_utc, and with it the SOURCE_LOCK sha that reviewer acks
            # and the seal are bound to
            _out('lock build: %s unchanged (closure, dynamic imports, params and ENDPOINT_LOCK sha are equal)'
                 % p.relative_to(Path(ctx.repo_root).resolve()).as_posix())
            return common.EXIT_PASS
    tmp = p.with_name(p.name + '.tmp')
    tmp.write_text(common.dumps_pretty(sl), encoding='utf-8')
    os.replace(str(tmp), str(p))
    rel = p.relative_to(Path(ctx.repo_root).resolve()).as_posix()
    if getattr(args, 'json', False):
        print(common.dumps_pretty({'path': rel, 'source_lock': sl}), end='')
    else:
        _out('lock build: wrote %s (%d closure files; native %s; runner %s; params %s)'
             % (rel, len(sl['closure_sha256']), sl['native_closure_sha'][:12], sl['runner_closure_sha'][:12],
                (sl['params_sha256'] or 'none')[:12]))
        for s in sl['dynamic_imports']:
            _out('  dynamic import (listed in dynamic_imports; review it): %s' % s)
        for s in b['optional_missing']:
            _out('  optional import not installed here: %s' % s)
    return common.EXIT_PASS


# --------------------------------------------------------------------------------------
# LOCK-01: schema, required fields, no __FILL__, enums
# --------------------------------------------------------------------------------------

class _V:
    """Collects schema problems."""

    def __init__(self, lock: dict):
        self.lock = lock
        self.problems: List[str] = []

    def bad(self, msg: str) -> None:
        self.problems.append(msg)

    def get(self, path: str) -> Tuple[bool, Any]:
        cur: Any = self.lock
        for part in path.split('.'):
            if not isinstance(cur, dict) or part not in cur:
                return False, None
            cur = cur[part]
        return True, cur

    def req(self, path: str, kind: str, enum: Optional[Sequence[Any]] = None, nullable: bool = False) -> Any:
        present, v = self.get(path)
        if not present:
            self.bad('%s is missing' % path)
            return None
        if v is None:
            if not nullable:
                self.bad('%s is null' % path)
            return None
        self.typed(path, v, kind, enum)
        return v

    def typed(self, path: str, v: Any, kind: str, enum: Optional[Sequence[Any]] = None) -> None:
        ok = {
            'str': isinstance(v, str) and bool(v.strip()),
            'bool': isinstance(v, bool),
            'int': is_int(v),
            'num': is_num(v),
            'list': isinstance(v, list),
            'dict': isinstance(v, dict),
            'sha': strip_sha(v) is not None,
            'hex': isinstance(v, str) and bool(HEX_RE.match(v)),
            'utc': common.normalize_utc(v) is not None,
        }[kind]
        if not ok:
            self.bad('%s must be %s, got %r' % (path, kind, v if not isinstance(v, (dict, list)) else type(v).__name__))
            return
        if enum is not None and v not in enum:
            self.bad('%s=%r is not one of %s' % (path, v, '|'.join(str(e) for e in enum)))

    def str_list(self, path: str, v: Any, enum: Optional[Sequence[str]] = None, nonempty: bool = False) -> None:
        if not isinstance(v, list):
            self.bad('%s must be a list' % path)
            return
        if nonempty and not v:
            self.bad('%s must not be empty' % path)
        for i, e in enumerate(v):
            if not isinstance(e, str) or not e:
                self.bad('%s[%d] must be a non-empty string' % (path, i))
            elif enum is not None and e not in enum:
                self.bad('%s[%d]=%r is not one of %s' % (path, i, e, '|'.join(enum)))


def lock_schema_problems(lock: dict, delivery_name: Optional[str]) -> List[str]:
    v = _V(lock)
    if lock.get('schema') != common.SCHEMA_ENDPOINT_LOCK:
        v.bad('schema is %r, not %s' % (lock.get('schema'), common.SCHEMA_ENDPOINT_LOCK))
    name = v.req('delivery', 'str')
    if delivery_name and name is not None and name != delivery_name:
        v.bad('delivery %r differs from the directory name %r' % (name, delivery_name))
    pid = v.req('protocol_id', 'str')
    if isinstance(pid, str) and isinstance(name, str):
        m = PROTOCOL_RE.match(pid)
        if not m or m.group('base') != name:
            v.bad('protocol_id %r must be <delivery>-p<N>' % pid)
    v.req('created_utc', 'utc')
    v.req('author_agent', 'str', AGENTS)
    td = v.req('tier_declared', 'int', (0, 1, 2))
    del td
    v.req('intent', 'str', INTENTS)
    v.req('claim_level_target', 'str', LEVELS)
    # task
    v.req('task', 'dict')
    task_id = v.req('task.id', 'str')
    fam = v.req('task.family', 'str', TASK_FAMILIES)
    v.req('task.format', 'dict')
    kind = v.req('task.format.kind', 'str', FORMAT_KINDS)
    for k in ('sep_hex', 'terminator_hex', 'eos_hex'):
        v.req('task.format.%s' % k, 'hex')
    gc = v.req('task.format.gen_cap_bytes', 'int')
    if is_int(gc) and gc <= 0:
        v.bad('task.format.gen_cap_bytes must be > 0')
    v.req('task.format.normalize_leading_zeros', 'bool')
    if kind == 'addition' and isinstance(fam, str) and fam not in ADDITION_FAMILIES:
        v.bad('task.format.kind addition with task.family %s' % fam)
    # objective distance
    v.req('objective_distance', 'dict')
    odf = v.req('objective_distance.task_family', 'str', TASK_FAMILIES)
    if isinstance(odf, str) and isinstance(fam, str) and odf != fam:
        v.bad('objective_distance.task_family %s differs from task.family %s' % (odf, fam))
    v.req('objective_distance.output', 'str', OUTPUT_ENUM)
    v.req('objective_distance.held', 'str', HELD_ENUM)
    v.req('objective_distance.native_only_path', 'bool')
    v.req('objective_distance.feedback', 'str', FEEDBACK_ENUM)
    v.req('objective_distance.old_protection_measured', 'bool')
    # units (required non-null from E2 by LOCK-02)
    present, units = v.get('units')
    if not present:
        v.bad('units is missing')
    elif units is not None:
        v.req('units', 'dict')
        if isinstance(units, dict):
            v.req('units.witness', 'str', WITNESS_ENUM)
            v.req('units.commutative', 'bool')
    # split
    v.req('split', 'dict')
    for k in ('taught_manifest', 'held_manifest', 'old_manifest', 'new_manifest'):
        v.req('split.%s' % k, 'str')
    v.req('split.split_sha256', 'sha')
    v.req('split.held_set_sha256', 'sha')
    # exposure streams
    es = v.req('exposure_streams', 'list')
    for i, s in enumerate(es or []):
        if not isinstance(s, dict):
            v.bad('exposure_streams[%d] must be an object' % i)
            continue
        if not isinstance(s.get('path_glob'), str) or not s.get('path_glob'):
            v.bad('exposure_streams[%d].path_glob must be a non-empty string' % i)
        if not isinstance(s.get('feedback_bearing'), bool):
            v.bad('exposure_streams[%d].feedback_bearing must be a bool' % i)
        if not isinstance(s.get('phase'), str) or not s.get('phase'):
            v.bad('exposure_streams[%d].phase must be a non-empty string' % i)
    # arms
    arms = v.req('arms', 'list')
    ids: List[str] = []
    if isinstance(arms, list):
        if not arms:
            v.bad('arms must not be empty')
        for i, a in enumerate(arms):
            if not isinstance(a, dict):
                v.bad('arms[%d] must be an object' % i)
                continue
            aid = a.get('id')
            if not isinstance(aid, str) or not aid:
                v.bad('arms[%d].id must be a non-empty string' % i)
            else:
                if aid in ids:
                    v.bad('arm id %s is duplicated' % aid)
                ids.append(aid)
            if a.get('role') not in ARM_ROLES:
                v.bad('arms[%d].role %r is not one of %s' % (i, a.get('role'), '|'.join(ARM_ROLES)))
            for k in ('zero_channels', 'declared_changed', 'teaching_withheld'):
                if k not in a:
                    v.bad('arms[%d].%s is missing' % (i, k))
                else:
                    v.str_list('arms[%d].%s' % (i, k), a.get(k))
            if 'min_realized_dose' not in a:
                v.bad('arms[%d].min_realized_dose is missing' % i)
            elif a.get('min_realized_dose') is not None and not is_num(a.get('min_realized_dose')):
                v.bad('arms[%d].min_realized_dose must be a number or null' % i)
    for k in ('causal_control_arm', 'positive_control_arm'):
        v.req(k, 'str')
    ports = v.req('ports', 'list')
    for i, p in enumerate(ports or []):
        if not isinstance(p, dict) or not isinstance(p.get('id'), str) or not p.get('id'):
            v.bad('ports[%d] must be {id, signs[]}' % i)
            continue
        v.str_list('ports[%d].signs' % i, p.get('signs'), nonempty=True)
    # primary
    v.req('primary', 'dict')
    ptid = v.req('primary.task_id', 'str')
    if isinstance(ptid, str) and isinstance(task_id, str) and ptid != task_id:
        v.bad('primary.task_id %s differs from task.id %s' % (ptid, task_id))
    v.req('primary.metric', 'str')
    v.req('primary.unconditional', 'bool')
    pool = v.req('primary.pool', 'str', POOLS)
    cp = v.req('primary.checkpoint', 'str')
    strata = v.req('primary.strata_counted', 'list')
    if strata is not None:
        v.str_list('primary.strata_counted', strata, STRATA, nonempty=True)
    cand = v.req('primary.candidate_arm', 'str')
    if isinstance(cand, str) and ids and cand not in ids:
        v.bad('primary.candidate_arm %s is not an arm id' % cand)
    comps = v.req('primary.comparator_arms', 'list')
    if comps is not None:
        v.str_list('primary.comparator_arms', comps)
        for c in comps:
            if isinstance(c, str) and ids and c not in ids:
                v.bad('primary.comparator_arms entry %s is not an arm id' % c)
    # CEILING_*/FLOOR_* ids mark evaluator testbed arms (excluded from learner-outcome detection and
    # MET-01): such an arm may be listed, but only with role other and never as candidate, comparator
    # or control.
    roles_by_id = {a.get('id'): a.get('role') for a in (arms or []) if isinstance(a, dict)}
    learner_refs = [('primary.candidate_arm', cand), ('causal_control_arm', lock.get('causal_control_arm')),
                    ('positive_control_arm', lock.get('positive_control_arm'))] + [
        ('primary.comparator_arms', c) for c in (comps if isinstance(comps, list) else [])]
    for aid, role in roles_by_id.items():
        if common.is_evaluator_arm(aid) and role != 'other':
            v.bad('arm %s uses the evaluator prefix CEILING_/FLOOR_ but has role %r (only other)' % (aid, role))
    for where, ref in learner_refs:
        if common.is_evaluator_arm(ref):
            v.bad('%s %s is an evaluator (CEILING_/FLOOR_) arm' % (where, ref))
    v.req('primary.ladder', 'str', ('auto',))
    for k in ('chance', 'margin_abs', 'absolute_floor', 'per_world_floor'):
        v.req('primary.%s' % k, 'num')
    agg = v.req('primary.aggregator', 'dict')
    if isinstance(agg, dict):
        at = agg.get('type')
        if at not in AGGREGATORS:
            v.bad('primary.aggregator.type %r is not one of %s' % (at, '|'.join(AGGREGATORS)))
        elif at == 'paired_lcb' and not (is_num(agg.get('alpha')) and 0 < agg['alpha'] < 1):
            v.bad('primary.aggregator.alpha must be a number in (0, 1) for paired_lcb')
        elif at == 'k_of_n' and not (is_int(agg.get('k')) and agg['k'] >= 1):
            v.bad('primary.aggregator.k must be an integer >= 1 for k_of_n')
    mw = v.req('primary.min_worlds', 'int')
    if is_int(mw) and mw < 1:
        v.bad('primary.min_worlds must be >= 1')
    nip = v.req('primary.n_items_per_pool', 'dict')
    if isinstance(nip, dict):
        if not nip:
            v.bad('primary.n_items_per_pool must not be empty')
        for k, n in nip.items():
            if k not in POOLS:
                v.bad('primary.n_items_per_pool key %r is not a pool' % k)
            if not (is_int(n) and n > 0):
                v.bad('primary.n_items_per_pool.%s must be an integer > 0' % k)
        if isinstance(pool, str) and pool not in nip:
            v.bad('primary.n_items_per_pool lacks the primary pool %s' % pool)
    v.req('primary.rule_type', 'str', RULE_TYPES)
    gates = v.req('gates', 'list')
    for i, g in enumerate(gates or []):
        if not isinstance(g, dict) or not isinstance(g.get('id'), str) or not g.get('id'):
            v.bad('gates[%d] must be {id, description}' % i)
        elif not isinstance(g.get('description'), str) or not g.get('description'):
            v.bad('gates[%d].description must be a non-empty string' % i)
    guards = v.req('guards', 'list')
    gids: List[str] = []
    for i, g in enumerate(guards or []):
        if not isinstance(g, dict):
            v.bad('guards[%d] must be an object' % i)
            continue
        if not isinstance(g.get('id'), str) or not g.get('id'):
            v.bad('guards[%d].id must be a non-empty string' % i)
        elif g['id'] in gids:
            v.bad('guard id %s is duplicated' % g['id'])
        else:
            gids.append(g['id'])
        if g.get('kind') not in GUARD_KINDS:
            v.bad('guards[%d].kind %r is not one of %s' % (i, g.get('kind'), '|'.join(GUARD_KINDS)))
        if g.get('pool') not in POOLS:
            v.bad('guards[%d].pool %r is not a pool' % (i, g.get('pool')))
        if g.get('kind') == 'old_drop' and not is_num(g.get('max_drop')):
            v.bad('guards[%d].max_drop must be a number' % i)
        if g.get('kind') == 'old_acq' and not is_num(g.get('margin')):
            v.bad('guards[%d].margin must be a number' % i)
    present, udi = v.get('udi')
    if not present:
        v.bad('udi is missing (null when not used)')
    elif udi is not None:
        if not isinstance(udi, dict) or not isinstance(udi.get('variants_manifest'), str) or not is_num(udi.get('alpha')):
            v.bad('udi must be null or {variants_manifest, alpha}')
    cps = v.req('checkpoints', 'dict')
    if isinstance(cps, dict):
        for k in ('prefix', 'old_end', 'final'):
            if k not in cps:
                v.bad('checkpoints.%s is missing' % k)
        for k, n in cps.items():
            if not (is_int(n) and n >= 0):
                v.bad('checkpoints.%s must be a record index (integer >= 0)' % k)
        if isinstance(cp, str) and cp not in cps:
            v.bad('primary.checkpoint %s is not a key of checkpoints' % cp)
    v.req('stage_chain', 'bool')
    v.req('worlds', 'dict')
    for k in ('technical', 'dev', 'rehearsal'):
        wl = v.req('worlds.%s' % k, 'list')
        if wl is not None:
            v.str_list('worlds.%s' % k, wl)
    v.req('worlds.official', 'dict')
    mode = v.req('worlds.official.mode', 'str', OFFICIAL_MODES)
    oids = v.req('worlds.official.ids', 'list')
    if oids is not None:
        v.str_list('worlds.official.ids', oids)
    on = v.req('worlds.official.n', 'int')
    if is_int(on) and on < 0:
        v.bad('worlds.official.n must be >= 0')
    if mode == 'listed' and isinstance(oids, list) and is_int(on) and on > 0 and len(oids) != on:
        v.bad('worlds.official: listed mode with %d ids but n=%d' % (len(oids), on))
    if mode == 'sealed' and is_int(on) and on <= 0:
        v.bad('worlds.official: sealed mode needs n > 0')
    v.req('worlds.official_role_intended', 'str', OFFICIAL_ROLES)
    v.req('schedule', 'dict')
    v.req('schedule.paired_atomic', 'bool')
    v.req('budget', 'dict')
    for k in ('max_records', 'max_cpu_h'):
        b = v.req('budget.%s' % k, 'num')
        if is_num(b) and b <= 0:
            v.bad('budget.%s must be > 0' % k)
    v.req('admission_thresholds', 'dict')
    snr = v.req('admission_thresholds.analytic_input_snr', 'str' if isinstance(
        dig(lock, 'admission_thresholds.analytic_input_snr'), str) else 'num')
    if isinstance(snr, str) and snr != 'not_derivable':
        v.bad("admission_thresholds.analytic_input_snr must be a number or 'not_derivable'")
    for k in ('input_contrast_min', 'prompt_id_margin_min', 'taught_acq_margin_min'):
        v.req('admission_thresholds.%s' % k, 'num')
    band = v.req('admission_thresholds.shuf_band', 'list')
    if band is not None and not (len(band) == 2 and all(is_num(x) for x in band) and band[0] < band[1]):
        v.bad('admission_thresholds.shuf_band must be [lo, hi] with lo < hi')
    v.req('testbed', 'dict')
    for k in ('ceiling_grid', 'floor_grid'):
        grid = v.req('testbed.%s' % k, 'list')
        for i, e in enumerate(grid or []):
            if not isinstance(e, dict) or not isinstance(e.get('id'), str) or not e.get('id'):
                v.bad('testbed.%s[%d] must be {id, learner, ...}' % (k, i))
        if isinstance(grid, list) and not grid:
            v.bad('testbed.%s must not be empty' % k)
    v.req('files', 'dict')
    v.req('files.admit', 'str')
    v.req('files.params', 'str')
    for k in ('evaluator', 'generators', 'runner', 'native', 'shared', 'assets'):
        fv = v.req('files.%s' % k, 'list')
        if fv is not None:
            v.str_list('files.%s' % k, fv, nonempty=k in ('runner', 'native'))
    efs = v.req('endpoint_files_sha256', 'dict')
    for k, s in (efs or {}).items():
        if strip_sha(s) is None:
            v.bad('endpoint_files_sha256[%s] must be a sha256' % k)
    v.req('lineage', 'dict')
    par = v.req('lineage.parent', 'str', nullable=True)
    if isinstance(par, str) and not PARENT_RE.match(par.strip()):
        v.bad("lineage.parent %r must be '<dir>@<sha>' or null" % par)
    for k in ('changes_declared', 'priors_added'):
        lv = v.req('lineage.%s' % k, 'list')
        if lv is not None:
            v.str_list('lineage.%s' % k, lv)
    preds = v.req('predictions', 'list')
    pids: List[str] = []
    for i, p in enumerate(preds or []):
        if not isinstance(p, dict):
            v.bad('predictions[%d] must be an object' % i)
            continue
        if not isinstance(p.get('id'), str) or not p.get('id'):
            v.bad('predictions[%d].id must be a non-empty string' % i)
        elif p['id'] in pids:
            v.bad('prediction id %s is duplicated' % p['id'])
        else:
            pids.append(p['id'])
        if not isinstance(p.get('target'), str) or not p.get('target'):
            v.bad('predictions[%d].target must be a non-empty string' % i)
        for k in ('point', 'interval90', 'p_pass', 'basis_findings', 'base_rate_forecast'):
            if k not in p:
                v.bad('predictions[%d].%s is missing' % (i, k))
        if p.get('point') is not None and not is_num(p.get('point')):
            v.bad('predictions[%d].point must be a number or null' % i)
        if p.get('p_pass') is not None and not is_num(p.get('p_pass')):
            v.bad('predictions[%d].p_pass must be a number or null' % i)
        iv = p.get('interval90')
        if iv is not None and not (isinstance(iv, list) and len(iv) == 2 and all(is_num(x) for x in iv)):
            v.bad('predictions[%d].interval90 must be [lo, hi] numbers or null' % i)
        if not isinstance(p.get('basis_findings'), list):
            v.bad('predictions[%d].basis_findings must be a list' % i)
        if not isinstance(p.get('base_rate_forecast'), dict):
            v.bad('predictions[%d].base_rate_forecast must be {value, source}' % i)
    v.req('e3_cadence', 'dict')
    v.req('e3_cadence.waiver_user_decision', 'str', nullable=True)
    v.req('constraints_sha256', 'sha')
    ud = v.req('user_decisions', 'list')
    if ud is not None:
        v.str_list('user_decisions', ud)
    v.req('coprimary', 'list')
    for p in fill_paths(lock):
        v.bad('%s still holds __FILL__' % p)
    return v.problems


@check('LOCK-01')
def check_lock_01(ctx: common.Ctx) -> Result:
    """Schema; required fields non-null; no '__FILL__' values; enums valid."""
    problems = lock_schema_problems(ctx.lock, ctx.delivery_name)
    if problems:
        return Result('LOCK-01', FAIL, '%d schema problem(s): %s' % (len(problems), '; '.join(problems[:12])),
                      evidence=problems[:60])
    return Result('LOCK-01', PASS, 'ENDPOINT_LOCK.json matches minifly.endpoint_lock.v1 (no __FILL__ left)')


# --------------------------------------------------------------------------------------
# LOCK-02: contract block complete
# --------------------------------------------------------------------------------------

def required_roles(ctx: common.Ctx, level: str) -> Tuple[List[str], str]:
    table = ctx.kb.constraints().get('required_arms_by_level')
    if not isinstance(table, dict):
        raise MissingInput('CONSTRAINTS.required_arms_by_level missing')
    if level not in LEVELS:
        raise InvalidInput('claim_level_target %r is not one of %s' % (level, '|'.join(LEVELS)))
    for lv in reversed(LEVELS[:LEVELS.index(level) + 1]):
        roles = table.get(lv)
        if isinstance(roles, list):
            return [str(r) for r in roles], lv
    return [], 'none'


@check('LOCK-02')
def check_lock_02(ctx: common.Ctx) -> Result:
    lock = ctx.lock
    problems: List[str] = []
    level = lock.get('claim_level_target')
    if level not in LEVELS:
        problems.append('claim_level_target %r is not set to E0..E4' % (level,))
    if lock.get('intent') not in INTENTS:
        problems.append('intent %r is not set to diagnostic|promotion' % (lock.get('intent'),))
    es = lock.get('exposure_streams')
    if not isinstance(es, list) or not es:
        problems.append('exposure_streams are not declared')
    else:
        for i, s in enumerate(es):
            if not (isinstance(s, dict) and isinstance(s.get('path_glob'), str) and s.get('path_glob')
                    and isinstance(s.get('feedback_bearing'), bool)):
                problems.append('exposure_streams[%d] lacks path_glob/feedback_bearing' % i)
    arms = [a for a in (lock.get('arms') or []) if isinstance(a, dict)]
    by_id = {a.get('id'): a for a in arms}
    for k in ('causal_control_arm', 'positive_control_arm'):
        if lock.get(k) not in by_id:
            problems.append('%s %r is not an existing arm' % (k, lock.get(k)))
    cand = dig(lock, 'primary.candidate_arm')
    if cand not in by_id:
        problems.append('primary.candidate_arm %r is not an existing arm' % (cand,))
    elif by_id[cand].get('role') != 'candidate':
        problems.append('primary.candidate_arm %s has role %r, not candidate' % (cand, by_id[cand].get('role')))
    # Round-1 repair (verifier probe E2; SPEC 20.14): role 'candidate' belongs to primary.candidate_arm alone. A second
    # 'candidate' arm would be neither the tested arm nor a decision comparator, so relabelling a control 'candidate'
    # would take it out of the rule.
    extra = [a.get('id') for a in arms if a.get('role') == 'candidate' and a.get('id') != cand]
    if extra:
        problems.append('arm(s) %s have role candidate but are not primary.candidate_arm %s (only the tested arm may '
                        'have role candidate)' % (', '.join(str(x) for x in extra), cand))
    for c in dig(lock, 'primary.comparator_arms') or []:
        if c not in by_id:
            problems.append('comparator arm %r is not an existing arm' % (c,))
    basis = 'none'
    if level in LEVELS:
        roles, basis = required_roles(ctx, level)
        present = set(a.get('role') for a in arms)
        for r in roles:
            if r not in present:
                problems.append('claim level %s requires an arm with role %s (CONSTRAINTS.required_arms_by_level.%s)'
                                % (level, r, basis))
    for a in arms:
        if a.get('role') in ('lesion', 'no_prior_teaching'):
            zc = a.get('zero_channels')
            if not isinstance(zc, list) or not [z for z in zc if isinstance(z, str) and z and z != FILL]:
                problems.append('arm %s (role %s) needs non-empty zero_channels' % (a.get('id'), a.get('role')))
        if a.get('role') == 'no_prior_teaching':
            tw = a.get('teaching_withheld')
            if not isinstance(tw, list) or not [t for t in tw if isinstance(t, str) and t and t != FILL]:
                problems.append('arm %s (no_prior_teaching) must declare teaching_withheld' % a.get('id'))
    if level in ('E2', 'E3', 'E4'):
        u = lock.get('units')
        if not (isinstance(u, dict) and u.get('witness') in WITNESS_ENUM and isinstance(u.get('commutative'), bool)):
            problems.append('claim level %s requires a units block {witness, commutative}' % level)
    if problems:
        return Result('LOCK-02', FAIL, 'contract block incomplete: %s' % '; '.join(problems[:12]),
                      evidence=problems[:40])
    return Result('LOCK-02', PASS, 'contract block complete for %s (required roles from %s)' % (level, basis))


# --------------------------------------------------------------------------------------
# LOCK-03: objective gate or forced component
# --------------------------------------------------------------------------------------

def gate_type_of(lock: dict) -> Tuple[str, List[str]]:
    prim = lock.get('primary') if isinstance(lock.get('primary'), dict) else {}
    why: List[str] = []
    if prim.get('metric') not in OBJECTIVE_METRICS:
        why.append('primary.metric %r is not exact_own_eos|balanced_exact_own_eos' % (prim.get('metric'),))
    if prim.get('unconditional') is not True:
        why.append('primary.unconditional is %r, not true' % (prim.get('unconditional'),))
    outputs = []
    if 'output' in prim:
        outputs.append(('primary.output', prim.get('output')))
    outputs.append(('objective_distance.output', dig(lock, 'objective_distance.output')))
    for name, val in outputs:
        if val != OBJECTIVE_OUTPUT:
            why.append('%s is %r, not %s' % (name, val, OBJECTIVE_OUTPUT))
    return ('objective' if not why else 'component'), why


@check('LOCK-03')
def check_lock_03(ctx: common.Ctx) -> Result:
    lock = ctx.lock
    gt, why = gate_type_of(lock)
    data = {'gate_type': gt}
    if gt == 'objective':
        return Result('LOCK-03', PASS, 'gate_type objective (unconditional %s, own bytes and own EOS)'
                      % dig(lock, 'primary.metric'), data=data)
    problems = []
    level = lock.get('claim_level_target')
    if level in ('E2', 'E3', 'E4'):
        problems.append('claim_level_target %s requires an objective gate' % level)
    for where, val in (('ENDPOINT_LOCK.gate_type', lock.get('gate_type')),
                       ('ENDPOINT_LOCK.primary.gate_type', dig(lock, 'primary.gate_type'))):
        if val == 'objective':
            problems.append('%s says objective' % where)
    cp = ctx.path('CLAIM.json')
    if cp.is_file():
        claim = ctx.read_json('CLAIM.json')
        if isinstance(claim, dict) and claim.get('gate_type') == 'objective':
            problems.append('CLAIM.json gate_type says objective')
    detail = 'forced to component: %s' % '; '.join(why)
    if problems:
        return Result('LOCK-03', FAIL, '%s; but %s' % (detail, '; '.join(problems)), evidence=why + problems, data=data)
    return Result('LOCK-03', PASS, '%s (claim level %s does not need an objective gate)' % (detail, level), data=data)


# --------------------------------------------------------------------------------------
# LOCK-04: thresholds meet CONSTRAINTS floors; OLD guards
# --------------------------------------------------------------------------------------

EPS = 1e-12


def old_pool_exists(lock: dict) -> bool:
    om = dig(lock, 'split.old_manifest')
    nip = dig(lock, 'primary.n_items_per_pool') or {}
    return bool(om) or (isinstance(nip, dict) and 'OLD' in nip) or dig(lock, 'primary.pool') == 'OLD'


@check('LOCK-04')
def check_lock_04(ctx: common.Ctx) -> Result:
    lock = ctx.lock
    fl = floors(ctx)
    problems: List[str] = []
    checked: List[str] = []

    def need(name: str, val: Any, floor_key: str, higher_is_stricter: bool = True, offset: float = 0.0) -> None:
        fv = fl.get(floor_key)
        if not is_num(fv):
            problems.append('CONSTRAINTS.threshold_floors.%s missing or not numeric' % floor_key)
            return
        if not is_num(val):
            problems.append('%s is %r, not a number' % (name, val))
            return
        ok = (val - offset >= fv - EPS) if higher_is_stricter else (val <= fv + EPS)
        checked.append('%s=%s' % (name, val))
        if not ok:
            problems.append('%s=%s %s floor %s=%s' % (name, val, 'below' if higher_is_stricter else 'above',
                                                      floor_key, fv))

    prim = lock.get('primary') if isinstance(lock.get('primary'), dict) else {}
    need('primary.margin_abs', prim.get('margin_abs'), 'margin_abs')
    chance = prim.get('chance')
    if is_num(chance):
        need('primary.absolute_floor - chance', prim.get('absolute_floor'), 'absolute_floor_above_chance',
             offset=chance)
    else:
        problems.append('primary.chance is %r, not a number' % (chance,))
    need('primary.per_world_floor', prim.get('per_world_floor'), 'per_world_floor')
    role = dig(lock, 'worlds.official_role_intended')
    if role != 'PILOT':
        need('primary.min_worlds', prim.get('min_worlds'), 'min_worlds_nonpilot')
    agg = prim.get('aggregator') if isinstance(prim.get('aggregator'), dict) else {}
    if agg.get('type') == 'paired_lcb':
        need('primary.aggregator.alpha', agg.get('alpha'), 'alpha_max', higher_is_stricter=False)
    at = lock.get('admission_thresholds') if isinstance(lock.get('admission_thresholds'), dict) else {}
    for k in ('input_contrast_min', 'prompt_id_margin_min', 'taught_acq_margin_min'):
        need('admission_thresholds.%s' % k, at.get(k), k)
    band, fband = at.get('shuf_band'), fl.get('shuf_band')
    if not (isinstance(fband, list) and len(fband) == 2 and all(is_num(x) for x in fband)):
        problems.append('CONSTRAINTS.threshold_floors.shuf_band missing')
    elif not (isinstance(band, list) and len(band) == 2 and all(is_num(x) for x in band)):
        problems.append('admission_thresholds.shuf_band is %r, not [lo, hi]' % (band,))
    else:
        checked.append('shuf_band=%s' % band)
        if band[0] < fband[0] - EPS or band[1] > fband[1] + EPS:
            problems.append('admission_thresholds.shuf_band %s is wider than the floor band %s' % (band, fband))
    guards = [g for g in (lock.get('guards') or []) if isinstance(g, dict)]
    drops = [g for g in guards if g.get('kind') == 'old_drop']
    acqs = [g for g in guards if g.get('kind') == 'old_acq']
    for g in drops:
        need('guards.%s.max_drop' % g.get('id'), g.get('max_drop'), 'old_max_drop', higher_is_stricter=False)
    for g in acqs:
        need('guards.%s.margin' % g.get('id'), g.get('margin'), 'old_acq_margin')
    if old_pool_exists(lock):
        if not any(g.get('pool') == 'OLD' for g in drops):
            problems.append('an OLD pool exists but no old_drop guard on pool OLD')
        if not any(g.get('pool') == 'OLD' for g in acqs):
            problems.append('an OLD pool exists but no old_acq guard on pool OLD')
    if problems:
        return Result('LOCK-04', FAIL, 'thresholds below CONSTRAINTS floors: %s' % '; '.join(problems[:12]),
                      evidence=problems[:40])
    return Result('LOCK-04', PASS, 'thresholds meet the CONSTRAINTS floors (%s)' % ', '.join(checked[:14]))


# --------------------------------------------------------------------------------------
# LOCK-05: SOURCE_LOCK equals a fresh lock build; closure complete; no collisions
# --------------------------------------------------------------------------------------

def _norm_closure(d: Any) -> Dict[str, Optional[str]]:
    if not isinstance(d, dict):
        return {}
    return {common.norm_rel(k): strip_sha(v) for k, v in d.items()}


@check('LOCK-05')
def check_lock_05(ctx: common.Ctx) -> Result:
    sl = ctx.source_lock
    b = build_closure(ctx)
    problems: List[str] = list(b['problems'])
    if sl.get('schema') != common.SCHEMA_SOURCE_LOCK:
        problems.append('SOURCE_LOCK schema is %r' % (sl.get('schema'),))
    if sl.get('delivery') != ctx.delivery_name:
        problems.append('SOURCE_LOCK delivery %r is not %r' % (sl.get('delivery'), ctx.delivery_name))
    if strip_sha(sl.get('endpoint_lock_sha256')) != ctx.endpoint_lock_sha256:
        problems.append('SOURCE_LOCK endpoint_lock_sha256 %s is stale (ENDPOINT_LOCK is %s): run lock build'
                        % (str(sl.get('endpoint_lock_sha256'))[:12], (ctx.endpoint_lock_sha256 or '')[:12]))
    if not isinstance(sl.get('closure_sha256'), dict):
        problems.append('SOURCE_LOCK closure_sha256 is not an object')
    have = _norm_closure(sl.get('closure_sha256'))
    want = dict(b['closure_sha256'])
    for p in sorted(set(want) - set(have)):
        problems.append('closure file %s missing from SOURCE_LOCK.closure_sha256' % p)
    for p in sorted(set(have) - set(want)):
        problems.append('SOURCE_LOCK.closure_sha256 lists %s, which is not in the fresh closure' % p)
    for p in sorted(set(have) & set(want)):
        if have[p] != want[p]:
            problems.append('%s sha256 %s differs from SOURCE_LOCK %s' % (p, want[p][:12], str(have[p])[:12]))
    if 'native_closure_sha256' in sl and _norm_closure(sl.get('native_closure_sha256')) != b['native_closure_sha256']:
        problems.append('SOURCE_LOCK.native_closure_sha256 differs from the fresh native closure')
    for k in ('native_closure_sha', 'runner_closure_sha', 'params_sha256'):
        if strip_sha(sl.get(k)) != b[k]:
            problems.append('SOURCE_LOCK.%s %s differs from a fresh build %s' % (k, str(sl.get(k))[:12],
                                                                                 str(b[k])[:12]))
    dyn = sl.get('dynamic_imports')
    if not isinstance(dyn, list):
        problems.append('SOURCE_LOCK.dynamic_imports must be a list')
    else:
        declared = set(str(x) for x in dyn)
        for s in b['dynamic_imports']:
            if s not in declared:
                problems.append('non-literal import not listed in dynamic_imports: %s' % s)
        for s in sorted(declared - set(b['dynamic_imports'])):
            problems.append('dynamic_imports lists %s, which a fresh build does not find' % s)
    if problems:
        return Result('LOCK-05', FAIL, 'SOURCE_LOCK is not a fresh, complete closure: %s' % '; '.join(problems[:10]),
                      evidence=problems[:60])
    return Result('LOCK-05', PASS, 'SOURCE_LOCK equals a fresh lock build (%d closure files, %d dynamic imports '
                  'listed)' % (len(want), len(b['dynamic_imports'])))


# --------------------------------------------------------------------------------------
# LOCK-06: timing
# --------------------------------------------------------------------------------------

@check('LOCK-06')
def check_lock_06(ctx: common.Ctx) -> Result:
    lrel = lock_rel(ctx)
    srel = ctx.repo_rel_of('SOURCE_LOCK.json')
    rpre = delivery_prefix(ctx) + 'results'
    problems: List[str] = []
    c_lock = ctx.git.first_commit_adding(lrel)
    if c_lock is None:
        problems.append('ENDPOINT_LOCK.json is not committed')
    c_res = first_commit_adding_under(ctx, [rpre])
    if c_res is not None and c_lock is not None and not strictly_before(ctx, c_lock, c_res):
        problems.append('the first commit adding results/** (%s) does not follow the lock commit %s'
                        % (c_res.sha[:12], c_lock.sha[:12]))
    if c_res is not None and c_lock is None:
        problems.append('results/** committed at %s with no committed ENDPOINT_LOCK' % c_res.sha[:12])
    # SPEC 6.2: the held set is fixed in the lock (split.held_set_sha256) before any stream exists
    split = ctx.lock.get('split') if isinstance(ctx.lock.get('split'), dict) else {}
    tdir = split.get('taught_manifest') if isinstance(split.get('taught_manifest'), str) and split.get('taught_manifest') \
        else 'manifests/TAUGHT'
    c_str = first_commit_adding_under(ctx, [delivery_prefix(ctx) + common.norm_rel(tdir)])
    if c_str is not None and (c_lock is None or not strictly_before(ctx, c_lock, c_str)):
        problems.append('the first commit adding a TAUGHT stream (%s) does not follow the lock commit %s: the held '
                        'set is fixed in the lock before any stream exists' % (c_str.sha[:12],
                                                                              c_lock.sha[:12] if c_lock else 'none'))
    shared = []
    for c in ctx.git.log([rpre]):
        res_paths = [p for p in c.paths if common.glob_match(p, rpre)]
        res_paths = [p for p in res_paths if not is_empty_keep(p, ctx.git.file_at(c.sha, p))]
        if res_paths and any(p in (lrel, srel) for p in c.paths):
            shared.append(c.sha[:12])
    if shared:
        problems.append('lock and results share commit(s) %s' % ', '.join(shared))
    # native_closure_sha in rows -> the SOURCE_LOCK commit with that closure precedes the rows
    shas: Set[str] = set()
    missing_sha: List[str] = []
    for label, row in current_outcome_rows(ctx):
        if common.is_evaluator_arm(row.get('arm')):
            continue
        s = row.get('native_closure_sha')
        if strip_sha(s):
            shas.add(strip_sha(s))
        elif len(missing_sha) < 5:
            missing_sha.append('%s (arm %r)' % (label, row.get('arm')))
    if missing_sha:
        problems.append('learner-arm rows without native_closure_sha cannot be timed: %s' % ', '.join(missing_sha))
    first_rows: Dict[str, common.Commit] = {}
    if shas:
        for c in ctx.git.log([rpre]):
            pending = [s for s in shas if s not in first_rows]
            if not pending:
                break
            for p in c.paths:
                if not p.startswith(rpre + '/'):
                    continue
                drel = p[len(delivery_prefix(ctx)):]
                name = drel.rsplit('/', 1)[-1]
                if not (is_outputs_name(name) or is_receipt_rel(drel) or common.RESULT_BUNDLE_RE.match(name)):
                    continue
                data = ctx.git.file_at(c.sha, p)
                if data is None:
                    continue
                rows = outcome_rows_from_bytes(drel, data)
                found = set(strip_sha(r.get('native_closure_sha')) for r in rows
                            if not common.is_evaluator_arm(r.get('arm')))
                for s in pending:
                    if s in found and s not in first_rows:
                        first_rows[s] = c
    for s in sorted(shas):
        c_sl = content_commit(ctx, srel, lambda o, s=s: isinstance(o, dict) and strip_sha(o.get('native_closure_sha')) == s)
        if c_sl is None:
            problems.append('no committed SOURCE_LOCK has native_closure_sha %s found in rows' % s[:12])
            continue
        c_rows = first_rows.get(s)
        if c_rows is not None and not strictly_before(ctx, c_sl, c_rows):
            problems.append('rows with native_closure_sha %s were committed at %s, not after its SOURCE_LOCK commit %s'
                            % (s[:12], c_rows.sha[:12], c_sl.sha[:12]))
    if problems:
        return Result('LOCK-06', FAIL, 'lock timing: %s' % '; '.join(problems[:8]), evidence=problems[:40])
    return Result('LOCK-06', PASS, 'lock committed at %s before results (%s); %d native closure(s) locked before '
                  'their rows' % (c_lock.sha[:12] if c_lock else '-', c_res.sha[:12] if c_res else 'none yet',
                                  len(shas)))


# --------------------------------------------------------------------------------------
# LOCK-07: native purity over the native and runner closure
# --------------------------------------------------------------------------------------

def word_parts(name: str) -> List[str]:
    s = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', name)
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', s)
    return [p.lower() for p in re.split(r'[^A-Za-z0-9]+', s) if p]


def _identifiers(tree: ast.AST) -> Iterable[Tuple[str, int]]:
    for node in ast.walk(tree):
        ln = getattr(node, 'lineno', 0)
        if isinstance(node, ast.Name):
            yield node.id, ln
        elif isinstance(node, ast.Attribute):
            yield node.attr, ln
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield node.name, ln
        elif isinstance(node, ast.arg):
            yield node.arg, ln
        elif isinstance(node, ast.keyword) and node.arg:
            yield node.arg, getattr(node.value, 'lineno', 0)
        elif isinstance(node, ast.alias):
            for part in node.name.split('.'):
                yield part, ln
            if node.asname:
                yield node.asname, ln
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = getattr(node, 'module', None)
            if mod:
                for part in mod.split('.'):
                    yield part, ln


def _literal_value(s: str) -> Any:
    try:
        return ast.literal_eval(s)
    except Exception:  # noqa: BLE001
        return s


def forbidden_hits(info: _ModInfo, forb: dict) -> List[str]:
    tree = info.tree
    if tree is None:
        return []
    al = info.aliases
    hits: List[str] = []
    imports = [str(x) for x in (forb.get('imports') or [])]
    calls = [str(x) for x in (forb.get('calls') or [])]
    idents = [str(x) for x in (forb.get('identifiers') or [])]

    def mod_hit(full: str) -> Optional[str]:
        for f in imports:
            if full == f or full.startswith(f + '.'):
                return f
        return None

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                f = mod_hit(a.name)
                if f:
                    hits.append('%s:%d: import %s (forbidden %s)' % (info.rel, node.lineno, a.name, f))
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            for full in [node.module] + ['%s.%s' % (node.module, a.name) for a in node.names]:
                f = mod_hit(full)
                if f:
                    hits.append('%s:%d: from-import %s (forbidden %s)' % (info.rel, node.lineno, full, f))
                    break
        elif isinstance(node, (ast.Attribute, ast.Name)) and isinstance(getattr(node, 'ctx', None), ast.Load):
            d = dotted_name(node, al)
            if d and '.' in d:
                f = mod_hit(d)
                if f:
                    hits.append('%s:%d: uses %s (forbidden %s)' % (info.rel, node.lineno, d, f))
        if isinstance(node, ast.Call):
            d = dotted_name(node.func, al)
            last = node.func.attr if isinstance(node.func, ast.Attribute) else (
                node.func.id if isinstance(node.func, ast.Name) else None)
            for c in calls:
                name = c[:-1] if c.endswith('(') else c
                if name.startswith('.'):
                    if isinstance(node.func, ast.Attribute) and node.func.attr == name[1:]:
                        hits.append('%s:%d: call %s (forbidden %s)' % (info.rel, node.lineno, d or '?.' + name[1:], c))
                elif '.' in name:
                    if d == name:
                        hits.append('%s:%d: call %s (forbidden %s)' % (info.rel, node.lineno, d, c))
                elif last == name:
                    hits.append('%s:%d: call %s (forbidden %s)' % (info.rel, node.lineno, d or name, c))
    plain = [i for i in idents if '=' not in i]
    assigns = [i for i in idents if '=' in i]
    for ident, ln in _identifiers(tree):
        parts = word_parts(ident)
        for f in plain:
            fl = f.lower()
            if fl in parts or (fl + 's') in parts:
                hits.append('%s:%d: identifier %s (forbidden %s)' % (info.rel, ln, ident, f))
                break
    for spec in assigns:
        key, val = spec.split('=', 1)
        key, lit = key.strip(), _literal_value(val.strip())
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == key and isinstance(node.value, ast.Constant) \
                    and node.value.value == lit and type(node.value.value) is type(lit):
                hits.append('%s:%d: %s (forbidden %s)' % (info.rel, getattr(node.value, 'lineno', 0), spec, spec))
            elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
                    and node.value.value == lit and type(node.value.value) is type(lit):
                for t in node.targets:
                    n = t.attr if isinstance(t, ast.Attribute) else (t.id if isinstance(t, ast.Name) else None)
                    if n == key:
                        hits.append('%s:%d: %s = %r (forbidden %s)' % (info.rel, node.lineno, n, lit, spec))
    return sorted(set(hits))


def _contains_call(node: ast.AST, pred: Callable[[ast.Call], bool]) -> bool:
    return any(isinstance(n, ast.Call) and pred(n) for n in ast.walk(node))


def heuristic_hits(info: _ModInfo) -> List[str]:
    """supplied_arithmetic_heuristics (CONSTRAINTS): int() over decoded operand bytes, carry or digit
    tables, '+' applied to parsed operands. Heuristic: REVIEW_REQUIRED, never proof."""
    tree = info.tree
    if tree is None:
        return []
    al = info.aliases
    hits: List[str] = []

    def is_int_call(n: ast.AST) -> bool:
        return isinstance(n, ast.Call) and dotted_name(n.func, al) in ('int', 'builtins.int')

    def decodes(n: ast.Call) -> bool:
        f = n.func
        return (isinstance(f, ast.Attribute) and f.attr == 'decode') or dotted_name(f, al) in ('chr', 'str')

    int_names: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and is_int_call(node.value):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    int_names.add(t.id)
    for node in ast.walk(tree):
        ln = getattr(node, 'lineno', 0)
        if is_int_call(node) and any(_contains_call(a, decodes) for a in node.args):
            hits.append('%s:%d: int() over decoded bytes: %s' % (info.rel, ln, ast.unparse(node)[:80]))
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
            left_ord = isinstance(node.left, ast.Call) and dotted_name(node.left.func, al) == 'ord'
            right = node.right
            right_digit = (isinstance(right, ast.Constant) and right.value == 48) or (
                isinstance(right, ast.Call) and dotted_name(right.func, al) == 'ord' and right.args
                and isinstance(right.args[0], ast.Constant) and right.args[0].value in ('0', b'0'))
            if left_ord and right_digit:
                hits.append('%s:%d: ord() digit decoding: %s' % (info.rel, ln, ast.unparse(node)[:80]))
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            sides = [node.left, node.right]
            if all(is_int_call(s) or (isinstance(s, ast.Name) and s.id in int_names) for s in sides):
                hits.append("%s:%d: '+' applied to parsed operands: %s" % (info.rel, ln, ast.unparse(node)[:80]))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in (
                'split', 'partition', 'rpartition') and node.args:
            a0 = node.args[0]
            if isinstance(a0, ast.Constant) and a0.value in ('+', b'+'):
                hits.append("%s:%d: prompt split on '+': %s" % (info.rel, ln, ast.unparse(node)[:80]))
        names: List[Tuple[str, int]] = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append((node.name, ln))
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.append((t.id, ln))
                elif isinstance(t, ast.Attribute):
                    names.append((t.attr, ln))
        for n, l2 in names:
            parts = word_parts(n)
            if any(p in ('carry', 'carries', 'digit', 'digits') for p in parts):
                hits.append('%s:%d: carry/digit table or helper %s' % (info.rel, l2, n))
    return sorted(set(hits))


def review_addresses(review: Optional[dict], fm_ids: Sequence[str]) -> Optional[str]:
    if not isinstance(review, dict):
        return None
    for f in review.get('findings') or []:
        if not isinstance(f, dict):
            continue
        fms = f.get('fm_ids') or []
        st = f.get('status')
        if any(x in fms for x in fm_ids) and isinstance(st, str) and st and st.upper() != 'OPEN':
            return str(f.get('id') or '?')
    return None


@check('LOCK-07')
def check_lock_07(ctx: common.Ctx) -> Result:
    lock = ctx.lock
    cons = ctx.kb.constraints()
    forb = cons.get('native_forbidden')
    if not isinstance(forb, dict):
        raise MissingInput('CONSTRAINTS.native_forbidden missing')
    b = build_closure(ctx)
    scope = set(b['native_closure_sha256']) | set(b['runner_closure_sha256'])
    evaluator, _missing = expand_entries(ctx, files_entries(lock, 'evaluator') + files_entries(lock, 'admit'))
    evaluator_set = set(evaluator)
    problems: List[str] = []
    reach = sorted(scope & evaluator_set)
    for r in reach:
        problems.append('evaluator module %s is reachable from the native/runner closure' % r)
    scanned = sorted(scope - evaluator_set)
    hard: List[str] = []
    heur: List[str] = []
    for rel in scanned:
        info = b['infos'].get(rel)
        if info is None:
            continue
        if info.error:
            problems.append(info.error)
            continue
        hard.extend(forbidden_hits(info, forb))
        heur.extend(heuristic_hits(info))
    problems.extend(hard)
    data = {'scanned': scanned, 'heuristic_hits': heur[:50]}
    if problems:
        return Result('LOCK-07', FAIL, 'native purity: %s' % '; '.join(problems[:8]), evidence=problems[:60], data=data)
    if heur:
        tier = current_tier(ctx)
        if tier < 2:
            return Result('LOCK-07', WARN, 'REVIEW_REQUIRED (tier %d): %d supplied-arithmetic heuristic hit(s): %s'
                          % (tier, len(heur), '; '.join(heur[:6])), evidence=heur[:60], data=data)
        fid = review_addresses(read_review(ctx), ('FM-12', 'FM-23'))
        if fid:
            return Result('LOCK-07', PASS, '%d heuristic hit(s) addressed by REVIEW finding %s (FM-12/FM-23)'
                          % (len(heur), fid), evidence=heur[:60], data=data)
        return Result('LOCK-07', FAIL, 'tier 2: %d supplied-arithmetic heuristic hit(s) not addressed by a REVIEW '
                      'finding with FM-12/FM-23 and status other than OPEN: %s' % (len(heur), '; '.join(heur[:6])),
                      evidence=heur[:60], data=data)
    return Result('LOCK-07', PASS, 'native/runner closure (%d files) free of forbidden imports, calls and identifiers; '
                  'no evaluator module reachable' % len(scanned), data=data)


# --------------------------------------------------------------------------------------
# LOCK-08: literal integrity booleans
# --------------------------------------------------------------------------------------

def integrity_name(n: str) -> bool:
    return bool(INTEGRITY_SUFFIX_RE.search(n) or INTEGRITY_PREFIX_RE.match(n))


def literal_flag(node: Optional[ast.AST]) -> Optional[str]:
    if isinstance(node, ast.Constant):
        v = node.value
        if isinstance(v, bool):
            return repr(v)
        if isinstance(v, (int, float)) and v == 0:
            return repr(v)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)) and isinstance(node.operand,
                                                                                                    ast.Constant):
        v = node.operand.value
        if isinstance(v, (int, float)) and not isinstance(v, bool) and v == 0:
            return '-%r' % v if isinstance(node.op, ast.USub) else repr(v)
    return None


def _target_name(t: ast.AST) -> Optional[str]:
    if isinstance(t, ast.Name):
        return t.id
    if isinstance(t, ast.Attribute):
        return t.attr
    if isinstance(t, ast.Subscript):
        sl = t.slice
        sl = getattr(sl, 'value', sl) if not isinstance(sl, ast.Constant) else sl
        if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
            return sl.value
    return None


def literal_integrity_hits(rel: str, tree: ast.AST) -> List[str]:
    hits: List[str] = []

    def hit(name: Optional[str], value: Optional[ast.AST], ln: int) -> None:
        lit = literal_flag(value)
        if name and lit is not None and integrity_name(name):
            hits.append('%s:%d: %s = %s' % (rel, ln, name, lit))

    def assign_pair(t: ast.AST, v: ast.AST, ln: int) -> None:
        if isinstance(t, (ast.Tuple, ast.List)) and isinstance(v, (ast.Tuple, ast.List)) and len(t.elts) == len(v.elts):
            for te, ve in zip(t.elts, v.elts):
                assign_pair(te, ve, ln)
            return
        hit(_target_name(t), v, ln)

    for node in ast.walk(tree):
        ln = getattr(node, 'lineno', 0)
        if isinstance(node, ast.Assign):
            for t in node.targets:
                assign_pair(t, node.value, ln)
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            hit(_target_name(node.target), node.value, ln)
        elif isinstance(node, ast.NamedExpr):
            hit(_target_name(node.target), node.value, ln)
        elif isinstance(node, ast.Dict):
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and isinstance(k.value, str):
                    hit(k.value, v, getattr(k, 'lineno', ln))
        elif isinstance(node, ast.Call):
            f = node.func
            fname = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else None)
            if fname in ('dict', 'update'):
                for kw in node.keywords:
                    if kw.arg:
                        hit(kw.arg, kw.value, ln)
            if fname in ('setattr', 'setdefault', '__setitem__') and len(node.args) >= 2:
                ki = 1 if fname == 'setattr' else 0
                if len(node.args) > ki + 1 or (fname != 'setattr' and len(node.args) >= 2):
                    k = node.args[ki]
                    v = node.args[ki + 1] if len(node.args) > ki + 1 else None
                    if isinstance(k, ast.Constant) and isinstance(k.value, str):
                        hit(k.value, v, ln)
    return sorted(set(hits))


@check('LOCK-08')
def check_lock_08(ctx: common.Ctx) -> Result:
    hits: List[str] = []
    errors: List[str] = []
    n = 0
    for rel in ctx.delivery_files():
        if not rel.endswith('.py'):
            continue
        n += 1
        try:
            tree = ast.parse(ctx.path(rel).read_bytes(), filename=rel)
        except (SyntaxError, ValueError) as e:
            errors.append('%s does not parse (%s): it cannot be checked' % (rel, e))
            continue
        hits.extend(literal_integrity_hits(rel, tree))
    problems = errors + hits
    if problems:
        return Result('LOCK-08', FAIL, 'literal integrity values in delivery code (the checker computes them): %s'
                      % '; '.join(problems[:8]), evidence=problems[:60])
    return Result('LOCK-08', PASS, '%d delivery .py file(s): no literal True/False/0 assigned to an integrity name' % n)


# --------------------------------------------------------------------------------------
# LOCK-09: E3 cadence
# --------------------------------------------------------------------------------------

def e3_opening(lock: dict) -> Tuple[bool, List[str]]:
    why: List[str] = []
    if dig(lock, 'primary.pool') != 'HELD':
        why.append('primary.pool is %r, not HELD' % dig(lock, 'primary.pool'))
    strata = dig(lock, 'primary.strata_counted')
    if not (isinstance(strata, list) and strata and all(s in E3_STRATA for s in strata)):
        why.append('strata_counted %r not within S2..S4' % (strata,))
    if lock.get('claim_level_target') != 'E3':
        why.append('claim_level_target %r is not E3' % lock.get('claim_level_target'))
    if not common._official_planned(lock):
        why.append('no official run planned')
    if dig(lock, 'worlds.official_role_intended') not in ('MAIN', 'SEALED_CONFIRM'):
        why.append('official_role_intended %r is not MAIN|SEALED_CONFIRM' % dig(lock, 'worlds.official_role_intended'))
    return (not why), why


@check('LOCK-09')
def check_lock_09(ctx: common.Ctx) -> Result:
    lock = ctx.lock
    try:
        kbmod = importlib.import_module('kb')
    except Exception as e:  # noqa: BLE001
        raise MissingInput('kb.deliveries_since_last_e3 unavailable (module kb: %s)' % common.summarize_exception(e, 1))
    fn = getattr(kbmod, 'deliveries_since_last_e3', None)
    if not callable(fn):
        raise MissingInput('kb.deliveries_since_last_e3 unavailable (function missing)')
    d = fn(ctx.kb)
    cmax = ctx.kb.constraints().get('e3_cadence_max')
    if not is_num(cmax):
        raise MissingInput('CONSTRAINTS.e3_cadence_max missing')
    if not is_num(d):
        raise InvalidInput('kb.deliveries_since_last_e3 returned %r' % (d,))
    data = {'deliveries_since_last_e3': d, 'e3_cadence_max': cmax}
    if d < cmax:
        return Result('LOCK-09', PASS, 'E3 cadence not due: %s deliveries since the last held-E3 opening (max %s)'
                      % (d, cmax), data=data)
    opening, why = e3_opening(lock)
    if opening:
        return Result('LOCK-09', PASS, 'E3 cadence due (%s >= %s) and this lock opens held E3' % (d, cmax), data=data)
    cited = []
    w = dig(lock, 'e3_cadence.waiver_user_decision')
    if isinstance(w, str):
        cited.append(w)
    cited.extend(lock_ud_ids(lock))
    scopes = ['e3_waiver:%s' % n for n in delivery_ids(ctx)]
    uid = ud_with_scope(ctx, cited, scopes) if cited else None
    if uid:
        return Result('LOCK-09', PASS, 'E3 cadence due (%s >= %s); not E3-opening (%s); owner waiver %s'
                      % (d, cmax, '; '.join(why), uid), data=data)
    return Result('LOCK-09', FAIL, 'E3 cadence due: %s deliveries since the last held-E3 opening (max %s); the lock is '
                  'not E3-opening (%s) and cites no UD with scope %s' % (d, cmax, '; '.join(why), scopes[0]),
                  evidence=why, data=data)


# --------------------------------------------------------------------------------------
# LOCK-10: constraints_sha256 and governed fields
# --------------------------------------------------------------------------------------

@check('LOCK-10')
def check_lock_10(ctx: common.Ctx) -> Result:
    lock = ctx.lock
    cons = ctx.kb.constraints()
    problems: List[str] = []
    want = ctx.kb.constraints_sha256()
    have = strip_sha(lock.get('constraints_sha256'))
    if have != want:
        problems.append('constraints_sha256 %s is not the current KB/CONSTRAINTS.json %s'
                        % (str(lock.get('constraints_sha256'))[:12], want[:12]))
    cited = lock_ud_ids(lock)
    deviations: List[Tuple[str, str]] = []
    for block, ckey in GOVERNED_BLOCKS:
        if block in lock:
            if common.canonical_json(lock.get(block)) != common.canonical_json(cons.get(ckey)):
                deviations.append((block.split('_')[0], 'lock.%s differs from CONSTRAINTS.%s' % (block, ckey)))
    fb = dig(lock, 'objective_distance.feedback')
    if fb is not None and fb != 'gradual_warmup':
        deviations.append(('warmup', 'objective_distance.feedback %r is not gradual_warmup (CONSTRAINTS.warmup %s)'
                           % (fb, common.canonical_json(cons.get('warmup')).decode('utf-8'))))
    covered = []
    for field, why in deviations:
        scopes = ['governed:%s:%s' % (field, n) for n in delivery_ids(ctx)]
        uid = ud_with_scope(ctx, cited, scopes) if cited else None
        if uid:
            covered.append('%s (UD %s)' % (why, uid))
        else:
            problems.append('%s and no cited UD has scope %s' % (why, scopes[0]))
    if problems:
        return Result('LOCK-10', FAIL, '; '.join(problems[:8]), evidence=problems)
    return Result('LOCK-10', PASS, 'constraints_sha256 %s current; governed fields %s' % (
        want[:12], ('deviate only under UD: %s' % '; '.join(covered)) if covered else 'equal the constraints'))


# --------------------------------------------------------------------------------------
# LOCK-11: predictions
# --------------------------------------------------------------------------------------

def _prediction_ok(p: dict) -> Tuple[bool, List[str]]:
    why: List[str] = []
    has_p = p.get('p_pass') is not None
    iv = p.get('interval90')
    has_point = p.get('point') is not None and iv is not None
    if has_p:
        if not is_num(p['p_pass']) or not (0.02 - EPS <= p['p_pass'] <= 0.98 + EPS):
            why.append('p_pass %r is not within [0.02, 0.98]' % (p['p_pass'],))
    if p.get('point') is not None or iv is not None:
        if not (is_num(p.get('point')) and isinstance(iv, list) and len(iv) == 2 and all(is_num(x) for x in iv)):
            why.append('point plus interval90 must both be numbers')
        elif not (iv[0] <= p['point'] <= iv[1]):
            why.append('point %s lies outside interval90 %s' % (p['point'], iv))
    if not has_p and not has_point:
        why.append('needs p_pass and/or point plus interval90')
    return (not why), why


@check('LOCK-11')
def check_lock_11(ctx: common.Ctx) -> Result:
    lock = ctx.lock
    preds = lock.get('predictions')
    problems: List[str] = []
    if not isinstance(preds, list) or not preds:
        return Result('LOCK-11', FAIL, 'no predictions registered (one on the primary and one per guard)')
    guard_ids = [g.get('id') for g in (lock.get('guards') or []) if isinstance(g, dict) and g.get('id')]
    targets_ok: Set[str] = set()
    finding_ids: Optional[Set[str]] = None
    for i, p in enumerate(preds):
        if not isinstance(p, dict):
            problems.append('predictions[%d] is not an object' % i)
            continue
        pid = p.get('id') or ('#%d' % i)
        t = p.get('target')
        if t != 'primary' and not (isinstance(t, str) and t.startswith('guard:') and t[6:] in guard_ids):
            problems.append('%s target %r is neither primary nor guard:<id of a lock guard>' % (pid, t))
        ok, why = _prediction_ok(p)
        problems.extend('%s: %s' % (pid, w) for w in why)
        bf = p.get('base_rate_forecast')
        if not (isinstance(bf, dict) and is_num(bf.get('value')) and 0 <= bf['value'] <= 1
                and isinstance(bf.get('source'), str) and bf['source'].strip() and FILL not in bf['source']):
            problems.append('%s: base_rate_forecast {value in [0,1], source} missing' % pid)
            ok = False
        basis = p.get('basis_findings')
        if not isinstance(basis, list) or not basis:
            problems.append("%s: basis_findings must list F-ids or ['none']" % pid)
            ok = False
        elif 'none' in basis:
            if len(basis) != 1:
                problems.append("%s: basis_findings mixes 'none' with ids" % pid)
                ok = False
        else:
            if finding_ids is None:
                finding_ids = set(str(r.get('id')) for r in ctx.kb.rows('FINDINGS') if r.get('id'))
            absent = [b for b in basis if str(b) not in finding_ids]
            if absent:
                problems.append('%s: basis_findings %s do not exist in FINDINGS' % (pid, ', '.join(map(str, absent))))
                ok = False
        if ok and isinstance(t, str):
            targets_ok.add(t)
    if 'primary' not in targets_ok:
        problems.append('no valid prediction on the primary')
    for g in guard_ids:
        if 'guard:%s' % g not in targets_ok:
            problems.append('no valid prediction for guard %s' % g)
    if problems:
        return Result('LOCK-11', FAIL, 'predictions: %s' % '; '.join(problems[:10]), evidence=problems[:40])
    return Result('LOCK-11', PASS, '%d prediction(s): primary and %d guard(s) covered' % (len(preds), len(guard_ids)))


# --------------------------------------------------------------------------------------
# LOCK-12: analytic input SNR
# --------------------------------------------------------------------------------------

@check('LOCK-12')
def check_lock_12(ctx: common.Ctx) -> Result:
    lock = ctx.lock
    lin = lock.get('lineage') if isinstance(lock.get('lineage'), dict) else {}
    parent = lin.get('parent')
    changes = [str(c) for c in (lin.get('changes_declared') or [])]
    trig = [c for c in changes if c in SNR_CHANGES]
    if parent and not trig:
        return Result('LOCK-12', NOT_APPLICABLE, 'lineage parent %s declared and no frontend/read/init change' % parent)
    why = 'no lineage parent' if not parent else 'changes_declared includes %s' % ','.join(trig)
    fl = floors(ctx)
    fmin = fl.get('analytic_input_snr_min')
    if not is_num(fmin):
        raise MissingInput('CONSTRAINTS.threshold_floors.analytic_input_snr_min missing')
    snr = dig(lock, 'admission_thresholds.analytic_input_snr')
    if snr == 'not_derivable':
        return Result('LOCK-12', PASS, "%s: analytic_input_snr 'not_derivable', so ADM-INPUT is mandatory" % why,
                      data={'adm_input_mandatory': True})
    if not is_num(snr):
        return Result('LOCK-12', FAIL, "%s: analytic_input_snr %r is neither a number nor 'not_derivable'" % (why, snr))
    if snr < fmin - EPS:
        return Result('LOCK-12', FAIL, '%s: analytic input SNR %s below the floor %s' % (why, snr, fmin))
    return Result('LOCK-12', PASS, '%s: analytic input SNR %s >= floor %s' % (why, snr, fmin))


# --------------------------------------------------------------------------------------
# LOCK-13: lineage (undeclared parent)
# --------------------------------------------------------------------------------------

def _parent_dir(lock: Optional[dict]) -> Optional[str]:
    if not isinstance(lock, dict):
        return None
    par = dig(lock, 'lineage.parent')
    if not isinstance(par, str) or not par.strip():
        return None
    m = PARENT_RE.match(par.strip())
    return (m.group('dir') if m else par.split('@', 1)[0]).strip().rstrip('/')


def _lock_of(ctx: common.Ctx, dir_rel: str) -> Optional[dict]:
    p = Path(ctx.repo_root) / dir_rel / 'ENDPOINT_LOCK.json'
    if not p.is_file():
        return None
    try:
        obj = common.load_json_file(p)
    except (MissingInput, InvalidInput):
        return None
    return obj if isinstance(obj, dict) else None


def _match_member(name: str, members: Sequence[str]) -> Optional[str]:
    if name in members:
        return name
    hits = [m for m in members if m.rsplit('/', 1)[-1] == name.rsplit('/', 1)[-1]]
    return hits[0] if len(hits) == 1 else (name if not hits else hits[0])


def lineage_chain(ctx: common.Ctx, start: str, members: Sequence[str], limit: int = 64) -> List[str]:
    """[start, parent(start), parent(parent), ...] as repo-relative dirs, following lineage.parent."""
    out: List[str] = []
    cur: Optional[str] = start
    while cur and cur not in out and len(out) < limit:
        out.append(cur)
        nxt = _parent_dir(_lock_of(ctx, cur))
        cur = _match_member(nxt, members) if nxt else None
    return out


@check('LOCK-13')
def check_lock_13(ctx: common.Ctx) -> Result:
    lock = ctx.lock
    fam_id, fam_members = common.family_of(ctx)
    neighbors = common.closure_jaccard_neighbors(ctx)
    g = common.family_graph(ctx)
    members = g['members']
    me = ctx.delivery_rel
    data = {'family_id': fam_id, 'family_members': fam_members}
    problems: List[str] = []
    pdir = _parent_dir(lock)
    ancestors: List[str] = []
    if pdir:
        pm = _match_member(pdir, members)
        if not (Path(ctx.repo_root) / pm).is_dir():
            problems.append('declared parent %s does not exist in the repo' % pdir)
        else:
            par = dig(lock, 'lineage.parent')
            m = PARENT_RE.match(str(par).strip())
            if m:
                ctx.git.file_at(m.group('sha'), pm + '/ENDPOINT_LOCK.json')  # MissingInput for an unknown commit
            ancestors = lineage_chain(ctx, pm, members)
    try:
        c_me = ctx.git.first_commit_adding(lock_rel(ctx))
        idx = log_index(ctx)
        git_ok = True
    except MissingInput:
        c_me, idx, git_ok = None, {}, False
    flagged: List[str] = []
    for m, j in neighbors:
        if j < 0.5:
            continue
        if m in ancestors or m.rsplit('/', 1)[-1] in [a.rsplit('/', 1)[-1] for a in ancestors]:
            continue
        if me in lineage_chain(ctx, m, members):
            continue  # a descendant of this delivery
        older = True
        if git_ok:
            try:
                c_m = ctx.git.first_commit_adding(m + '/ENDPOINT_LOCK.json')
            except MissingInput:
                c_m = None
            if c_me is not None and c_m is not None:
                older = idx.get(c_m.sha, 0) <= idx.get(c_me.sha, 0)
            elif c_me is not None and c_m is None:
                older = False  # not committed yet: younger than this committed lock
        else:
            dm = re.search(r'_([0-9]{8})$', m.rsplit('/', 1)[-1])
            dme = re.search(r'_([0-9]{8})$', me.rsplit('/', 1)[-1])
            if dm and dme and dm.group(1) > dme.group(1):
                older = False
        if older:
            flagged.append('%s (Jaccard %.2f)' % (m, j))
    if flagged:
        problems.append('undeclared parent: %s %s native-closure Jaccard >= 0.5 with this delivery but %s neither the '
                        'declared parent (%s) nor its ancestor' % (', '.join(flagged), 'has' if len(flagged) == 1 else
                                                                  'have', 'is' if len(flagged) == 1 else 'are',
                                                                  pdir or 'null'))
    scope = g.get('scope') or ''
    if problems:
        return Result('LOCK-13', FAIL, '%s; family %s; %s' % ('; '.join(problems), fam_id, scope),
                      evidence=flagged, data=data)
    return Result('LOCK-13', PASS, 'family %s (%d member(s)); parent %s; no undeclared parent; %s'
                  % (fam_id, len(fam_members), pdir or 'null', scope), data=data)


# --------------------------------------------------------------------------------------
# AMD-01..05
# --------------------------------------------------------------------------------------

def lock_transitions(ctx: common.Ctx) -> Tuple[List[Tuple[str, str, str]], List[str]]:
    """Consecutive ENDPOINT_LOCK content changes (old sha, new sha, where) over history plus the
    worktree, and problems (deletions)."""
    trans: List[Tuple[str, str, str]] = []
    problems: List[str] = []
    prev: Optional[str] = None
    for c, data in lock_versions(ctx):
        if data is None:
            if prev is not None:
                problems.append('ENDPOINT_LOCK.json deleted at commit %s' % c.sha[:12])
            continue
        s = common.sha256_bytes(data)
        if prev is not None and s != prev:
            trans.append((prev, s, 'commit %s' % c.sha[:12]))
        prev = s
    cur = ctx.endpoint_lock_sha256
    if prev is not None and cur is not None and cur != prev:
        trans.append((prev, cur, 'worktree (uncommitted)'))
    return trans, problems


@check('AMD-01')
def check_amd_01(ctx: common.Ctx) -> Result:
    rows = amendments(ctx)
    trans, problems = lock_transitions(ctx)
    pairs = set()
    for i, r in enumerate(rows):
        o, n = strip_sha(r.get('old_lock_sha256')), strip_sha(r.get('new_lock_sha256'))
        if not o or not n:
            problems.append('AMENDMENTS row %d (%s) lacks old/new lock sha256' % (i + 1, r.get('id')))
            continue
        pairs.add((o, n))
    linked = 0
    for o, n, where in trans:
        if (o, n) in pairs:
            linked += 1
        else:
            problems.append('ENDPOINT_LOCK changed %s -> %s at %s without an AMENDMENTS row' % (o[:12], n[:12], where))
    tset = set((o, n) for o, n, _w in trans)
    for o, n in sorted(pairs - tset):
        problems.append('AMENDMENTS row %s -> %s links no observed ENDPOINT_LOCK change' % (o[:12], n[:12]))
    if problems:
        return Result('AMD-01', FAIL, '; '.join(problems[:8]), evidence=problems[:40])
    return Result('AMD-01', PASS, '%d ENDPOINT_LOCK change(s), all linked by AMENDMENTS rows' % len(trans))


def objective_distance_raised(old: Any, new: Any) -> List[str]:
    out: List[str] = []
    old = old if isinstance(old, dict) else {}
    if not isinstance(new, dict):
        return ['objective_distance missing in the amended lock']
    for axis, ranks in DISTANCE_RANKS.items():
        ov, nv = old.get(axis), new.get(axis)
        if canonical_eq(ov, nv):
            continue
        orank = ranks.get(ov) if not isinstance(ov, (dict, list)) else None
        nrank = ranks.get(nv) if not isinstance(nv, (dict, list)) else None
        if nrank is None or orank is None or nrank > orank:
            out.append('objective_distance.%s %r -> %r' % (axis, ov, nv))
    return out


def canonical_eq(a: Any, b: Any) -> bool:
    try:
        return common.canonical_json(a) == common.canonical_json(b)
    except InvalidInput:
        return False


def _coprimary_keeps(new_lock: dict, old_lock: dict, old_sha: str) -> bool:
    cps = new_lock.get('coprimary')
    if not isinstance(cps, list):
        return False
    for e in cps:
        if isinstance(e, str) and strip_sha(e) == old_sha:
            return True
        if not isinstance(e, dict):
            continue
        if any(isinstance(v, str) and strip_sha(v) == old_sha for v in e.values()):
            return True
        if old_lock.get('protocol_id') and e.get('protocol_id') == old_lock.get('protocol_id'):
            return True
        if 'primary' in e and canonical_eq(e.get('primary'), old_lock.get('primary')):
            return True
    return False


def _protocol_n(pid: Any) -> Optional[int]:
    m = PROTOCOL_RE.match(pid) if isinstance(pid, str) else None
    return int(m.group('n')) if m else None


def _amendment_pair(ctx: common.Ctx, row: dict) -> Tuple[Optional[dict], Optional[dict], List[str]]:
    o, n = strip_sha(row.get('old_lock_sha256')), strip_sha(row.get('new_lock_sha256'))
    why = []
    old_lock = lock_content_by_sha(ctx, o) if o else None
    new_lock = lock_content_by_sha(ctx, n) if n else None
    if old_lock is None:
        why.append('old lock content %s not found in history' % (o or '?')[:12])
    if new_lock is None:
        why.append('new lock content %s not found in history or worktree' % (n or '?')[:12])
    return old_lock, new_lock, why


@check('AMD-02')
def check_amd_02(ctx: common.Ctx) -> Result:
    rows = amendments(ctx)
    if not rows:
        return Result('AMD-02', NOT_APPLICABLE, 'no AMENDMENTS rows')
    problems: List[str] = []
    notes: List[str] = []
    scopes = ['amendment:%s' % n for n in delivery_ids(ctx)]
    for i, r in enumerate(rows):
        rid = r.get('id') or ('row %d' % (i + 1))
        old_lock, new_lock, why = _amendment_pair(ctx, r)
        if why:
            problems.append('%s: %s' % (rid, '; '.join(why)))
            continue
        changed = leaf_diff(old_lock, new_lock)
        sections = sorted(set(path_top(p) for p in changed if path_top(p) in AMD02_SECTIONS))
        triggers: List[str] = []
        if sections:
            c = amendment_commit(ctx, r)
            ev = learner_outcome_before(ctx, c, include_worktree=c is None)
            if ev:
                triggers.append('changes %s after a learner-arm outcome (%s)' % (','.join(sections), ev))
        raised = objective_distance_raised(old_lock.get('objective_distance'), new_lock.get('objective_distance'))
        if raised:
            triggers.append('raises %s' % '; '.join(raised))
        if not triggers:
            notes.append('%s: no UD needed' % rid)
            continue
        miss: List[str] = []
        ud = r.get('user_decision')
        uid = ud_with_scope(ctx, [ud] if isinstance(ud, str) else [], scopes) if ud else None
        if not uid:
            miss.append('a UD with scope %s (user_decision %r)' % (scopes[0], ud))
        o = strip_sha(r.get('old_lock_sha256')) or ''
        if not _coprimary_keeps(new_lock, old_lock, o) or r.get('original_endpoint_kept_as_coprimary') is not True:
            miss.append('the original endpoint in coprimary[] (and original_endpoint_kept_as_coprimary true)')
        on, nn = _protocol_n(old_lock.get('protocol_id')), _protocol_n(new_lock.get('protocol_id'))
        if on is None or nn is None or nn <= on or r.get('new_protocol_id') != new_lock.get('protocol_id'):
            miss.append('a bumped protocol_id (%r -> %r; row new_protocol_id %r)' % (
                old_lock.get('protocol_id'), new_lock.get('protocol_id'), r.get('new_protocol_id')))
        if miss:
            problems.append('%s %s but lacks %s' % (rid, ' and '.join(triggers), '; '.join(miss)))
        else:
            notes.append('%s: UD %s, co-primary kept, protocol bumped' % (rid, uid))
    if problems:
        return Result('AMD-02', FAIL, '; '.join(problems[:6]), evidence=problems[:30])
    return Result('AMD-02', PASS, '%d amendment(s): %s' % (len(rows), '; '.join(notes[:6])))


def _entry_passes(e: Any) -> Optional[bool]:
    if not isinstance(e, dict):
        return None
    for k in ('verdict', 'status', 'result'):
        if isinstance(e.get(k), str):
            return e[k].upper() == 'PASS'
    for k in ('pass', 'passed', 'passes'):
        if isinstance(e.get(k), bool):
            return e[k]
    return None


def testbed_shows_invalid(tb: Any) -> Tuple[bool, str]:
    """True when a TESTBED result shows ceiling FAIL (no ceiling passes) or floor PASS."""
    if not isinstance(tb, dict):
        return False, 'TESTBED is not an object'
    # scorer.py (WP3) shape: result, ceiling_any_pass, floors_all_fail, configs[{arm, kind, declared, status}]
    for k in ('result', 'verdict', 'testbed_verdict', 'stop_reason'):
        v = tb.get(k)
        if isinstance(v, str) and v.upper().startswith('CANNOT_TEST'):
            return True, '%s %s' % (k, v)
    for k in ('ceiling_any_pass', 'any_ceiling_pass', 'ceiling_pass', 'ceiling_passed'):
        if tb.get(k) is False:
            return True, '%s false' % k
    if tb.get('floors_all_fail') is False:
        return True, 'floors_all_fail false'
    for k in ('floor_pass', 'any_floor_pass', 'best_floor_pass', 'floor_passed'):
        if tb.get(k) is True:
            return True, '%s true' % k
    cfgs = [c for c in (tb.get('configs') or []) if isinstance(c, dict)]
    if any(c.get('kind') == 'floor' and str(c.get('status')).upper() == 'PASS' for c in cfgs):
        return True, 'a floor config passes'
    ceil = [c for c in cfgs if c.get('kind') == 'ceiling' and c.get('declared', True)]
    if ceil and all(str(c.get('status')).upper() == 'FAIL' for c in ceil):
        return True, 'every declared ceiling config fails'
    for k in ('ceiling', 'ceilings', 'ceiling_results', 'ceiling_grid'):
        v = tb.get(k)
        entries = list(v.values()) if isinstance(v, dict) else (v if isinstance(v, list) else None)
        if entries:
            res = [_entry_passes(e) for e in entries]
            if res and all(x is False for x in res):
                return True, 'no %s config passes' % k
    for k in ('floor', 'floors', 'floor_results', 'floor_grid', 'best_floor'):
        v = tb.get(k)
        entries = [v] if k == 'best_floor' else (list(v.values()) if isinstance(v, dict) else
                                                  (v if isinstance(v, list) else []))
        if any(_entry_passes(e) is True for e in entries):
            return True, 'a %s config passes' % k
    return False, 'TESTBED shows neither ceiling FAIL nor floor PASS'


def testbed_evidence(ctx: common.Ctx, row: dict, old_sha: str) -> Tuple[bool, str]:
    cands = []
    ev = row.get('evidence')
    if isinstance(ev, str) and ev.strip().endswith('.json'):
        cands.append(common.norm_rel(ev.strip()))
    if 'results/admission/TESTBED.json' not in cands:
        cands.append('results/admission/TESTBED.json')
    c_new = None
    new = strip_sha(row.get('new_lock_sha256'))
    if new:
        c_new = content_commit(ctx, lock_rel(ctx), lambda b: common.sha256_bytes(b) == new, parse_json=False)
    idx = log_index(ctx)
    for drel in cands:
        versions: List[Tuple[Optional[common.Commit], bytes]] = []
        rel = ctx.repo_rel_of(drel)
        for c in ctx.git.log([rel]):
            data = ctx.git.file_at(c.sha, rel)
            if data is not None:
                versions.append((c, data))
        if ctx.path(drel).is_file():
            versions.append((None, ctx.path(drel).read_bytes()))
        for c, data in versions:
            try:
                tb = common.loads_strict(data, what=drel)
            except InvalidInput:
                continue
            ok, why = testbed_shows_invalid(tb)
            if not ok:
                continue
            tsha = strip_sha(tb.get('endpoint_lock_sha256')) if isinstance(tb, dict) else None
            if tsha:
                if tsha == old_sha:
                    return True, '%s (%s) on the original endpoint %s' % (drel, why, old_sha[:12])
                continue
            if c is not None and (c_new is None or idx.get(c.sha, 1 << 30) < idx.get(c_new.sha, 1 << 30)):
                return True, '%s@%s (%s) committed before the amended lock' % (drel, c.sha[:12], why)
    return False, 'no TESTBED result on the original endpoint %s shows ceiling FAIL or floor PASS' % old_sha[:12]


AMD_ROW_FIELDS = ('id', 'utc', 'old_lock_sha256', 'new_lock_sha256', 'new_protocol_id', 'fields_changed',
                  'results_seen', 'reason', 'evidence', 'objective_distance_delta',
                  'original_endpoint_kept_as_coprimary', 'user_decision', 'author_agent')


@check('AMD-03')
def check_amd_03(ctx: common.Ctx) -> Result:
    rows = amendments(ctx)
    if not rows:
        return Result('AMD-03', NOT_APPLICABLE, 'no AMENDMENTS rows')
    problems: List[str] = []
    ids: List[str] = []
    for i, r in enumerate(rows):
        rid = r.get('id') or ('row %d' % (i + 1))
        for f in AMD_ROW_FIELDS:
            if f not in r:
                problems.append('%s lacks %s' % (rid, f))
        if r.get('id') in ids:
            problems.append('amendment id %s is duplicated' % r.get('id'))
        ids.append(r.get('id'))
        if not isinstance(r.get('fields_changed'), list) or not isinstance(r.get('results_seen'), list):
            problems.append('%s: fields_changed and results_seen must be lists' % rid)
            continue
        reason = r.get('reason')
        if reason not in AMD_REASONS:
            problems.append('%s: reason %r is not one of %s' % (rid, reason, '|'.join(AMD_REASONS)))
        old_lock, new_lock, why = _amendment_pair(ctx, r)
        if why:
            problems.append('%s: %s' % (rid, '; '.join(why)))
        else:
            changed = [p for p in leaf_diff(old_lock, new_lock) if path_top(p) not in AMD_ADMIN_FIELDS]
            unlisted = [p for p in changed if not path_covered(p, r.get('fields_changed') or [])]
            if unlisted:
                problems.append('%s: fields_changed omits %s' % (rid, ', '.join(unlisted[:6])))
        if reason == 'testbed_invalid':
            ok, how = testbed_evidence(ctx, r, strip_sha(r.get('old_lock_sha256')) or '')
            if not ok:
                problems.append('%s: testbed_invalid needs %s' % (rid, how))
        c = amendment_commit(ctx, r)
        computed = outcome_files_before(ctx, c, include_worktree=c is None)
        try:
            seen = sorted(set(common.norm_rel(x) for x in r.get('results_seen') or []))
        except InvalidInput as e:
            problems.append('%s: results_seen: %s' % (rid, e))
            continue
        # every raw outcome file must be listed; a checker-written score file (SCORES/VERDICT/TESTBED) that was
        # committed before the amendment may also be listed (it was seen); nothing else may be
        scored_seen = set(outcome_files_before(ctx, c, include_worktree=c is None, checker_written=True))
        if set(seen) - scored_seen != set(computed):
            extra = sorted(set(seen) - set(computed) - scored_seen)
            lack = sorted(set(computed) - set(seen))
            problems.append('%s: results_seen differs from the outcome files committed before it (missing %s; '
                            'extra %s)' % (rid, ', '.join(lack[:6]) or '-', ', '.join(extra[:6]) or '-'))
    if problems:
        return Result('AMD-03', FAIL, '; '.join(problems[:6]), evidence=problems[:40])
    return Result('AMD-03', PASS, '%d amendment row(s): reasons valid, evidence present, results_seen computed-equal'
                  % len(rows))


def official_world_ids(ctx: common.Ctx) -> Tuple[List[str], str]:
    """Listed official world ids plus seal-derived ones (when the seal is revealed)."""
    lock = ctx.lock
    ids = [str(x) for x in (dig(lock, 'worlds.official.ids') or []) if isinstance(x, (str, int)) and str(x)]
    how = 'listed' if ids else ''
    try:
        seal = importlib.import_module('seal')
        derived = seal.derived_world_ids(ctx)
    except (MissingInput, InvalidInput, ImportError, AttributeError):
        derived = None
    if derived:
        for d in derived:
            if d not in ids:
                ids.append(d)
        how = (how + '+' if how else '') + 'seal-derived'
    return ids, how


_TOKEN_CACHE: Dict[str, Any] = {}


def _token_re(wid: str):
    rx = _TOKEN_CACHE.get(wid)
    if rx is None:
        rx = re.compile(r'(?<![A-Za-z0-9_])%s(?![A-Za-z0-9_])' % re.escape(wid))
        _TOKEN_CACHE[wid] = rx
    return rx


def json_value_hits(obj: Any, ids: Set[str], key: str = '') -> Set[str]:
    out: Set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k) in ids:
                out.add(str(k))
            out |= json_value_hits(v, ids, str(k))
    elif isinstance(obj, list):
        for v in obj:
            out |= json_value_hits(v, ids, key)
    elif isinstance(obj, str):
        if obj in ids:
            out.add(obj)
    elif is_int(obj) and 'world' in key.lower() and str(obj) in ids:
        out.add(str(obj))
    return out


def _stream_contains(fh: Any, needles: Sequence[bytes], chunk: int = 1 << 20) -> bool:
    """True when any needle occurs in the stream (read in chunks with overlap; bounded memory)."""
    keep = max((len(n) for n in needles), default=1) - 1
    tail = b''
    while True:
        block = fh.read(chunk)
        if not block:
            return False
        buf = tail + block
        if any(n in buf for n in needles):
            return True
        tail = buf[-keep:] if keep > 0 else b''


def world_hits_in_bytes(rel: str, data: bytes, ids: Set[str]) -> Set[str]:
    """World ids present in one file version: JSON-aware for .json/.jsonl[.gz] and zip members, token
    match for text, and path segments. Compressed content is searched as a stream first and fully
    decompressed only when an id's bytes occur in it."""
    hits: Set[str] = set(seg for seg in rel.split('/') if seg in ids)
    name = rel.rsplit('/', 1)[-1]
    needles = [i.encode('utf-8') for i in ids]
    try:
        if common.RESULT_BUNDLE_RE.match(name) or name.endswith('.zip'):
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    mrel = '%s!%s' % (rel, info.filename)
                    hits |= set(seg for seg in info.filename.split('/') if seg in ids)
                    with zf.open(info) as mf:
                        raw = gzip.GzipFile(fileobj=mf) if info.filename.endswith('.gz') else mf
                        found = _stream_contains(raw, needles)
                    if found:
                        hits |= world_hits_in_bytes(mrel, zf.read(info), ids)
            return hits
        if name.endswith('.gz'):
            with gzip.GzipFile(fileobj=io.BytesIO(data)) as gf:
                if not _stream_contains(gf, needles):
                    return hits
            data = gzip.decompress(data)
            name = name[:-3]
    except (zipfile.BadZipFile, OSError, EOFError) as e:
        raise InvalidInput('%s unreadable: %s' % (rel, e))
    if not any(i.encode('utf-8') in data for i in ids):
        return hits
    if name.endswith('.json') or name.endswith('.jsonl'):
        try:
            if name.endswith('.json'):
                hits |= json_value_hits(common.loads_strict(data, what=rel), ids)
            else:
                for row in jsonl_rows_from_bytes(data, rel):
                    hits |= json_value_hits(row, ids)
            return hits
        except InvalidInput:
            pass  # fall back to a token scan
    text = data.decode('utf-8', 'replace')
    for i in ids:
        if _token_re(i).search(text):
            hits.add(i)
    return hits


@check('AMD-04')
def check_amd_04(ctx: common.Ctx) -> Result:
    ids, how = official_world_ids(ctx)
    if not ids:
        return Result('AMD-04', NOT_APPLICABLE, 'no official world ids listed or seal-derived yet')
    idset = set(ids)
    problems: List[str] = []
    rows = amendments(ctx)
    if rows:
        cs = [amendment_commit(ctx, r) for r in rows]
        if any(c is None for c in cs):
            latest = None
        else:
            idx = log_index(ctx)
            latest = max(cs, key=lambda c: idx.get(c.sha, -1))
        for c, p, drel in committed_results_files(ctx, commits_before(ctx, latest)):
            data = ctx.git.file_at(c.sha, p)
            if data is None:
                continue
            h = world_hits_in_bytes(drel, data, idset)
            if h:
                problems.append('official world(s) %s in %s@%s, committed before the latest amendment'
                                % (','.join(sorted(h)), drel, c.sha[:12]))
        if latest is None:
            for rel in ctx.delivery_files():
                if rel.startswith('results/'):
                    h = world_hits_in_bytes(rel, ctx.path(rel).read_bytes(), idset)
                    if h:
                        problems.append('official world(s) %s in %s before the pending amendment'
                                        % (','.join(sorted(h)), rel))
    me = ctx.delivery_rel
    excludes = tuple(common.DEFAULT_SCAN_EXCLUDES) + (me, common.KB_REL, common.KIT_REL_DEFAULT, '.claude')
    scan = ctx.repo_scan(patterns=common.DEFAULT_SCAN_PATTERNS, excludes=excludes)
    other: List[str] = []
    for f in scan.files:
        rel = scan.rel(f)
        try:
            data = f.read_bytes()
        except OSError as e:
            raise MissingInput('repo scan could not read %s: %s' % (rel, e))
        h = world_hits_in_bytes(rel, data, idset)
        if h:
            other.append('%s in %s' % (','.join(sorted(h)), rel))
    for o in other[:20]:
        problems.append('official world id %s (another delivery)' % o)
    if problems:
        return Result('AMD-04', FAIL, '%s; %s' % ('; '.join(problems[:6]), scan.scope()), evidence=problems[:40])
    return Result('AMD-04', PASS, '%d %s official world id(s) appear in no earlier outcome artifact and no other '
                  'delivery; %s' % (len(ids), how, scan.scope()))


def split_manifest_paths(ctx: common.Ctx, lk: dict) -> Dict[str, Optional[str]]:
    """{split key: repo-relative path or None} for the HELD/OLD/NEW manifests a lock version names."""
    split = lk.get('split') if isinstance(lk.get('split'), dict) else {}
    out: Dict[str, Optional[str]] = {}
    for key, _pool in common.SPLIT_MANIFESTS:
        v = split.get(key)
        try:
            nk = common.norm_rel(v) if isinstance(v, str) and v else None
        except InvalidInput:
            nk = None
        if nk is None:
            out[key] = None
            continue
        rr, _how = resolve_entry(ctx, nk)
        out[key] = rr or ctx.repo_rel_of(nk)
    return out


def split_hash_problem(lk: dict, blobs: Dict[str, Optional[bytes]], where: str) -> Optional[str]:
    """Why the split manifests `blobs` do not match the lock version `lk` (SPEC 6.2: the held set is fixed in the
    lock by split.held_set_sha256; SCHEMAS.md section 1 gives both formulas, common.split_hashes)."""
    split = lk.get('split') if isinstance(lk.get('split'), dict) else {}
    absent = [k for k, _p in common.SPLIT_MANIFESTS if blobs.get(k) is None]
    if absent:
        return 'split manifest(s) %s absent %s while the lock fixes them' % (', '.join(absent), where)
    held, sp = common.split_hashes(blobs)
    bad = []
    if held != strip_sha(split.get('held_set_sha256')):
        bad.append('held_set_sha256 (recomputed %s)' % str(held)[:12])
    if sp != strip_sha(split.get('split_sha256')):
        bad.append('split_sha256 (recomputed %s)' % str(sp)[:12])
    if bad:
        return ('split manifests %s do not hash to the lock\'s %s: the held set is fixed in the lock and changes '
                'only with an amendment' % (where, ', '.join(bad)))
    return None


@check('AMD-05')
def check_amd_05(ctx: common.Ctx) -> Result:
    lock = ctx.lock
    efs = lock.get('endpoint_files_sha256')
    if not isinstance(efs, dict):
        return Result('AMD-05', FAIL, 'endpoint_files_sha256 is not an object')
    problems: List[str] = []
    # every endpoint-side file is endpoint-locked
    keys_norm: Set[str] = set()
    for k in efs:
        try:
            nk = common.norm_rel(k)
        except InvalidInput as e:
            problems.append('endpoint_files_sha256 key %r: %s' % (k, e))
            continue
        keys_norm.add(nk)
        keys_norm.add(ctx.repo_rel_of(nk))
    for key in ('admit', 'evaluator', 'generators'):
        for e in files_entries(lock, key):
            if common.has_glob(e):
                found, _m = expand_entries(ctx, [e])
                ents = [f[len(delivery_prefix(ctx)):] if f.startswith(delivery_prefix(ctx)) else f for f in found]
            else:
                ents = [e]
            for x in ents:
                xn = common.norm_rel(x)
                if xn not in keys_norm and ctx.repo_rel_of(xn) not in keys_norm:
                    problems.append('files.%s %s is not in endpoint_files_sha256' % (key, x))
    # worktree equals the lock
    repo_paths: Dict[str, str] = {}
    for k, v in efs.items():
        try:
            nk = common.norm_rel(k)
        except InvalidInput:
            continue
        rr, _how = resolve_entry(ctx, nk)
        repo_paths[k] = rr or ctx.repo_rel_of(nk)
        want = strip_sha(v)
        if rr is None:
            problems.append('endpoint file %s is missing' % k)
            continue
        got = common.sha256_file(ctx.repo_path(rr))
        if got != want:
            problems.append('endpoint file %s sha256 %s differs from the lock %s (changed without an amendment)'
                            % (k, got[:12], str(v)[:12]))
    # split manifests in the worktree equal the lock's split hashes
    split_paths = split_manifest_paths(ctx, lock)
    wt_blobs = {k: (ctx.repo_path(rp).read_bytes() if rp and ctx.repo_path(rp).is_file() else None)
                for k, rp in split_paths.items()}
    sp = split_hash_problem(lock, wt_blobs, 'in the worktree')
    if sp:
        problems.append(sp)
    # history after the first lock commit
    lrel = lock_rel(ctx)
    c_lock = ctx.git.first_commit_adding(lrel)
    checked = 0
    split_checked = 0
    if c_lock is not None:
        paths = [lrel] + sorted(set(repo_paths.values()) | set(p for p in split_paths.values() if p))
        idx = log_index(ctx)
        start = idx.get(c_lock.sha, 0)
        prev_lock_sha: Optional[str] = None
        prev_efs: Dict[str, Optional[str]] = {}
        prev_split: Dict[str, Any] = {}
        pairs = set((strip_sha(r.get('old_lock_sha256')), strip_sha(r.get('new_lock_sha256'))) for r in amendments(ctx))
        for c in ctx.git.log(paths):
            if idx.get(c.sha, -1) < start:
                continue
            ldata = ctx.git.file_at(c.sha, lrel)
            if ldata is None:
                continue
            lk = parse_lock_bytes(ldata, 'ENDPOINT_LOCK.json@%s' % c.sha[:12]) or {}
            lsha = common.sha256_bytes(ldata)
            cur_efs = lk.get('endpoint_files_sha256') if isinstance(lk.get('endpoint_files_sha256'), dict) else {}
            for k, v in cur_efs.items():
                try:
                    nk = common.norm_rel(k)
                except InvalidInput:
                    continue
                rp = repo_paths.get(k) or ctx.repo_rel_of(nk)
                data = ctx.git.file_at(c.sha, rp)
                checked += 1
                if data is None:
                    problems.append('endpoint file %s absent at commit %s while locked' % (k, c.sha[:12]))
                elif common.sha256_bytes(data) != strip_sha(v):
                    problems.append('endpoint file %s at commit %s differs from the lock in effect (changed after the '
                                    'lock without an amendment)' % (k, c.sha[:12]))
                if prev_lock_sha is not None and k in prev_efs and strip_sha(prev_efs[k]) != strip_sha(v) \
                        and (prev_lock_sha, lsha) not in pairs:
                    problems.append('the lock re-pinned %s at commit %s without an AMENDMENTS row' % (k, c.sha[:12]))
            # the split manifests at this commit equal the split hashes of the lock in effect
            c_paths = split_manifest_paths(ctx, lk)
            blobs = {k: (ctx.git.file_at(c.sha, rp) if rp else None) for k, rp in c_paths.items()}
            split_checked += 1
            sp = split_hash_problem(lk, blobs, 'at commit %s' % c.sha[:12])
            if sp:
                problems.append(sp)
            cur_split = lk.get('split') if isinstance(lk.get('split'), dict) else {}
            if prev_lock_sha is not None and prev_lock_sha != lsha and (prev_lock_sha, lsha) not in pairs and any(
                    strip_sha(prev_split.get(h)) != strip_sha(cur_split.get(h)) for h in ('held_set_sha256', 'split_sha256')):
                problems.append('the lock re-pinned the split hashes at commit %s without an AMENDMENTS row' % c.sha[:12])
            if prev_lock_sha != lsha:
                prev_lock_sha, prev_efs = lsha, dict(cur_efs)
                prev_split = dict(cur_split)
    if problems:
        return Result('AMD-05', FAIL, '; '.join(sorted(set(problems))[:8]), evidence=sorted(set(problems))[:40])
    return Result('AMD-05', PASS, '%d endpoint file(s) and the split manifests unchanged since the lock (%d committed '
                  'endpoint versions and %d committed split versions checked)' % (len(efs), checked, split_checked))


# --------------------------------------------------------------------------------------
# WIT-01: external witness
# --------------------------------------------------------------------------------------

def world_prefixes(ctx: common.Ctx, role: str) -> List[str]:
    lock = ctx.lock
    pre = delivery_prefix(ctx)
    out = [pre + 'results/%s' % role]
    if role == 'official':
        ids, _how = official_world_ids(ctx)
    else:
        ids = [str(x) for x in (dig(lock, 'worlds.%s' % role) or []) if isinstance(x, (str, int))]
    for w in ids:
        out.append(pre + 'results/receipts/%s' % w)
    return out


def worktree_outputs(ctx: common.Ctx, role: str) -> List[str]:
    pre = delivery_prefix(ctx)
    prefixes = [p[len(pre):] for p in world_prefixes(ctx, role)]
    return [f for f in ctx.delivery_files() if any(common.glob_match(f, p) for p in prefixes)
            and not (f.endswith('/.gitkeep') and ctx.path(f).stat().st_size == 0)]


def first_output_commit(ctx: common.Ctx, role: str) -> Optional[common.Commit]:
    return first_commit_adding_under(ctx, world_prefixes(ctx, role))


@check('WIT-01')
def check_wit_01(ctx: common.Ctx) -> Result:
    tier = current_tier(ctx)
    if tier < 2:
        return Result('WIT-01', NOT_APPLICABLE, 'tier %d < 2' % tier)
    lock = ctx.lock
    fam = reviewer_family(lock)
    if fam is None:
        return Result('WIT-01', FAIL, 'lock.author_agent %r is not codex|claude: no reviewer namespace' %
                      lock.get('author_agent'))
    review = read_review(ctx)
    if review is None:
        raise MissingInput('review/REVIEW.json absent: no external witness attestation')
    l1 = ctx.endpoint_lock_sha256
    problems: List[str] = []
    notes: List[str] = []
    if strip_sha(review.get('reviewed_endpoint_lock_sha256')) != l1:
        problems.append('REVIEW.reviewed_endpoint_lock_sha256 %s is not the current ENDPOINT_LOCK %s'
                        % (str(review.get('reviewed_endpoint_lock_sha256'))[:12], (l1 or '')[:12]))
    rrel = ctx.repo_rel_of(REVIEW_REL)
    c_att = content_commit(ctx, rrel, lambda o: isinstance(o, dict)
                           and strip_sha(o.get('reviewed_endpoint_lock_sha256')) == l1)
    if c_att is None:
        problems.append('no commit introduces a REVIEW.json attesting the current ENDPOINT_LOCK')
    else:
        br = in_namespace(ctx.git.remote_branches_containing(c_att.sha), fam)
        ok, why = common.commit_is_family(ctx.git, c_att, fam)
        if not ok:
            # the acting family is derived from the trailer AND the namespace (SPEC 20.15)
            problems.append('attestation commit %s is not the reviewer family %s (%s)' % (c_att.sha[:12], fam, why))
        c_dev = first_output_commit(ctx, 'dev')
        if c_dev is not None and not strictly_before(ctx, c_att, c_dev):
            problems.append('attestation commit %s is not an ancestor of the first dev output commit %s'
                            % (c_att.sha[:12], c_dev.sha[:12]))
        notes.append('lock attested at %s (%s)' % (c_att.sha[:12], ','.join(br[:2]) or '-'))
    official = ctx.stage == 'official' or ctx.options.get('launch_stage') == 'official' or bool(
        worktree_outputs(ctx, 'official'))
    if official:
        l2 = ctx.source_lock_sha256
        acks = [a for a in (review.get('source_lock_acks') or []) if isinstance(a, dict)]
        acked = l2 is not None and any(strip_sha(a.get('sha')) == l2 for a in acks)
        if not acked:
            msg = 'review.source_lock_acks lacks the final SOURCE_LOCK %s' % (l2 or 'none')[:12]
            if ctx.dry and not worktree_outputs(ctx, 'official'):
                notes.append('[dry: deferred to the official gate] %s' % msg)
            else:
                problems.append(msg)
        else:
            c_ack = content_commit(ctx, rrel, lambda o: isinstance(o, dict) and any(
                isinstance(a, dict) and strip_sha(a.get('sha')) == l2 for a in (o.get('source_lock_acks') or [])))
            if c_ack is None:
                problems.append('the SOURCE_LOCK ack is not committed')
            else:
                ok, why = common.commit_is_family(ctx.git, c_ack, fam)
                if not ok:
                    problems.append('SOURCE_LOCK ack commit %s is not the reviewer family %s (%s)'
                                    % (c_ack.sha[:12], fam, why))
                c_off = first_output_commit(ctx, 'official')
                if c_off is not None and not strictly_before(ctx, c_ack, c_off):
                    problems.append('SOURCE_LOCK ack commit %s is not an ancestor of the first official output %s'
                                    % (c_ack.sha[:12], c_off.sha[:12]))
                notes.append('SOURCE_LOCK %s acked at %s' % (l2[:12], c_ack.sha[:12]))
    if problems:
        return Result('WIT-01', FAIL, 'external witness: %s' % '; '.join(problems[:6]), evidence=problems)
    status = WARN if any(n.startswith('[dry') for n in notes) else PASS
    return Result('WIT-01', status, 'reviewer namespace origin/%s/*: %s' % (fam, '; '.join(notes)))
