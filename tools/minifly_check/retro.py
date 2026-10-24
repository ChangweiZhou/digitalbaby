"""MRK-1 retro (WP6): read-only RETRO_ANNOTATION rows for legacy deliveries (SPEC 12.4).

    MFC retro (--all | <dir> | <ref>:<dir>) --out DIR

Rows are {checker_sha256, delivery, check_id, status: PASS | FLAG | NOT_APPLICABLE, reason, evidence_paths}.
They are annotations, never verdicts. Historical code is never imported or executed: files are read as
bytes (worktree) or through `git cat-file` / `git diff --name-only` (branches such as origin/charming-head,
never checked out), and Python files are only parsed with `ast`. Output goes only to --out, which must lie
outside every delivery directory. Retro exits nonzero only when it crashes (or refuses an invalid --out).

Acceptance (retro --all) follows SPEC 12.4: recall on the artifact-permitting FLAG list, and the false-flag
limit read twice, scoped to each quiet item's named aspect and unscoped with every quiet-delivery FLAG
adjudicated (MUST_STAY_QUIET, QUIET_ADJUDICATIONS). Commit timing in a shallow clone is NOT_APPLICABLE at the
grafted boundary, never counted.
"""
from __future__ import annotations

import ast
import gzip
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import common
import governance

SCHEMA = 'minifly.retro.v1'
ADAPTERS = ('R-LOCKTIME', 'R-CLOSURE', 'R-LITERAL', 'R-FAILOPEN', 'R-CLAIM', 'R-POINTER', 'R-CTRL', 'R-CATALOG',
            'R-AUTH', 'R-PATHS', 'R-INERT')
FLAG, PASS, NA = 'FLAG', 'PASS', 'NOT_APPLICABLE'
BULK = 'NOT_APPLICABLE_BULK_IMPORT'
LEARNED_BANK_REF = 'origin/charming-head'
LEARNED_BANK_DIR = 'minifly_learned_bank_v1_20261024'
MAX_BYTES = common.MAX_SCAN_BYTES
TEXT_EXT = ('.json', '.jsonl', '.md', '.py')
INFRA_TOP = ('.git', '.claude', '.github', 'tools', 'PROJECT_DOCUMENTS', 'studies', '__pycache__')

# SPEC 12.4 acceptance lists (embedded constants).
MUST_FLAG = (
    {'id': 'CTRL-minifly_b', 'adapter': 'R-CTRL', 'delivery': 'minifly_b',
     'what': 'minifly_b/results/science receipts (gz): N_old_rel identical to W'},
    {'id': 'CLOSURE-v6r1', 'adapter': 'R-CLOSURE', 'delivery': 'minifly_v6_r1_dev_20261023', 'match': 'audit_results.py',
     'what': 'audit_results.py outside SOURCE_LOCK'},
    {'id': 'LITERAL-v6r1', 'adapter': 'R-LITERAL', 'delivery': 'minifly_v6_r1_dev_20261023', 'match': 'audit_results.py',
     'what': 'audit_results.py line 97 literals'},
    {'id': 'FAILOPEN-lb', 'adapter': 'R-FAILOPEN', 'delivery': LEARNED_BANK_REF + ':' + LEARNED_BANK_DIR,
     'match': 'summarize.py', 'what': "LEARNED-BANK summarize.py 'is not False' and len(verdict)"},
    {'id': 'CLOSURE-lb', 'adapter': 'R-CLOSURE', 'delivery': LEARNED_BANK_REF + ':' + LEARNED_BANK_DIR,
     'what': 'LEARNED-BANK make_lock.py closure'},
    {'id': 'CLAIM-lb', 'adapter': 'R-CLAIM', 'delivery': LEARNED_BANK_REF + ':' + LEARNED_BANK_DIR,
     'match': 'onfirmation', 'what': "LEARNED-BANK 'Native confirmation'"},
    {'id': 'LOCKTIME-lb', 'adapter': 'R-LOCKTIME', 'delivery': LEARNED_BANK_REF + ':' + LEARNED_BANK_DIR,
     'what': 'LEARNED-BANK lock timing (main run 1.7 minutes after the lock)'},
    {'id': 'CLAIM-cls', 'adapter': 'R-CLAIM', 'delivery': 'minifly_cls_dev_20261017', 'match': 'largely',
     'what': "minifly_cls_dev_20261017 'largely met'"},
    {'id': 'POINTER-extmem', 'adapter': 'R-POINTER',
     'delivery': 'PROJECT_DOCUMENTS/CURRENT_EXTERNAL_MEMORY_EXPERIMENT_DESIGN.json', 'match': 'confirmation_passes',
     'what': 'CURRENT_EXTERNAL_MEMORY_EXPERIMENT_DESIGN.json confirmation_passes'},
    {'id': 'CATALOG-p036', 'adapter': 'R-CATALOG', 'delivery': 'PROJECT_DOCUMENTS/FULL151_VERSION_CENSUS_20261023',
     'match': 'FIXED_1', 'what': 'P036 +22.10pp without FIXED_1'},
    {'id': 'CATALOG-levels', 'adapter': 'R-CATALOG', 'delivery': 'PROJECT_DOCUMENTS/FULL151_VERSION_CENSUS_20261023',
     'match': 'evidence_level', 'what': '40 free-text evidence_level strings'},
    {'id': 'LOCKTIME-43d04b5e', 'adapter': 'R-LOCKTIME', 'commit': '43d04b5e', 'what': 'lock and results in 43d04b5e'},
    {'id': 'LOCKTIME-3065bb1a', 'adapter': 'R-LOCKTIME', 'commit': '3065bb1a', 'what': 'lock and results in 3065bb1a'},
    {'id': 'LOCKTIME-43b07e89', 'adapter': 'R-LOCKTIME', 'commit': '43b07e89', 'what': 'lock and results in 43b07e89'},
    {'id': 'AUTH-count', 'adapter': 'R-AUTH', 'aggregate': True,
     'what': '14 of 39 cycle deliveries without AUTHORIZATION'},
    {'id': 'PATHS-count', 'adapter': 'R-PATHS', 'aggregate': True, 'what': 'count of /Users paths'},
)
# --------------------------------------------------------------------------------------
# Quiet list (SPEC 12.4 "Must stay quiet") and how the false-flag limit is measured.
#
# SPEC 12.4 names, for two of the three quiet deliveries, the ASPECT that must stay quiet (rcenter "round-1
# statistics", SAMPLE-GEN "SELECTION_LOCK-before-confirm ordering"); the third (V51) is named as a whole and is a
# delivery of audit reconciliation records. A flag on that aspect is a false flag by definition (the spec says the
# aspect is sound). A flag elsewhere in the same delivery is not automatically false: it can be a true finding about
# something else. So acceptance is measured twice and both readings must hold (acceptance()):
#   scoped:   only the adapters that judge the named aspect, run on the aspect's own files ('aspect_paths',
#             relative to the delivery; None means the whole delivery). Every FLAG here counts as false.
#   unscoped: every applied adapter on the whole delivery. Each FLAG is matched against QUIET_ADJUDICATIONS below
#             (a reviewed true finding or a false positive, with evidence); an unmatched FLAG counts as false.
#             The raw rate (every FLAG counted as false) is reported as well.
# 'adapters' maps each scoped adapter to why it judges the aspect; 'excluded' says why every other adapter does
# not. 'path_only' adapters have no sound whole-delivery stand-in (their delivery-level row is dominated by files
# outside the aspect), so they count only through aspect rows, which run_retro always computes.
# R-PATHS and R-AUTH are repo-wide counts in the FLAG list ("count of /Users paths", "14 of 39 cycle
# deliveries"), not defect judgements of one delivery's aspect: scoring them as quiet-list flags would make the
# spec contradict itself (V51 holds /Users paths that the R-PATHS count must include).
COUNT_ADAPTERS = ('R-PATHS', 'R-AUTH')
_COUNT_WHY = 'repo-wide count in the FLAG list (SPEC 12.4), not a judgement of this aspect'
MUST_STAY_QUIET = (
    {'delivery': 'studies/rcenter-survivor-rounds-20261003', 'aspect': 'round-1 statistics',
     # the frozen terminal analysis, the registered statistical plan, the official-world receipts and the report
     'aspect_paths': ('results/', 'receipts/science/', 'protocol/ROUND1_DESIGN_CYCLE3.md',
                      'source/runtime/analyze.py', 'source/runtime/analysis_entry.py',
                      'source/runtime/validate_analysis.py', 'source/private_official_policy/analysis_bootstrap.py'),
     'adapters': {
         'R-FAILOPEN': 'reducers that turn the registered contrasts into gate verdicts (is-not-False, n_ok == len)',
         'R-INERT': 'two arm summaries with identical numbers would be a contrast without a manipulation',
         'R-CTRL': 'control receipts bit-identical to W would make the W-minus-control contrasts vacuous',
         'R-CLAIM': 'soft or confirm wording on the registered gates in the report and the final design',
         'R-LITERAL': 'a literal integrity boolean in the analysis code would stand in for a computed check'},
     'path_only': ('R-LITERAL',),
     'excluded': {
         'R-LOCKTIME': 'commit timing of the official lock: provenance, not the statistics',
         'R-CLOSURE': 'completeness of the source lock: code provenance, not the statistics',
         'R-CATALOG': 'catalog and memo quotation; the aspect has no catalog or memo',
         'R-POINTER': 'pointer files only', 'R-PATHS': _COUNT_WHY, 'R-AUTH': _COUNT_WHY}},
    {'delivery': 'addition_sample_generalization_dev_v2_20261011', 'aspect': 'SELECTION_LOCK-before-confirm ordering',
     'aspect_paths': ('results/',),
     'adapters': {
         'R-LOCKTIME': 'the only adapter that judges whether a lock was committed before the results it governs; '
                       'run with SELECTION_LOCK.json as the lock and the confirm artifacts as the results'},
     # results/SELECTION_LOCK.json sits under results/, so the generic lock/result split would call it a result
     'locktime': {'locks': r'^results/SELECTION_LOCK\.json$',
                  'results': r'^results/(CONFIRM_[^/]*|confirm/.*)$'},
     'excluded': {
         'R-CLOSURE': 'source-lock completeness, not the order of selection and confirmation',
         'R-LITERAL': 'literal booleans in code, not ordering', 'R-FAILOPEN': 'reducer code, not ordering',
         'R-CLAIM': 'wording, not ordering', 'R-CTRL': 'receipt digests, not ordering',
         'R-INERT': 'summary identity, not ordering', 'R-CATALOG': 'catalog quotation, not ordering',
         'R-POINTER': 'pointer files only', 'R-PATHS': _COUNT_WHY, 'R-AUTH': _COUNT_WHY}},
    {'delivery': 'PROJECT_DOCUMENTS/V51_CLAUDE_AUDIT_RECONCILIATION_20261020', 'aspect': 'audit reconciliation records',
     'aspect_paths': None,
     'adapters': {a: 'the spec names the whole delivery; every adapter that judges records or code applies'
                  for a in ADAPTERS if a not in COUNT_ADAPTERS + ('R-POINTER',)},
     'excluded': {'R-POINTER': 'pointer files only (V51 is a directory)', 'R-PATHS': _COUNT_WHY, 'R-AUTH': _COUNT_WHY}},
)
# Reviewed FLAGs on quiet deliveries outside their aspect (retro implementer's adjudication of 2026-10-24, read from
# the artifacts cited; annotations for the owner, never verdicts). 'match' strings must all occur in the row's
# reason or evidence paths, otherwise the FLAG is unadjudicated and counts as a false flag.
QUIET_ADJUDICATIONS = (
    {'delivery': 'studies/rcenter-survivor-rounds-20261003', 'check_id': 'R-LOCKTIME',
     'match': ('share commit 34246217',),
     'verdict': 'true_finding', 'aspect_related': False,
     'evidence': 'protocol/OFFICIAL_LOCK.json and the first official-world receipts receipts/science/310001/ are both '
                 'first added in commit 34246217 (2026-10-03 18:00 -0700), so git cannot show the official lock '
                 'before the official run. Mitigation: the frozen source and the three ROUND1_DESIGN_CYCLE*.md were '
                 'committed earlier in 630b8f65 (14:21 -0700), the lock self-reports created_utc '
                 '2026-10-03T21:42:46Z, and its execution_persistence field records '
                 "'verified_private_only_owner_authorized_GitHub_deferral' (private persistence is not visible in "
                 'this clone). This concerns timing, not the round-1 statistics.'},
    {'delivery': 'studies/rcenter-survivor-rounds-20261003', 'check_id': 'R-LITERAL',
     'match': ('operations/transport.py',),
     'verdict': 'false_positive', 'aspect_related': False,
     'evidence': 'every remaining literal is written after a raising check on the same code path (check-then-record), '
                 'e.g. operations/transport.py:191-193 asserts the manifest sha256 in the public tree and in the '
                 'private readback zip before recording private_readback_verified/github_tree_verified = True; '
                 'audits/CYCLE2_SOURCE_FINAL_CHECK.py:82 asserts the native modules are not imported before '
                 'no_native_imports = True (line 91). A failed check aborts before the literal is written, so these '
                 'are fail-closed, not the FM-14 fail-open shape. They still match LOCK-08 as written, and assert '
                 'is stripped under python -O. None sits in the round-1 analysis code (the aspect run passes).'},
    {'delivery': 'addition_sample_generalization_dev_v2_20261011', 'check_id': 'R-LITERAL',
     'match': ('ops/science_20261011/science.py', 'cycles.py'),
     'verdict': 'true_finding', 'aspect_related': False,
     'evidence': "FINAL_AUDIT.json integrity flags written as literals: ops/science_20261011/science.py:245 package() "
                 "records 'sources_verified': True and 'no_test_feedback': True with no check in that function; "
                 "cycles.py:164-165 records 'sources_verified': True and 'no_ODD_CYCLE_learning': True although "
                 "config.lock() (config.py:34-44) only hashes files and compares nothing. Same shape as the V6 "
                 "audit_results.py:97 incident (LOCK-08, FM-14). Unrelated to the SELECTION_LOCK ordering."},
    {'delivery': 'addition_sample_generalization_dev_v2_20261011', 'check_id': 'R-PATHS',
     'match': ('absolute local paths',), 'verdict': 'true_finding', 'aspect_related': False,
     'evidence': 'committed operations records carry absolute /Users/... paths of a local Downloads copy, e.g. '
                 'cycles/cycle2/operations/COV-R.task.json and cycles/cycle1_capture_failure/FAILURE.json; R-PATHS is '
                 'the repo-wide census of SPEC 12.4, not a judgement of the selection ordering.'},
    {'delivery': 'PROJECT_DOCUMENTS/V51_CLAUDE_AUDIT_RECONCILIATION_20261020', 'check_id': 'R-PATHS',
     'match': ('absolute local paths',), 'verdict': 'true_finding', 'aspect_related': False,
     'evidence': 'byte/check.py:3 and context/check_context.py:4 hard-code a /Users/... Downloads path; '
                 'PROJECT_DOCUMENTS/KB/ANALYSIS_20261024/UNDERSTAND_DIGEST.md:365 already records byte/check.py:3,27 '
                 'as a reproducibility defect. A census count, not a judgement of the reconciliation records.'},
)
PROSPECTIVE_ONLY = (
    ('minifly_stop_only_confirm_v1_20261023', 'LAD-02', 'STOP-ONLY weak comparator'),
    ('minifly_v5_1_dev_20261020', 'ADM-FIX', 'v5.1 port fixture density'),
    ('minifly_v5_1_dev_20261020', 'ADM-GATE', 'v5.1 plumbing gate without a negative fixture'),
)
ARCHIVE_DESIGNS_MIN = 20
RECALL_MIN = 0.90
FALSE_FLAG_MAX = 0.15

LITERAL_NAME_RE = re.compile(r'(_ok|_passed|_passes|overlap|_verified|_independent)$|^(no_|all_)')
# Standard-library keyword flags that end in _ok but carry no integrity meaning (mkdir(exist_ok=True), ...).
STDLIB_KW_FLAGS = ('exist_ok', 'missing_ok', 'dirs_exist_ok')
LOCK_NAME_RE = re.compile(r'(?i)(lock[^/]*\.json$)|((^|_)(design|spec|protocol|plan|prereg\w*)[^/]*\.(md|json)$)')
RESULT_NAME_RE = re.compile(r'(?i)^(summary|results?|verdict|scores|final_metrics|final_audit|metrics|report)[^/]*'
                            r'\.(json|jsonl|md|csv|gz)$')
SOFT_RE = re.compile(r'(?i)\b(largely|mostly|in part|nearly|promising|essentially|approved|'
                     r'met in part|substantially|broadly|close to|encouraging)\b')
# 'partial(ly)' is soft gate wording only next to a verdict word ("partially met", "partial PASS", "PASS (partial)");
# "PASS: four partial synthetic native events" describes the test input, not the gate.
_VERDICT_W = r'(?:pass\w*|met|success\w*|support\w*|confirm\w*|achiev\w*|satisf\w*|fulfil\w*|meets?)'
PARTIAL_SOFT_RE = re.compile(r'(?i)\bpartial(?:ly)?\b(?:[ \t]+\w+)?[ \t]+' + _VERDICT_W + r'\b|\b' + _VERDICT_W +
                             r'[ \t]*(?:\([ \t]*|[ \t]+)partial(?:ly)?\b')
# CLM-06 shape: soft words in a sentence that names a gate id, the primary gate or a gate verdict.
GATE_TOKEN_RE = re.compile(r'\bG\d+\b|\b[A-Z]{2,}-\d+\b|(?i:\bprimary[ _-]gate\b|\bprimary_gate\b)|\bPASS\b|\bFAIL\b|'
                           r'\bmet\b')
SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?;])\s+|\s+[|]\s+')
LEGIT_CONFIRM_RE = re.compile(r'(?i)\b(sealed|fresh)\b')
CONFIRM_RE = re.compile(r'(?i)\bconfirm\w*')
DEV_WORLD_RE = re.compile(r'\b(D\d+)\b|(?i:\b(dev|tuning|development)\b)')
# A confirm word counts as confirm wording ON a dev/tuning world only when the dev-world reference is bound to it in
# one clause. Designs routinely list DEV and confirmation as separate stages ("DEV and conditional confirmation",
# "DEV/CONFIRM receipts", "After confirmation, retire ... from future development"); those are not claims.
_DEVW = r'(?:D\d+|dev|tuning|development)'
_CLAUSE_STOP = r'(?:and|or|nor|then|than|before|after|vs|versus|not|no|but|to|into|plus)'
DEV_CONFIRM_BOUND_RES = (
    # a dev-world word modifying a confirm word: "development confirmation", "dev-confirmed", "D3 confirm run"
    re.compile(r'(?i)\b' + _DEVW + r'\b(?:[ \t-]+(?!' + _CLAUSE_STOP + r'\b)\w+){0,2}?[ \t-]+confirm\w*'),
    # a confirm word placed on/in/with a dev world: "confirmation on D2", "confirmed on the dev worlds"
    re.compile(r'(?i)\bconfirm\w*(?:[ \t]+(?!' + _CLAUSE_STOP + r'\b)[\w-]+){0,3}?[ \t]+'
               r'(?:on|in|at|with|using|from|over)[ \t]+(?:(?!' + _CLAUSE_STOP + r'\b)[\w-]+[ \t]+){0,3}?'
               + _DEVW + r'\b'),
    # a confirm word whose parenthetical names a dev world: "Native confirmation (T-PAR, D2, 150 passes)"
    re.compile(r'(?i)\bconfirm\w*[ \t]*\([^)]*\b' + _DEVW + r'\b'),
)
# Test code: literals there are fixture inputs (often deliberately invalid), not the delivery's integrity claims.
TEST_FILE_RE = re.compile(r'(?i)(^|/)(tests?|testing)/|(^|/)(test_[^/]*|[^/]*_tests?|conftest)\.py$')
# Base names that declare an object a deliberately invalid fixture: bad = copy.deepcopy(entry); bad['x_verified'] = True
NEGATIVE_FIXTURE_RE = re.compile(r'(?i)^(bad|bogus|broken|evil|tamper\w*|forg\w*|corrupt\w*|mutat\w*|fake\w*|'
                                 r'invalid\w*|wrong\w*)$')
# A technical-cycle directory inside a delivery (receipts/cycle3/, cycles/cycle1/, preflight/).
TECH_CYCLE_SEG_RE = re.compile(r'(?i)^(cycle[_-]?\d+[^/]*|preflight)$')
NEGATION_RE = re.compile(r"(?i)\b(not|no|never|without|cannot|isn't|wasn't|nor)\b[^.;:]{0,40}confirm")
POINTER_BOOL_RE = re.compile(r'(?i)(confirm|pass|passes|passed|success|qualified)')
ABS_PATH_RE = re.compile(rb'/Users/[^\s"\'<>]|/home/[A-Za-z][^\s"\'<>]*|(?<![A-Za-z0-9])[A-Za-z]:\\\\[A-Za-z]|'
                         rb'(?<![\w.~/-])/tmp/')
USERS_RE = re.compile(rb'/Users/[^\s"\'<>]')
FINDING_TOKEN_RE = re.compile(r'\b[A-Z][A-Z0-9]*_[A-Z0-9_]+\b')
EVIDENCE_ENUM = ('E0', 'E1', 'E2', 'E3', 'E4', 'NONE')
IDENTITY_KEY_RE = re.compile(r'(?i)^(arm|arms|name|label|id|branch|variant|condition|seed|path|file|files|utc|time|'
                             r'timestamp|created|created_utc|started|finished|wall|elapsed|duration|sha|sha256|digest|'
                             r'hash|run|run_id|attempt|attempt_id|pid|host|world|task|role)$|(_s|_sec|_seconds|_time|'
                             r'_utc|_sha256|_sha|_digest|_path)$')
NOISE_TOKENS = {'design', 'dev', 'confirm', 'pilot', 'review', 'trial', 'rehearsal', 'minifly', 'revised'}


# ======================================================================================
# Sources: the worktree (read from disk) and git refs (read through git, never checked out)
# ======================================================================================

class History:
    """First-add commits of every path in the available (possibly shallow) history."""

    def __init__(self, commits: List[dict], shallow: Set[str]):
        self.commits = commits                 # oldest first: {sha, ct, parents, added}
        self.shallow = shallow
        self.first_add: Dict[str, int] = {}
        self.dir_first: Dict[str, int] = {}
        for i, c in enumerate(commits):
            for p in c['added']:
                self.first_add.setdefault(p, i)
                d = governance.delivery_dir_of(p)
                if d:
                    self.dir_first.setdefault(d, i)
        self._bulk: Dict[int, Optional[str]] = {}

    def bulk_reason(self, i: int) -> Optional[str]:
        if i in self._bulk:
            return self._bulk[i]
        c = self.commits[i]
        why = None
        if c['sha'] in self.shallow:
            # git rev-parse --is-shallow-repository is true and this commit is a grafted boundary: the files it adds
            # may be much older, so no commit order can be read from it. Not counted (NOT_APPLICABLE).
            why = ('shallow clone: commit %s is a grafted boundary whose parents are absent from this clone, so the '
                   'files it adds may be older and their commit order cannot be read (git fetch --unshallow would '
                   'deepen it; retro never fetches)' % c['sha'][:8])
        elif not c['parents']:
            why = 'root commit %s: no earlier history' % c['sha'][:8]
        else:
            units: Set[frozenset] = set()
            designs: Set[str] = set()
            for p in c['added']:
                d = governance.delivery_dir_of(p)
                if d and self._new_dir(d, i):
                    units.add(lineage_stem(d.rsplit('/', 1)[-1]))
                base = p.rsplit('/', 1)[-1]
                if re.search(r'(?i)(design|protocol)[^/]*\.md$', base):
                    designs.add(base)
            if len(units) >= 3:
                why = '%s: commit %s adds %d delivery dirs' % (BULK, c['sha'][:8], len(units))
            elif len(designs) >= ARCHIVE_DESIGNS_MIN:
                # an archive commit: each distinct archived design document stands for a separate delivery
                why = '%s: commit %s adds %d distinct archived design documents' % (BULK, c['sha'][:8], len(designs))
        self._bulk[i] = why
        return why

    def _new_dir(self, d: str, i: int) -> bool:
        return self.dir_first.get(d, i) >= i


def lineage_stem(name: str) -> frozenset:
    toks = [t for t in re.split(r'[_\-.]', name.lower()) if t]
    keep = [t for t in toks if t not in NOISE_TOKENS and not re.fullmatch(r'\d{8}', t)]
    return frozenset(keep or toks)


def _read_shallow(root: Path) -> Set[str]:
    try:
        proc = common.run_git(root, ['rev-parse', '--git-path', 'shallow'])
        p = Path(proc.stdout.decode().strip())
        p = p if p.is_absolute() else Path(root) / p
        return set(ln.strip() for ln in p.read_text().splitlines() if ln.strip()) if p.is_file() else set()
    except (common.MissingInput, OSError):
        return set()


def load_history(root: Path, ref: str = 'HEAD') -> Optional[History]:
    try:
        proc = common.run_git(root, ['log', '--reverse', '--no-renames', '--diff-filter=A', '--name-only',
                                     '--format=%x1e%H%x1f%ct%x1f%P', ref], timeout=120)
    except common.MissingInput:
        return None
    commits = []
    for rec in proc.stdout.decode('utf-8', 'replace').split('\x1e'):
        if not rec.strip():
            continue
        head, _, rest = rec.partition('\n')
        parts = head.split('\x1f')
        if len(parts) < 3:
            continue
        try:
            ct = int(parts[1])
        except ValueError:
            ct = 0
        commits.append({'sha': parts[0].strip(), 'ct': ct, 'parents': parts[2].split(),
                        'added': [ln.strip() for ln in rest.splitlines() if ln.strip()]})
    # commits that add nothing are still useful for "next commit after the lock"
    try:
        proc = common.run_git(root, ['log', '--reverse', '--no-renames', '--name-only', '--format=%x1e%H%x1f%ct', ref],
                              timeout=120)
        touched = []
        for rec in proc.stdout.decode('utf-8', 'replace').split('\x1e'):
            if rec.strip():
                head, _, rest = rec.partition('\n')
                sha, _, ct = head.partition('\x1f')
                touched.append((sha.strip(), int(ct or 0), [ln.strip() for ln in rest.splitlines() if ln.strip()]))
    except (common.MissingInput, ValueError):
        touched = []
    h = History(commits, _read_shallow(root))
    h.touched = touched  # type: ignore[attr-defined]
    return h


class Source:
    label = 'abstract'

    def files(self, prefix: str) -> List[str]:
        raise NotImplementedError

    def exists(self, path: str) -> bool:
        raise NotImplementedError

    def read(self, path: str) -> Optional[bytes]:
        raise NotImplementedError

    def history(self) -> Optional[History]:
        raise NotImplementedError

    def delivery_label(self, d: str) -> str:
        return d


class WorktreeSource(Source):
    label = 'worktree'

    def __init__(self, root: Path, with_history: bool = True):
        self.root = Path(root).resolve()
        self._files: Dict[str, List[str]] = {}
        self._hist: Any = None if with_history else False

    def files(self, prefix: str) -> List[str]:
        prefix = prefix.strip('/')
        if prefix not in self._files:
            base = self.root / prefix if prefix else self.root
            out = []
            if base.is_file():
                out = [prefix]
            elif base.is_dir():
                for dirpath, dirnames, filenames in os.walk(base):
                    dirnames[:] = sorted(d for d in dirnames if d not in common.SKIP_DIR_NAMES)
                    for fn in filenames:
                        r = Path(dirpath, fn).relative_to(self.root).as_posix()
                        if not common._is_cache_junk(r) and not os.path.islink(os.path.join(dirpath, fn)):
                            out.append(r)
            self._files[prefix] = sorted(out)
        return list(self._files[prefix])

    def exists(self, path: str) -> bool:
        return (self.root / path).is_file()

    def read(self, path: str) -> Optional[bytes]:
        p = self.root / path
        try:
            if not p.is_file() or p.stat().st_size > MAX_BYTES:
                return None
            return p.read_bytes()
        except OSError:
            return None

    def history(self) -> Optional[History]:
        if self._hist is False:
            return None
        if self._hist is None:
            self._hist = load_history(self.root, 'HEAD') or False
        return self._hist or None


class RefSource(Source):
    """A git ref read through `git diff --name-only <empty> <ref>` and `git cat-file --batch`."""

    def __init__(self, root: Path, ref: str):
        self.root = Path(root).resolve()
        self.ref = ref
        self.label = ref
        self.git = common.RealGit(self.root)
        self._files: Optional[List[str]] = None
        self._cache: Dict[str, Optional[bytes]] = {}
        self._hist: Any = None

    def ref_exists(self) -> bool:
        try:
            proc = common.run_git(self.root, ['rev-parse', '--verify', '--quiet', self.ref + '^{commit}'], ok=(0, 1, 128))
            return proc.returncode == 0
        except common.MissingInput:
            return False

    def files(self, prefix: str) -> List[str]:
        if self._files is None:
            self._files = self.git.list_files(self.ref)
            self._set = set(self._files)
        prefix = prefix.strip('/')
        if not prefix:
            return list(self._files)
        pre = prefix + '/'
        return [p for p in self._files if p.startswith(pre) or p == prefix]

    def exists(self, path: str) -> bool:
        self.files('')
        return path in self._set

    def prefetch(self, paths: Sequence[str]) -> None:
        need = [p for p in paths if p not in self._cache]
        if need:
            self._cache.update(self.git.files_at(self.ref, need))

    def read(self, path: str) -> Optional[bytes]:
        if path not in self._cache:
            self.prefetch([path])
        data = self._cache.get(path)
        return data if data is None or len(data) <= MAX_BYTES else None

    def history(self) -> Optional[History]:
        if self._hist is None:
            self._hist = load_history(self.root, self.ref) or False
        return self._hist or None

    def delivery_label(self, d: str) -> str:
        return '%s:%s' % (self.ref, d)


# ======================================================================================
# Annotation rows
# ======================================================================================

def row(delivery: str, check_id: str, status: str, reason: str, evidence: Iterable[str] = ()) -> dict:
    return {'checker_sha256': _CHECKER_SHA[0], 'delivery': delivery, 'check_id': check_id, 'status': status,
            'reason': reason, 'evidence_paths': sorted(set(str(e) for e in evidence))[:50]}


_CHECKER_SHA = ['']


class Delivery:
    def __init__(self, src: Source, path: str, kind: str = 'dir'):
        self.src = src
        self.path = path.rstrip('/')
        self.kind = kind
        self.label = src.delivery_label(self.path)
        self._files: Optional[List[str]] = None
        self._ast: Dict[str, Any] = {}

    def files(self) -> List[str]:
        if self._files is None:
            self._files = [self.path] if self.kind == 'file' else self.src.files(self.path)
        return self._files

    def rel(self, p: str) -> str:
        return p[len(self.path) + 1:] if p.startswith(self.path + '/') else p

    def py_files(self) -> List[str]:
        return [p for p in self.files() if p.endswith('.py')]

    def tree(self, p: str) -> Any:
        if p not in self._ast:
            data = self.src.read(p)
            try:
                self._ast[p] = ast.parse(data.decode('utf-8', 'replace'), filename=p) if data is not None else None
            except (SyntaxError, ValueError):
                self._ast[p] = None
        return self._ast[p]


# ======================================================================================
# AST helpers (parsing only; nothing is executed)
# ======================================================================================

def _target_name(t: Any) -> Optional[str]:
    if isinstance(t, ast.Name):
        return t.id
    if isinstance(t, ast.Attribute):
        return t.attr
    if isinstance(t, ast.Subscript):
        sl = t.slice
        if type(sl).__name__ == 'Index':  # Python < 3.9 AST shape
            sl = getattr(sl, 'value', sl)
        if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
            return sl.value
    return None


def _is_literal_flag(v: Any) -> bool:
    if not isinstance(v, ast.Constant):
        return False
    x = v.value
    return x is True or x is False or (type(x) in (int, float) and x == 0)


def _base_name(t: Any) -> Optional[str]:
    """The variable a subscript/attribute target hangs off: bad['x']['y'] -> 'bad'."""
    while isinstance(t, (ast.Subscript, ast.Attribute)):
        t = t.value
    return t.id if isinstance(t, ast.Name) else None


def is_test_module(path: str, tree: Any) -> bool:
    """Test code: by path (tests/, test_*.py, *_test.py, conftest.py), a unittest.TestCase subclass or a pytest import.
    Importing unittest.mock alone does not make a module a test: legacy finalize/cycle scripts use mock.patch and
    still write FINAL_AUDIT literals (earned_authority_v1_20261007/finalize_exercise.py:45)."""
    if TEST_FILE_RE.search(path):
        return True
    if tree is None:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(a.name.split('.')[0] == 'pytest' for a in node.names):
            return True
        if isinstance(node, ast.ImportFrom) and (node.module or '').split('.')[0] == 'pytest' and not node.level:
            return True
        if isinstance(node, ast.ClassDef) and any(_name_of(b) == 'TestCase' for b in node.bases):
            return True
    return False


def literal_hits(tree: Any) -> List[Tuple[int, str, str]]:
    """LOCK-08 shape: a literal True/False/0/0.0 assigned to a name or key matching the integrity pattern.
    Skipped (genuine false positives of the shape on real history, never the V6 audit_results.py:97 kind):
    plain-name initializers that the module later reassigns with a computed value; dict/keyword/subscript
    initializers whose key the module later computes (stats = dict(no_write_events=0) ... s['no_write_events'] += 1);
    and literals written into a deliberately invalid fixture (bad = copy.deepcopy(e); bad['x_verified'] = True)."""
    hits: List[Tuple[int, str, str]] = []
    if tree is None:
        return hits
    name_assigns: Dict[str, List[Tuple[int, bool]]] = {}
    computed_keys: Set[str] = set()
    negative_values: Set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            flat = [tt for t in targets for tt in (t.elts if isinstance(t, (ast.Tuple, ast.List)) else [t])]
            if node.value is not None and flat and all(isinstance(tt, ast.Name) and NEGATIVE_FIXTURE_RE.match(tt.id)
                                                       for tt in flat):
                negative_values.add(id(node.value))
            for tt in flat:
                computed = isinstance(node, ast.AugAssign) or (node.value is not None and
                                                               not _is_literal_flag(node.value))
                if isinstance(tt, (ast.Subscript, ast.Attribute)) and computed:
                    k = _target_name(tt)
                    if k:
                        computed_keys.add(k)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            val = node.value
            for t in targets:
                for tt in (t.elts if isinstance(t, (ast.Tuple, ast.List)) else [t]):
                    n = _target_name(tt)
                    if n is None:
                        continue
                    lit = val is not None and _is_literal_flag(val) and not isinstance(node, ast.AugAssign)
                    if isinstance(tt, ast.Name):
                        name_assigns.setdefault(n, []).append((node.lineno, lit))
                    elif lit and LITERAL_NAME_RE.search(n) and n not in computed_keys \
                            and not NEGATIVE_FIXTURE_RE.match(_base_name(tt) or ''):
                        hits.append((node.lineno, n, ast.dump(val)))
        elif isinstance(node, (ast.For, ast.comprehension)):
            tgt = node.target
            for tt in (tgt.elts if isinstance(tgt, (ast.Tuple, ast.List)) else [tgt]):
                if isinstance(tt, ast.Name):
                    name_assigns.setdefault(tt.id, []).append((getattr(node, 'lineno', 0), False))
        elif isinstance(node, ast.Dict):
            if id(node) in negative_values:
                continue
            for k, v in zip(node.keys, node.values):
                if isinstance(k, ast.Constant) and isinstance(k.value, str) and LITERAL_NAME_RE.search(k.value) \
                        and _is_literal_flag(v) and k.value not in computed_keys:
                    hits.append((getattr(k, 'lineno', node.lineno), k.value, repr(v.value)))
        elif isinstance(node, ast.Call):
            if id(node) in negative_values:
                continue
            for kw in node.keywords:
                if kw.arg and kw.arg not in STDLIB_KW_FLAGS and LITERAL_NAME_RE.search(kw.arg) \
                        and _is_literal_flag(kw.value) and kw.arg not in computed_keys:
                    hits.append((getattr(kw.value, 'lineno', node.lineno), kw.arg, repr(kw.value.value)))
    for n, assigns in name_assigns.items():
        if not LITERAL_NAME_RE.search(n):
            continue
        lits = [ln for ln, lit in assigns if lit]
        if lits and all(lit for _, lit in assigns):
            for ln in lits:
                hits.append((ln, n, 'literal (never computed)'))
    return sorted(set(hits))


def _name_of(n: Any) -> Optional[str]:
    if isinstance(n, ast.Name):
        return n.id
    if isinstance(n, ast.Attribute):
        return n.attr
    return None


def _raise_guard_compares(tree: Any) -> Set[int]:
    """ids of Compare nodes in the test of `if <test>: raise ...` (not under a `not`). There `x is not False` is
    fail-closed: anything but an explicit False raises (if d.get('development') is not False: raise ...)."""
    out: Set[int] = set()

    def visit(n: Any, negated: bool) -> None:
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not):
            negated = not negated
        if isinstance(n, ast.Compare) and not negated:
            out.add(id(n))
        for c in ast.iter_child_nodes(n):
            visit(c, negated)
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and node.body and isinstance(node.body[0], ast.Raise):
            visit(node.test, False)
    return out


def failopen_hits(tree: Any) -> List[Tuple[int, str]]:
    hits: List[Tuple[int, str]] = []
    if tree is None:
        return hits
    guards = _raise_guard_compares(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            for op, comp in zip(node.ops, node.comparators):
                if isinstance(op, (ast.IsNot, ast.NotEq)) and isinstance(comp, ast.Constant) and comp.value is False \
                        and id(node) not in guards:
                    hits.append((node.lineno, "'is not False': None or a missing gate passes"))
            if len(node.ops) == 1 and isinstance(node.ops[0], ast.Eq):
                pairs = ((node.left, node.comparators[0]), (node.comparators[0], node.left))
                for a, b in pairs:
                    if isinstance(b, ast.Call) and _name_of(b.func) == 'len' and _name_of(a) and \
                            re.search(r'(?i)(ok|pass|good|success)', _name_of(a) or ''):
                        hits.append((node.lineno, "'%s == len(...)': all-of-N passes for any N, including 1"
                                     % _name_of(a)))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'get' \
                and len(node.args) == 2 and isinstance(node.args[1], ast.Constant) and node.args[1].value is True \
                and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str) \
                and re.search(r'(?i)(gate|ok|pass|valid)', node.args[0].value):
            hits.append((node.lineno, ".get(%r, True): a missing gate defaults to pass" % node.args[0].value))
    return sorted(set(hits))


def imports_of(tree: Any) -> Tuple[List[Tuple[str, Optional[str], int]], List[List[str]]]:
    """([(module, imported_name_or_None, level)], [string constants of sys.path.insert/append arguments])."""
    mods: List[Tuple[str, Optional[str], int]] = []
    paths: List[List[str]] = []
    if tree is None:
        return mods, paths
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                mods.append((a.name, None, 0))
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                mods.append((node.module or '', a.name, node.level or 0))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ('insert', 'append') \
                and isinstance(node.func.value, ast.Attribute) and node.func.value.attr == 'path' \
                and _name_of(node.func.value.value) == 'sys':
            consts = [n.value for n in ast.walk(node) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
            if consts:
                paths.append(consts)
    return mods, paths


# ======================================================================================
# Adapters
# ======================================================================================

def _norm_join(base_dir: str, rel: str) -> Optional[str]:
    parts: List[str] = [p for p in base_dir.split('/') if p]
    for p in rel.replace('\\', '/').split('/'):
        if p in ('', '.'):
            continue
        if p == '..':
            if not parts:
                return None
            parts.pop()
        else:
            parts.append(p)
    return '/'.join(parts)


def _lock_entries(d: Delivery, p: str) -> Dict[str, str]:
    """{path key: 64-hex sha256} entries anywhere in JSON file p (keys whose basename has an extension)."""
    data = d.src.read(p)
    try:
        obj = json.loads(data.decode('utf-8')) if data else None
    except (ValueError, UnicodeDecodeError):
        return {}
    found: Dict[str, str] = {}

    def walk(o: Any) -> None:
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(k, str) and isinstance(v, str) and re.fullmatch(r'[0-9a-f]{64}', v) \
                        and '.' in k.rsplit('/', 1)[-1]:
                    found[k] = v
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(obj)
    return found


def _key_candidates(d: Delivery, lock_path: str, k: str) -> List[str]:
    k = k.replace('\\', '/')
    lock_dir = lock_path.rsplit('/', 1)[0]
    cands = [_norm_join(lock_dir, k), _norm_join(d.path, k), common.norm_rel(k) if not k.startswith('/') and
             '..' not in k.split('/') else None]
    return [c for c in cands if c]


def lock_hashes(d: Delivery, p: str) -> Set[str]:
    """Candidate repo paths of the files that lock file p hashes ({path: 64-hex sha256} entries anywhere in it)."""
    out: Set[str] = set()
    for k in _lock_entries(d, p):
        out.update(_key_candidates(d, p, k))
    return out


def lock_closure_hashes(d: Delivery, p: str, depth: int = 2) -> Set[str]:
    """lock_hashes(p) plus the entries of hashed JSON closure manifests whose bytes still match the hash the lock
    records for them (SOURCE_LOCK.json -> PARENT_CLOSURE.json -> parent modules), followed to `depth` levels. A
    manifest whose bytes differ from the recorded hash is not followed."""
    out: Set[str] = set()
    todo = [(p, 0)]
    seen: Set[str] = set()
    while todo:
        f, lvl = todo.pop()
        if f in seen:
            continue
        seen.add(f)
        for k, hv in _lock_entries(d, f).items():
            cands = _key_candidates(d, f, k)
            out.update(cands)
            if lvl >= depth or not k.endswith('.json'):
                continue
            for c in cands:
                data = d.src.read(c)
                if data is not None and hashlib.sha256(data).hexdigest() == hv:
                    todo.append((c, lvl + 1))
                    break
    return out


def locked_paths(d: Delivery) -> Tuple[Set[str], List[str]]:
    """Repo paths hashed by the delivery's source/code locks (and by the closure manifests they hash), and the
    lock files used."""
    out: Set[str] = set()
    used: List[str] = []
    lock_files = [p for p in d.files() if re.search(r'(?i)lock[^/]*\.json$', p.rsplit('/', 1)[-1])]
    primary = [p for p in lock_files if p.rsplit('/', 1)[-1] in ('SOURCE_LOCK.json', 'LOCK.json')]
    # the main pre-run lock wins: post-hoc locks (AUDIT_SOURCE_LOCK.json, ...) never extend its closure
    for p in (primary or lock_files):
        found = lock_closure_hashes(d, p)
        if found:
            used.append(p)
            out |= found
    return out, used


def _resolve_module(src: Source, search: List[str], mod: str, name: Optional[str]) -> List[str]:
    out = []
    parts = [x for x in mod.split('.') if x]
    for base in search:
        for cand_parts in ([parts + [name]] if name else []) + [parts]:
            if not cand_parts:
                continue
            stem = '/'.join([base] + cand_parts if base else cand_parts)
            for c in (stem + '.py', stem + '/__init__.py'):
                if src.exists(c):
                    out.append(c)
                    break
            if out:
                return out
    return out


def adapter_closure(d: Delivery) -> List[dict]:
    locked, used = locked_paths(d)
    if not used:
        return [row(d.label, 'R-CLOSURE', NA, 'no source/code lock with file hashes in this delivery')]
    src = d.src
    entries = sorted(p for p in locked if p.endswith('.py') and p.startswith(d.path + '/') and src.exists(p))
    seen: Set[str] = set()
    outside: Dict[str, str] = {}
    queue = list(entries)
    extra_dirs: List[str] = []
    steps = 0
    while queue and steps < 400:
        steps += 1
        f = queue.pop(0)
        if f in seen:
            continue
        seen.add(f)
        tree = d.tree(f) if f in d.files() else _parse(src, f)
        mods, spaths = imports_of(tree)
        fdir = f.rsplit('/', 1)[0] if '/' in f else ''
        for consts in spaths:
            for s in consts:
                for base in (fdir, fdir.rsplit('/', 1)[0] if '/' in fdir else '', ''):
                    c = _norm_join(base, s)
                    if c and src.files(c) and c not in extra_dirs:
                        extra_dirs.append(c)
        search = [fdir] + [x for x in extra_dirs if x != fdir] + [d.path]
        for mod, name, level in mods:
            if level:
                base = fdir
                for _ in range(level - 1):
                    base = base.rsplit('/', 1)[0] if '/' in base else ''
                res = _resolve_module(src, [base], mod, name)
            else:
                res = _resolve_module(src, search, mod, name)
            for r in res:
                if r not in locked and r not in outside:
                    outside[r] = f
                if r not in seen:
                    queue.append(r)
    # Rule B: verdict/gate-computing scripts outside the lock
    verdict_scripts = []
    for p in d.py_files():
        if p in locked or re.search(r'(^|/)(tests?/|test_[^/]*\.py$)', p):
            continue
        tree = d.tree(p)
        if tree is None:
            continue
        writes = any(isinstance(n, ast.Constant) and isinstance(n.value, str) and
                     re.search(r'(?i)(summary|verdict|final_audit|gates?)[^/"\']*\.json', n.value) for n in ast.walk(tree))
        assigns_verdict = any(isinstance(n, (ast.Assign,)) and any(_target_name(t) in ('verdict', 'overall', 'gates_ok')
                                                                   for t in n.targets) for n in ast.walk(tree))
        if writes and assigns_verdict:
            verdict_scripts.append(p)
    reasons = []
    if outside:
        reasons.append('modules reachable from locked code but not in %s: %s'
                       % (', '.join(d.rel(u) for u in used), ', '.join('%s (imported by %s)' % (k, d.rel(v))
                                                                    for k, v in sorted(outside.items())[:8])))
    if verdict_scripts:
        reasons.append('verdict/gate-computing scripts outside the lock: %s'
                       % ', '.join(d.rel(p) for p in verdict_scripts[:8]))
    if reasons:
        return [row(d.label, 'R-CLOSURE', FLAG, '; '.join(reasons), list(outside) + verdict_scripts + used)]
    return [row(d.label, 'R-CLOSURE', PASS, 'locked closure of %d entry files complete; no verdict script outside %s'
                % (len(entries), ', '.join(d.rel(u) for u in used)), used)]


def _parse(src: Source, p: str) -> Any:
    data = src.read(p)
    try:
        return ast.parse(data.decode('utf-8', 'replace'), filename=p) if data is not None else None
    except (SyntaxError, ValueError):
        return None


def adapter_literal(d: Delivery) -> List[dict]:
    pys = d.py_files()
    if not pys:
        return [row(d.label, 'R-LITERAL', NA, 'no Python files')]
    hits, unparsed, tests = [], 0, 0
    for p in pys:
        t = d.tree(p)
        if t is None:
            unparsed += 1
            continue
        if is_test_module(d.rel(p), t):
            tests += 1  # fixture literals in test code are inputs, not the delivery's integrity claims
            continue
        for ln, name, val in literal_hits(t):
            hits.append('%s:%d %s=%s' % (p, ln, name, val[:20]))
    note = (' (%d unparseable files)' % unparsed) if unparsed else ''
    note += (' (%d test modules skipped)' % tests) if tests else ''
    checked = len(pys) - unparsed - tests
    if hits:
        return [row(d.label, 'R-LITERAL', FLAG, 'integrity booleans written as literals: %s%s'
                    % ('; '.join(hits[:8]), note), [h.split(':')[0] for h in hits])]
    if checked == 0:
        return [row(d.label, 'R-LITERAL', NA, 'no non-test Python files%s' % note)]
    return [row(d.label, 'R-LITERAL', PASS, 'no literal integrity booleans in %d Python files%s' % (checked, note))]


def adapter_failopen(d: Delivery) -> List[dict]:
    pys = d.py_files()
    if not pys:
        return [row(d.label, 'R-FAILOPEN', NA, 'no Python files')]
    hits = []
    for p in pys:
        for ln, what in failopen_hits(d.tree(p)):
            hits.append('%s:%d %s' % (p, ln, what))
    if hits:
        return [row(d.label, 'R-FAILOPEN', FLAG, 'fail-open reducer patterns: %s' % '; '.join(hits[:8]),
                    [h.split(':')[0] for h in hits])]
    return [row(d.label, 'R-FAILOPEN', PASS, 'no fail-open reducer pattern in %d Python files' % len(pys))]


def dev_confirm_wording(ln: str, in_cycle: bool) -> bool:
    """Confirm wording ON a dev/tuning world: a confirm word, not negated and not sealed/fresh, whose dev-world
    reference is bound to it in one clause (DEV_CONFIRM_BOUND_RES). In a cycle note, a heading that names a
    confirmation and no dev world at all also counts (a legacy cycle is never SEALED_CONFIRM); a heading that lists
    DEV and confirmation as separate stages ("## 10. Finite DEV and confirmation roster") does not."""
    if not CONFIRM_RE.search(ln) or NEGATION_RE.search(ln) or LEGIT_CONFIRM_RE.search(ln):
        return False
    if any(rx.search(ln) for rx in DEV_CONFIRM_BOUND_RES):
        return True
    return bool(in_cycle and ln.lstrip().startswith('#') and not DEV_WORLD_RE.search(ln))


def adapter_claim(d: Delivery) -> List[dict]:
    mds = [p for p in d.files() if p.lower().endswith('.md')]
    if not mds:
        return [row(d.label, 'R-CLAIM', NA, 'no Markdown claim text')]
    soft, conf = [], []
    for p in mds:
        data = d.src.read(p)
        if data is None:
            continue
        in_cycle = bool(re.search(r'(?i)(^|/)cycles?/|/cycle[^/]*\.md$', p))
        for i, ln in enumerate(data.decode('utf-8', 'replace').splitlines(), 1):
            for sent in SENTENCE_SPLIT_RE.split(ln):
                if (SOFT_RE.search(sent) or PARTIAL_SOFT_RE.search(sent)) and GATE_TOKEN_RE.search(sent):
                    soft.append('%s:%d %s' % (p, i, sent.strip()[:100]))
                    break
            if dev_confirm_wording(ln, in_cycle):
                conf.append('%s:%d %s' % (p, i, ln.strip()[:100]))
    reasons = []
    if soft:
        reasons.append('soft wording on gate lines: %s' % '; '.join(soft[:5]))
    if conf:
        reasons.append('confirm wording on dev/tuning worlds (legacy worlds are never SEALED_CONFIRM): %s'
                       % '; '.join(conf[:5]))
    if reasons:
        return [row(d.label, 'R-CLAIM', FLAG, ' | '.join(reasons), [x.split(':')[0] for x in soft + conf])]
    return [row(d.label, 'R-CLAIM', PASS, 'no soft gate wording or dev-world confirm wording in %d Markdown files'
                % len(mds))]


def adapter_pointer(d: Delivery) -> List[dict]:
    data = d.src.read(d.path)
    try:
        obj = json.loads(data.decode('utf-8')) if data else None
    except (ValueError, UnicodeDecodeError):
        return [row(d.label, 'R-POINTER', FLAG, 'pointer does not parse as JSON', [d.path])]
    hits: List[str] = []

    def walk(o: Any, path: str) -> None:
        if isinstance(o, dict):
            for k, v in o.items():
                kp = '%s/%s' % (path, k)
                if isinstance(v, bool) and POINTER_BOOL_RE.search(str(k)):
                    hits.append('%s: %s' % (kp, json.dumps(v)))
                walk(v, kp)
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, '%s/%d' % (path, i))
    walk(obj, '')
    if hits:
        return [row(d.label, 'R-POINTER', FLAG, 'bare boolean keys in a pointer (PTR-03): %s' % ', '.join(hits[:8]),
                    [d.path])]
    return [row(d.label, 'R-POINTER', PASS, 'no bare confirm/pass/success boolean keys', [d.path])]


DIGEST_KEYS = ('end_state_digests', 'end_graph_digests', 'state_digests', 'channel_digests')


def _receipt_branch_digests(text: str) -> Optional[Dict[str, str]]:
    """Per-branch end digests of a receipt, decoding only that member (receipts carry large row arrays)."""
    dec = json.JSONDecoder()
    for key in DIGEST_KEYS:
        i = text.find('"%s"' % key)
        while i >= 0:
            j = text.find(':', i + len(key) + 2)
            if j < 0:
                break
            k = j + 1
            while k < len(text) and text[k] in ' \t\r\n':
                k += 1
            try:
                v, _ = dec.raw_decode(text, k)
            except ValueError:
                v = None
            if isinstance(v, dict) and 'W' in v and len(v) >= 2:
                return {b: json.dumps(x, sort_keys=True) for b, x in v.items()}
            i = text.find('"%s"' % key, i + 1)
    return None


def adapter_ctrl(d: Delivery) -> List[dict]:
    cands = [p for p in d.files() if (d.rel(p).startswith('results/') or '/results/' in d.rel(p))
             and p.endswith(('.json', '.json.gz')) and ('receipt' in p.lower() or '/science/' in p
                                                        or '/receipts/' in p or p.endswith('.gz'))]
    if not cands:
        return [row(d.label, 'R-CTRL', NA, 'no per-branch control receipts')]
    same: Dict[str, int] = {}
    present: Dict[str, int] = {}
    n = 0
    examples: List[str] = []
    for p in cands:
        data = d.src.read(p)
        if data is None:
            continue
        try:
            if p.endswith('.gz'):
                data = gzip.decompress(data)
            text = data.decode('utf-8')
        except (OSError, ValueError, UnicodeDecodeError, EOFError):
            continue
        dig = _receipt_branch_digests(text)
        if dig is None:
            continue
        n += 1
        for b, v in dig.items():
            if b == 'W':
                continue
            present[b] = present.get(b, 0) + 1
            if v == dig['W']:
                same[b] = same.get(b, 0) + 1
                if len(examples) < 5:
                    examples.append(p)
    if n == 0:
        return [row(d.label, 'R-CTRL', NA, 'no receipt with per-branch end digests including W')]
    if same:
        detail = ', '.join('%s identical to W in %d/%d receipts' % (b, c, present[b]) for b, c in sorted(same.items()))
        return [row(d.label, 'R-CTRL', FLAG, 'control/lesion branches end bit-identical to W (manipulation not '
                    'delivered): %s' % detail, examples)]
    return [row(d.label, 'R-CTRL', PASS, '%d receipts: every non-W branch ends different from W' % n)]


def _kb_narrowed(kb_root: Path) -> List[Tuple[str, List[str], List[str]]]:
    """[(finding id, number strings of its claim, caveat tokens)] for narrowed or superseded findings."""
    kb = common.KB(kb_root)
    try:
        rows = kb.rows('FINDINGS')
    except (common.MissingInput, common.InvalidInput):
        return []
    out = []
    for r in rows:
        if r.get('kind', 'finding') != 'finding' or not r.get('id'):
            continue
        try:
            st = kb.finding_status(r['id'])
        except (common.MissingInput, common.InvalidInput):
            continue
        if st not in ('narrowed', 'superseded'):
            continue
        claim = ' '.join(str(x) for x in (r.get('claim'), (r.get('effect') or {}).get('text')) if x)
        nums = sorted(sorted(set(re.findall(r'[+-]?\d+\.\d+', claim))), key=len, reverse=True)[:2]
        toks: List[str] = []
        for rr in rows:
            if rr.get('refers_to') == r['id'] and rr.get('kind') in ('narrowing', 'supersession'):
                toks += FINDING_TOKEN_RE.findall(str(rr.get('caveat') or '') + ' ' + str(rr.get('claim') or ''))
        toks += FINDING_TOKEN_RE.findall(str(r.get('caveat') or ''))
        toks = [t for t in dict.fromkeys(toks) if re.search(r'\d', t)] or list(dict.fromkeys(toks))[:1]
        if nums and toks:
            out.append((r['id'], nums, toks))
    return out


def adapter_catalog(d: Delivery, kb_root: Optional[Path]) -> List[dict]:
    docs = [p for p in d.files() if re.search(r'(?i)(^|/)[^/]*(catalog|memo)[^/]*\.(md|json)$', p)]
    if not docs:
        return [row(d.label, 'R-CATALOG', NA, 'no catalog or memo')]
    out = []
    narrowed = _kb_narrowed(kb_root) if kb_root else []
    hits = []
    for p in docs:
        if not p.lower().endswith('.md'):
            continue
        data = d.src.read(p)
        if data is None:
            continue
        for i, ln in enumerate(data.decode('utf-8', 'replace').splitlines(), 1):
            for fid, nums, toks in narrowed:
                # a number quote, not a substring of another number (fpsyg.2022.1032003 is not +22.10)
                if any(re.search(r'(?<![\d.])' + re.escape(n.lstrip('+')) + r'(?![\d])', ln) for n in nums) \
                        and not any(t in ln for t in toks):
                    hits.append('%s:%d quotes %s (%s) without its narrowing caveat %s' % (p, i, fid, nums[0], toks[0]))
    if narrowed:
        if hits:
            out.append(row(d.label, 'R-CATALOG', FLAG, 'narrowed finding quoted without caveat: %s' % '; '.join(hits[:5]),
                           [h.split(':')[0] for h in hits]))
        else:
            out.append(row(d.label, 'R-CATALOG', PASS, 'every quote of a narrowed finding carries its caveat'))
    else:
        out.append(row(d.label, 'R-CATALOG', NA, 'no narrowed finding in KB FINDINGS to trace'))
    levels: Dict[str, int] = {}
    lvl_files = []
    for p in docs:
        if not p.lower().endswith('.json'):
            continue
        data = d.src.read(p)
        try:
            obj = json.loads(data.decode('utf-8')) if data else None
        except (ValueError, UnicodeDecodeError):
            continue
        found = []

        def walk(o: Any) -> None:
            if isinstance(o, dict):
                v = o.get('evidence_level')
                if isinstance(v, str):
                    found.append(v)
                for x in o.values():
                    walk(x)
            elif isinstance(o, list):
                for x in o:
                    walk(x)
        walk(obj)
        if found:
            lvl_files.append(p)
            for v in found:
                levels[v] = levels.get(v, 0) + 1
    if levels:
        free = sorted(v for v in levels if v not in EVIDENCE_ENUM)
        if free:
            out.append(row(d.label, 'R-CATALOG', FLAG, '%d distinct free-text evidence_level strings (closed enum is %s): '
                           'e.g. %s' % (len(free), '|'.join(EVIDENCE_ENUM), '; '.join(free[:3])), lvl_files))
        else:
            out.append(row(d.label, 'R-CATALOG', PASS, 'evidence_level values use the closed enum', lvl_files))
    return out


def adapter_auth(d: Delivery) -> List[dict]:
    rels = [d.rel(p) for p in d.files()]
    cyc = any(re.match(r'(?i)cycles?/', r) or ('/' not in r and re.match(r'(?i)cycle', r)) for r in rels)
    if not cyc:
        return [row(d.label, 'R-AUTH', NA, 'no cycles in this delivery')]
    auth = [r for r in rels if 'authorization' in r.rsplit('/', 1)[-1].lower()]
    if auth:
        return [row(d.label, 'R-AUTH', PASS, 'cycle-bearing delivery with %s' % ', '.join(auth[:3]),
                    [d.path + '/' + a for a in auth[:5]])]
    return [row(d.label, 'R-AUTH', FLAG, 'cycle-bearing delivery without any AUTHORIZATION record')]


def adapter_paths(d: Delivery) -> List[dict]:
    files = [p for p in d.files() if p.endswith(TEXT_EXT)]
    if not files:
        return [row(d.label, 'R-PATHS', NA, 'no text files in the bounded scan patterns')]
    hit_files, users = [], 0
    for p in files:
        data = d.src.read(p)
        if data is None:
            continue
        if ABS_PATH_RE.search(data):
            hit_files.append(p)
            users += len(USERS_RE.findall(data))
    scope = 'scan scope: %d files matching %s, files over %d bytes skipped' % (len(files), ','.join(TEXT_EXT), MAX_BYTES)
    if hit_files:
        r = row(d.label, 'R-PATHS', FLAG, '%d files with absolute local paths (%d /Users/ occurrences); %s'
                % (len(hit_files), users, scope), hit_files)
        r['_users'] = users
        r['_files'] = len(hit_files)
        return [r]
    return [row(d.label, 'R-PATHS', PASS, 'no absolute local paths; ' + scope)]


def _normalize_summary(o: Any) -> Any:
    if isinstance(o, dict):
        return {k: _normalize_summary(v) for k, v in sorted(o.items()) if not IDENTITY_KEY_RE.search(str(k))}
    if isinstance(o, list):
        return [_normalize_summary(v) for v in o]
    return o


def _numeric_leaves(o: Any) -> int:
    if isinstance(o, bool):
        return 0
    if isinstance(o, (int, float)):
        return 1
    if isinstance(o, dict):
        return sum(_numeric_leaves(v) for v in o.values())
    if isinstance(o, list):
        return sum(_numeric_leaves(v) for v in o)
    return 0


# A file-name token that names a world, job, attempt or content hash rather than an arm (202610129001, 7d18564a...).
ID_TOKEN_RE = re.compile(r'(?i)^\d+$|\d{6,}|[0-9a-f]{8,}')


def _arm_variants(a: str, b: str) -> bool:
    """File names that differ in exactly one '_'-separated token, and that token names an arm (PAR_LEARN_D2 vs
    PAR_FROZEN_D2). Two worlds' or jobs' certificates (202610129001_X vs 202610129032_X) are not arms: their
    count-only content is identical by design."""
    ta = re.split(r'_', a.rsplit('/', 1)[-1].rsplit('.', 1)[0])
    tb = re.split(r'_', b.rsplit('/', 1)[-1].rsplit('.', 1)[0])
    diff = [(x, y) for x, y in zip(ta, tb) if x != y]
    return len(ta) == len(tb) >= 2 and len(diff) == 1 and not any(ID_TOKEN_RE.search(t) for t in diff[0])


def adapter_inert(d: Delivery) -> List[dict]:
    res = [p for p in d.files() if (d.rel(p).startswith('results/') or '/results/' in d.rel(p)) and p.endswith('.json')
           and '/receipts/' not in '/' + d.rel(p) and not p.endswith('.audit.json')]
    if not res:
        return [row(d.label, 'R-INERT', NA, 'no JSON result summaries')]
    groups: Dict[Tuple[str, str], List[str]] = {}
    checked = 0
    for p in res:
        data = d.src.read(p)
        if data is None or len(data) > 2 * 1024 * 1024:
            continue
        try:
            obj = json.loads(data.decode('utf-8'))
        except (ValueError, UnicodeDecodeError):
            continue
        norm = _normalize_summary(obj)
        if _numeric_leaves(norm) < 3:
            continue
        checked += 1
        h = hashlib.sha256(json.dumps(norm, sort_keys=True).encode()).hexdigest()
        groups.setdefault((p.rsplit('/', 1)[0], h), []).append(p)
    same = []
    for v in groups.values():
        pairs = [(a, b) for i, a in enumerate(v) for b in v[i + 1:] if _arm_variants(a, b)]
        if pairs:
            same.append(sorted(set(x for pr in pairs for x in pr)))
    if same:
        detail = '; '.join(' = '.join(d.rel(x) for x in v[:3]) for v in same[:5])
        return [row(d.label, 'R-INERT', FLAG, 'arm/run summaries with identical numeric content in one results dir: %s'
                    % detail, [x for v in same for x in v])]
    if checked == 0:
        return [row(d.label, 'R-INERT', NA, 'no result summary with numeric content')]
    return [row(d.label, 'R-INERT', PASS, '%d numeric result summaries, no two arm variants identical within a '
                'directory' % checked)]


def _tech_cycle_prefix(rel: str) -> Optional[str]:
    """'receipts/cycle3' for receipts/cycle3/pilot/x.json: the first technical-cycle directory on the path."""
    parts = rel.split('/')
    for i, seg in enumerate(parts[:-1]):
        if TECH_CYCLE_SEG_RE.match(seg):
            return '/'.join(parts[:i + 1])
    return None


def _lock_and_result_paths(d: Delivery, siblings: Sequence[Delivery]) -> Tuple[List[str], List[str], List[str]]:
    """(locks, governed results, lock-bound technical-cycle receipts).

    A result inside a technical-cycle directory (receipts/cycle3/, cycles/cycle1/, preflight/) of which the
    delivery's own lock hashes at least one file is an input the lock accepted (the MRK-1 typed-cycle order:
    technical cycles precede the lock), not a result the lock governs, so it is not timed against the lock. A lock
    that hashes non-cycle results gains nothing from this: those stay governed results."""
    locks, docs, results, own_locks = [], [], [], []
    for p in d.files():
        rel = d.rel(p)
        name = rel.rsplit('/', 1)[-1]
        if '/results/' in '/' + rel or rel.startswith(('results/', 'receipts/')) or '/receipts/' in '/' + rel \
                or RESULT_NAME_RE.match(name):
            results.append(p)
        elif re.search(r'(?i)lock[^/]*\.json$', name):
            locks.append(p)
            own_locks.append(p)
        elif LOCK_NAME_RE.search(name):
            docs.append(p)
    for s in siblings:
        for p in s.files():
            name = p.rsplit('/', 1)[-1]
            if re.search(r'(?i)lock[^/]*\.json$', name):
                locks.append(p)
            elif LOCK_NAME_RE.search(name):
                docs.append(p)
    bound: Set[str] = set()
    for lp in own_locks:
        for hp in lock_hashes(d, lp):
            if hp.startswith(d.path + '/'):
                pre = _tech_cycle_prefix(d.rel(hp))
                if pre:
                    bound.add(pre)
    governed = [p for p in results if _tech_cycle_prefix(d.rel(p)) not in bound]
    lock_bound = [p for p in results if bound and _tech_cycle_prefix(d.rel(p)) in bound]
    # an explicit lock file is the lock; design/spec documents stand in only when no lock file exists
    return (locks or docs), governed, lock_bound


def adapter_locktime(d: Delivery, siblings: Sequence[Delivery]) -> List[dict]:
    h = d.src.history()
    if h is None:
        return [row(d.label, 'R-LOCKTIME', NA, 'git history unavailable')]
    locks, results, lock_bound = _lock_and_result_paths(d, siblings)
    if not locks:
        return [row(d.label, 'R-LOCKTIME', NA, 'no lock, design or spec artifact')]
    note = ''
    if lock_bound:
        dirs = sorted(set(_tech_cycle_prefix(d.rel(p)) or '' for p in lock_bound))
        note = ('; %d technical-cycle receipts in %s are hashed by the lock as accepted inputs and are not timed '
                'against it' % (len(lock_bound), ', '.join(dirs[:4])))
    if not results:
        return [row(d.label, 'R-LOCKTIME', NA, 'no result artifact' + (' governed by the lock' + note if note else ''))]
    return locktime_between(d, h, locks, results, note)


def locktime_between(d: Delivery, h: History, locks: Sequence[str], results: Sequence[str],
                     note: str = '') -> List[dict]:
    """Commit order of the first lock artifact and the first result artifact (first-add commits)."""
    if not locks:
        return [row(d.label, 'R-LOCKTIME', NA, 'no lock artifact' + note)]
    if not results:
        return [row(d.label, 'R-LOCKTIME', NA, 'no result artifact' + note)]
    li = [h.first_add[p] for p in locks if p in h.first_add]
    ri = [h.first_add[p] for p in results if p in h.first_add]
    if not li or not ri:
        return [row(d.label, 'R-LOCKTIME', NA, 'lock or results not added in the available history' + note)]
    cl, cr = min(li), min(ri)
    for i in sorted({cl, cr}):
        why = h.bulk_reason(i)
        if why:
            which = sorted(set(p for p in list(locks) + list(results) if h.first_add.get(p) == i))
            return [row(d.label, 'R-LOCKTIME', NA, '%s; first-added there: %s%s'
                        % (why, ', '.join(d.rel(p) for p in which[:3]) + (' (+%d more)' % (len(which) - 3)
                                                                         if len(which) > 3 else ''), note),
                        [h.commits[i]['sha']] + which[:3])]
    lc, rc = h.commits[cl], h.commits[cr]
    gap = ''
    touched = getattr(h, 'touched', [])
    pre = d.path + '/'
    for idx, (sha, ct, paths) in enumerate(touched):
        if sha == lc['sha']:
            for sha2, ct2, paths2 in touched[idx + 1:]:
                if any(p.startswith(pre) for p in paths2):
                    gap = '; next delivery commit %s %.1f minutes after the lock' % (sha2[:8], (ct2 - ct) / 60.0)
                    break
            break
    lock_files = sorted(set(p for p in locks if h.first_add.get(p) == cl))[:3]
    res_files = sorted(set(p for p in results if h.first_add.get(p) == cr))[:3]
    if cl == cr:
        return [row(d.label, 'R-LOCKTIME', FLAG, 'lock and results share commit %s (%s with %s)%s%s'
                    % (lc['sha'][:8], ', '.join(lock_files), ', '.join(res_files), gap, note),
                    [lc['sha']] + lock_files)]
    if cr < cl:
        late = [p for p in results if h.first_add.get(p, 1 << 30) == cl]
        extra = ('; lock commit %s also adds results %s' % (lc['sha'][:8], ', '.join(late[:2]))) if late else ''
        return [row(d.label, 'R-LOCKTIME', FLAG, 'results committed in %s before the first lock artifact in %s%s%s%s'
                    % (rc['sha'][:8], lc['sha'][:8], extra, gap, note), [rc['sha'], lc['sha']] + res_files)]
    return [row(d.label, 'R-LOCKTIME', PASS, 'lock %s precedes the first results %s by %.1f minutes%s'
                % (lc['sha'][:8], rc['sha'][:8], (rc['ct'] - lc['ct']) / 60.0, note), [lc['sha'], rc['sha']])]


# ======================================================================================
# Enumeration and orchestration
# ======================================================================================

def enumerate_worktree(src: WorktreeSource) -> List[Delivery]:
    root = src.root
    out: List[Delivery] = []
    for p in sorted(root.iterdir()):
        if not p.is_dir() or p.name.startswith('.') or p.name in INFRA_TOP:
            continue
        out.append(Delivery(src, p.name))
    for base in ('studies', 'PROJECT_DOCUMENTS'):
        bd = root / base
        if not bd.is_dir():
            continue
        for p in sorted(bd.iterdir()):
            if not p.is_dir() or p.name in ('KB',) or p.name.startswith('.'):
                continue
            if base == 'PROJECT_DOCUMENTS' and not (governance.DATE_DIR_RE.search(p.name)
                                                    or common.V1_DIR_RE.match(p.name)):
                continue
            out.append(Delivery(src, '%s/%s' % (base, p.name)))
    pd = root / 'PROJECT_DOCUMENTS'
    if pd.is_dir():
        for p in sorted(pd.glob('CURRENT_*.json')):
            out.append(Delivery(src, 'PROJECT_DOCUMENTS/%s' % p.name, kind='file'))
    return out


def _siblings(d: Delivery, all_d: Sequence[Delivery]) -> List[Delivery]:
    if d.kind != 'dir':
        return []
    parent = d.path.rsplit('/', 1)[0] if '/' in d.path else ''
    stem = lineage_stem(d.path.rsplit('/', 1)[-1])
    out = []
    for o in all_d:
        if o is d or o.kind != 'dir' or o.src is not d.src:
            continue
        op = o.path.rsplit('/', 1)[0] if '/' in o.path else ''
        if op == parent and 'design' in o.path.rsplit('/', 1)[-1].lower() and lineage_stem(o.path.rsplit('/', 1)[-1]) == stem:
            out.append(o)
    return out


def annotate(d: Delivery, all_d: Sequence[Delivery], kb_root: Optional[Path]) -> List[dict]:
    if d.kind == 'file':
        if re.search(r'(^|/)CURRENT_[^/]*\.json$', d.path):
            rows = adapter_pointer(d) + adapter_paths(d)
        else:
            rows = adapter_paths(d)
        return rows
    if d.src.read(d.path + '/ENDPOINT_LOCK.json') is not None and common.V1_DIR_RE.match(d.path.rsplit('/', 1)[-1]):
        return [row(d.label, a, NA, 'schema v1 delivery: checked by the gates, not by retro') for a in ADAPTERS]
    rows: List[dict] = []
    rows += adapter_locktime(d, _siblings(d, all_d))
    rows += adapter_closure(d)
    rows += adapter_literal(d)
    rows += adapter_failopen(d)
    rows += adapter_claim(d)
    rows += adapter_ctrl(d)
    rows += adapter_catalog(d, kb_root)
    rows += adapter_auth(d)
    rows += adapter_paths(d)
    rows += adapter_inert(d)
    rows.append(row(d.label, 'R-POINTER', NA, 'not a pointer file'))
    d._ast.clear()  # parsed trees are not kept across deliveries
    return rows


# ======================================================================================
# Quiet-list aspects (SPEC 12.4) and acceptance
# ======================================================================================

class AspectDelivery(Delivery):
    """One quiet-list delivery restricted to the files of its named aspect (prefixes relative to the delivery;
    a prefix ending in '/' is a directory, any other prefix is one file)."""

    def __init__(self, base: Delivery, prefixes: Optional[Sequence[str]]):
        super().__init__(base.src, base.path, base.kind)
        self.label = base.label
        self.prefixes = tuple(prefixes or ())

    def files(self) -> List[str]:
        if self._files is None:
            allf = [self.path] if self.kind == 'file' else self.src.files(self.path)
            if self.prefixes:
                allf = [p for p in allf if any(self.rel(p).startswith(x) if x.endswith('/') else self.rel(p) == x
                                               for x in self.prefixes)]
            self._files = allf
        return self._files


def run_adapter(a: str, d: Delivery, kb_root: Optional[Path], siblings: Sequence[Delivery] = ()) -> List[dict]:
    if a == 'R-LOCKTIME':
        return adapter_locktime(d, siblings)
    if a == 'R-CATALOG':
        return adapter_catalog(d, kb_root)
    if a == 'R-POINTER':
        return adapter_pointer(d) if d.kind == 'file' else [row(d.label, a, NA, 'not a pointer file')]
    fn = {'R-CLOSURE': adapter_closure, 'R-LITERAL': adapter_literal, 'R-FAILOPEN': adapter_failopen,
          'R-CLAIM': adapter_claim, 'R-CTRL': adapter_ctrl, 'R-AUTH': adapter_auth, 'R-PATHS': adapter_paths,
          'R-INERT': adapter_inert}[a]
    return fn(d)


def _hash_chain_note(d: Delivery, locks: Sequence[str], results: Sequence[str]) -> str:
    """In-artifact ordering evidence: a result JSON that names a lock's exact sha256 (self-reported, not commits)."""
    digests = {}
    for lp in locks:
        data = d.src.read(lp)
        if data is not None:
            digests[hashlib.sha256(data).hexdigest()] = lp
    hits = []
    for rp in results:
        if not rp.endswith('.json'):
            continue
        data = d.src.read(rp)
        if data is None or len(data) > 2 * 1024 * 1024:
            continue
        text = data.decode('utf-8', 'replace')
        for dg, lp in digests.items():
            if dg in text:
                hits.append('%s names sha256 %s... of %s' % (d.rel(rp), dg[:12], d.rel(lp)))
    if not hits:
        return ''
    return ('; in-artifact ordering evidence (self-reported, not commit order): %s' % '; '.join(hits[:3]))


def quiet_aspect_rows(q: dict, d: Delivery, all_d: Sequence[Delivery], kb_root: Optional[Path]) -> List[dict]:
    """The scoped adapters of one quiet item run on its aspect files (rows labelled with the delivery)."""
    ad = AspectDelivery(d, q.get('aspect_paths'))
    out: List[dict] = []
    for a in q['adapters']:
        if a == 'R-LOCKTIME' and q.get('locktime'):
            h = d.src.history()
            locks = [p for p in ad.files() if re.search(q['locktime']['locks'], ad.rel(p))]
            results = [p for p in ad.files() if re.search(q['locktime']['results'], ad.rel(p))]
            if h is None:
                rs = [row(d.label, a, NA, 'git history unavailable')]
            else:
                rs = locktime_between(ad, h, locks, results)
            if rs and rs[0]['status'] == NA:
                rs[0]['reason'] += _hash_chain_note(ad, locks, results)
            out += rs
        else:
            out += run_adapter(a, ad, kb_root, _siblings(d, all_d) if a == 'R-LOCKTIME' else ())
    ad._ast.clear()
    return out


def adjudicate(r: dict) -> Optional[dict]:
    for a in QUIET_ADJUDICATIONS:
        if a['delivery'] == r['delivery'] and a['check_id'] == r['check_id'] and \
                all(m in r['reason'] or any(m in e for e in r['evidence_paths']) for m in a['match']):
            return a
    return None


def _rate(n: int, k: int) -> Optional[float]:
    return (n / float(k)) if k else None


def acceptance(rows: List[dict], scope_all: bool, aspect_rows: Optional[Dict[str, List[dict]]] = None) -> dict:
    items = []
    hit = permit = 0
    for it in MUST_FLAG:
        cands = [r for r in rows if r['check_id'] == it['adapter']]
        if 'delivery' in it:
            cands = [r for r in cands if r['delivery'] == it['delivery']]
        if 'commit' in it:
            cands = [r for r in cands if any(e.startswith(it['commit']) for e in r['evidence_paths'])
                     or it['commit'] in r['reason']]
        applied = [r for r in cands if r['status'] != NA]
        flagged = [r for r in applied if r['status'] == FLAG and
                   ('match' not in it or it['match'] in r['reason'] or any(it['match'] in e for e in r['evidence_paths']))]
        bulk_only = cands and not applied
        permits = bool(applied) or ('commit' in it and not cands and scope_all)
        if it.get('aggregate'):
            permits = bool(applied)
        status = 'flagged' if flagged else ('not artifact-permitting' if not permits else 'missed')
        if status != 'not artifact-permitting':
            permit += 1
            hit += int(bool(flagged))
        items.append({'id': it['id'], 'adapter': it['adapter'], 'what': it['what'], 'status': status,
                      'rows': [r['delivery'] + ' ' + r['status'] for r in (flagged or cands)[:3]],
                      'note': ('only NOT_APPLICABLE rows (%s)' % cands[0]['reason'][:120]) if bulk_only else ''})
    # scoped: the adapters that judge each quiet item's named aspect, on the aspect's files (see MUST_STAY_QUIET)
    quiet = []
    applied_q = ff = 0
    for q in MUST_STAY_QUIET:
        if aspect_rows is not None and q['delivery'] in aspect_rows:
            rs_all, basis = aspect_rows[q['delivery']], 'aspect rows (scoped adapters on the aspect files)'
        else:
            rs_all = [r for r in rows if r['delivery'] == q['delivery'] and r['check_id'] in q['adapters']
                      and r['check_id'] not in q.get('path_only', ())]
            basis = 'delivery rows (aspect rows not computed; path-only adapters %s not counted)' % (
                ', '.join(q.get('path_only', ())) or 'none')
        rs = [r for r in rs_all if r['status'] != NA]
        flags = [r for r in rs if r['status'] == FLAG]
        applied_q += len(rs)
        ff += len(flags)
        quiet.append({'delivery': q['delivery'], 'aspect': q['aspect'], 'applied': len(rs),
                      'status': 'false-flagged' if flags else ('quiet' if rs else 'not measurable (NOT_APPLICABLE)'),
                      'scoped_adapters': dict(q['adapters']), 'excluded_adapters': dict(q.get('excluded', {})),
                      'aspect_paths': list(q.get('aspect_paths') or []) or 'whole delivery', 'basis': basis,
                      'applied_rows': ['%s %s: %s' % (r['check_id'], r['status'], r['reason'][:200]) for r in rs],
                      'not_applicable': ['%s: %s' % (r['check_id'], r['reason'][:1000]) for r in rs_all
                                         if r['status'] == NA],
                      'false_flags': [r['check_id'] + ': ' + r['reason'][:160] for r in flags]})
    # unscoped: every applied adapter on the whole quiet delivery; each FLAG adjudicated (unmatched = false)
    un_applied = un_flags = un_false = 0
    un_list = []
    for q in MUST_STAY_QUIET:
        rs = [r for r in rows if r['delivery'] == q['delivery'] and r['status'] != NA and r['check_id'] in ADAPTERS]
        un_applied += len(rs)
        for r in rs:
            if r['status'] != FLAG:
                continue
            un_flags += 1
            a = adjudicate(r)
            verdict = a['verdict'] if a else 'unadjudicated (counted as a false flag)'
            un_false += int(verdict != 'true_finding')
            un_list.append({'delivery': r['delivery'], 'check_id': r['check_id'], 'verdict': verdict,
                            'aspect_related': a['aspect_related'] if a else None,
                            'evidence': a['evidence'] if a else '', 'reason': r['reason'][:300],
                            'evidence_paths': r['evidence_paths'][:6]})
    recall = _rate(hit, permit)
    ffr = _rate(ff, applied_q)
    ffr_adj = _rate(un_false, un_applied)
    auth = [r for r in rows if r['check_id'] == 'R-AUTH' and r['status'] != NA]
    paths = [r for r in rows if r['check_id'] == 'R-PATHS' and r['status'] == FLAG]
    ok = scope_all and recall is not None and recall >= RECALL_MIN
    return {'must_flag': items, 'recall': recall, 'recall_min': RECALL_MIN, 'flagged': hit, 'artifact_permitting': permit,
            'must_stay_quiet': quiet, 'false_flag_rate': ffr, 'false_flag_max': FALSE_FLAG_MAX,
            'quiet_applied': applied_q, 'quiet_flags': ff,
            'false_flag_rate_unscoped': _rate(un_flags, un_applied),
            'quiet_applied_unscoped': un_applied, 'quiet_flags_unscoped': un_flags,
            'false_flag_rate_unscoped_adjudicated': ffr_adj, 'quiet_false_flags_unscoped_adjudicated': un_false,
            'quiet_unscoped_flags': un_list,
            'auth_counts': {'cycle_deliveries': len(auth), 'without_authorization':
                            sum(1 for r in auth if r['status'] == FLAG)},
            'paths_counts': {'deliveries': len(paths), 'files': sum(r.get('_files', 0) for r in paths),
                             'users_occurrences': sum(r.get('_users', 0) for r in paths)},
            'accepted_scoped': bool(ok and ffr is not None and ffr <= FALSE_FLAG_MAX),
            'accepted_unscoped_adjudicated': bool(ok and ffr_adj is not None and ffr_adj <= FALSE_FLAG_MAX),
            'accepted': bool(ok and ffr is not None and ffr <= FALSE_FLAG_MAX and ffr_adj is not None
                             and ffr_adj <= FALSE_FLAG_MAX),
            'acceptance_reading': 'accepted needs recall >= %.0f%% and BOTH false-flag readings <= %.0f%%: scoped '
                                  '(every FLAG of an aspect adapter on the aspect files is false) and unscoped '
                                  'adjudicated (every applied adapter on the whole quiet delivery; a FLAG is false '
                                  'unless QUIET_ADJUDICATIONS records it as a true finding). The raw unscoped rate '
                                  'counts every FLAG as false and is reported, not gated.'
                                  % (100 * RECALL_MIN, 100 * FALSE_FLAG_MAX),
            'scope_note': '' if scope_all else 'single-delivery run: acceptance is measured only by retro --all'}


def _parse_target(root: Path, target: str) -> Tuple[Source, str]:
    if ':' in target and not os.path.exists(target):
        ref, _, path = target.partition(':')
        src = RefSource(root, ref)
        if not src.ref_exists():
            raise common.MissingInput('git ref %s not found' % ref)
        return src, path.strip('/')
    p = Path(target)
    p = p if p.is_absolute() else (root / p if (root / p).exists() else Path.cwd() / p)
    p = p.resolve()
    try:
        rel = p.relative_to(root).as_posix()
    except ValueError:
        raise common.InvalidInput('%s is outside the repository %s' % (p, root))
    return WorktreeSource(root), rel


def run_retro(root: Path, target: Optional[str], all_: bool, with_ref: bool = True) -> Tuple[List[dict], dict]:
    root = Path(root).resolve()
    _CHECKER_SHA[0] = common.checker_sha256()
    kb_root = root if (root / common.KB_REL).is_dir() else None
    rows: List[dict] = []
    scope: dict = {'repo_root': str(root), 'mode': 'all' if all_ else 'dir'}
    if all_:
        wt = WorktreeSource(root)
        ds = enumerate_worktree(wt)
        hist = wt.history()
        scope['history'] = ('%d commits available%s' % (len(hist.commits), ' (shallow clone)' if hist.shallow else '')
                            if hist else 'unavailable')
        scope['shallow_repository'] = _is_shallow(root)
        scope['shallow_boundaries'] = sorted(hist.shallow) if hist else []
        ref_ds: List[Delivery] = []
        if with_ref:
            rs = RefSource(root, LEARNED_BANK_REF)
            if rs.ref_exists():
                tops = sorted(set(p.split('/', 1)[0] for p in rs.files('') if '/' in p))
                for t in tops:
                    if t in INFRA_TOP or t.startswith('.') or (root / t).exists():
                        continue
                    ref_ds.append(Delivery(rs, t))
                scope['ref'] = '%s: %d deliveries absent from the worktree' % (LEARNED_BANK_REF, len(ref_ds))
                if ref_ds:
                    rs.prefetch([p for dd in ref_ds for p in dd.files()])
            else:
                scope['ref'] = '%s absent' % LEARNED_BANK_REF
                rows += [row('%s:%s' % (LEARNED_BANK_REF, LEARNED_BANK_DIR), a, NA, 'git ref %s is absent: LEARNED-BANK '
                             'cannot be read' % LEARNED_BANK_REF) for a in ADAPTERS]
        all_d = ds + ref_ds
        by_label = {d.label: d for d in all_d}
        aspect_rows: Dict[str, List[dict]] = {}
        for d in all_d:
            rows += annotate(d, all_d, kb_root)
        for q in MUST_STAY_QUIET:
            if q['delivery'] in by_label:
                aspect_rows[q['delivery']] = quiet_aspect_rows(q, by_label[q['delivery']], all_d, kb_root)
        for dname, cid, what in PROSPECTIVE_ONLY:
            if (root / dname).is_dir():
                rows.append(row(dname, cid, NA, 'prospective-only (%s): legacy artifacts lack stream_hex and production '
                                'params' % what))
        scope['deliveries'] = len(all_d)
        return rows, dict(acceptance(rows, True, aspect_rows), scope=scope)
    else:
        src, rel = _parse_target(root, target or '')
        if isinstance(src, WorktreeSource):
            p = root / rel
            kind = 'file' if p.is_file() else 'dir'
            if not p.exists():
                raise common.MissingInput('%s does not exist' % p)
        else:
            kind = 'dir' if src.files(rel) else 'file'
            src.prefetch(src.files(rel) or [rel])
        d = Delivery(src, rel, kind)
        rows += annotate(d, [d], kb_root)
        scope['deliveries'] = 1
    return rows, dict(acceptance(rows, all_), scope=scope)


def _is_shallow(root: Path) -> Optional[bool]:
    """git rev-parse --is-shallow-repository (read-only); None when git cannot answer."""
    try:
        out = common.run_git(root, ['rev-parse', '--is-shallow-repository']).stdout.decode().strip()
    except common.MissingInput:
        return None
    return {'true': True, 'false': False}.get(out)


def cmd_retro(args: Any) -> int:
    """MFC retro (--all | <dir>) --out DIR."""
    root = Path(args.repo_root).resolve() if getattr(args, 'repo_root', None) else common.git_toplevel(Path.cwd())
    out = Path(args.out)
    out = out if out.is_absolute() else (Path.cwd() / out)
    why = governance._refuse_out(root, out)
    if why:
        sys.stderr.write('retro refuses --out %s: %s\n' % (out, why))
        return common.EXIT_MISSING
    if out.exists() and not out.is_dir():
        sys.stderr.write('retro --out %s is not a directory\n' % out)
        return common.EXIT_MISSING
    t0 = time.perf_counter()
    head0 = _git_head(root)  # the history the rows were read from; a commit landing mid-run is reported below
    rows, acc = run_retro(root, getattr(args, 'dir', None), bool(getattr(args, 'all', False)))
    head1 = _git_head(root)
    clean_rows = [{k: v for k, v in r.items() if not k.startswith('_')} for r in rows]
    utc = common.utc_now()
    out.mkdir(parents=True, exist_ok=True)
    full = {'schema': SCHEMA, 'record_type': 'RETRO_ANNOTATION', 'is_verdict': False, 'utc': utc,
            'checker_sha256': common.checker_sha256(), 'retro_py_sha256': common.sha256_file(Path(__file__)),
            'git_head': head0, 'runtime_s': round(time.perf_counter() - t0, 2), 'acceptance': acc,
            'counts': _counts(clean_rows), 'rows': clean_rows,
            'note': 'RETRO_ANNOTATION rows are annotations, never verdicts (SPEC 12.4)'}
    if head1 != head0:
        full['git_head_after_run'] = head1
    p_json = out / ('RETRO_%s.json' % utc)
    p_rows = out / ('RETRO_ANNOTATIONS_%s.jsonl' % utc)
    p_json.write_text(common.dumps_pretty(full), encoding='utf-8')
    with open(p_rows, 'w', encoding='utf-8') as fh:
        for r in clean_rows:
            fh.write(json.dumps(r, sort_keys=True, ensure_ascii=False) + '\n')
    if getattr(args, 'json', False):
        print(json.dumps({k: full[k] for k in ('schema', 'utc', 'acceptance', 'counts', 'runtime_s')}, indent=2))
    else:
        print(_format_acceptance(acc, full['counts']))
        print('retro: %d rows written to %s and %s (%.1fs)' % (len(clean_rows), p_json, p_rows, full['runtime_s']))
    return common.EXIT_PASS


def _git_head(root: Path) -> Optional[str]:
    try:
        return common.run_git(root, ['rev-parse', 'HEAD']).stdout.decode().strip() or None
    except common.MissingInput:
        return None


def _counts(rows: List[dict]) -> dict:
    c: Dict[str, Dict[str, int]] = {}
    for r in rows:
        c.setdefault(r['check_id'], {}).setdefault(r['status'], 0)
        c[r['check_id']][r['status']] += 1
    return c


def _format_acceptance(acc: dict, counts: dict) -> str:
    lines = ['RETRO acceptance (SPEC 12.4)%s' % (' - ' + acc['scope_note'] if acc['scope_note'] else '')]
    for it in acc['must_flag']:
        lines.append('  %-24s %-14s %-24s %s' % (it['id'], it['adapter'], it['status'], it['what']))
    rec = acc['recall']
    lines.append('  recall %s (%d/%d artifact-permitting; need >= %.0f%%)'
                 % ('n/a' if rec is None else '%.1f%%' % (100 * rec), acc['flagged'], acc['artifact_permitting'],
                    100 * acc['recall_min']))
    for q in acc['must_stay_quiet']:
        lines.append('  quiet %-60s %-26s applied %d, false flags %d [%s]%s'
                     % (q['delivery'], q['aspect'][:26], q['applied'], len(q['false_flags']), q['status'],
                        (': ' + '; '.join(q['false_flags'][:2])) if q['false_flags'] else ''))
        for na in q['not_applicable'][:2]:
            lines.append('      NOT_APPLICABLE %s' % na[:220])
    ffr = acc['false_flag_rate']

    def pct(x: Optional[float]) -> str:
        return 'n/a' if x is None else '%.1f%%' % (100 * x)
    lines.append('  false-flag rate, scoped: %s (%d/%d applied aspect checks; need <= %.0f%%)'
                 % (pct(ffr), acc['quiet_flags'], acc['quiet_applied'], 100 * acc['false_flag_max']))
    lines.append('  false-flag rate, unscoped adjudicated: %s (%d/%d applied checks on the quiet deliveries; need <= '
                 '%.0f%%); raw unscoped (every FLAG counted false): %s (%d/%d)'
                 % (pct(acc['false_flag_rate_unscoped_adjudicated']), acc['quiet_false_flags_unscoped_adjudicated'],
                    acc['quiet_applied_unscoped'], 100 * acc['false_flag_max'], pct(acc['false_flag_rate_unscoped']),
                    acc['quiet_flags_unscoped'], acc['quiet_applied_unscoped']))
    for f in acc['quiet_unscoped_flags']:
        lines.append('    FLAG %-14s %-60s %s' % (f['check_id'], f['delivery'][:60], f['verdict']))
    lines.append('  R-AUTH: %d of %d cycle deliveries without AUTHORIZATION; R-PATHS: %d files with absolute paths '
                 '(%d /Users/ occurrences) in %d deliveries'
                 % (acc['auth_counts']['without_authorization'], acc['auth_counts']['cycle_deliveries'],
                    acc['paths_counts']['files'], acc['paths_counts']['users_occurrences'],
                    acc['paths_counts']['deliveries']))
    lines.append('  accepted: %s (scoped %s, unscoped adjudicated %s)'
                 % (acc['accepted'], acc['accepted_scoped'], acc['accepted_unscoped_adjudicated']))
    lines.append('  counts: %s' % ', '.join('%s %s' % (k, '/'.join('%s=%d' % kv for kv in sorted(v.items())))
                                          for k, v in sorted(counts.items())))
    return '\n'.join(lines)
