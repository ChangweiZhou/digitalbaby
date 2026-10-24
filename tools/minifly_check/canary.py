"""MRK-1 role-targeted canary (WP4): `MFC canary <dir> --seed HEX [--n 20] [--out DIR]` and CAN-01..03
(SPEC 6.4 item 4, 7.4, 12.2).

The delivery is copied to a temporary directory (with the KB files and the closure files outside the delivery
that its SOURCE_LOCK names). Each operator of SPEC 12.2 (M01-M07, M09-M15, M17) picks its target by arm role
or behaviour, never by a declaration that names the canary, and random.Random(seed) chooses the instance. For
every mutant only the operator's target check ids run (plus the program decide where the operator says so),
with git checks disabled: copies answer history queries from the original repository's read-only history
(never mutated; git-history operators are exercised only in the corpus) and refuse worktree git queries.

    CAN-01  the unmutated copy reproduces the original statuses of every target (and decide)
    CAN-02  every applicable mutant is caught: a target goes from non-blocking to FAIL/MISSING, or decide
            meets the operator's expectation (not PASS / INCOMPLETE / INVALID TREATMENT_NOT_DELIVERED)
    CAN-03  at least 50% of the operators are applicable

The only file written inside the delivery is checks/CANARY_<utc>.json. Delivery code is never imported or
executed. These operators are public: they test checker regression and declaration minimization, never
held-out generalization (SPEC 12.1 L4). Python 3.9+, standard library only.
"""
from __future__ import annotations

import gzip
import io
import json
import os
import random
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import admitcheck
import common
from common import FAIL, MISSING, NOT_APPLICABLE, PASS, InvalidInput, MissingInput, Result, check

SCHEMA_CANARY_REPORT = 'minifly.canary_report.v1'
DEFAULT_N = 20
MIN_APPLICABLE_FRACTION = 0.5
LEARNER_ROLES = ('official', 'rehearsal', 'dev')  # decisive-role preference order
POINTER_REL = '_canary/POINTER.json'  # pointer copy, inside the copy root and outside the delivery
FIXTURES_REL = admitcheck.FIXTURES_REL
SOFT_PHRASE = ' (largely met)'
BACKWARD_LINE = b'\nx.backward()\n'
LITERAL_LINE = b'\nexposure_overlap = 0\n'


# --------------------------------------------------------------------------------------
# Copies: history-only git view, context, file rewriting
# --------------------------------------------------------------------------------------

class HistoryOnlyGit(common.GitView):
    """Read-only history of the original repository; worktree queries are disabled in canary copies."""
    mode = 'canary'

    def __init__(self, inner: common.GitView):
        self.inner = inner

    def head(self):
        return self.inner.head()

    def is_shallow(self):
        return self.inner.is_shallow()

    def log(self, paths=None):
        return self.inner.log(paths)

    def first_commit_adding(self, glob):
        return self.inner.first_commit_adding(glob)

    def first_commit_where(self, path, pred_bytes):
        return self.inner.first_commit_where(path, pred_bytes)

    def is_ancestor(self, a, b):
        return self.inner.is_ancestor(a, b)

    def file_at(self, sha, path):
        return self.inner.file_at(sha, path)

    def files_at(self, sha, paths):
        return self.inner.files_at(sha, paths)

    def remote_branches_containing(self, sha):
        return self.inner.remote_branches_containing(sha)

    def diff_lines(self, base, head, path):
        return self.inner.diff_lines(base, head, path)

    def changed_paths(self, base, head):
        return self.inner.changed_paths(base, head)

    def list_files(self, sha, prefix=''):
        return self.inner.list_files(sha, prefix)

    def commit(self, sha):
        return self.inner.commit(sha)

    def dirty_paths(self, paths):
        raise MissingInput('git worktree queries are disabled in canary copies')

    def ls_files(self, paths):
        raise MissingInput('git worktree queries are disabled in canary copies')


class CopyCtx(common.Ctx):
    """A Ctx over a canary copy that reports the original's git mode."""

    @property
    def git_mode(self) -> str:
        return self.options.get('canary_git_mode') or 'real'


def replace_bytes(path: Path, data: bytes) -> None:
    """Write by unlink-then-create, so a hard-linked copy never changes the file it was linked from."""
    p = Path(path)
    if p.exists() or p.is_symlink():
        p.unlink()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


def write_json(path: Path, obj: Any) -> None:
    replace_bytes(path, (json.dumps(obj, indent=2, ensure_ascii=False) + '\n').encode('utf-8'))


def jsonl_bytes(rows: Sequence[dict]) -> bytes:
    return ''.join(json.dumps(r, sort_keys=True, ensure_ascii=False) + '\n' for r in rows).encode('utf-8')


def rewrite_rows(cctx: common.Ctx, logical: str, fn: Callable[[List[dict]], Optional[List[dict]]]) -> bool:
    """Apply fn to the rows of one logical JSONL path (plain, .gz or RESULT_BUNDLE member), keeping its format."""
    for lg, kind, ref, gz in cctx.row_sources(logical):
        if lg != logical:
            continue
        rows = [r for _l, _n, r in cctx.iter_rows_with_source(lg)]
        new = fn(rows)
        if new is None:
            return False
        data = jsonl_bytes(new)
        if kind == 'file':
            replace_bytes(cctx.path(ref), gzip.compress(data) if gz else data)
        else:
            z, member = ref
            zp = cctx.path(z)
            buf = io.BytesIO()
            with zipfile.ZipFile(str(zp)) as zin, zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zout:
                for info in zin.infolist():
                    payload = zin.read(info.filename)
                    if info.filename == member:
                        payload = gzip.compress(data) if gz else data
                    zout.writestr(info, payload)
            replace_bytes(zp, buf.getvalue())
        return True
    return False


def _link_or_copy(src: str, dst: str) -> str:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return dst


_IGNORE = shutil.ignore_patterns('__pycache__', '*.pyc', '.git')


def make_pristine(ctx: common.Ctx, root: Path) -> List[str]:
    """Copy the delivery, the top-level KB files and the repo files outside the delivery that SOURCE_LOCK
    closure_sha256 or lock files.shared name. Returns the repo-relative extra paths copied."""
    drel = ctx.delivery_rel
    shutil.copytree(str(ctx.delivery), str(root / drel), ignore=_IGNORE)
    kb = Path(ctx.repo_root) / common.KB_REL
    if kb.is_dir():
        (root / common.KB_REL).mkdir(parents=True, exist_ok=True)
        for p in sorted(kb.iterdir()):
            if p.is_file():
                shutil.copy2(str(p), str(root / common.KB_REL / p.name))
    extra: List[str] = []
    names: List[str] = []
    try:
        sl = ctx.source_lock
        clo = sl.get('closure_sha256')
        if isinstance(clo, dict):
            names.extend(str(k) for k in clo)
    except (MissingInput, InvalidInput):
        pass
    shared = admitcheck.dig(ctx.lock, 'files.shared')
    if isinstance(shared, list):
        names.extend(str(x) for x in shared if isinstance(x, str))
    for n in sorted(set(names)):
        try:
            rel = common.norm_rel(n)
        except InvalidInput:
            continue
        if not rel or rel == drel or rel.startswith(drel + '/'):
            continue
        src = Path(ctx.repo_root) / rel
        dst = root / rel
        if dst.exists():
            continue
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(src), str(dst))
            extra.append(rel)
        elif src.is_dir():
            shutil.copytree(str(src), str(dst), ignore=_IGNORE)
            extra.append(rel + '/')
    return extra


def clone_mutant(pristine: Path, dest: Path) -> None:
    shutil.copytree(str(pristine), str(dest), copy_function=_link_or_copy, ignore=_IGNORE)


# --------------------------------------------------------------------------------------
# Environment shared by the operators
# --------------------------------------------------------------------------------------

class Env:
    def __init__(self, ctx: common.Ctx, pristine: Path, base_options: dict):
        self.ctx = ctx
        self.pristine = pristine
        self.drel = ctx.delivery_rel
        self.base_options = dict(base_options)
        self._ctxs: Dict[Tuple[str, str], common.Ctx] = {}
        self.pctx = self.copy_ctx(pristine)
        self.lock = self.pctx.lock
        self._cells: Dict[str, Any] = {}
        self._decide: Dict[str, Any] = {}
        self._role: Any = ()

    def copy_ctx(self, root: Path, extra: Optional[dict] = None) -> common.Ctx:
        opts = dict(self.base_options)
        opts.update(extra or {})
        opts['canary_git_mode'] = self.ctx.git_mode
        return CopyCtx(repo_root=Path(root), delivery=Path(root) / self.drel, git=HistoryOnlyGit(self.ctx.git),
                       kb=common.KB(root), rules=self.ctx.rules, stage=self.ctx.stage, tier=self.ctx.tier,
                       dry=self.ctx.dry, options=opts)

    def original_ctx(self, extra: Optional[dict] = None) -> common.Ctx:
        key = ('orig', json.dumps(extra or {}, sort_keys=True))
        if key not in self._ctxs:
            opts = dict(self.base_options)
            opts.update(extra or {})
            self._ctxs[key] = common.Ctx(repo_root=self.ctx.repo_root, delivery=self.ctx.delivery, git=self.ctx.git,
                                         kb=common.KB(self.ctx.repo_root), rules=self.ctx.rules, stage=self.ctx.stage,
                                         tier=self.ctx.tier, dry=self.ctx.dry, options=opts)
        return self._ctxs[key]

    def pristine_ctx(self, extra: Optional[dict] = None) -> common.Ctx:
        key = ('pristine', json.dumps(extra or {}, sort_keys=True))
        if key not in self._ctxs:
            self._ctxs[key] = self.copy_ctx(self.pristine, extra)
        return self._ctxs[key]

    def decisive_role(self) -> Optional[str]:
        if self._role == ():
            scorer = admitcheck.lazy('scorer')
            self._role = None
            for r in LEARNER_ROLES:
                if scorer.role_has_rows(self.pctx, r):
                    self._role = r
                    break
        return self._role

    def base_cells(self, role: str) -> dict:
        if role not in self._cells:
            scorer = admitcheck.lazy('scorer')
            self._cells[role] = scorer.score_role(self.pctx, role)['cells']
        return self._cells[role]


def decide_outcome(c: common.Ctx, role: Optional[str]) -> Optional[dict]:
    if role is None:
        return None
    scorer = admitcheck.lazy('scorer')
    try:
        v = scorer.evaluate_role(c, role)
    except (MissingInput, InvalidInput, common.CheckerError) as e:
        return {'role': role, 'verdict': 'ERROR', 'shadow_verdict': None, 'invalid_reason': None,
                'error': '%s: %s' % (type(e).__name__, str(e)[:300])}
    return {'role': role, 'verdict': v.get('verdict'), 'shadow_verdict': v.get('shadow_verdict'),
            'invalid_reason': v.get('invalid_reason')}


DECIDE_EXPECT = {
    'not_pass': (lambda v: v['verdict'] != 'PASS' and v.get('shadow_verdict') != 'PASS',
                 'decide does not PASS (verdict and shadow)'),
    'incomplete': (lambda v: v['verdict'] == 'INCOMPLETE', 'decide gives INCOMPLETE'),
    'invalid_ctrl': (lambda v: v['verdict'] == 'INVALID' and v.get('invalid_reason') == 'TREATMENT_NOT_DELIVERED',
                     'decide gives INVALID TREATMENT_NOT_DELIVERED'),
}


# --------------------------------------------------------------------------------------
# Operators (SPEC 12.2)
# --------------------------------------------------------------------------------------

class Op:
    oid = ''
    title = ''
    targets: Tuple[str, ...] = ()
    decide: Optional[str] = None

    def options(self, env: Env, root: Path) -> dict:
        return {}

    def find(self, env: Env) -> Tuple[List[Any], str]:
        raise NotImplementedError

    def apply(self, env: Env, cctx: common.Ctx, target: Any, rng: random.Random) -> str:
        raise NotImplementedError


def _cand(lock: dict) -> str:
    return admitcheck.candidate_arm(lock)


def _fixture_rows(c: common.Ctx) -> List[dict]:
    if not c.row_sources(FIXTURES_REL):
        return []
    return list(c.iter_rows(FIXTURES_REL))


def _taught_files(env: Env) -> Dict[Tuple[str, str], str]:
    """{(world, arm): logical TAUGHT path}."""
    split = env.lock.get('split') if isinstance(env.lock.get('split'), dict) else {}
    tdir = common.norm_rel(split.get('taught_manifest') or 'manifests/TAUGHT')
    out = {}
    for logical, _k, _r, _gz in env.pctx.row_sources(tdir + '/*/*.jsonl'):
        parts = logical[len(tdir) + 1:].split('/')
        if len(parts) == 2 and parts[1].endswith('.jsonl'):
            out[(parts[0], parts[1][:-len('.jsonl')])] = logical
    return out


class M01(Op):
    oid, title = 'M01', "copy W's post_first channel digests and writes into the causal-control arm receipt"
    targets, decide = ('ADM-CTRL-ZERO',), 'invalid_ctrl'

    def find(self, env):
        lock = env.lock
        cand = _cand(lock)
        by = admitcheck.arm_by_id(lock)
        arms = []
        cc = lock.get('causal_control_arm')
        if isinstance(cc, str) and cc in by and cc != cand:
            arms.append(cc)
        arms += [a['id'] for a in admitcheck.control_arms(lock) if a['id'] not in arms]
        rc = admitcheck.receipts(env.pctx)
        t = sorted((w, a) for (w, a) in rc if a in arms and (w, cand) in rc)
        return t, 'no (world, causal-control arm) receipt with a candidate receipt beside it'

    def apply(self, env, cctx, target, rng):
        w, a = target
        cand = _cand(env.lock)
        rp = cctx.path('results/receipts/%s/%s.json' % (w, a))
        R = common.load_json_file(rp)
        Wr = common.load_json_file(cctx.path('results/receipts/%s/%s.json' % (w, cand)))
        cd = R.setdefault('channel_digests', {})
        cd['post_first'] = json.loads(json.dumps((Wr.get('channel_digests') or {}).get('post_first')))
        R['writes_by_channel'] = json.loads(json.dumps(Wr.get('writes_by_channel')))
        write_json(rp, R)
        return 'receipt %s/%s given the candidate post_first digests and writes' % (w, a)


class M02(Op):
    oid, title = 'M02', 'negate a directional delta'
    targets = ('ADM-DIR',)

    def find(self, env):
        rows = _fixture_rows(env.pctx)
        return [i for i, r in enumerate(rows) if 'port' in r and 'gate_id' not in r], 'no directional fixture row'

    def apply(self, env, cctx, target, rng):
        out = {}

        def fn(rows):
            r = rows[target]
            d = r.get('target_margin_delta_vs_passive')
            r['target_margin_delta_vs_passive'] = -d if admitcheck.is_num(d) and d != 0 else -1.0
            out['d'] = (d, r['target_margin_delta_vs_passive'])
            return rows
        rewrite_rows(cctx, FIXTURES_REL, fn)
        return 'FIXTURES row %d: target_margin_delta_vs_passive %r -> %r' % (target, out['d'][0], out['d'][1])


def _numeric_leaves(obj: Any, path: Tuple[str, ...] = ()) -> List[Tuple[str, ...]]:
    out = []
    if isinstance(obj, dict):
        for k in sorted(obj, key=str):
            out.extend(_numeric_leaves(obj[k], path + (str(k),)))
    elif admitcheck.is_num(obj) and obj != 0:
        out.append(path)
    return out


class M03(Op):
    oid, title = 'M03', 'perturb a fixture param by 40%'
    targets = ('ADM-FIX',)

    def find(self, env):
        t = []
        for i, r in enumerate(_fixture_rows(env.pctx)):
            if isinstance(r.get('params'), dict):
                t.extend((i, p) for p in _numeric_leaves(r['params']))
        return t, 'no fixture row with a non-zero numeric param'

    def apply(self, env, cctx, target, rng):
        i, path = target
        out = {}

        def fn(rows):
            cur = rows[i]['params']
            for k in path[:-1]:
                cur = cur[k]
            old = cur[path[-1]]
            cur[path[-1]] = float(old) * 1.4
            out['v'] = (old, cur[path[-1]])
            return rows
        rewrite_rows(cctx, FIXTURES_REL, fn)
        return 'FIXTURES row %d params.%s %r -> %r' % (i, '.'.join(path), out['v'][0], out['v'][1])


class M04(Op):
    oid, title = 'M04', "delete a plumbing gate's negative fixture"
    targets = ('ADM-GATE',)

    def find(self, env):
        gates = set(g.get('id') for g in (env.lock.get('gates') or []) if isinstance(g, dict))
        t = sorted(set(r.get('gate_id') for r in _fixture_rows(env.pctx)
                       if r.get('kind') == 'negative' and r.get('gate_id') in gates))
        return t, 'no declared plumbing gate with a negative fixture'

    def apply(self, env, cctx, target, rng):
        out = {}

        def fn(rows):
            keep = [r for r in rows if not (r.get('gate_id') == target and r.get('kind') == 'negative')]
            out['n'] = len(rows) - len(keep)
            return keep
        rewrite_rows(cctx, FIXTURES_REL, fn)
        return 'deleted the %d negative fixture(s) of gate %s' % (out['n'], target)


def _emitted(stream: bytes, eos: bytes) -> bytes:
    pos = stream.find(eos)
    return stream if pos < 0 else stream[:pos]


def _stream_for(pred: bytes, eos: bytes, cap: int) -> Tuple[str, str, bool]:
    st = (pred + eos * cap)[:cap]
    pos = st.find(eos)
    em = st if pos < 0 else st[:pos]
    return st.hex(), em.hex(), pos >= 0


class M05(Op):
    oid, title = 'M05', "replace the candidate's rows with the best shortcut's predictions (expect not PASS)"
    decide = 'not_pass'

    def find(self, env):
        role = env.decisive_role()
        if role is None:
            return [], 'no learner-arm rows (dev, rehearsal or official)'
        base = decide_outcome(env.pristine_ctx(), role)
        if not base or DECIDE_EXPECT['not_pass'][0](base):
            return [], 'decide on the unmutated %s rows does not PASS (%s/%s): the substitution has nothing to break' % (
                role, base and base.get('verdict'), base and base.get('shadow_verdict'))
        return [role], ''

    def apply(self, env, cctx, role, rng):
        scorer = admitcheck.lazy('scorer')
        decide = admitcheck.lazy('decide')
        lock = env.lock
        fmt = admitcheck.dig(lock, 'task.format')
        eos = bytes.fromhex(fmt['eos_hex'])
        cap = int(fmt['gen_cap_bytes'])
        pr = lock['primary']
        cand, pool, ck = pr['candidate_arm'], pr['pool'], pr['checkpoint']
        cells = env.base_cells(role)
        taught = scorer.load_taught(env.pctx, lock)
        best = {}
        for w in sorted(cells.get('worlds') or {}):
            b = decide.ladder_best(cells, w, pool, ck)
            if b is not None:
                best[w] = b[0]
        tidx: Dict[str, Any] = {}
        mode: Dict[str, bytes] = {}
        logical = 'results/%s/OUTPUTS.jsonl' % role
        # MODE_REPEAT per world: the candidate's modal output byte repeated to its modal emitted length
        stats: Dict[str, Tuple[dict, dict]] = {}
        for r in cctx.iter_rows(logical):
            if r.get('arm') == cand and r.get('pool') == pool and r.get('checkpoint') == ck:
                em = _emitted(bytes.fromhex(r.get('stream_hex') or ''), eos)
                bc, lc = stats.setdefault(r.get('world'), ({}, {}))
                lc[len(em)] = lc.get(len(em), 0) + 1
                for byte in em:
                    bc[byte] = bc.get(byte, 0) + 1
        for w, (bc, lc) in stats.items():
            ml = min(lc.items(), key=lambda kv: (-kv[1], kv[0]))[0] if lc else 0
            mb = min(bc.items(), key=lambda kv: (-kv[1], kv[0]))[0] if bc else None
            mode[w] = (bytes([mb]) * ml) if mb is not None else b''
        count = {'n': 0}

        def predict(w: str, r: dict, m: str) -> bytes:
            if m.startswith('FIXED_'):
                return bytes.fromhex(r.get('stream_hex') or '')[:int(m[6:])]
            if m == 'MODE_REPEAT':
                return mode.get(w, b'')
            if w not in tidx:
                tidx[w] = scorer.TaughtIndex(taught.get((w, cand)) or [], fmt, 'TAUGHT/%s/%s' % (w, cand))
            ti = tidx[w]
            ph = r.get('prompt_hex') or ''
            preds = ti.predictions(ph, bytes.fromhex(ph), True)
            return preds.get(m) or ti.const or b''

        def fn(rows):
            for r in rows:
                w = r.get('world')
                if r.get('arm') == cand and r.get('pool') == pool and r.get('checkpoint') == ck and w in best:
                    sh, eh, oe = _stream_for(predict(w, r, best[w]), eos, cap)
                    r['stream_hex'], r['emitted_hex'], r['own_eos'] = sh, eh, oe
                    count['n'] += 1
            return rows
        rewrite_rows(cctx, logical, fn)
        return '%s: %d candidate %s/%s rows replaced by the best shortcut per world %s' % (
            role, count['n'], pool, ck, best)


class M06(Op):
    oid, title = 'M06', "drop a comparator arm's rows"
    targets, decide = ('LAD-01',), 'incomplete'

    def find(self, env):
        role = env.decisive_role()
        if role is None:
            return [], 'no learner-arm rows'
        comps = admitcheck.str_list(admitcheck.dig(env.lock, 'primary.comparator_arms'))
        if not comps:
            return [], 'no declared comparator arm'
        seen = set()
        for r in env.pctx.iter_rows('results/%s/OUTPUTS.jsonl' % role):
            if r.get('arm') in comps:
                seen.add((role, r.get('arm'), str(r.get('world'))))
        return sorted(seen), 'no comparator-arm rows in %s' % role

    def apply(self, env, cctx, target, rng):
        role, arm, w = target
        out = {}

        def fn(rows):
            keep = [r for r in rows if not (r.get('arm') == arm and str(r.get('world')) == w)]
            out['n'] = len(rows) - len(keep)
            return keep
        rewrite_rows(cctx, 'results/%s/OUTPUTS.jsonl' % role, fn)
        return 'dropped %d %s rows of comparator %s in world %s' % (out['n'], role, arm, w)


class M07(Op):
    oid, title = 'M07', 'insert a commuted held prompt into a feedback-bearing TAUGHT stream'
    targets = ('ADM-EXPO',)

    def _held(self, env) -> List[dict]:
        split = env.lock.get('split') if isinstance(env.lock.get('split'), dict) else {}
        rel = split.get('held_manifest')
        if not admitcheck.s(rel) or not env.pctx.row_sources(common.norm_rel(rel)):
            return []
        return [r for r in env.pctx.iter_rows(common.norm_rel(rel)) if isinstance(r.get('prompt_hex'), str)]

    def find(self, env):
        decls, _bad = admitcheck._stream_decls(env.lock)
        fb_globs = [g for g, fb in decls if fb]
        if not self._held(env):
            return [], 'no held manifest rows'
        files = sorted(set(src[0] for src in env.pctx.row_sources('**/*.jsonl')
                           if any(common.glob_match(src[0], g) for g in fb_globs)))
        return files, 'no feedback-bearing declared stream file'

    def apply(self, env, cctx, target, rng):
        addition = admitcheck.lazy('addition')
        fmt = admitcheck.dig(env.lock, 'task.format')
        held = sorted(self._held(env), key=lambda r: (str(r.get('item_id')), r.get('prompt_hex')))
        h = rng.choice(held)
        ph = h['prompt_hex'].lower()
        exact = env.lock.get('claim_level_target') == 'E2' or not addition.is_addition(fmt)
        pr = None if exact else addition.parse(ph, fmt)
        if pr is not None:
            sep, term = bytes.fromhex(fmt['sep_hex']), bytes.fromhex(fmt['terminator_hex'])
            ph2 = (pr[1].encode('ascii') + sep + pr[0].encode('ascii') + term).hex()
            ans = addition.expected(ph2, fmt)
            how = 'commuted'
        else:
            ph2, ans, how = ph, h.get('expected_hex'), 'exact (E2 or non-addition)'
        parts = target.split('/')
        first = next(iter(cctx.iter_rows(target)), {})
        row = {'world': first.get('world', parts[-2] if len(parts) >= 2 else None),
               'arm': first.get('arm', parts[-1][:-len('.jsonl')]), 'prompt_hex': ph2, 'answer_hex': ans,
               'phase': 'calibration', 'feedback_bearing': True, 'presentations': 1}
        rewrite_rows(cctx, target, lambda rows: rows + [row])
        return '%s held item %s inserted as %s into %s' % (how, h.get('item_id'), ph2, target)


class M09(Op):
    oid, title = 'M09', "put 'largely met' in the headline sentence"
    targets = ('CLM-06', 'CLM-05')

    def find(self, env):
        p = env.pctx.path('REPORT.md')
        if p.is_file() and any(ln.strip() for ln in p.read_text(encoding='utf-8', errors='replace').splitlines()):
            return ['REPORT.md'], ''
        return [], 'no REPORT.md headline'

    def apply(self, env, cctx, target, rng):
        p = cctx.path('REPORT.md')
        lines = p.read_text(encoding='utf-8', errors='replace').splitlines(True)
        for i, ln in enumerate(lines):
            if ln.strip():
                nl = '\n' if ln.endswith('\n') else ''
                lines[i] = ln.rstrip('\r\n') + SOFT_PHRASE + nl
                break
        replace_bytes(p, ''.join(lines).encode('utf-8'))
        return "REPORT.md headline sentence gains '%s'" % SOFT_PHRASE.strip()


class M10(Op):
    oid, title = 'M10', 'set worlds_role SEALED_CONFIRM in a CLAIM copy'
    targets = ('CLM-09',)

    def find(self, env):
        p = env.pctx.path('CLAIM.json')
        if not p.is_file():
            return [], 'no CLAIM.json'
        c = common.load_json_file(p)
        if not isinstance(c, dict):
            return [], 'CLAIM.json is not an object'
        if c.get('worlds_role') == 'SEALED_CONFIRM':
            return [], 'CLAIM.worlds_role is already SEALED_CONFIRM'
        return ['CLAIM.json'], ''

    def apply(self, env, cctx, target, rng):
        p = cctx.path('CLAIM.json')
        c = common.load_json_file(p)
        old = c.get('worlds_role')
        c['worlds_role'] = 'SEALED_CONFIRM'
        write_json(p, c)
        return 'CLAIM.worlds_role %r -> SEALED_CONFIRM' % (old,)


def _delivery_py(c: common.Ctx, entries: Optional[List[str]] = None) -> List[str]:
    files = [f for f in c.delivery_files() if f.endswith('.py')]
    if entries is None:
        return sorted(files)
    out = set()
    for e in entries:
        try:
            rel = common.norm_rel(e)
        except InvalidInput:
            continue
        d = c.delivery_rel
        if d and rel.startswith(d + '/'):
            rel = rel[len(d) + 1:]
        for f in files:
            if f == rel or (common.has_glob(rel) and common.glob_match(f, rel)):
                out.add(f)
    return sorted(out)


class M11(Op):
    oid, title = 'M11', 'append x.backward() to a native file copy'
    targets = ('LOCK-07',)

    def find(self, env):
        nat = admitcheck.dig(env.lock, 'files.native')
        nat = [nat] if isinstance(nat, str) else (nat if isinstance(nat, list) else [])
        return _delivery_py(env.pctx, [x for x in nat if isinstance(x, str)]), 'no files.native .py file in the delivery'

    def apply(self, env, cctx, target, rng):
        p = cctx.path(target)
        replace_bytes(p, p.read_bytes() + BACKWARD_LINE)
        return 'appended x.backward() to %s' % target


class M12(Op):
    oid, title = 'M12', 'set one emitted_hex to null'
    targets = ('MET-01',)

    def find(self, env):
        scorer = admitcheck.lazy('scorer')
        roles = [r for r in ('dev', 'rehearsal', 'official') if scorer.role_has_rows(env.pctx, r)]
        if not roles and scorer.role_has_rows(env.pctx, 'admission'):
            roles = ['admission']
        return ['results/%s/OUTPUTS.jsonl' % r for r in roles], 'no OUTPUTS rows'

    def apply(self, env, cctx, target, rng):
        out = {}

        def fn(rows):
            i = rng.randrange(len(rows))
            out['i'] = (i, rows[i].get('arm'), rows[i].get('item_id'))
            rows[i]['emitted_hex'] = None
            return rows
        rewrite_rows(cctx, target, fn)
        return '%s row %d (arm %s item %s): emitted_hex -> null' % ((target,) + out['i'])


class M13(Op):
    oid, title = 'M13', "add a literal 'exposure_overlap = 0'"
    targets = ('LOCK-08',)

    def find(self, env):
        return _delivery_py(env.pctx), 'no .py file in the delivery'

    def apply(self, env, cctx, target, rng):
        p = cctx.path(target)
        replace_bytes(p, p.read_bytes() + LITERAL_LINE)
        return 'appended exposure_overlap = 0 to %s' % target


class M14(Op):
    oid, title = 'M14', "add 'confirmation_passes': true to a pointer copy"
    targets = ('PTR-03',)

    def options(self, env, root):
        return {'pointer': str(Path(root) / POINTER_REL)}

    def find(self, env):
        if (env.pristine / POINTER_REL).is_file():
            return ['pointer'], ''
        return [], 'no pointer naming the delivery and no CLAIM.json to copy one from'

    def apply(self, env, cctx, target, rng):
        p = Path(cctx.repo_root) / POINTER_REL
        obj = common.load_json_file(p)
        obj['confirmation_passes'] = True
        write_json(p, obj)
        return "pointer copy gains 'confirmation_passes': true"


class M15(Op):
    oid, title = 'M15', "relabel the causal-control arm's role to 'other' and clear its zero_channels"
    targets = ('LOCK-02', 'ADM-CTRL-ZERO', 'ADM-CTRL-ISO')

    def find(self, env):
        cc = env.lock.get('causal_control_arm')
        if isinstance(cc, str) and cc in admitcheck.arm_by_id(env.lock):
            return [cc], ''
        return [], 'lock.causal_control_arm names no arm'

    def apply(self, env, cctx, target, rng):
        p = cctx.path('ENDPOINT_LOCK.json')
        lock = common.load_json_file(p)
        for a in lock.get('arms') or []:
            if isinstance(a, dict) and a.get('id') == target:
                old = (a.get('role'), a.get('zero_channels'))
                a['role'] = 'other'
                a['zero_channels'] = []
        write_json(p, lock)
        return 'arm %s role/zero_channels %r -> other/[]' % (target, old)


class M17(Op):
    oid, title = 'M17', 'invert SHUF labels'
    targets = ('ADM-SHUF',)

    def find(self, env):
        shufs = [a['id'] for a in admitcheck.lock_arms(env.lock) if a.get('role') == 'shuf']
        tf = _taught_files(env)
        return sorted((w, a, lg) for (w, a), lg in tf.items() if a in shufs), 'no SHUF arm TAUGHT manifest'

    def apply(self, env, cctx, target, rng):
        addition = admitcheck.lazy('addition')
        w, a, logical = target
        fmt = admitcheck.dig(env.lock, 'task.format')
        mode = rng.choice(['truth', 'derange'])
        cand = _cand(env.lock)
        truth_map: Dict[str, str] = {}
        if not addition.is_addition(fmt):
            tf = _taught_files(env)
            for r in env.pctx.iter_rows(tf.get((w, cand), '')) if (w, cand) in tf else []:
                truth_map.setdefault(r.get('prompt_hex'), r.get('answer_hex'))
        labels = sorted(set(v for v in truth_map.values() if isinstance(v, str)))

        def truth_of(ph: str) -> Optional[str]:
            if addition.is_addition(fmt):
                return addition.expected(ph, fmt)
            return truth_map.get(ph)

        def wrong_of(t: str) -> str:
            if addition.is_addition(fmt):
                return str(int(bytes.fromhex(t).decode('ascii')) + 1).encode('ascii').hex()
            others = [x for x in labels if x != t]
            return others[0] if others else (t + '00')

        def fn(rows):
            for r in rows:
                t = truth_of(r.get('prompt_hex'))
                if t is None:
                    continue
                r['answer_hex'] = t if mode == 'truth' else wrong_of(t)
            return rows
        rewrite_rows(cctx, logical, fn)
        return 'SHUF %s/%s labels set to %s' % (w, a, 'the truth (shuffle undone)' if mode == 'truth'
                                               else 'a wrong answer for every orbit (derangement)')


OPERATORS: Tuple[Op, ...] = (M01(), M02(), M03(), M04(), M05(), M06(), M07(), M09(), M10(), M11(), M12(), M13(),
                             M14(), M15(), M17())


# --------------------------------------------------------------------------------------
# Running the canary
# --------------------------------------------------------------------------------------

def _rule(ctx: common.Ctx, rid: str) -> dict:
    idx = common.rule_index(ctx.rules or common.load_rules())
    if rid not in idx:
        raise common.CheckerError('canary target %s is not in RULES.json' % rid)
    return idx[rid]


def eval_targets(ctx: common.Ctx, c: common.Ctx, op: Op, role: Optional[str]) -> dict:
    out: Dict[str, Any] = {}
    for rid in op.targets:
        res = common.run_check(c, _rule(ctx, rid))
        out[rid] = [{'status': r.status, 'detail': r.detail[:300]} for r in res]
    if op.decide:
        out['decide'] = decide_outcome(c, role)
    return out


def _statuses(v: Any) -> List[str]:
    return [x['status'] for x in v] if isinstance(v, list) else []


def _blocking(v: Any) -> bool:
    return any(st in common.BLOCKING_STATUSES for st in _statuses(v))


def _decide_key(d: Optional[dict]) -> Optional[tuple]:
    if not d:
        return None
    return (d.get('verdict'), d.get('shadow_verdict'), d.get('invalid_reason'))


def caught(op: Op, base: dict, mut: dict) -> Tuple[bool, str]:
    for rid in op.targets:
        if not _blocking(base.get(rid)) and _blocking(mut.get(rid)):
            return True, '%s %s -> %s' % (rid, ','.join(_statuses(base.get(rid))), ','.join(_statuses(mut.get(rid))))
    if op.decide and base.get('decide') and mut.get('decide'):
        fn, what = DECIDE_EXPECT[op.decide]
        if not fn(base['decide']) and fn(mut['decide']):
            return True, '%s (%s -> %s)' % (what, _decide_key(base['decide']), _decide_key(mut['decide']))
    return False, 'no target went from non-blocking to blocking: %s' % {
        k: ','.join(_statuses(v)) for k, v in mut.items() if k != 'decide'}


def _pointer_seed(ctx: common.Ctx, pristine: Path) -> Optional[str]:
    """Create the pointer copy for M14 in the pristine root: an existing minifly.pointer.v1 file under
    PROJECT_DOCUMENTS/ naming this delivery, else one copied from CLAIM.json fields. Returns its source."""
    drel = ctx.delivery_rel
    pdir = Path(ctx.repo_root) / 'PROJECT_DOCUMENTS'
    if pdir.is_dir():
        for p in sorted(pdir.glob('*.json')):
            try:
                obj = common.load_json_file(p)
            except (MissingInput, InvalidInput):
                continue
            if isinstance(obj, dict) and obj.get('schema') == 'minifly.pointer.v1' and \
                    admitcheck.s(obj.get('delivery')) and common.norm_rel(obj['delivery']) == drel:
                write_json(pristine / POINTER_REL, obj)
                return 'PROJECT_DOCUMENTS/%s' % p.name
    cp = ctx.path('CLAIM.json')
    if not cp.is_file():
        return None
    try:
        claim = common.load_json_file(cp)
    except (MissingInput, InvalidInput):
        return None
    if not isinstance(claim, dict):
        return None
    ptr = {'schema': 'minifly.pointer.v1', 'track': 'canary', 'delivery': drel, 'claim': drel + '/CLAIM.json',
           'claim_sha256': common.sha256_file(cp)}
    for k in ('level_earned', 'primary_gate', 'gate_type', 'worlds_role', 'objective_metric'):
        ptr[k] = claim.get(k)
    ptr.update({'narrowing_rows': [], 'updated_utc': common.utc_now(), 'previous_snapshot': None,
                'user_decision': None})
    write_json(pristine / POINTER_REL, ptr)
    return 'generated from CLAIM.json'


def canary_run(ctx: common.Ctx) -> dict:
    """The canary of SPEC 12.2 for ctx.delivery (cached per ctx; a failure is cached and re-raised)."""
    key = 'wp4_canary'
    if key in ctx._cache:
        got = ctx._cache[key]
        if isinstance(got, BaseException):
            raise got
        return got
    try:
        out = _canary_run(ctx)
    except BaseException as e:  # noqa: BLE001 - cache the failure so CAN-01..03 report the same cause
        ctx._cache[key] = e
        raise
    ctx._cache[key] = out
    return out


def _canary_run(ctx: common.Ctx) -> dict:
    seed = ctx.options.get('canary_seed')
    if not isinstance(seed, str) or not re.fullmatch(r'[0-9a-fA-F]+', seed or ''):
        raise MissingInput('CAN-01..03 run only under MFC canary <dir> --seed HEX (seed chosen after the lock)')
    n_req = ctx.options.get('canary_n', DEFAULT_N)
    if not admitcheck.is_int(n_req) or n_req < 1:
        raise InvalidInput('--n must be a positive integer')
    if ctx.delivery is None or not common.is_v1_delivery(ctx.delivery):
        raise MissingInput('canary runs only on v1 deliveries')
    rng = random.Random(int(seed, 16))
    keep = ctx.options.get('canary_out')
    if keep:
        base_name = 'canary_%s_%s' % (ctx.delivery_name, common.utc_now())
        work = Path(keep) / base_name
        k = 1
        while work.exists():
            work = Path(keep) / ('%s_%d' % (base_name, k))
            k += 1
        work.mkdir(parents=True)
    else:
        work = Path(tempfile.mkdtemp(prefix='minifly_canary_'))
    base_opts = {k: v for k, v in ctx.options.items() if not k.startswith('canary_')}
    base_opts['invoked_by'] = 'canary'
    try:
        pristine = work / 'pristine'
        pristine.mkdir()
        extra = make_pristine(ctx, pristine)
        ptr_src = _pointer_seed(ctx, pristine)
        env = Env(ctx, pristine, base_opts)
        ops_report: List[dict] = []
        applicable: List[Tuple[Op, List[Any]]] = []
        errors: List[str] = []
        for op in OPERATORS:
            ent = {'id': op.oid, 'title': op.title, 'targets': list(op.targets), 'decide': op.decide}
            try:
                tlist, why = op.find(env)
            except (MissingInput, InvalidInput, common.CheckerError, KeyError, ValueError, TypeError) as e:
                ent.update({'applicable': False, 'error': '%s: %s' % (type(e).__name__, str(e)[:300])})
                errors.append('%s target discovery failed: %s' % (op.oid, ent['error']))
                ops_report.append(ent)
                continue
            ent['applicable'] = bool(tlist)
            ent['n_targets'] = len(tlist)
            if not tlist:
                ent['reason'] = why
            else:
                applicable.append((op, tlist))
            ops_report.append(ent)
        role = env.decisive_role()
        repro: List[dict] = []
        baselines: Dict[str, dict] = {}
        for op, _t in applicable:
            o_ctx = env.original_ctx(op.options(env, pristine))
            p_ctx = env.pristine_ctx(op.options(env, pristine))
            orig = eval_targets(ctx, o_ctx, op, role)
            base = eval_targets(ctx, p_ctx, op, role)
            baselines[op.oid] = base
            for rid in list(op.targets) + (['decide'] if op.decide else []):
                a = _decide_key(orig.get(rid)) if rid == 'decide' else _statuses(orig.get(rid))
                b = _decide_key(base.get(rid)) if rid == 'decide' else _statuses(base.get(rid))
                repro.append({'op': op.oid, 'id': rid, 'original': a, 'copy': b, 'equal': a == b})
        total = max(n_req, len(applicable)) if applicable else 0
        mutants: List[dict] = []
        for i in range(total):
            op, tlist = applicable[i % len(applicable)]
            target = tlist[rng.randrange(len(tlist))]
            mdir = work / ('m%03d_%s' % (i, op.oid))
            ent: Dict[str, Any] = {'index': i, 'op': op.oid, 'target': _jsonable(target)}
            try:
                clone_mutant(pristine, mdir)
                mctx0 = env.copy_ctx(mdir, op.options(env, mdir))
                ent['description'] = op.apply(env, mctx0, target, rng)
                mctx = env.copy_ctx(mdir, op.options(env, mdir))
                res = eval_targets(ctx, mctx, op, role)
                ok, why = caught(op, baselines[op.oid], res)
                ent.update({'checks': {k: _statuses(v) for k, v in res.items() if k != 'decide'},
                            'decide': res.get('decide'), 'caught': ok, 'why': why})
            except (MissingInput, InvalidInput, common.CheckerError, OSError, KeyError, ValueError, TypeError,
                    IndexError) as e:
                ent.update({'caught': False, 'why': 'mutant could not be applied or evaluated: %s: %s'
                            % (type(e).__name__, str(e)[:300])})
            finally:
                if not keep:
                    shutil.rmtree(str(mdir), ignore_errors=True)
            mutants.append(ent)
        return {'seed': seed.lower(), 'n_requested': n_req, 'operators': ops_report, 'errors': errors,
                'n_operators': len(OPERATORS), 'n_applicable': len(applicable),
                'applicable_fraction': len(applicable) / float(len(OPERATORS)), 'decisive_role': role,
                'reproduction': repro, 'mutants': mutants, 'n_mutants': len(mutants),
                'detected': sum(1 for m in mutants if m.get('caught')), 'copied_outside_delivery': extra,
                'pointer_copy': ptr_src, 'copies_kept': str(work) if keep else None}
    finally:
        if not keep:
            shutil.rmtree(str(work), ignore_errors=True)


def _jsonable(t: Any) -> Any:
    if isinstance(t, tuple):
        return [_jsonable(x) for x in t]
    return t


@check('CAN-01')
def check_can_01(ctx: common.Ctx) -> Result:
    run = canary_run(ctx)
    rep = run['reproduction']
    if not rep:
        return Result('CAN-01', MISSING, 'no applicable operator: no target status could be reproduced')
    bad = [r for r in rep if not r['equal']]
    if bad:
        return Result('CAN-01', FAIL, 'the unmutated copy does not reproduce the original statuses: %s' % '; '.join(
            '%s %s original %s copy %s' % (r['op'], r['id'], r['original'], r['copy']) for r in bad[:8]),
            evidence=['%s %s' % (r['op'], r['id']) for r in bad])
    return Result('CAN-01', PASS, 'the unmutated copy reproduces the original statuses of %d target(s) across %d '
                  'operator(s)' % (len(rep), run['n_applicable']))


@check('CAN-02')
def check_can_02(ctx: common.Ctx) -> Result:
    run = canary_run(ctx)
    problems = list(run['errors'])
    if not run['mutants'] and not problems:
        return Result('CAN-02', NOT_APPLICABLE, 'no applicable mutant')
    surv = [m for m in run['mutants'] if not m.get('caught')]
    for m in surv:
        problems.append('mutant %d %s survived (%s): %s' % (m['index'], m['op'], m.get('description', ''), m['why']))
    if problems:
        return Result('CAN-02', FAIL, '%d of %d mutants survived (each survivor becomes an incident): %s'
                      % (len(surv), run['n_mutants'], '; '.join(problems[:6])), evidence=problems[:40])
    return Result('CAN-02', PASS, 'all %d applicable mutants caught (seed %s)' % (run['n_mutants'], run['seed']))


@check('CAN-03')
def check_can_03(ctx: common.Ctx) -> Result:
    run = canary_run(ctx)
    frac = run['applicable_fraction']
    na = ['%s (%s)' % (o['id'], o.get('reason') or o.get('error')) for o in run['operators'] if not o.get('applicable')]
    if frac < MIN_APPLICABLE_FRACTION:
        return Result('CAN-03', FAIL, '%d of %d operators applicable (%.0f%% < %.0f%%); not applicable: %s'
                      % (run['n_applicable'], run['n_operators'], 100 * frac, 100 * MIN_APPLICABLE_FRACTION,
                         '; '.join(na)), evidence=na)
    return Result('CAN-03', PASS, '%d of %d operators applicable (%.0f%%)%s' % (
        run['n_applicable'], run['n_operators'], 100 * frac, ('; not applicable: %s' % '; '.join(na)) if na else ''))


def build_report(ctx: common.Ctx, results: Sequence[Result]) -> dict:
    run = ctx._cache.get('wp4_canary')
    err = None
    if isinstance(run, BaseException):
        err = '%s: %s' % (type(run).__name__, run)
        run = None
    run = run or {}
    rc = common.exit_code(results)
    try:
        head = ctx.git.head()
    except MissingInput as e:
        head = 'unavailable: %s' % e
    rep = {
        'schema': SCHEMA_CANARY_REPORT, 'delivery': ctx.delivery_rel, 'utc': common.utc_now(),
        'seed': ctx.options.get('canary_seed'), 'n_requested': ctx.options.get('canary_n'),
        'n_mutants': run.get('n_mutants', 0), 'detected': run.get('detected', 0),
        'checker_sha256': common.checker_sha256(), 'checker_version': common.CHECKER_VERSION,
        'git_mode': ctx.git_mode, 'git_head': head, 'endpoint_lock_sha256': ctx.endpoint_lock_sha256,
        'source_lock_sha256': ctx.source_lock_sha256, 'tier': ctx.tier,
        'decisive_role': run.get('decisive_role'), 'applicable_fraction': run.get('applicable_fraction'),
        'operators': run.get('operators', []), 'reproduction': run.get('reproduction', []),
        'mutants': run.get('mutants', []), 'copied_outside_delivery': run.get('copied_outside_delivery', []),
        'pointer_copy': run.get('pointer_copy'), 'copies_kept': run.get('copies_kept'), 'error': err,
        'results': [r.to_dict() for r in results], 'exit_code': rc, 'verdict': common.VERDICT_BY_EXIT[rc],
    }
    return rep


def write_report(ctx: common.Ctx, rep: dict) -> Path:
    d = ctx.path('checks')
    d.mkdir(parents=True, exist_ok=True)
    path = d / ('CANARY_%s.json' % rep['utc'])
    k = 1
    while path.exists():
        path = d / ('CANARY_%s_%d.json' % (rep['utc'], k))
        k += 1
    path.write_text(common.dumps_pretty(common._jsonable(rep)), encoding='utf-8')
    return path


def cmd_canary(args: Any) -> int:
    seed = getattr(args, 'seed', None)
    if not isinstance(seed, str) or not re.fullmatch(r'[0-9a-fA-F]{4,}', seed.strip()):
        sys.stderr.write('canary: --seed must be hex (at least 4 digits), chosen by the reviewer after reading the lock\n')
        return common.EXIT_ERROR
    n = getattr(args, 'n', DEFAULT_N)
    if not admitcheck.is_int(n) or n < 1:
        sys.stderr.write('canary: --n must be a positive integer\n')
        return common.EXIT_ERROR
    out = getattr(args, 'out', None)
    ctx = common.make_ctx(args, delivery=args.dir, stage='canary',
                          options={'canary_seed': seed.strip().lower(), 'canary_n': n, 'invoked_by': 'canary'})
    ok, why = common.v1_status(ctx.delivery)
    if not ok:
        sys.stderr.write('legacy delivery: use retro (%s)\n' % why)
        return common.EXIT_MISSING
    if out:
        op = Path(out).resolve()
        try:
            op.relative_to(Path(ctx.repo_root).resolve())
            sys.stderr.write('canary: --out %s must lie outside the repository (copies would look like deliveries)\n' % op)
            return common.EXIT_ERROR
        except ValueError:
            pass
        ctx.options['canary_out'] = str(op)
    results = common.run_bundle(ctx, 'canary')
    rep = build_report(ctx, results)
    path = write_report(ctx, rep)
    rc = rep['exit_code']
    if getattr(args, 'json', False):
        print(json.dumps(common._jsonable(rep), indent=2, ensure_ascii=False))
    else:
        print(common.format_results(results))
        for m in rep['mutants']:
            print('  %-4s %-4s %-8s %s' % (m['index'], m['op'], 'caught' if m.get('caught') else 'SURVIVED',
                                           (m.get('description') or '')[:150]))
        print('canary: %s (exit %d) %d/%d mutants caught; report %s' % (
            rep['verdict'], rc, rep['detected'], rep['n_mutants'], path.relative_to(ctx.repo_root).as_posix()))
    return rc
