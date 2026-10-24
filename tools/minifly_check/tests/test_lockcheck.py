"""Unit tests for lockcheck.py (WP2): init, lock build, LOCK-01..13, AMD-01..05, WIT-01.

Synthetic TimelineGit fixtures in temp dirs only; delivery code is parsed, never run. stdlib only.
"""
import argparse
import contextlib
import copy
import hashlib
import io
import json
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
if str(KIT) not in sys.path:
    sys.path.insert(0, str(KIT))

import common  # noqa: E402
import lockcheck  # noqa: E402
from common import FAIL, MISSING, NOT_APPLICABLE, PASS, WARN  # noqa: E402

DNAME = 'minifly_unit_dev_v1_20261101'
CODEX = ['origin/codex/work']
CLAUDE = ['origin/claude/review', 'origin/codex/work']
REVIEWER_TRAILER = {'Acting-Agent': ['claude:s1']}  # the reviewer's commits (WIT-01 reads trailer AND namespace)
WP2_IDS = (['LOCK-%02d' % i for i in range(1, 14)] + ['AMD-%02d' % i for i in range(1, 6)] + ['WIT-01'])

CONSTRAINTS = {
    'schema': 'minifly.constraints.v1',
    'native_forbidden': {
        'imports': ['torch.autograd', 'torch.optim', 'torch.nn.MultiheadAttention', 'transformers', 'jax',
                    'sklearn.linear_model'],
        'calls': ['.backward(', 'scaled_dot_product_attention(', 'numpy.linalg.lstsq(', 'numpy.linalg.pinv(',
                  'torch.linalg.lstsq('],
        'identifiers': ['transformer', 'attention', 'elm', 'requires_grad=True'],
    },
    'supplied_arithmetic_heuristics': ['int() over decoded operand bytes', 'carry or digit tables',
                                       "'+' applied to parsed operands"],
    'io': {'raw_bytes': True, 'own_eos_required': True},
    'warmup': {'start_ceiling': '1/16', 'ramp': 'gradual'},
    'threshold_floors': {
        'margin_abs': 0.05, 'absolute_floor_above_chance': 0.05, 'per_world_floor': -0.05, 'min_worlds_nonpilot': 8,
        'min_items_per_stratum': 8, 'input_contrast_min': 2.0, 'prompt_id_margin_min': 0.1,
        'taught_acq_margin_min': 0.05, 'min_realized_dose': 0.05, 'shuf_band': [0.35, 0.65], 'old_max_drop': 0.1,
        'old_acq_margin': 0.05, 'analytic_input_snr_min': 0.5, 'oc_min_power': 0.8, 'alpha_max': 0.05},
    'tier_thresholds': {'t0_max_cpu_h': 1, 't1_max_cpu_h': 4, 't1_max_records': 20000, 'family_t1_max_cpu_h': 12},
    'default_grant_caps': {'max_cpu_h': 2, 'max_records': 5000, 'max_cycles': 3},
    'e3_cadence_max': 2,
    'e3_cadence_counts_legacy': False,
    'review_debt_max_open': 3,
    'required_arms_by_level': {'E1': ['candidate', 'shuf'], 'E2': ['candidate', 'shuf', 'positive_control'],
                               'E3': ['candidate', 'no_prior_teaching', 'shuf', 'oracle_plumbing']},
    'testbed_floor_grid': [{'id': 'FRF_W256_LR0.001', 'learner': 'FROZEN_RANDOM_FEATURES', 'width': 256,
                            'readout_lr': 0.001}],
    'testbed_ceiling_grid': [{'id': 'EXACT_BP', 'learner': 'EXACT_BP'},
                             {'id': 'ORACLE_FEATURES', 'learner': 'ORACLE_FEATURES'}],
    'confirm_alpha_spending': 'geometric_0.05_2^-j',
}

NATIVE_PY = '''"""Tiny native learner stub (parsed, never executed)."""
import os

import helpers


class Learner:
    def __init__(self, width):
        self.width = width
        self.state = [0] * width

    def update(self, byte):
        self.state[byte % self.width] += 1
        return helpers.clip(self.state[0])
'''

HELPERS_PY = '''def clip(v):
    return max(0, min(255, v))
'''

RUN_PY = '''"""Runner stub (parsed, never executed)."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'lib'))

import native
import util


def main():
    learner = native.Learner(16)
    print(json.dumps(util.encode(learner.update(1))))


if __name__ == '__main__':
    main()
'''

UTIL_PY = '''def encode(v):
    return '%02x' % v
'''

ADMIT_PY = '''"""Evaluator stub (parsed, never executed)."""
import json


def main():
    print(json.dumps({'status': None}))
'''


def sha(data):
    if isinstance(data, str):
        data = data.encode('utf-8')
    return hashlib.sha256(data).hexdigest()


def jdump(obj):
    return json.dumps(obj, indent=1, sort_keys=True) + '\n'


def jsonl(rows):
    return ''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows)


def setp(obj, dotted, value):
    parts = dotted.split('.')
    cur = obj
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


class Fx:
    """TimelineGit fixture: <root>/repo is the worktree; commit() snapshots changed files into history/."""

    def __init__(self, root):
        self.root = Path(root)
        self.repo = self.root / 'repo'
        self.repo.mkdir(parents=True, exist_ok=True)
        self.commits = []
        self.committed = {}
        self.n = 0

    def write(self, rel, data):
        p = self.repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, (dict, list)):
            data = jdump(data)
        if isinstance(data, str):
            data = data.encode('utf-8')
        p.write_bytes(data)
        return p

    def read(self, rel):
        return (self.repo / rel).read_bytes()

    def read_json(self, rel):
        return json.loads(self.read(rel).decode('utf-8'))

    def delete(self, rel):
        (self.repo / rel).unlink()

    def _files(self):
        out = {}
        for p in sorted(self.repo.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:
                out[p.relative_to(self.repo).as_posix()] = p.read_bytes()
        return out

    def commit(self, paths=None, branches=CODEX, trailers=None, utc=None, author='Pencilbard'):
        files = self._files()
        if paths is None:
            paths = sorted([p for p, b in files.items() if self.committed.get(p) != b]
                           + [p for p in self.committed if p not in files])
        self.n += 1
        csha = hashlib.sha1(('commit-%d-%s' % (self.n, self.root)).encode()).hexdigest()
        for p in paths:
            if p in files:
                h = self.root / 'history' / csha / p
                h.parent.mkdir(parents=True, exist_ok=True)
                h.write_bytes(files[p])
                self.committed[p] = files[p]
            else:
                self.committed.pop(p, None)
        self.commits.append({'sha': csha, 'utc': utc or '202611%02dT%02d0000Z' % (1 + self.n // 24, self.n % 24),
                             'author': author, 'trailers': trailers or {}, 'paths': list(paths),
                             'branches': list(branches)})
        (self.root / 'timeline.json').write_text(json.dumps({'head': csha, 'commits': self.commits}, indent=1))
        return csha

    def git(self):
        return common.TimelineGit(self.root)

    def ctx(self, name=DNAME, stage='roster', tier=None, dry=False, options=None):
        root = self.repo.resolve()
        return common.Ctx(repo_root=root, delivery=root / name, git=self.git(), kb=common.KB(root),
                          rules=common.load_rules(), stage=stage, tier=tier, dry=dry, options=dict(options or {}))

    def args(self, **kw):
        ns = argparse.Namespace(repo_root=str(self.repo), git='timeline:%s' % self.root, json=False, extra=[])
        for k, v in kw.items():
            setattr(ns, k, v)
        return ns


def write_kb(fx, constraints=None, findings=None, uds=None, ledger=None):
    kb = 'PROJECT_DOCUMENTS/KB/'
    fx.write(kb + 'CONSTRAINTS.json', constraints or CONSTRAINTS)
    fx.write(kb + 'FINDINGS.jsonl', jsonl(findings if findings is not None else [
        {'id': 'F-TEST-1', 'kind': 'finding', 'status': 'active', 'claim': 'synthetic'}]))
    fx.write(kb + 'USER_DECISIONS.jsonl', jsonl(uds or []))
    fx.write(kb + 'DELIVERY_LEDGER.jsonl', jsonl(ledger or []))
    fx.write(kb + 'CONFIRM_LEDGER.jsonl', '')
    fx.write(kb + 'REVIEW_DEBT.jsonl', '')
    return [kb + n for n in ('CONSTRAINTS.json', 'FINDINGS.jsonl', 'USER_DECISIONS.jsonl', 'DELIVERY_LEDGER.jsonl',
                             'CONFIRM_LEDGER.jsonl', 'REVIEW_DEBT.jsonl')]


SPLIT_ITEMS = {'HELD': [('h%d' % i, '%d+%d=' % (i % 10, (i + 3) % 10)) for i in range(1, 11)],
               'OLD': [('o%d' % i, '%d+5=' % (i % 10)) for i in range(1, 11)],
               'NEW': [('n%d' % i, '5+%d=' % (i % 10)) for i in range(1, 11)]}


def write_manifests(fx, name=DNAME):
    """HELD/OLD/NEW manifests (SCHEMAS.md 5.2); h1 is 3+4= (outputs_rows' item)."""
    paths = []
    for pool, items in SPLIT_ITEMS.items():
        rows = []
        for iid, prompt in items:
            a, b = prompt[:-1].split('+')
            rows.append({'item_id': iid, 'prompt_hex': prompt.encode().hex(), 'expected_hex': str(int(a) + int(b)).encode().hex()})
        if pool == 'HELD':
            rows[0] = {'item_id': 'h1', 'prompt_hex': '332b343d', 'expected_hex': '37'}
        rel = '%s/manifests/%s.jsonl' % (name, pool)
        fx.write(rel, jsonl(rows))
        paths.append(rel)
    return paths


def split_shas(fx, name=DNAME):
    blobs = {}
    for key, pool in common.SPLIT_MANIFESTS:
        p = fx.repo / name / 'manifests' / ('%s.jsonl' % pool)
        blobs[key] = p.read_bytes() if p.is_file() else None
    return common.split_hashes(blobs)


def base_lock(fx, name=DNAME, **over):
    lock = json.loads((KIT / 'templates' / 'ENDPOINT_LOCK.template.json').read_text(encoding='utf-8'))
    cons_sha = sha(fx.read('PROJECT_DOCUMENTS/KB/CONSTRAINTS.json'))
    lock.update({'delivery': name, 'protocol_id': name + '-p1', 'created_utc': '20261101T000000Z',
                 'author_agent': 'codex', 'tier_declared': 1, 'intent': 'diagnostic', 'claim_level_target': 'E3',
                 'constraints_sha256': cons_sha, 'positive_control_arm': 'ORACLE'})
    lock['task'].update({'id': 'add1', 'family': 'addition_single_digit'})
    lock['task']['format']['gen_cap_bytes'] = 4
    lock['objective_distance'].update({'task_family': 'addition_single_digit', 'held': 'unseen_composition'})
    lock['units']['witness'] = 'pair_rectangle'
    held_sha, split_sha = split_shas(fx, name)
    lock['split'].update({'split_sha256': split_sha or 'a' * 64, 'held_set_sha256': held_sha or 'b' * 64})
    lock['exposure_streams'][0]['phase'] = 'teach'
    arms = {a['id']: a for a in lock['arms']}
    arms['NPT'].update({'zero_channels': ['old_relation'], 'teaching_withheld': ['held_pairs']})
    arms['ORACLE']['declared_changed'] = ['readout']
    lock['ports'] = [{'id': 'old_relation', 'signs': ['+', '-']}]
    lock['primary'].update({'task_id': 'add1', 'chance': 0.1, 'absolute_floor': 0.15,
                            'n_items_per_pool': {'TAUGHT': 10, 'OLD': 10, 'NEW': 10, 'HELD': 10}})
    lock['gates'] = [{'id': 'stop_gate', 'description': 'stops at EOS'}]
    lock['checkpoints'] = {'prefix': 10, 'old_end': 20, 'final': 30}
    lock['worlds'] = {'technical': ['wt_unit_1'], 'dev': ['wd_unit_1', 'wd_unit_2'], 'rehearsal': ['wr_unit_1'],
                      'official': {'mode': 'listed', 'ids': ['wo_unit_%d' % i for i in range(1, 9)], 'n': 8},
                      'official_role_intended': 'MAIN'}
    lock['budget'] = {'max_records': 1000, 'max_cpu_h': 1}
    lock['admission_thresholds']['analytic_input_snr'] = 0.9
    lock['testbed'] = {'ceiling_grid': CONSTRAINTS['testbed_ceiling_grid'],
                       'floor_grid': CONSTRAINTS['testbed_floor_grid']}
    lock['files'].update({'native': ['native.py'], 'runner': ['run.py']})
    lock['endpoint_files_sha256'] = {'admit.py': sha(fx.read(name + '/admit.py'))}
    lock['lineage'] = {'parent': None, 'changes_declared': [], 'priors_added': []}
    lock['predictions'] = [
        {'id': 'P1', 'target': 'primary', 'point': 0.3, 'interval90': [0.1, 0.5], 'p_pass': 0.4,
         'basis_findings': ['F-TEST-1'], 'base_rate_forecast': {'value': 0.5, 'source': 'laplace 0/0'}},
        {'id': 'P2', 'target': 'guard:old_drop', 'point': 0.02, 'interval90': [0.0, 0.08], 'p_pass': 0.7,
         'basis_findings': ['none'], 'base_rate_forecast': {'value': 0.5, 'source': 'laplace 0/0'}},
        {'id': 'P3', 'target': 'guard:old_acq', 'point': 0.2, 'interval90': [0.1, 0.3], 'p_pass': 0.6,
         'basis_findings': ['none'], 'base_rate_forecast': {'value': 0.5, 'source': 'laplace 0/0'}},
    ]
    for k, v in over.items():
        setp(lock, k, v)
    return lock


def write_code(fx, name=DNAME, native=NATIVE_PY, run=RUN_PY):
    fx.write(name + '/admit.py', ADMIT_PY)
    fx.write(name + '/params.json', {'lr': 0.1})
    fx.write(name + '/native.py', native)
    fx.write(name + '/helpers.py', HELPERS_PY)
    fx.write(name + '/run.py', run)
    fx.write(name + '/lib/util.py', UTIL_PY)


def build_source_lock(fx, name=DNAME, allow_problems=False):
    ctx = fx.ctx(name)
    b = lockcheck.build_closure(ctx)
    if b['problems'] and not allow_problems:
        raise AssertionError('closure problems: %s' % b['problems'])
    sl = lockcheck.source_lock_from_build(ctx, b)
    fx.write(name + '/SOURCE_LOCK.json', sl)
    return sl


def outputs_rows(sl, world='wd_unit_1', role='dev', arm='W'):
    return [{'world': world, 'world_role': role, 'arm': arm, 'pool': 'HELD', 'checkpoint': 'final', 'item_id': 'h1',
             'prompt_hex': '332b343d', 'expected_hex': '37', 'stream_hex': '370a0000', 'emitted_hex': '37',
             'own_eos': True, 'answer_path': 'native', 'native_closure_sha': sl['native_closure_sha'],
             'params_sha256': sl['params_sha256'], 'attempt_id': 'ATT-1'}]


def write_review(fx, name=DNAME, **over):
    d = fx.repo / name
    rv = {'schema': 'minifly.review.v1', 'delivery': name, 'reviewer_agent': 'claude', 'author_agent': 'codex',
          'reviewed_endpoint_lock_sha256': sha((d / 'ENDPOINT_LOCK.json').read_bytes()),
          'source_lock_acks': [{'sha': sha((d / 'SOURCE_LOCK.json').read_bytes()), 'utc': '20261103T000000Z',
                                'note': 'acked'}],
          'findings': [], 'decision': 'ADMIT'}
    rv.update(over)
    fx.write(name + '/review/REVIEW.json', rv)
    return rv


def standard_history(fx, name=DNAME, lock_over=None, native=NATIVE_PY, run=RUN_PY, results=True, review=True,
                     allow_problems=False):
    """c1 KB + lock + admit + params; c2 code; c3 SOURCE_LOCK; c4 review (claude); c5 dev results."""
    write_kb(fx)
    write_code(fx, name, native=native, run=run)
    man = write_manifests(fx, name)
    fx.write(name + '/ENDPOINT_LOCK.json', base_lock(fx, name, **(lock_over or {})))
    fx.commit([p for p in fx._files() if p.startswith('PROJECT_DOCUMENTS/')] +
              [name + '/ENDPOINT_LOCK.json', name + '/admit.py', name + '/params.json'] + man)
    fx.commit()
    sl = build_source_lock(fx, name, allow_problems=allow_problems)
    fx.commit()
    if review:
        write_review(fx, name)
        fx.commit(branches=CLAUDE, trailers={'Acting-Agent': ['claude:s1']})
    if results:
        fx.write(name + '/results/dev/OUTPUTS.jsonl', jsonl(outputs_rows(sl)))
        fx.write(name + '/results/receipts/wd_unit_1/W.json',
                 {'world': 'wd_unit_1', 'world_role': 'dev', 'arm': 'W', 'records': 10,
                  'native_closure_sha': sl['native_closure_sha'], 'params_sha256': sl['params_sha256']})
        fx.commit()
    return sl


def run(ctx, cid):
    rule = common.rule_index(ctx.rules)[cid]
    return common.run_check(ctx, rule)


def status(ctx, cid):
    return run(ctx, cid)[0].status


class KBStub:
    """Install a stub kb module (deliveries_since_last_e3) for LOCK-09 and init."""

    def __init__(self, value):
        self.value = value

    def __enter__(self):
        self.prev = sys.modules.get('kb', '__absent__')
        mod = types.ModuleType('kb')
        mod.deliveries_since_last_e3 = lambda kb, v=self.value: v
        sys.modules['kb'] = mod
        return mod

    def __exit__(self, *exc):
        if self.prev == '__absent__':
            sys.modules.pop('kb', None)
        else:
            sys.modules['kb'] = self.prev


class TmpCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix='mfc_wp2_'))
        self.fx = Fx(self.tmp / 'fx')
        self.kbstub = KBStub(0)
        self.kbstub.__enter__()

    def tearDown(self):
        self.kbstub.__exit__(None, None, None)
        shutil.rmtree(self.tmp, ignore_errors=True)

    def assertStatus(self, ctx, cid, want, contains=None):
        res = run(ctx, cid)
        got = [r.status for r in res]
        self.assertIn(want, got, '%s: %s' % (cid, [(r.status, r.detail) for r in res]))
        if contains:
            self.assertTrue(any(contains in r.detail for r in res), '%s detail lacks %r: %s'
                            % (cid, contains, [r.detail for r in res]))
        return res


# ------------------------------------------------------------------------------------------------
class TestRegistry(unittest.TestCase):
    def test_all_wp2_ids_registered(self):
        import seal  # noqa: F401
        for cid in WP2_IDS + ['SEAL-%02d' % i for i in range(1, 6)] + ['CONF-01']:
            self.assertIn(cid, common.CHECKS, cid)
        rules = common.rule_index(common.load_rules())
        for cid in WP2_IDS:
            fn, err = common.resolve_check(rules[cid])
            self.assertIsNotNone(fn, err)

    def test_sources_parse_as_python39(self):
        import ast
        for f in ('lockcheck.py', 'seal.py', 'tests/test_lockcheck.py', 'tests/test_seal.py'):
            ast.parse((KIT / f).read_text(encoding='utf-8'), feature_version=(3, 9))


# ------------------------------------------------------------------------------------------------
class TestCleanLock(TmpCase):
    def test_clean_lock_all_pass(self):
        standard_history(self.fx)
        ctx = self.fx.ctx(stage='roster')
        ctx.tier = common.compute_tier(ctx)[0]
        self.assertEqual(ctx.tier, 2)  # an official run is planned
        bad = []
        for cid in WP2_IDS:
            for r in run(ctx, cid):
                if r.status not in (PASS, NOT_APPLICABLE):
                    bad.append((cid, r.status, r.detail))
        self.assertEqual(bad, [])
        na = [cid for cid in WP2_IDS if status(ctx, cid) == NOT_APPLICABLE]
        self.assertEqual(sorted(na), ['AMD-02', 'AMD-03'])

    def test_lock_bundle_via_run_bundle_has_no_wp2_blockers(self):
        standard_history(self.fx)
        ctx = self.fx.ctx(stage='lock')
        res = common.run_bundle(ctx, 'lock')
        wp2 = [(r.id, r.status, r.detail) for r in res if r.id in WP2_IDS and r.status not in (PASS, NOT_APPLICABLE)]
        self.assertEqual(wp2, [])


# ------------------------------------------------------------------------------------------------
class TestLock01to04(TmpCase):
    def setUp(self):
        super().setUp()
        standard_history(self.fx)

    def edit_lock(self, fn):
        p = DNAME + '/ENDPOINT_LOCK.json'
        lock = self.fx.read_json(p)
        fn(lock)
        self.fx.write(p, lock)
        return self.fx.ctx()

    def test_lock01_fill_and_enum(self):
        ctx = self.edit_lock(lambda l: (setp(l, 'primary.chance', '__FILL__'), setp(l, 'intent', 'confirm')))
        res = self.assertStatus(ctx, 'LOCK-01', FAIL, 'primary.chance')
        self.assertIn('intent', res[0].detail)

    def test_lock01_null_required_and_bad_protocol(self):
        ctx = self.edit_lock(lambda l: (setp(l, 'budget.max_cpu_h', None), setp(l, 'protocol_id', 'x-p1')))
        res = self.assertStatus(ctx, 'LOCK-01', FAIL, 'budget.max_cpu_h is null')
        self.assertIn('protocol_id', res[0].detail)

    def test_lock01_evaluator_prefix_arm(self):
        def f(l):
            l['arms'][0]['id'] = 'CEILING_W'
            l['primary']['candidate_arm'] = 'CEILING_W'
        res = self.assertStatus(self.edit_lock(f), 'LOCK-01', FAIL, 'evaluator prefix')
        self.assertIn('primary.candidate_arm CEILING_W is an evaluator', res[0].detail)

    def test_lock01_testbed_arm_listed_as_other_is_fine(self):
        def f(l):
            l['arms'].append({'id': 'CEILING_EXACT_BP', 'role': 'other', 'zero_channels': [],
                              'declared_changed': [], 'teaching_withheld': [], 'min_realized_dose': None})
        self.assertStatus(self.edit_lock(f), 'LOCK-01', PASS)

    def test_lock02_missing_role_and_zero_channels(self):
        def f(l):
            l['arms'] = [a for a in l['arms'] if a['role'] != 'shuf']
            for a in l['arms']:
                if a['role'] == 'no_prior_teaching':
                    a['zero_channels'] = []
        res = self.assertStatus(self.edit_lock(f), 'LOCK-02', FAIL, 'role shuf')
        self.assertIn('zero_channels', res[0].detail)

    def test_lock02_role_candidate_only_on_the_primary_candidate(self):
        # Round-1 repair (verifier probe E2): a control relabelled 'candidate' is neither tested nor compared.
        def f(l):
            for a in l['arms']:
                if a['role'] == 'shuf':
                    a['role'] = 'candidate'
        res = self.assertStatus(self.edit_lock(f), 'LOCK-02', FAIL,
                                'have role candidate but are not primary.candidate_arm')
        self.assertIn('role shuf', res[0].detail)   # the required shuf role is now missing too

    def test_lock02_units_required_for_e2(self):
        def f(l):
            l['claim_level_target'] = 'E2'
            l['units'] = None
            l['arms'].append({'id': 'POS', 'role': 'positive_control', 'zero_channels': [], 'declared_changed': ['x'],
                              'teaching_withheld': [], 'min_realized_dose': None})
        self.assertStatus(self.edit_lock(f), 'LOCK-02', FAIL, 'units block')

    def test_lock03_objective(self):
        res = self.assertStatus(self.fx.ctx(), 'LOCK-03', PASS)
        self.assertEqual(res[0].data['gate_type'], 'objective')

    def test_lock03_conditional_primary_forced_component_fails_at_e3(self):
        ctx = self.edit_lock(lambda l: setp(l, 'primary.unconditional', False))
        res = self.assertStatus(ctx, 'LOCK-03', FAIL, 'forced to component')
        self.assertEqual(res[0].data['gate_type'], 'component')

    def test_lock03_component_allowed_at_e1(self):
        ctx = self.edit_lock(lambda l: (setp(l, 'primary.metric', 'teacher_forced_acc'),
                                        setp(l, 'claim_level_target', 'E1')))
        self.assertStatus(ctx, 'LOCK-03', PASS, 'forced to component')

    def test_lock03_claim_says_objective(self):
        ctx = self.edit_lock(lambda l: (setp(l, 'objective_distance.output', 'own_bytes_evaluator_length'),
                                        setp(l, 'claim_level_target', 'E1')))
        self.fx.write(DNAME + '/CLAIM.json', {'gate_type': 'objective'})
        self.assertStatus(self.fx.ctx(), 'LOCK-03', FAIL, 'CLAIM.json gate_type')

    def test_lock04_floors(self):
        def f(l):
            l['primary']['margin_abs'] = 0.04
            l['primary']['aggregator']['alpha'] = 0.1
            l['guards'] = [g for g in l['guards'] if g['kind'] != 'old_acq']
            l['admission_thresholds']['shuf_band'] = [0.3, 0.7]
        res = self.assertStatus(self.edit_lock(f), 'LOCK-04', FAIL, 'margin_abs')
        for s in ('alpha', 'old_acq', 'shuf_band'):
            self.assertIn(s, res[0].detail)

    def test_lock04_absolute_floor_above_chance(self):
        self.assertStatus(self.edit_lock(lambda l: setp(l, 'primary.absolute_floor', 0.12)), 'LOCK-04', FAIL,
                          'absolute_floor')

    def test_lock04_pilot_skips_min_worlds(self):
        def f(l):
            l['primary']['min_worlds'] = 2
            l['worlds']['official_role_intended'] = 'PILOT'
        self.assertStatus(self.edit_lock(f), 'LOCK-04', PASS)


# ------------------------------------------------------------------------------------------------
class TestLockBuildAndLock05(TmpCase):
    def test_lock_build_writes_source_lock_and_lock05_passes(self):
        write_kb(self.fx)
        write_code(self.fx)
        self.fx.write(DNAME + '/ENDPOINT_LOCK.json', base_lock(self.fx))
        self.fx.commit()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = lockcheck.cmd_lock_build(self.fx.args(action='build', dir=DNAME))
        self.assertEqual(rc, 0, out.getvalue())
        sl = self.fx.read_json(DNAME + '/SOURCE_LOCK.json')
        self.assertEqual(sl['schema'], common.SCHEMA_SOURCE_LOCK)
        self.assertEqual(sl['checker_version'], common.CHECKER_VERSION)
        self.assertEqual(sl['endpoint_lock_sha256'], sha(self.fx.read(DNAME + '/ENDPOINT_LOCK.json')))
        keys = set(sl['closure_sha256'])
        for f in ('native.py', 'helpers.py', 'run.py', 'lib/util.py', 'admit.py'):
            self.assertIn('%s/%s' % (DNAME, f), keys)
        self.assertEqual(set(sl['native_closure_sha256']), {DNAME + '/native.py', DNAME + '/helpers.py'})
        self.assertEqual(sl['params_sha256'], sha(self.fx.read(DNAME + '/params.json')))
        self.assertEqual(sl['dynamic_imports'], [])
        self.assertIn('json', sl['env']['stdlib_modules'])
        self.assertTrue(sl['env']['python'])
        self.assertTrue(all(len(v) == 64 for v in sl['closure_sha256'].values()))
        self.assertStatus(self.fx.ctx(stage='technical'), 'LOCK-05', PASS)
        before = self.fx.read(DNAME + '/SOURCE_LOCK.json')
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = lockcheck.cmd_lock_build(self.fx.args(action='build', dir=DNAME))
        self.assertEqual(rc, 0)
        self.assertIn('unchanged', out.getvalue())
        self.assertEqual(self.fx.read(DNAME + '/SOURCE_LOCK.json'), before)  # the sha reviewers ack is stable
        self.fx.write(DNAME + '/params.json', {'lr': 0.2})
        with contextlib.redirect_stdout(io.StringIO()):
            rc = lockcheck.cmd_lock_build(self.fx.args(action='build', dir=DNAME))
        self.assertEqual(rc, 0)
        self.assertEqual(self.fx.read_json(DNAME + '/SOURCE_LOCK.json')['params_sha256'],
                         sha(self.fx.read(DNAME + '/params.json')))

    def test_closure_missing_bootstrap_fails(self):
        native = NATIVE_PY.replace('import helpers\n', 'import helpers\nimport bootstrap\n')
        write_kb(self.fx)
        write_code(self.fx, native=native)
        self.fx.write(DNAME + '/bootstrap.py', 'SEED_SALT = "x"\n')
        self.fx.write(DNAME + '/ENDPOINT_LOCK.json', base_lock(self.fx))
        self.fx.commit()
        sl = build_source_lock(self.fx)
        # a hand-made lock script that hashed only the files it knew about
        sl['closure_sha256'].pop(DNAME + '/bootstrap.py')
        sl['native_closure_sha256'].pop(DNAME + '/bootstrap.py')
        self.fx.write(DNAME + '/SOURCE_LOCK.json', sl)
        res = self.assertStatus(self.fx.ctx(stage='technical'), 'LOCK-05', FAIL, 'bootstrap.py')
        self.assertIn('missing from SOURCE_LOCK', res[0].detail)

    def test_stale_source_lock_after_code_change(self):
        standard_history(self.fx)
        self.fx.write(DNAME + '/helpers.py', HELPERS_PY + '\n\ndef extra():\n    return 1\n')
        self.assertStatus(self.fx.ctx(stage='technical'), 'LOCK-05', FAIL, 'helpers.py sha256')

    def test_sys_path_insert_to_sibling_delivery_fails(self):
        sib = 'minifly_sib_dev_v1_20261020'
        native = NATIVE_PY.replace(
            'import helpers\n',
            'import sys\nimport helpers\nsys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '
            "'..', '%s'))\nimport core\n" % sib)
        write_kb(self.fx)
        write_code(self.fx, native=native)
        self.fx.write(sib + '/core.py', 'def step(x):\n    return x\n')
        self.fx.write(DNAME + '/ENDPOINT_LOCK.json', base_lock(self.fx))
        self.fx.commit()
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = lockcheck.cmd_lock_build(self.fx.args(action='build', dir=DNAME))
        self.assertEqual(rc, 1)
        self.assertIn('collision', err.getvalue())
        self.assertFalse((self.fx.repo / DNAME / 'SOURCE_LOCK.json').exists())
        build_source_lock(self.fx, allow_problems=True)
        res = self.assertStatus(self.fx.ctx(stage='technical'), 'LOCK-05', FAIL, 'collision')
        self.assertIn(sib + '/core.py', res[0].detail)

    def test_non_literal_sys_path_and_unresolved_import_fail(self):
        run = RUN_PY.replace("sys.path.insert(0, os.path.join(HERE, 'lib'))",
                             "sys.path.insert(0, os.environ['EXTRA'])\nsys.path.insert(0, os.path.join(HERE, 'lib'))")
        run = run.replace('import util\n', 'import util\nimport no_such_module_xyz\n')
        write_kb(self.fx)
        write_code(self.fx, run=run)
        self.fx.write(DNAME + '/ENDPOINT_LOCK.json', base_lock(self.fx))
        self.fx.commit()
        build_source_lock(self.fx, allow_problems=True)
        res = self.assertStatus(self.fx.ctx(stage='technical'), 'LOCK-05', FAIL, 'non-literal sys.path')
        self.assertIn('no_such_module_xyz', res[0].detail + ' '.join(res[0].evidence))

    def test_optional_import_and_dynamic_import_listed(self):
        run = RUN_PY.replace('import util\n', 'import util\nimport importlib\ntry:\n    import no_such_optional_xyz\n'
                             'except ImportError:\n    no_such_optional_xyz = None\n')
        run = run.replace('    learner = native.Learner(16)\n',
                          '    learner = native.Learner(16)\n    mod = importlib.import_module(sys.argv[1])\n')
        write_kb(self.fx)
        write_code(self.fx, run=run)
        self.fx.write(DNAME + '/ENDPOINT_LOCK.json', base_lock(self.fx))
        self.fx.commit()
        sl = build_source_lock(self.fx)
        self.assertEqual(len(sl['dynamic_imports']), 1)
        self.assertIn('importlib.import_module(sys.argv[1])', sl['dynamic_imports'][0])
        self.assertTrue(sl['dynamic_imports'][0].startswith(DNAME + '/run.py:'))
        self.assertStatus(self.fx.ctx(stage='technical'), 'LOCK-05', PASS)
        sl['dynamic_imports'] = []
        self.fx.write(DNAME + '/SOURCE_LOCK.json', sl)
        self.assertStatus(self.fx.ctx(stage='technical'), 'LOCK-05', FAIL, 'not listed in dynamic_imports')

    def test_imports_inside_match_cases_are_followed(self):
        if sys.version_info < (3, 10):
            self.skipTest('match statements need Python 3.10')
        run = RUN_PY.replace('def main():\n', 'def main():\n    match sys.argv:\n        case [_, "x"]:\n'
                             '            import extra_mod\n        case _:\n            pass\n')
        write_kb(self.fx)
        write_code(self.fx, run=run)
        self.fx.write(DNAME + '/extra_mod.py', 'y = 2\n')
        self.fx.write(DNAME + '/ENDPOINT_LOCK.json', base_lock(self.fx))
        self.fx.commit()
        sl = build_source_lock(self.fx)
        self.assertIn(DNAME + '/extra_mod.py', sl['closure_sha256'])

    def test_literal_import_module_and_relative_package_import(self):
        run = RUN_PY.replace('import util\n', 'import util\nimport importlib\nimportlib.import_module("pkg.mod")\n')
        write_kb(self.fx)
        write_code(self.fx, run=run)
        self.fx.write(DNAME + '/pkg/__init__.py', 'from . import mod\n')
        self.fx.write(DNAME + '/pkg/mod.py', 'from .sub import thing\n')
        self.fx.write(DNAME + '/pkg/sub.py', 'thing = 1\n')
        self.fx.write(DNAME + '/ENDPOINT_LOCK.json', base_lock(self.fx))
        self.fx.commit()
        sl = build_source_lock(self.fx)
        for f in ('pkg/__init__.py', 'pkg/mod.py', 'pkg/sub.py'):
            self.assertIn('%s/%s' % (DNAME, f), sl['closure_sha256'])
        self.assertEqual(sl['dynamic_imports'], [])

    def test_python39_fallbacks_classify_without_importing(self):
        from unittest import mock
        import importlib.metadata as md
        write_kb(self.fx)
        write_code(self.fx)
        self.fx.write(DNAME + '/ENDPOINT_LOCK.json', base_lock(self.fx))
        self.fx.commit()
        with mock.patch.object(sys, 'stdlib_module_names', (), create=True):
            cb = lockcheck.ClosureBuilder(self.fx.ctx())
            self.assertEqual(cb.classify_external('json'), 'stdlib')
            self.assertEqual(cb.classify_external('posixpath'), 'stdlib')
            self.assertEqual(cb.classify_external('no_such_module_xyz'), 'unresolved')
            self.assertNotIn('no_such_module_xyz', sys.modules)
        saved = getattr(md, 'packages_distributions', None)
        try:
            if saved is not None:
                del md.packages_distributions
            pd = lockcheck._packages_distributions()
            self.assertIsInstance(pd, dict)
        finally:
            if saved is not None:
                md.packages_distributions = saved

    def test_missing_source_lock_is_missing(self):
        write_kb(self.fx)
        write_code(self.fx)
        self.fx.write(DNAME + '/ENDPOINT_LOCK.json', base_lock(self.fx))
        self.fx.commit()
        self.assertStatus(self.fx.ctx(stage='technical'), 'LOCK-05', MISSING)


# ------------------------------------------------------------------------------------------------
class TestLock06(TmpCase):
    def test_lock_and_results_in_same_commit_fails(self):
        write_kb(self.fx)
        write_code(self.fx)
        self.fx.write(DNAME + '/ENDPOINT_LOCK.json', base_lock(self.fx))
        self.fx.commit([p for p in self.fx._files() if p.startswith('PROJECT_DOCUMENTS/')])
        sl = build_source_lock(self.fx)
        self.fx.write(DNAME + '/results/dev/OUTPUTS.jsonl', jsonl(outputs_rows(sl)))
        self.fx.commit()  # lock, code, SOURCE_LOCK and results in one commit
        res = self.assertStatus(self.fx.ctx(stage='technical'), 'LOCK-06', FAIL, 'share commit')
        self.assertIn('does not follow the lock commit', res[0].detail)

    def test_rows_before_their_source_lock_fail(self):
        write_kb(self.fx)
        write_code(self.fx)
        self.fx.write(DNAME + '/ENDPOINT_LOCK.json', base_lock(self.fx))
        self.fx.commit()
        ctx = self.fx.ctx()
        b = lockcheck.build_closure(ctx)
        rows = outputs_rows({'native_closure_sha': b['native_closure_sha'], 'params_sha256': b['params_sha256']})
        self.fx.write(DNAME + '/results/dev/OUTPUTS.jsonl', jsonl(rows))
        self.fx.commit()
        build_source_lock(self.fx)
        self.fx.commit()
        self.assertStatus(self.fx.ctx(stage='technical'), 'LOCK-06', FAIL, 'not after its SOURCE_LOCK commit')

    def test_gitkeep_with_lock_is_fine_and_clean_passes(self):
        write_kb(self.fx)
        write_code(self.fx)
        self.fx.write(DNAME + '/ENDPOINT_LOCK.json', base_lock(self.fx))
        self.fx.write(DNAME + '/results/.gitkeep', b'')
        self.fx.commit()
        sl = build_source_lock(self.fx)
        self.fx.commit()
        self.fx.write(DNAME + '/results/dev/OUTPUTS.jsonl', jsonl(outputs_rows(sl)))
        self.fx.commit()
        self.assertStatus(self.fx.ctx(stage='technical'), 'LOCK-06', PASS)

    def test_streams_follow_the_lock(self):
        # SPEC 6.2: the held set is fixed in the lock before any stream exists
        write_kb(self.fx)
        write_code(self.fx)
        self.fx.write(DNAME + '/manifests/TAUGHT/wd_unit_1/W.jsonl', '{}\n')
        self.fx.write(DNAME + '/ENDPOINT_LOCK.json', base_lock(self.fx))
        self.fx.commit()
        self.assertStatus(self.fx.ctx(stage='technical'), 'LOCK-06', FAIL, 'before any stream exists')
        self.fx = Fx(self.tmp / 'fx2')
        write_kb(self.fx)
        write_code(self.fx)
        self.fx.write(DNAME + '/ENDPOINT_LOCK.json', base_lock(self.fx))
        self.fx.commit()
        self.fx.write(DNAME + '/manifests/TAUGHT/wd_unit_1/W.jsonl', '{}\n')
        self.fx.commit()
        self.assertStatus(self.fx.ctx(stage='technical'), 'LOCK-06', PASS)

    def test_shallow_is_missing(self):
        standard_history(self.fx)
        tl = json.loads((self.fx.root / 'timeline.json').read_text())
        tl['shallow'] = True
        (self.fx.root / 'timeline.json').write_text(json.dumps(tl))
        self.assertStatus(self.fx.ctx(stage='technical'), 'LOCK-06', MISSING)


# ------------------------------------------------------------------------------------------------
class TestLock07(TmpCase):
    def test_backward_in_native_fails(self):
        native = NATIVE_PY + '\n\ndef train(loss):\n    loss.backward()\n'
        standard_history(self.fx, native=native)
        self.assertStatus(self.fx.ctx(stage='technical'), 'LOCK-07', FAIL, '.backward(')

    def test_forbidden_import_alias_and_identifier(self):
        native = NATIVE_PY + ('\n\nimport torch.nn as nn\n\n\nclass TinyTransformer:\n    def __init__(self):\n'
                              '        self.attn = nn.MultiheadAttention(4, 1)\n')
        standard_history(self.fx, native=native, allow_problems=True)  # torch is not installed here
        res = self.assertStatus(self.fx.ctx(stage='technical'), 'LOCK-07', FAIL, 'torch.nn.MultiheadAttention')
        self.assertIn('TinyTransformer', ' '.join(res[0].evidence))

    def test_requires_grad_and_lstsq(self):
        native = NATIVE_PY + ('\n\nimport numpy as np\n\n\ndef fit(a, b, w):\n    w.requires_grad = True\n'
                              '    return np.linalg.lstsq(a, b)\n')
        standard_history(self.fx, native=native, allow_problems=True)  # numpy may not be installed
        res = self.assertStatus(self.fx.ctx(stage='technical'), 'LOCK-07', FAIL, 'numpy.linalg.lstsq')
        self.assertIn('requires_grad=True', ' '.join(res[0].evidence))

    def test_evaluator_reachable_from_runner_fails(self):
        run = RUN_PY.replace('import util\n', 'import util\nimport admit\n')
        standard_history(self.fx, run=run)
        self.assertStatus(self.fx.ctx(stage='technical'), 'LOCK-07', FAIL, 'evaluator module')

    def test_heuristics_warn_at_t1_fail_at_t2_unless_reviewed(self):
        native = NATIVE_PY + ("\n\ndef parse(prompt):\n    a, b = prompt.split(b'+')\n"
                              '    return int(a.decode()) + int(b.decode())\n')
        standard_history(self.fx, native=native)
        self.assertStatus(self.fx.ctx(stage='technical', tier=1), 'LOCK-07', WARN, 'REVIEW_REQUIRED')
        self.assertStatus(self.fx.ctx(stage='technical', tier=2), 'LOCK-07', FAIL, 'not addressed')
        write_review(self.fx, findings=[{'id': 'R1', 'fm_ids': ['FM-12'], 'status': 'OPEN', 'severity': 'MAJOR'}])
        self.assertStatus(self.fx.ctx(stage='technical', tier=2), 'LOCK-07', FAIL, 'not addressed')
        write_review(self.fx, findings=[{'id': 'R1', 'fm_ids': ['FM-12'], 'status': 'CONFIRMED', 'severity': 'MINOR'}])
        self.assertStatus(self.fx.ctx(stage='technical', tier=2), 'LOCK-07', PASS, 'addressed by REVIEW finding R1')

    def test_admit_with_forbidden_code_is_excluded(self):
        standard_history(self.fx)
        self.fx.write(DNAME + '/admit.py', ADMIT_PY + '\n\ndef ceiling(loss):\n    loss.backward()\n')
        self.assertStatus(self.fx.ctx(stage='technical'), 'LOCK-07', PASS)


# ------------------------------------------------------------------------------------------------
class TestLock08(TmpCase):
    def test_literal_exposure_overlap_zero_fails(self):
        run = RUN_PY.replace('def main():\n', 'def main():\n    exposure_overlap = 0\n')
        standard_history(self.fx, run=run)
        self.assertStatus(self.fx.ctx(), 'LOCK-08', FAIL, 'exposure_overlap = 0')

    def test_dict_keys_attributes_and_prefixes(self):
        standard_history(self.fx)
        self.fx.write(DNAME + '/review/summarize.py',
                      "summary = {'controls_independent': True}\nresult = dict(all_passed=False)\n"
                      "self_obj = object()\nreport = {}\nreport['no_leak'] = 0.0\n")
        res = self.assertStatus(self.fx.ctx(), 'LOCK-08', FAIL, 'controls_independent')
        ev = ' '.join(res[0].evidence)
        self.assertIn('all_passed', ev)
        self.assertIn('no_leak', ev)

    def test_computed_values_pass(self):
        run = RUN_PY.replace('def main():\n', 'def main():\n    exposure_overlap = len(set())\n    gate_ok = 1 > 0\n')
        standard_history(self.fx, run=run)
        self.assertStatus(self.fx.ctx(), 'LOCK-08', PASS)

    def test_unparseable_py_fails(self):
        standard_history(self.fx)
        self.fx.write(DNAME + '/broken.py', 'def (:\n')
        self.assertStatus(self.fx.ctx(), 'LOCK-08', FAIL, 'does not parse')


# ------------------------------------------------------------------------------------------------
class TestLock09to13(TmpCase):
    def setUp(self):
        super().setUp()
        standard_history(self.fx)

    def edit_lock(self, fn):
        p = DNAME + '/ENDPOINT_LOCK.json'
        lock = self.fx.read_json(p)
        fn(lock)
        self.fx.write(p, lock)
        return self.fx.ctx()

    def test_lock09_cadence(self):
        with KBStub(1):
            self.assertStatus(self.fx.ctx(), 'LOCK-09', PASS, 'not due')
        with KBStub(3):
            self.assertStatus(self.fx.ctx(), 'LOCK-09', PASS, 'opens held E3')
            ctx = self.edit_lock(lambda l: setp(l, 'claim_level_target', 'E2'))
            self.assertStatus(ctx, 'LOCK-09', FAIL, 'e3_waiver:%s' % DNAME)
            uds = [{'id': 'UD-20261101-1', 'scope': ['e3_waiver:%s' % DNAME], 'verbatim': 'waive'}]
            self.fx.write('PROJECT_DOCUMENTS/KB/USER_DECISIONS.jsonl', jsonl(uds))
            ctx = self.edit_lock(lambda l: setp(l, 'e3_cadence.waiver_user_decision', 'UD-20261101-1'))
            self.assertStatus(ctx, 'LOCK-09', PASS, 'owner waiver UD-20261101-1')

    def test_lock09_missing_kb_function(self):
        prev = sys.modules.get('kb')
        sys.modules['kb'] = None
        try:
            self.assertStatus(self.fx.ctx(), 'LOCK-09', MISSING, 'deliveries_since_last_e3')
        finally:
            sys.modules['kb'] = prev
        stub = types.ModuleType('kb')
        sys.modules['kb'] = stub
        try:
            self.assertStatus(self.fx.ctx(), 'LOCK-09', MISSING, 'function missing')
        finally:
            sys.modules['kb'] = prev

    def test_lock10_constraints_sha_and_governed(self):
        ctx = self.edit_lock(lambda l: setp(l, 'constraints_sha256', 'c' * 64))
        self.assertStatus(ctx, 'LOCK-10', FAIL, 'constraints_sha256')
        cons_sha = sha(self.fx.read('PROJECT_DOCUMENTS/KB/CONSTRAINTS.json'))
        ctx = self.edit_lock(lambda l: (setp(l, 'constraints_sha256', cons_sha),
                                        setp(l, 'warmup', {'start_ceiling': '1/2', 'ramp': 'gradual'})))
        self.assertStatus(ctx, 'LOCK-10', FAIL, 'governed:warmup:%s' % DNAME)
        self.fx.write('PROJECT_DOCUMENTS/KB/USER_DECISIONS.jsonl',
                      jsonl([{'id': 'UD-20261101-2', 'scope': ['governed:warmup:%s' % DNAME]}]))
        ctx = self.edit_lock(lambda l: setp(l, 'user_decisions', ['UD-20261101-2']))
        self.assertStatus(ctx, 'LOCK-10', PASS, 'UD-20261101-2')

    def test_lock11_predictions(self):
        def f(l):
            l['predictions'][0]['basis_findings'] = ['F-NOPE-9']
            l['predictions'][1]['p_pass'] = 0.99
            del l['predictions'][2]
        res = self.assertStatus(self.edit_lock(f), 'LOCK-11', FAIL, 'F-NOPE-9')
        self.assertIn('[0.02, 0.98]', res[0].detail)
        self.assertIn('guard old_acq', res[0].detail)

    def test_lock11_point_outside_interval_and_no_base_rate(self):
        def f(l):
            l['predictions'][0]['point'] = 0.9
            l['predictions'][0]['base_rate_forecast'] = {'value': 0.5}
        res = self.assertStatus(self.edit_lock(f), 'LOCK-11', FAIL, 'outside interval90')
        self.assertIn('base_rate_forecast', res[0].detail)

    def test_lock12_snr(self):
        self.assertStatus(self.fx.ctx(), 'LOCK-12', PASS, 'no lineage parent')
        self.assertStatus(self.edit_lock(lambda l: setp(l, 'admission_thresholds.analytic_input_snr', 0.027)),
                          'LOCK-12', FAIL, 'below the floor')
        res = self.assertStatus(self.edit_lock(lambda l: setp(l, 'admission_thresholds.analytic_input_snr',
                                                              'not_derivable')), 'LOCK-12', PASS, 'ADM-INPUT')
        self.assertTrue(res[0].data['adm_input_mandatory'])

    def test_lock12_not_applicable_with_parent_and_no_frontend_change(self):
        par = 'minifly_par_dev_v1_20261001'
        self.fx.write(par + '/README.md', 'parent\n')
        csha = self.fx.commit()
        ctx = self.edit_lock(lambda l: setp(l, 'lineage.parent', '%s@%s' % (par, csha)))
        self.assertStatus(ctx, 'LOCK-12', NOT_APPLICABLE)
        ctx = self.edit_lock(lambda l: setp(l, 'lineage.changes_declared', ['frontend']))
        self.assertStatus(ctx, 'LOCK-12', PASS, 'changes_declared includes frontend')


class TestLock13(TmpCase):
    def make_sibling(self, name, parent=None):
        write_code(self.fx, name)
        lk = base_lock(self.fx, name)
        lk['worlds']['official']['ids'] = ['wo_%s_%d' % (name[-8:], i) for i in range(8)]
        if parent:
            lk['lineage']['parent'] = parent
        self.fx.write(name + '/ENDPOINT_LOCK.json', lk)
        build_source_lock(self.fx, name)

    def test_undeclared_parent_fails_and_declared_passes(self):
        old = 'minifly_old_dev_v1_20261001'
        write_kb(self.fx)
        self.fx.commit()
        self.make_sibling(old)
        self.fx.commit()
        standard_history(self.fx)
        res = self.assertStatus(self.fx.ctx(stage='lock'), 'LOCK-13', FAIL, 'undeclared parent')
        self.assertIn(old, res[0].detail)
        self.assertTrue(res[0].data['family_id'].startswith('FAM-'))
        csha = self.fx.git().log()[1].sha
        lock = self.fx.read_json(DNAME + '/ENDPOINT_LOCK.json')
        lock['lineage']['parent'] = '%s@%s' % (old, csha)
        self.fx.write(DNAME + '/ENDPOINT_LOCK.json', lock)
        self.assertStatus(self.fx.ctx(stage='lock'), 'LOCK-13', PASS, 'parent %s' % old)

    def test_descendant_is_not_an_undeclared_parent(self):
        standard_history(self.fx)
        child = 'minifly_kid_dev_v1_20261201'
        head = self.fx.git().head()
        self.make_sibling(child, parent='%s@%s' % (DNAME, head))
        self.fx.commit()
        self.assertStatus(self.fx.ctx(stage='lock'), 'LOCK-13', PASS)
        self.assertStatus(self.fx.ctx(child, stage='lock'), 'LOCK-13', PASS)

    def test_declared_parent_missing(self):
        standard_history(self.fx, lock_over={'lineage.parent': 'minifly_ghost_dev_v1_20261001@abcdef1'})
        self.assertStatus(self.fx.ctx(stage='lock'), 'LOCK-13', FAIL, 'does not exist')


# ------------------------------------------------------------------------------------------------
class TestAmendments(TmpCase):
    def setUp(self):
        super().setUp()
        self.sl = standard_history(self.fx)
        self.lock_path = DNAME + '/ENDPOINT_LOCK.json'
        self.old = self.fx.read_json(self.lock_path)
        self.old_sha = sha(self.fx.read(self.lock_path))

    def amend(self, fn, row_over=None, write_row=True, protocol=True, coprimary=True):
        new = copy.deepcopy(self.old)
        fn(new)
        if protocol:
            new['protocol_id'] = DNAME + '-p2'
        if coprimary:
            new['coprimary'] = [{'protocol_id': self.old['protocol_id'], 'endpoint_lock_sha256': self.old_sha}]
        self.fx.write(self.lock_path, new)
        new_sha = sha(self.fx.read(self.lock_path))
        changed = sorted(set(p for p in lockcheck.leaf_diff(self.old, new)
                             if lockcheck.path_top(p) not in lockcheck.AMD_ADMIN_FIELDS))
        row = {'id': 'AMD-1', 'utc': '20261110T000000Z', 'old_lock_sha256': self.old_sha, 'new_lock_sha256': new_sha,
               'new_protocol_id': new['protocol_id'], 'fields_changed': changed,
               'results_seen': ['results/dev/OUTPUTS.jsonl', 'results/receipts/wd_unit_1/W.json'],
               'reason': 'operational', 'evidence': 'results/dev/OUTPUTS.jsonl', 'objective_distance_delta': {},
               'original_endpoint_kept_as_coprimary': True, 'user_decision': None, 'author_agent': 'codex'}
        row.update(row_over or {})
        if write_row:
            self.fx.write(DNAME + '/AMENDMENTS.jsonl', jsonl([row]))
        self.fx.commit()
        return row

    def test_margin_change_without_amendment_fails(self):
        lock = self.fx.read_json(self.lock_path)
        lock['primary']['margin_abs'] = 0.06
        self.fx.write(self.lock_path, lock)
        self.assertStatus(self.fx.ctx(), 'AMD-01', FAIL, 'worktree (uncommitted)')
        self.fx.commit()
        self.assertStatus(self.fx.ctx(), 'AMD-01', FAIL, 'without an AMENDMENTS row')

    def test_task_swap_after_outcomes_without_ud_fails(self):
        def swap(l):
            l['task']['id'] = 'add2'
            l['primary']['task_id'] = 'add2'
        self.amend(swap)
        ctx = self.fx.ctx()
        self.assertStatus(ctx, 'AMD-01', PASS)
        self.assertStatus(ctx, 'AMD-03', PASS)
        res = self.assertStatus(ctx, 'AMD-02', FAIL, 'learner-arm outcome')
        self.assertIn('amendment:%s' % DNAME, res[0].detail)
        # with the owner's UD the same amendment passes
        self.fx.write('PROJECT_DOCUMENTS/KB/USER_DECISIONS.jsonl',
                      jsonl([{'id': 'UD-20261110-1', 'scope': ['amendment:%s' % DNAME], 'verbatim': 'ok'}]))
        rows = self.fx.read(DNAME + '/AMENDMENTS.jsonl').decode()
        self.fx.write(DNAME + '/AMENDMENTS.jsonl', rows.replace('"user_decision": null', '"user_decision": "UD-20261110-1"'))
        self.assertStatus(self.fx.ctx(), 'AMD-02', PASS, 'UD UD-20261110-1')

    def test_task_swap_needs_protocol_bump_and_coprimary(self):
        self.fx.write('PROJECT_DOCUMENTS/KB/USER_DECISIONS.jsonl',
                      jsonl([{'id': 'UD-20261110-1', 'scope': ['amendment:%s' % DNAME]}]))

        def swap(l):
            l['task']['id'] = 'add2'
            l['primary']['task_id'] = 'add2'
        self.amend(swap, row_over={'user_decision': 'UD-20261110-1', 'new_protocol_id': DNAME + '-p1'},
                   protocol=False, coprimary=False)
        res = self.assertStatus(self.fx.ctx(), 'AMD-02', FAIL, 'bumped protocol_id')
        self.assertIn('coprimary', res[0].detail)

    def test_objective_distance_raise_needs_ud_even_before_outcomes(self):
        self.amend(lambda l: setp(l, 'objective_distance.held', 'taught_only'))
        self.assertStatus(self.fx.ctx(), 'AMD-02', FAIL, 'objective_distance.held')

    def test_testbed_change_without_learner_outcome_needs_no_ud(self):
        fx = Fx(self.tmp / 'fx2')
        self.fx = fx
        standard_history(fx, results=False)
        self.old = fx.read_json(self.lock_path)
        self.old_sha = sha(fx.read(self.lock_path))
        fx.write(DNAME + '/results/admission/OUTPUTS.jsonl', jsonl(outputs_rows(
            fx.read_json(DNAME + '/SOURCE_LOCK.json'), world='wt_unit_1', role='technical', arm='CEILING_EXACT_BP')))
        # the scorer.py (WP3) TESTBED.json shape
        fx.write(DNAME + '/results/admission/TESTBED.json', {
            'schema': 'minifly.testbed.v1', 'endpoint_lock_sha256': self.old_sha,
            'configs': [{'arm': 'CEILING_EXACT_BP', 'kind': 'ceiling', 'declared': True, 'status': 'FAIL'},
                        {'arm': 'FLOOR_F', 'kind': 'floor', 'declared': True, 'status': 'FAIL'}],
            'ceiling_any_pass': False, 'floors_all_fail': True, 'result': 'CANNOT_TEST_CLAIM_AS_SPECIFIED'})
        fx.commit()
        self.amend(lambda l: setp(l, 'testbed.ceiling_grid', [{'id': 'EXACT_BP2', 'learner': 'EXACT_BP'}]),
                   row_over={'reason': 'testbed_invalid', 'evidence': 'results/admission/TESTBED.json',
                             'results_seen': ['results/admission/OUTPUTS.jsonl']}, protocol=False, coprimary=False)
        ctx = fx.ctx()
        self.assertStatus(ctx, 'AMD-01', PASS)
        self.assertStatus(ctx, 'AMD-02', PASS, 'no UD needed')
        self.assertStatus(ctx, 'AMD-03', PASS)

    def _testbed_amendment(self, seen, name='fx3'):
        fx = Fx(self.tmp / name)
        self.fx = fx
        standard_history(fx, results=False)
        self.old = fx.read_json(self.lock_path)
        self.old_sha = sha(fx.read(self.lock_path))
        fx.write(DNAME + '/results/admission/OUTPUTS.jsonl', jsonl(outputs_rows(
            fx.read_json(DNAME + '/SOURCE_LOCK.json'), world='wt_unit_1', role='technical', arm='CEILING_EXACT_BP')))
        fx.write(DNAME + '/results/admission/TESTBED.json', {
            'schema': 'minifly.testbed.v1', 'endpoint_lock_sha256': self.old_sha,
            'configs': [{'arm': 'CEILING_EXACT_BP', 'kind': 'ceiling', 'declared': True, 'status': 'FAIL'}],
            'ceiling_any_pass': False, 'floors_all_fail': True, 'result': 'CANNOT_TEST_CLAIM_AS_SPECIFIED'})
        fx.commit()
        self.amend(lambda l: setp(l, 'testbed.ceiling_grid', [{'id': 'EXACT_BP2', 'learner': 'EXACT_BP'}]),
                   row_over={'reason': 'testbed_invalid', 'evidence': 'results/admission/TESTBED.json',
                             'results_seen': seen}, protocol=False, coprimary=False)
        return fx.ctx()

    def test_amd03_seen_checker_written_score_file_may_be_listed(self):
        # the TESTBED result that motivated the amendment existed before it, so listing it is truthful (C00c)
        ctx = self._testbed_amendment(['results/admission/OUTPUTS.jsonl', 'results/admission/TESTBED.json'])
        self.assertStatus(ctx, 'AMD-03', PASS)

    def test_amd03_listed_score_file_must_have_existed(self):
        ctx = self._testbed_amendment(['results/admission/OUTPUTS.jsonl', 'results/dev/SCORES.json'])
        self.assertStatus(ctx, 'AMD-03', FAIL, 'extra results/dev/SCORES.json')
        ctx = self._testbed_amendment(['results/admission/TESTBED.json'], name='fx4')
        self.assertStatus(ctx, 'AMD-03', FAIL, 'missing results/admission/OUTPUTS.jsonl')

    def test_amd03_reason_results_seen_fields_and_testbed(self):
        self.amend(lambda l: setp(l, 'primary.margin_abs', 0.07),
                   row_over={'reason': 'testbed_invalid', 'results_seen': ['results/dev/OUTPUTS.jsonl'],
                             'fields_changed': ['guards']})
        res = self.assertStatus(self.fx.ctx(), 'AMD-03', FAIL, 'results_seen differs')
        self.assertIn('results/receipts/wd_unit_1/W.json', res[0].detail)
        self.assertIn('fields_changed omits primary.margin_abs', res[0].detail)
        self.assertIn('testbed_invalid needs', res[0].detail)

    def test_testbed_shapes(self):
        ok = lockcheck.testbed_shows_invalid
        self.assertTrue(ok({'result': 'CANNOT_TEST_CLAIM_AS_SPECIFIED'})[0])
        self.assertTrue(ok({'floors_all_fail': False, 'ceiling_any_pass': True})[0])
        self.assertTrue(ok({'configs': [{'kind': 'floor', 'status': 'PASS'}]})[0])
        self.assertFalse(ok({'result': 'PASS', 'ceiling_any_pass': True, 'floors_all_fail': True,
                             'configs': [{'kind': 'ceiling', 'declared': True, 'status': 'PASS'},
                                         {'kind': 'floor', 'status': 'FAIL'}]})[0])
        self.assertFalse(ok({'result': 'MISSING', 'ceiling_any_pass': None})[0])

    def test_amd03_bad_reason(self):
        self.amend(lambda l: setp(l, 'primary.margin_abs', 0.07), row_over={'reason': 'improve_odds'})
        self.assertStatus(self.fx.ctx(), 'AMD-03', FAIL, 'improve_odds')

    def test_amd01_dangling_row(self):
        self.fx.write(DNAME + '/AMENDMENTS.jsonl', jsonl([{'id': 'AMD-9', 'old_lock_sha256': 'd' * 64,
                                                           'new_lock_sha256': 'e' * 64}]))
        self.assertStatus(self.fx.ctx(), 'AMD-01', FAIL, 'links no observed')

    def test_amd04_world_reuse(self):
        ctx = self.fx.ctx()
        self.assertStatus(ctx, 'AMD-04', PASS, 'scan scope')
        other = 'minifly_oth_dev_v1_20261005'
        self.fx.write(other + '/manifests/worlds.json', {'worlds': ['wo_unit_3']})
        self.assertStatus(self.fx.ctx(), 'AMD-04', FAIL, 'wo_unit_3')

    def test_amd04_official_world_before_amendment(self):
        self.fx.write(DNAME + '/results/receipts/wo_unit_2/W.json',
                      {'world': 'wo_unit_2', 'world_role': 'dev', 'arm': 'W', 'native_closure_sha':
                       self.sl['native_closure_sha']})
        self.fx.commit()
        self.amend(lambda l: setp(l, 'primary.margin_abs', 0.07), row_over={
            'results_seen': ['results/dev/OUTPUTS.jsonl', 'results/receipts/wd_unit_1/W.json',
                             'results/receipts/wo_unit_2/W.json']})
        self.assertStatus(self.fx.ctx(), 'AMD-04', FAIL, 'before the latest amendment')

    def test_amd04_not_applicable_without_ids(self):
        lock = self.fx.read_json(self.lock_path)
        lock['worlds']['official'] = {'mode': 'sealed', 'ids': [], 'n': 8}
        self.fx.write(self.lock_path, lock)
        self.assertStatus(self.fx.ctx(), 'AMD-04', NOT_APPLICABLE)

    def test_amd05_admit_edit_without_amendment(self):
        self.fx.write(DNAME + '/admit.py', ADMIT_PY + '\n# relaxed after a FAIL\n')
        self.assertStatus(self.fx.ctx(), 'AMD-05', FAIL, 'changed without an amendment')
        self.fx.commit()
        self.assertStatus(self.fx.ctx(), 'AMD-05', FAIL, 'differs from the lock in effect')

    def test_amd05_split_manifests_fixed_by_the_lock(self):
        # verifier round-3 probe F: the held set is fixed in the lock (SPEC 6.2) and recomputed at every stage
        self.assertStatus(self.fx.ctx(), 'AMD-05', PASS, 'split versions checked')
        held = DNAME + '/manifests/HELD.jsonl'
        orig = self.fx.read(held)
        self.fx.write(held, orig.replace(b'332b343d', b'322b373d'))       # one item swapped (2+7=)
        self.assertStatus(self.fx.ctx(), 'AMD-05', FAIL, 'do not hash to the lock\'s held_set_sha256')
        self.fx.commit()
        self.assertStatus(self.fx.ctx(), 'AMD-05', FAIL, 'at commit')
        self.fx.write(held, orig)
        self.fx.commit()                                                 # reverted, but the history keeps it
        self.assertStatus(self.fx.ctx(), 'AMD-05', FAIL, 'at commit')

    def test_amd05_split_manifest_deleted_or_rehashed(self):
        self.fx.delete(DNAME + '/manifests/OLD.jsonl')
        self.assertStatus(self.fx.ctx(), 'AMD-05', FAIL, 'old_manifest absent in the worktree')
        self.fx.commit()
        self.assertStatus(self.fx.ctx(), 'AMD-05', FAIL, 'absent at commit')

    def test_amd05_split_repin_needs_amendment(self):
        new = DNAME + '/manifests/NEW.jsonl'
        self.fx.write(new, self.fx.read(new) + b'{"item_id": "n11", "prompt_hex": "352b353d", "expected_hex": "3130"}\n')
        lock = self.fx.read_json(self.lock_path)
        held_sha, split_sha = split_shas(self.fx)
        lock['split'].update({'held_set_sha256': held_sha, 'split_sha256': split_sha})
        self.fx.write(self.lock_path, lock)
        self.assertNotIn('do not hash', ' '.join(r.detail for r in run(self.fx.ctx(), 'AMD-05')))
        self.fx.commit()
        self.assertStatus(self.fx.ctx(), 'AMD-05', FAIL, 're-pinned the split hashes')

    def test_amd05_repin_needs_amendment_and_completeness(self):
        self.fx.write(DNAME + '/admit.py', ADMIT_PY + '\n# fix\n')
        lock = self.fx.read_json(self.lock_path)
        lock['endpoint_files_sha256']['admit.py'] = sha(self.fx.read(DNAME + '/admit.py'))
        self.fx.write(self.lock_path, lock)
        self.fx.commit()
        self.assertStatus(self.fx.ctx(), 'AMD-05', FAIL, 're-pinned admit.py')
        lock['files']['evaluator'] = ['evalx.py']
        self.fx.write(DNAME + '/evalx.py', 'x = 1\n')
        self.fx.write(self.lock_path, lock)
        self.assertStatus(self.fx.ctx(), 'AMD-05', FAIL, 'files.evaluator evalx.py is not in endpoint_files_sha256')


# ------------------------------------------------------------------------------------------------
class TestWit01(TmpCase):
    def test_dev_output_before_attestation_fails(self):
        write_kb(self.fx)
        write_code(self.fx)
        self.fx.write(DNAME + '/ENDPOINT_LOCK.json', base_lock(self.fx))
        self.fx.commit([p for p in self.fx._files() if p.startswith('PROJECT_DOCUMENTS/')] +
                       [DNAME + '/ENDPOINT_LOCK.json', DNAME + '/admit.py', DNAME + '/params.json'])
        self.fx.commit()
        sl = build_source_lock(self.fx)
        self.fx.commit()
        self.fx.write(DNAME + '/results/dev/OUTPUTS.jsonl', jsonl(outputs_rows(sl)))
        self.fx.commit()
        write_review(self.fx)
        self.fx.commit(branches=CLAUDE, trailers=REVIEWER_TRAILER)
        self.assertStatus(self.fx.ctx(tier=2), 'WIT-01', FAIL, 'not an ancestor of the first dev output')

    def test_wrong_namespace_and_stale_lock(self):
        standard_history(self.fx, review=False)
        write_review(self.fx)
        # the reviewer's trailer on a commit only the author namespace contains (SPEC 20.15: trailer AND namespace)
        self.fx.commit(branches=['origin/codex/work'], trailers=REVIEWER_TRAILER)
        self.assertStatus(self.fx.ctx(tier=2), 'WIT-01', FAIL, 'origin/claude/*')
        rv = self.fx.read_json(DNAME + '/review/REVIEW.json')
        rv['reviewed_endpoint_lock_sha256'] = 'f' * 64
        self.fx.write(DNAME + '/review/REVIEW.json', rv)
        self.assertStatus(self.fx.ctx(tier=2), 'WIT-01', FAIL, 'is not the current ENDPOINT_LOCK')

    def test_tier_and_missing_review(self):
        standard_history(self.fx, review=False)
        self.assertStatus(self.fx.ctx(tier=1), 'WIT-01', NOT_APPLICABLE)
        self.assertStatus(self.fx.ctx(tier=2), 'WIT-01', MISSING, 'REVIEW.json absent')

    def test_official_needs_source_lock_ack(self):
        sl = standard_history(self.fx, review=False, results=False)
        write_review(self.fx, source_lock_acks=[])
        self.fx.commit(branches=CLAUDE, trailers=REVIEWER_TRAILER)
        self.fx.write(DNAME + '/results/dev/OUTPUTS.jsonl', jsonl(outputs_rows(sl)))
        self.fx.commit()
        self.assertStatus(self.fx.ctx(stage='roster', tier=2), 'WIT-01', PASS)
        self.assertStatus(self.fx.ctx(stage='official', tier=2), 'WIT-01', FAIL, 'source_lock_acks lacks')
        self.assertStatus(self.fx.ctx(stage='official', tier=2, dry=True), 'WIT-01', WARN, 'deferred')
        write_review(self.fx)
        self.fx.commit(branches=CLAUDE, trailers=REVIEWER_TRAILER)
        self.assertStatus(self.fx.ctx(stage='official', tier=2), 'WIT-01', PASS, 'acked')


# ------------------------------------------------------------------------------------------------
class TestInit(TmpCase):
    def setUp(self):
        super().setUp()
        ledger = [{'delivery': 'a', 'legacy': False, 'task_family': 'addition_single_digit', 'primary_gate': 'PASS'},
                  {'delivery': 'b', 'legacy': False, 'task_family': 'addition_single_digit', 'primary_gate': 'FAIL'},
                  {'delivery': 'c', 'legacy': False, 'task_family': 'addition_single_digit', 'primary_gate': 'FAIL'},
                  # a second `claim` run of c: one delivery counts once (latest row), as in `kb score`
                  {'delivery': 'c', 'legacy': False, 'task_family': 'addition_single_digit', 'primary_gate': 'FAIL',
                   'utc': '20261105T000000Z'},
                  {'delivery': 'f', 'legacy': False, 'task_family': 'addition_single_digit',
                   'primary_gate': 'NOT_OPENED'},
                  {'delivery': 'd', 'legacy': True, 'task_family': 'addition_single_digit', 'primary_gate': 'PASS'},
                  {'delivery': 'e', 'legacy': False, 'task_family': 'non_arithmetic', 'primary_gate': 'PASS'}]
        write_kb(self.fx, ledger=ledger)
        self.fx.commit()

    def init(self, name=DNAME, **kw):
        args = self.fx.args(dir=name, parent=kw.pop('parent', None), claim_level=kw.pop('claim_level', None),
                            task_family=kw.pop('task_family', None), agent=kw.pop('agent', None))
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = lockcheck.cmd_init(args)
        return rc, out.getvalue(), err.getvalue()

    def test_init_fills_and_refuses_overwrite(self):
        rc, out, err = self.init(claim_level='E3', task_family='addition_single_digit', agent='claude')
        self.assertEqual(rc, 0, err)
        lock = self.fx.read_json(DNAME + '/ENDPOINT_LOCK.json')
        self.assertEqual(lock['delivery'], DNAME)
        self.assertEqual(lock['protocol_id'], DNAME + '-p1')
        self.assertEqual(lock['author_agent'], 'claude')
        self.assertEqual(lock['constraints_sha256'], sha(self.fx.read('PROJECT_DOCUMENTS/KB/CONSTRAINTS.json')))
        self.assertEqual(lock['primary']['margin_abs'], 0.05)
        self.assertEqual(lock['primary']['min_worlds'], 8)
        self.assertEqual(lock['primary']['aggregator']['alpha'], 0.05)
        self.assertEqual(lock['testbed']['ceiling_grid'], CONSTRAINTS['testbed_ceiling_grid'])
        self.assertEqual(lock['testbed']['floor_grid'], CONSTRAINTS['testbed_floor_grid'])
        self.assertEqual(sorted(a['role'] for a in lock['arms']),
                         sorted(['candidate', 'no_prior_teaching', 'shuf', 'oracle_plumbing']))
        self.assertEqual(lock['task']['format']['sep_hex'], '2b')
        self.assertEqual(lock['objective_distance']['task_family'], 'addition_single_digit')
        self.assertIsNone(lock['lineage']['parent'])
        p1 = [p for p in lock['predictions'] if p['target'] == 'primary'][0]
        self.assertAlmostEqual(p1['base_rate_forecast']['value'], 0.4)  # (1+1)/(3+2), legacy row excluded
        self.assertIn('1 PASS of 3', p1['base_rate_forecast']['source'])
        self.assertIn('__FILL__', out)
        rc, _out, err = self.init()
        self.assertEqual(rc, common.EXIT_FAIL)
        self.assertIn('already exists', err)
        self.assertStatus(self.fx.ctx(stage='lock'), 'LOCK-01', FAIL, '__FILL__')

    def test_init_fills_split_hashes_from_existing_manifests(self):
        rc, out, _e = self.init(claim_level='E3', task_family='addition_single_digit', agent='claude')
        self.assertEqual(rc, 0)
        self.assertIn('split manifests', out)                    # none written yet: left __FILL__, with a note
        self.fx.delete(DNAME + '/ENDPOINT_LOCK.json')
        write_manifests(self.fx)
        rc, out, err = self.init(claim_level='E3', task_family='addition_single_digit', agent='claude')
        self.assertEqual(rc, 0, err)
        lock = self.fx.read_json(DNAME + '/ENDPOINT_LOCK.json')
        held_sha, split_sha = split_shas(self.fx)
        self.assertEqual((lock['split']['held_set_sha256'], lock['split']['split_sha256']), (held_sha, split_sha))
        self.assertEqual(held_sha, sha(self.fx.read(DNAME + '/manifests/HELD.jsonl')))

    def test_init_e2_adds_positive_control_and_parent(self):
        par = 'minifly_par_dev_v1_20261001'
        self.fx.write(par + '/README.md', 'p\n')
        csha = self.fx.commit()
        rc, _o, err = self.init(claim_level='E2', parent=par, task_family='non_arithmetic')
        self.assertEqual(rc, 0, err)
        lock = self.fx.read_json(DNAME + '/ENDPOINT_LOCK.json')
        roles = [a['role'] for a in lock['arms']]
        self.assertIn('positive_control', roles)
        self.assertEqual(lock['positive_control_arm'], 'POS')
        self.assertIn('no_prior_teaching', roles)  # the causal control is always kept
        self.assertEqual(lock['lineage']['parent'], '%s@%s' % (par, csha))
        self.assertEqual(lock['task']['format']['kind'], 'other')
        self.assertEqual(lock['author_agent'], '__FILL__')

    def test_init_cadence_message_and_bad_name(self):
        with KBStub(5):
            rc, out, _e = self.init(claim_level='E1')
        self.assertEqual(rc, 0)
        self.assertIn('E3 cadence DUE', out)
        rc, _o, err = self.init(name='not_a_v1_name')
        self.assertEqual(rc, common.EXIT_MISSING)
        self.assertIn('does not match', err)


if __name__ == '__main__':
    unittest.main()
