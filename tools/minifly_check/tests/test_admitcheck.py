"""Unit tests for admitcheck.py (WP4): ADM-*, AUT-*. Synthetic deliveries in temp dirs (TimelineGit); stdlib only.

The clean fixture (WP4Fixture.build) is a small single-digit addition E3 delivery (the WP3 reference design,
trimmed): technical world wt1 (admission rows, testbed CEILING_EXACT_BP / FLOOR_FRF, receipts, fixtures) and
dev worlds wd1..wd3 (arms W, NPT, SHUF, ORACLE). The candidate W is exact on 19/18/17 of 20 held items, the best
shortcut is DIGIT_NO_CARRY at 0.5, NPT is the causal control (zero channel pair_memory). It is reused by
test_review.py and test_canary.py.
"""
import contextlib
import io
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

KIT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
for p in (str(KIT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import common  # noqa: E402
import admitcheck  # noqa: E402

NAME = 'minifly_wp4fx_dev_v1_20261101'
EOS = b'\n'
CAP = 4
FMT = {'kind': 'addition', 'sep_hex': '2b', 'terminator_hex': '3d', 'eos_hex': '0a', 'gen_cap_bytes': CAP,
       'normalize_leading_zeros': True}
TECH = ['wt1']
DEV = ['wd1', 'wd2', 'wd3']
ARMS = ('W', 'NPT', 'SHUF', 'ORACLE')
POOLS = ('HELD', 'OLD', 'TAUGHT', 'NEW')
CKS = ('prefix', 'old_end', 'final')
H = [(a, b) for a in range(1, 6) for b in range(5, 9)]
T_ALL = [(x, y) for x in range(1, 10) for y in range(1, 10) if (x, y) not in H and (y, x) not in H]
OLD_ITEMS = [(1, 1), (1, 2), (2, 1), (1, 3), (3, 1), (9, 1), (9, 2), (9, 3), (1, 9), (2, 9)]
TAUGHT_ITEMS = [(1, 4), (4, 1), (2, 2), (2, 3), (3, 2), (9, 4), (9, 5), (9, 6), (3, 9), (4, 9)]
NEW_ITEMS = [(2, 4), (4, 2), (3, 3), (4, 4), (3, 4), (6, 9), (7, 9), (8, 9), (9, 7), (9, 8)]
POOL_ITEMS = {'HELD': H, 'OLD': OLD_ITEMS, 'TAUGHT': TAUGHT_ITEMS, 'NEW': NEW_ITEMS}
W_WRONG = {'wt1': 1, 'wd1': 1, 'wd2': 2, 'wd3': 3}
PARAMS = {'lr': 0.1, 'pair_density': 0.0354, 'width': 64, 'warmup': {'start_ceiling': '1/16', 'steps': 50}}
NATIVE_PY = b'''"""Synthetic native learner (never executed by the checker)."""
import json


class Learner:
    def __init__(self):
        self.pairs = {}
        self.writes = {"pair_memory": 0}

    def write_pair(self, key, value):
        self.pairs[key] = value
        self.writes["pair_memory"] += 1
'''
RUN_PY = b'''"""Synthetic runner (never executed by the checker)."""
import json
import os


def main():
    return os.environ.get("MINIFLY_ATTEMPT_ID")
'''
ADMIT_PY = b'''"""Synthetic admit.py (never executed by the checker)."""
import json
import os


def main():
    return os.environ.get("MINIFLY_ATTEMPT_ID")
'''
CONSTRAINTS = {
    'schema': 'minifly.constraints.v1',
    'tier_thresholds': {'t0_max_cpu_h': 1, 't1_max_cpu_h': 4, 't1_max_records': 20000, 'family_t1_max_cpu_h': 12},
    'threshold_floors': {'margin_abs': 0.05, 'absolute_floor_above_chance': 0.05, 'per_world_floor': -0.05,
                         'min_worlds_nonpilot': 8, 'min_items_per_stratum': 8, 'input_contrast_min': 2.0,
                         'prompt_id_margin_min': 0.1, 'taught_acq_margin_min': 0.05, 'min_realized_dose': 0.05,
                         'shuf_band': [0.35, 0.65], 'old_max_drop': 0.1, 'old_acq_margin': 0.05,
                         'analytic_input_snr_min': 0.5, 'oc_min_power': 0.8, 'alpha_max': 0.05},
    'default_grant_caps': {'max_cpu_h': 2, 'max_records': 5000, 'max_cycles': 3, 'official': False, 'push': False,
                           'adoption': False, 'pointer_update': False},
    'review_debt_max_open': 3,
    'required_arms_by_level': {'E1': ['candidate', 'shuf'], 'E2': ['candidate', 'shuf', 'positive_control'],
                               'E3': ['candidate', 'no_prior_teaching', 'shuf', 'oracle_plumbing']},
    'testbed_ceiling_grid': [{'id': 'EXACT_BP', 'learner': 'EXACT_BP'}],
    'testbed_floor_grid': [{'id': 'FRF', 'learner': 'FROZEN_RANDOM_FEATURES'}],
    'native_forbidden': {'imports': ['torch.autograd', 'torch.optim', 'transformers', 'jax'],
                         'calls': ['.backward(', 'scaled_dot_product_attention(', 'numpy.linalg.lstsq('],
                         'identifiers': ['transformer', 'attention', 'requires_grad=True']},
    'supplied_arithmetic_heuristics': ['int() over decoded operand bytes', 'carry or digit tables',
                                       "'+' applied to parsed operands"],
    'e3_cadence_max': 2,
}
VERBATIM = ('Approved: implementation, the technical cycle and dev cycles for this E3 design, at most 2 CPU-h, '
            '1000 records and 3 cycles. No official run yet.')
C1, C2, C3, C4, C5, C6 = ('1' * 40, '2' * 40, '3' * 40, '4' * 40, '5' * 40, '6' * 40)
CODEX = {'Acting-Agent': ['codex']}
CLAUDE = {'Acting-Agent': ['claude:s-review']}
CODEX_BR = ['origin/codex/wp4fx']
CLAUDE_BR = ['origin/claude/wp4fx-review']


def ph(a, b):
    return ('%d+%d=' % (a, b)).encode('ascii').hex()


def truth(a, b):
    return str(a + b).encode('ascii')


def stream_of(content):
    return (content + EOS * CAP)[:CAP]


def wrong_of(exp):
    return b'5' if exp != b'5' else b'4'


def d64(tag):
    return common.sha256_bytes(tag.encode('utf-8'))


def jsonl(rows):
    return ''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows)


def write(path, data):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, (dict, list)):
        data = json.dumps(data, indent=1)
    if isinstance(data, str):
        data = data.encode('utf-8')
    if p.exists():
        p.unlink()
    p.write_bytes(data)
    return p


def behaviour(world, arm, pool, idx):
    if arm in ('W',):
        if pool == 'HELD':
            return idx not in set(range(19, 19 - W_WRONG.get(world, 2), -1))
        if pool == 'OLD':
            return idx != 9
        return idx != 0
    if arm == 'NPT':
        return idx % 10 == 0
    if arm == 'SHUF':
        return idx % 5 == 0
    if arm in ('ORACLE', 'CEILING_EXACT_BP'):
        return True
    if arm == 'FLOOR_FRF':
        return idx % 7 == 0
    return False


def output_rows(worlds, role, arms, beh=behaviour):
    rows = []
    for w in worlds:
        for arm in arms:
            evaluator = common.is_evaluator_arm(arm)
            for pool in POOLS:
                for ck in CKS:
                    for idx, (a, b) in enumerate(POOL_ITEMS[pool]):
                        exp = truth(a, b)
                        st = stream_of(exp if beh(w, arm, pool, idx) else wrong_of(exp))
                        pos = st.find(EOS)
                        rows.append({'world': w, 'world_role': role, 'arm': arm, 'pool': pool, 'checkpoint': ck,
                                     'item_id': '%s%02d' % (pool[0].lower(), idx), 'prompt_hex': ph(a, b),
                                     'expected_hex': exp.hex(), 'stream_hex': st.hex(),
                                     'emitted_hex': st[:pos].hex() if pos >= 0 else st.hex(), 'own_eos': pos >= 0,
                                     'answer_path': 'evaluator' if evaluator else 'native',
                                     'native_closure_sha': None if evaluator else 'n' * 64,
                                     'params_sha256': None if evaluator else 'p' * 64, 'attempt_id': 'ATT-1'})
    return rows


def shuf_label(x, y):
    """Orbit-consistent: the truth for even canonical orbits, truth + 1 for odd ones (agreement about 0.5)."""
    orbits = sorted(set(tuple(sorted((a, b))) for a, b in T_ALL))
    k = orbits.index(tuple(sorted((x, y))))
    return truth(x, y) if k % 2 == 0 else str(x + y + 1).encode('ascii')


def taught_rows(world, arm, label=None, fb=True):
    out = []
    for (x, y) in T_ALL:
        ans = label(x, y) if label else truth(x, y)
        out.append({'world': world, 'arm': arm, 'prompt_hex': ph(x, y), 'answer_hex': ans.hex(), 'phase': 'teach',
                    'feedback_bearing': fb, 'presentations': 12})
    return out


def base_lock():
    return {
        'schema': common.SCHEMA_ENDPOINT_LOCK, 'delivery': NAME, 'protocol_id': NAME + '-p1',
        'created_utc': '20261101T000000Z', 'author_agent': 'codex', 'tier_declared': 1, 'intent': 'diagnostic',
        'claim_level_target': 'E3',
        'task': {'id': 'add1', 'family': 'addition_single_digit', 'format': dict(FMT)},
        'objective_distance': {'task_family': 'addition_single_digit', 'output': 'own_bytes_own_eos',
                               'held': 'unseen_composition', 'native_only_path': True, 'feedback': 'gradual_warmup',
                               'old_protection_measured': True},
        'units': {'witness': 'pair_rectangle', 'commutative': False},
        'split': {'taught_manifest': 'manifests/TAUGHT', 'held_manifest': 'manifests/HELD.jsonl',
                  'old_manifest': 'manifests/OLD.jsonl', 'new_manifest': 'manifests/NEW.jsonl',
                  'split_sha256': 'a' * 64, 'held_set_sha256': 'b' * 64},
        'exposure_streams': [{'path_glob': 'manifests/TAUGHT/*/*.jsonl', 'feedback_bearing': True, 'phase': 'teach'}],
        'arms': [{'id': 'W', 'role': 'candidate', 'zero_channels': [], 'declared_changed': [], 'teaching_withheld': [],
                  'min_realized_dose': None},
                 {'id': 'NPT', 'role': 'no_prior_teaching', 'zero_channels': ['pair_memory'], 'declared_changed': [],
                  'teaching_withheld': ['pair_teaching'], 'min_realized_dose': 0.05},
                 {'id': 'SHUF', 'role': 'shuf', 'zero_channels': [], 'declared_changed': ['pair_memory'],
                  'teaching_withheld': [], 'min_realized_dose': None},
                 {'id': 'ORACLE', 'role': 'oracle_plumbing', 'zero_channels': [], 'declared_changed': ['readout'],
                  'teaching_withheld': [], 'min_realized_dose': None}],
        'causal_control_arm': 'NPT', 'positive_control_arm': 'ORACLE',
        'ports': [{'id': 'old_relation', 'signs': ['+', '-']}],
        'primary': {'task_id': 'add1', 'metric': 'exact_own_eos', 'unconditional': True, 'pool': 'HELD',
                    'checkpoint': 'final', 'strata_counted': ['S2', 'S3'], 'candidate_arm': 'W',
                    'comparator_arms': ['NPT'], 'ladder': 'auto', 'chance': 0.1, 'margin_abs': 0.05,
                    'absolute_floor': 0.15, 'per_world_floor': -0.05, 'aggregator': {'type': 'all_worlds'},
                    'min_worlds': 3, 'n_items_per_pool': {'HELD': 20, 'OLD': 10, 'TAUGHT': 10, 'NEW': 10},
                    'rule_type': 'declarative'},
        'gates': [{'id': 'stop_gate', 'description': 'stop gate'}],
        'guards': [{'id': 'old_drop', 'kind': 'old_drop', 'pool': 'OLD', 'max_drop': 0.1},
                   {'id': 'old_acq', 'kind': 'old_acq', 'pool': 'OLD', 'margin': 0.05}],
        'udi': None, 'checkpoints': {'prefix': 100, 'old_end': 200, 'final': 300}, 'stage_chain': False,
        'worlds': {'technical': list(TECH), 'dev': list(DEV), 'rehearsal': [],
                   'official': {'mode': 'listed', 'ids': [], 'n': 0}, 'official_role_intended': 'MAIN'},
        'schedule': {'paired_atomic': True}, 'budget': {'max_records': 1000, 'max_cpu_h': 1},
        'admission_thresholds': {'analytic_input_snr': 1.7, 'input_contrast_min': 2.0, 'prompt_id_margin_min': 0.1,
                                 'taught_acq_margin_min': 0.05, 'shuf_band': [0.35, 0.65]},
        'testbed': {'ceiling_grid': [{'id': 'EXACT_BP', 'learner': 'EXACT_BP'}],
                    'floor_grid': [{'id': 'FRF', 'learner': 'FROZEN_RANDOM_FEATURES'}]},
        'files': {'admit': 'admit.py', 'evaluator': [], 'generators': [], 'runner': ['run.py'], 'native': ['native.py'],
                  'shared': [], 'assets': [], 'params': 'params.json'},
        'endpoint_files_sha256': {'admit.py': common.sha256_bytes(ADMIT_PY)},
        'lineage': {'parent': None, 'changes_declared': [], 'priors_added': []},
        'predictions': [], 'e3_cadence': {'waiver_user_decision': None}, 'constraints_sha256': None,
        'user_decisions': [], 'coprimary': [],
    }


def receipt(world, arm, lock_sha, **over):
    birth = {'pair_memory': d64('birth-pm'), 'stop': d64('birth-stop'), 'readout': d64('birth-ro')}
    post = {'pair_memory': d64('%s-pm1' % arm), 'stop': d64('post-stop'), 'readout': d64('post-ro')}
    writes = {'pair_memory': 12, 'stop': 3, 'readout': 40}
    if arm == 'NPT':
        post['pair_memory'] = birth['pair_memory']
        writes['pair_memory'] = 0
    if arm == 'ORACLE':
        post['readout'] = d64('oracle-ro')
    r = {'schema': 'minifly.receipt.v1', 'world': world, 'world_role': 'technical', 'arm': arm, 'phase': 'teach+test',
         'records': 100, 'taught_presentations': 1200, 'writes_by_channel': writes,
         'write_counter_site': 'native.py:Learner.write_pair',
         'channel_digests': {'birth': birth, 'post_first': post, 'end': dict(post)},
         'state_digest_end': d64('end-%s' % arm),
         'realized_dose': {'teaching_withheld_fraction': 1.0} if arm == 'NPT' else {},
         'native_closure_sha': 'n' * 64, 'params_sha256': 'p' * 64, 'endpoint_lock_sha256': lock_sha,
         'attempt_id': 'ATT-1'}
    r.update(over)
    return r


def fixture_rows_default(params=None):
    p = dict(params or PARAMS)
    rows = []
    for sign in ('+', '-'):
        rows.append({'port': 'old_relation', 'sign': sign, 'target_margin_delta_vs_passive': 0.031,
                     'zero_coef_digest_equals_passive': True, 'zero_replay_applications': 0, 'params': p,
                     'attempt_id': 'ATT-1'})
    rows.append({'gate_id': 'stop_gate', 'kind': 'negative', 'params': p, 'result': False, 'attempt_id': 'ATT-1'})
    rows.append({'gate_id': 'stop_gate', 'kind': 'positive', 'params': p, 'result': True, 'attempt_id': 'ATT-1'})
    return rows


def authorization(**over):
    a = {'schema': 'minifly.authorization.v1', 'verbatim': VERBATIM, 'utc': '20261101T000000Z',
         'session': 'owner-chat-1', 'channel': 'chat',
         'options_shown': [{'label': 'A: open the held E3 endpoint', 'text': 'Run the E3 design.', 'cost': '2 CPU-h',
                            'objective_distance': 'held E3'},
                           {'label': 'B: stop', 'text': 'Record the finding and stop.', 'cost': '0',
                            'objective_distance': 'none'}],
         'recommendation_shown': 'A', 'stages': {'implementation': True, 'technical': True, 'dev_cycles': True,
                                                 'official': False, 'push': False, 'push_lock': False,
                                                 'adoption': False, 'pointer_update': False},
         'caps': {'max_records': 1000, 'max_cpu_h': 2, 'max_cycles': 3}, 'defaults_applied': False,
         'conditionals': [], 'recorded_by': 'codex'}
    a.update(over)
    return a


class WP4Fixture:
    """A v1 delivery inside a TimelineGit fixture: fx/timeline.json, fx/repo/<NAME>/..."""

    def __init__(self, tmp):
        self.tmp = Path(tmp)
        self.fx = self.tmp / 'fx'
        self.root = self.fx / 'repo'
        self.d = self.root / NAME
        self.extra_commits = []

    # ---- files
    def w(self, rel, data):
        return write(self.d / rel, data)

    def wr(self, rel, data):
        return write(self.root / rel, data)

    def read(self, rel):
        return json.loads((self.d / rel).read_text(encoding='utf-8'))

    def lock_sha(self):
        return common.sha256_file(self.d / 'ENDPOINT_LOCK.json')

    def set_lock(self, lock):
        self.w('ENDPOINT_LOCK.json', lock)

    def build(self):
        self.wr('PROJECT_DOCUMENTS/KB/CONSTRAINTS.json', CONSTRAINTS)
        for led in ('FINDINGS', 'USER_DECISIONS', 'REVIEW_DEBT', 'INCIDENTS', 'DELIVERY_LEDGER', 'CONFIRM_LEDGER',
                    'CANARY_COMMITMENTS'):
            self.wr('PROJECT_DOCUMENTS/KB/%s.jsonl' % led, '')
        self.w('native.py', NATIVE_PY)
        self.w('run.py', RUN_PY)
        self.w('admit.py', ADMIT_PY)
        self.w('params.json', PARAMS)
        self.set_lock(base_lock())
        lsha = self.lock_sha()
        nat = {NAME + '/native.py': common.sha256_bytes(NATIVE_PY)}
        clo = dict(nat)
        clo[NAME + '/run.py'] = common.sha256_bytes(RUN_PY)
        clo[NAME + '/admit.py'] = common.sha256_bytes(ADMIT_PY)
        self.w('SOURCE_LOCK.json', {'schema': common.SCHEMA_SOURCE_LOCK, 'delivery': NAME, 'endpoint_lock_sha256': lsha,
                                    'built_utc': '20261101T010000Z', 'checker_version': common.CHECKER_VERSION,
                                    'closure_sha256': clo, 'native_closure_sha256': nat, 'dynamic_imports': [],
                                    'env': {'python': '3.9.18', 'packages': {}},
                                    'native_closure_sha': common.sha256_json(nat), 'runner_closure_sha': 'r' * 64,
                                    'params_sha256': common.sha256_file(self.d / 'params.json')})
        self.w('AUTHORIZATION.json', authorization())
        for pool, rel in (('HELD', 'manifests/HELD.jsonl'), ('OLD', 'manifests/OLD.jsonl'), ('NEW', 'manifests/NEW.jsonl')):
            self.w(rel, jsonl([{'item_id': '%s%02d' % (pool[0].lower(), i), 'prompt_hex': ph(a, b),
                                'expected_hex': truth(a, b).hex()} for i, (a, b) in enumerate(POOL_ITEMS[pool])]))
        for wld in TECH + DEV:
            self.w('manifests/TAUGHT/%s/W.jsonl' % wld, jsonl(taught_rows(wld, 'W')))
            self.w('manifests/TAUGHT/%s/ORACLE.jsonl' % wld, jsonl(taught_rows(wld, 'ORACLE')))
            self.w('manifests/TAUGHT/%s/NPT.jsonl' % wld, jsonl(taught_rows(wld, 'NPT', fb=False)))
            self.w('manifests/TAUGHT/%s/SHUF.jsonl' % wld, jsonl(taught_rows(wld, 'SHUF', label=shuf_label)))
        self.w('results/admission/OUTPUTS.jsonl', jsonl(output_rows(TECH, 'technical',
                                                                    ARMS + ('CEILING_EXACT_BP', 'FLOOR_FRF'))))
        self.w('results/admission/FIXTURES.jsonl', jsonl(fixture_rows_default()))
        for arm in ARMS:
            self.w('results/receipts/wt1/%s.json' % arm, receipt('wt1', arm, lsha))
        self.write_admission()
        self.w('results/dev/OUTPUTS.jsonl', jsonl(output_rows(DEV, 'dev', ARMS)))
        self.timeline()
        return self

    def write_admission(self, **over):
        a = {'schema': 'minifly.admission.v1', 'endpoint_lock_sha256': self.lock_sha(),
             'source_lock_sha256': common.sha256_file(self.d / 'SOURCE_LOCK.json'),
             'producer': {'script': 'admit.py', 'sha256': common.sha256_bytes(ADMIT_PY)}, 'worlds': list(TECH),
             'testbed_outputs': 'results/admission/OUTPUTS.jsonl', 'receipts_glob': 'results/receipts/wt1/*.json',
             'fixtures': 'results/admission/FIXTURES.jsonl',
             'input_contrast': {'metric_definition': 'L2 distance of post-read codes', 'analytic_snr': 1.7,
                                'same_input_diff_noise': 0.12, 'diff_input_same_noise': 0.41, 'ratio': 3.42,
                                'prompt_id_loo_acc': 0.31, 'chance': 0.01},
             'structure_meters': {'shared_operand_similarity': 0.44, 'nonshared_operand_similarity': 0.21,
                                  'modal_output_fraction': 0.18, 'code_overlap_birth': 0.02, 'code_overlap_end': 0.07},
             'parent_admission_sha': None, 'created_utc': '20261101T020000Z'}
        a.update(over)
        self.w('ADMISSION.json', a)

    # ---- history
    def _bucket(self, rel):
        r = rel[len(NAME) + 1:] if rel.startswith(NAME + '/') else None
        if r is None:
            return C1
        if r.startswith('results/dev/') or any(r.startswith('results/receipts/%s/' % w) for w in DEV):
            return C3
        if r.startswith('results/official/'):
            return C3
        if r.startswith('results/') or r in ('ADMISSION.json', 'SOURCE_LOCK.json'):
            return C2
        if r.startswith('review/') or r.startswith('checks/'):
            return None
        return C1

    def timeline(self, extra=None, exclude=()):
        """Commits C1 (lock and sources, codex), C2 (SOURCE_LOCK + admission), C3 (dev/official outputs), then
        `extra` commits [{sha, trailers, branches, paths}] inserted where their 'after' key says (default: end).
        review/ and checks/ files are committed only through `extra`."""
        extra = list(extra if extra is not None else self.extra_commits)
        buckets = {C1: [], C2: [], C3: []}
        claimed = set(p for e in extra for p in e.get('paths', []))
        for f in sorted(self.root.rglob('*')):
            if not f.is_file():
                continue
            rel = f.relative_to(self.root).as_posix()
            if rel in claimed or rel in exclude or '__pycache__' in rel:
                continue
            b = self._bucket(rel)
            if b is not None:
                buckets[b].append(rel)
        commits = [{'sha': C1, 'utc': '20261101T000000Z', 'author': 'Pencilbard', 'trailers': CODEX,
                    'branches': CODEX_BR, 'paths': buckets[C1]},
                   {'sha': C2, 'utc': '20261101T020000Z', 'author': 'Pencilbard', 'trailers': CODEX,
                    'branches': CODEX_BR, 'paths': buckets[C2]},
                   {'sha': C3, 'utc': '20261101T040000Z', 'author': 'Pencilbard', 'trailers': CODEX,
                    'branches': CODEX_BR, 'paths': buckets[C3]}]
        for e in extra:
            ent = {'sha': e['sha'], 'utc': e.get('utc', '20261101T050000Z'), 'author': 'Pencilbard',
                   'trailers': e.get('trailers', {}), 'branches': e.get('branches', []), 'paths': list(e['paths'])}
            after = e.get('after')
            if after is None:
                commits.append(ent)
            else:
                pos = [c['sha'] for c in commits].index(after)
                commits.insert(pos + 1, ent)
        commits = [c for c in commits if c['paths'] or c['sha'] not in (C1, C2, C3) or c['sha'] == C1]
        write(self.fx / 'timeline.json', {'head': commits[-1]['sha'], 'commits': commits})

    def ctx(self, stage='roster', tier=1, dry=False, options=None):
        git = common.TimelineGit(self.fx)
        return common.Ctx(repo_root=self.root, delivery=self.d, git=git, kb=common.KB(self.root),
                          rules=common.load_rules(), stage=stage, tier=tier, dry=dry, options=dict(options or {}))


def run(ctx, rid):
    idx = common.rule_index(ctx.rules)
    return common.run_check(ctx, idx[rid])


def status(ctx, rid):
    res = run(ctx, rid)
    return res[0].status, res[0].detail


class FixtureCase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp(prefix='mfc_wp4_')
        self.f = WP4Fixture(self.tmpdir).build()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def assertStatus(self, rid, want, ctx=None, contains=None):
        st, det = status(ctx or self.f.ctx(), rid)
        self.assertEqual(st, want, '%s: %s %s' % (rid, st, det))
        if contains:
            self.assertIn(contains, det)
        return det


ADM_IDS = ('ADM-01', 'ADM-TB-CEIL', 'ADM-TB-FLOOR', 'ADM-CTRL-ZERO', 'ADM-CTRL-ISO', 'ADM-CTRL-DOSE', 'ADM-DIR',
           'ADM-FIX', 'ADM-GATE', 'ADM-INPUT', 'ADM-ACQ', 'ADM-SHUF', 'ADM-EXPO', 'ADM-BUDGET')
AUT_IDS = ('AUT-01', 'AUT-02', 'AUT-03')


class CleanFixture(FixtureCase):
    def test_every_admission_and_authorization_check_passes(self):
        ctx = self.f.ctx()
        for rid in ADM_IDS + AUT_IDS:
            st, det = status(ctx, rid)
            self.assertEqual(st, common.PASS, '%s: %s' % (rid, det))
        self.assertEqual(status(ctx, 'ADM-REUSE')[0], common.NOT_APPLICABLE)
        self.assertEqual(status(ctx, 'AUT-04')[0], common.NOT_APPLICABLE)

    def test_registered_ids_match_rules(self):
        rules = common.load_rules()
        mine = sorted(r['id'] for r in rules['rules'] if r['module'] == 'admitcheck')
        self.assertEqual(mine, sorted(set(ADM_IDS) | {'ADM-REUSE', 'AUT-01', 'AUT-02', 'AUT-03', 'AUT-04'}))
        for rid in mine:
            fn, err = common.resolve_check(common.rule_index(rules)[rid])
            self.assertIsNotNone(fn, err)


class CtrlChecks(FixtureCase):
    def test_package_b_identical_digests_fail_zero(self):
        # FM-02 / C02a: the lesion arm writes the zeroed channel and its post_first equals W
        w = self.f.read('results/receipts/wt1/W.json')
        npt = self.f.read('results/receipts/wt1/NPT.json')
        npt['channel_digests']['post_first'] = dict(w['channel_digests']['post_first'])
        npt['writes_by_channel'] = dict(w['writes_by_channel'], pair_memory=288)
        self.f.w('results/receipts/wt1/NPT.json', npt)
        det = self.assertStatus('ADM-CTRL-ZERO', common.FAIL, contains='TREATMENT_NOT_DELIVERED')
        self.assertIn('288', det)
        self.assertIn('differs from birth', det)

    def test_candidate_never_writing_the_channel_fails(self):
        w = self.f.read('results/receipts/wt1/W.json')
        w['writes_by_channel']['pair_memory'] = 0
        w['channel_digests']['post_first']['pair_memory'] = w['channel_digests']['birth']['pair_memory']
        self.f.w('results/receipts/wt1/W.json', w)
        self.assertStatus('ADM-CTRL-ZERO', common.FAIL, contains='never writes')

    def test_write_counter_site_must_name_a_native_closure_function(self):
        npt = self.f.read('results/receipts/wt1/NPT.json')
        npt['write_counter_site'] = 'native.py:Learner.no_such_method'
        self.f.w('results/receipts/wt1/NPT.json', npt)
        self.assertStatus('ADM-CTRL-ZERO', common.FAIL, contains='not defined')
        npt['write_counter_site'] = 'run.py:main'
        self.f.w('results/receipts/wt1/NPT.json', npt)
        self.assertStatus('ADM-CTRL-ZERO', common.FAIL, contains='not in the native closure')
        npt['write_counter_site'] = NAME + '/native.py:Learner.write_pair'  # repo-relative form is accepted
        self.f.w('results/receipts/wt1/NPT.json', npt)
        self.assertStatus('ADM-CTRL-ZERO', common.PASS)
        npt['write_counter_site'] = {'path': 'native.py'}
        self.f.w('results/receipts/wt1/NPT.json', npt)
        self.assertStatus('ADM-CTRL-ZERO', common.FAIL, contains='is not <path>:<function>')

    def test_write_site_file_changed_since_lock_build(self):
        self.f.w('native.py', NATIVE_PY + b'\n# edited\n')
        self.assertStatus('ADM-CTRL-ZERO', common.FAIL, contains='changed since the closure')

    def test_receipts_required_in_admission_worlds_only(self):
        # a testbed-only technical world (evaluator CEILING_/FLOOR_ rows, no learner run) needs no receipts:
        # receipts are required in ADMISSION.worlds, which ADM-01 cross-checks against the receipts
        lock = self.f.read('ENDPOINT_LOCK.json')
        lock['worlds']['technical'] = ['wt1', 'wt_testbed_2']
        self.f.set_lock(lock)
        for cid in ('ADM-CTRL-ZERO', 'ADM-CTRL-ISO', 'ADM-CTRL-DOSE'):
            self.assertStatus(cid, common.PASS)
        adm = self.f.read('ADMISSION.json')
        adm['worlds'] = []  # unusable: every lock technical world is required (fail closed)
        self.f.w('ADMISSION.json', adm)
        self.assertStatus('ADM-CTRL-ZERO', common.MISSING, contains='wt_testbed_2')
        self.assertStatus('ADM-CTRL-DOSE', common.MISSING, contains='wt_testbed_2')

    def test_missing_control_receipt_is_missing(self):
        (self.f.d / 'results/receipts/wt1/NPT.json').unlink()
        self.assertStatus('ADM-CTRL-ZERO', common.MISSING)
        self.assertStatus('ADM-CTRL-DOSE', common.MISSING)

    def test_receipt_path_identity_mismatch_fails(self):
        npt = self.f.read('results/receipts/wt1/NPT.json')
        npt['arm'] = 'W'
        self.f.w('results/receipts/wt1/NPT.json', npt)
        self.assertStatus('ADM-CTRL-ZERO', common.FAIL, contains='its path says')

    def test_confound_on_undeclared_channel_fails_iso(self):
        # FM-03 / C03: the ZERO arm also changes the stop channel at post_first
        npt = self.f.read('results/receipts/wt1/NPT.json')
        npt['channel_digests']['post_first']['stop'] = d64('npt-stop-changed')
        self.f.w('results/receipts/wt1/NPT.json', npt)
        self.assertStatus('ADM-CTRL-ZERO', common.PASS)
        self.assertStatus('ADM-CTRL-ISO', common.FAIL, contains='stop')
        lock = self.f.read('ENDPOINT_LOCK.json')
        lock['arms'][1]['declared_changed'] = ['stop']
        self.f.set_lock(lock)
        self.assertStatus('ADM-CTRL-ISO', common.PASS)

    def test_iso_covers_the_causal_control_by_role(self):
        # M15-like: the control is relabelled and its zero_channels cleared; ISO still compares it with W
        lock = self.f.read('ENDPOINT_LOCK.json')
        lock['arms'][1]['role'] = 'other'
        lock['arms'][1]['zero_channels'] = []
        self.f.set_lock(lock)
        self.assertStatus('ADM-CTRL-ZERO', common.NOT_APPLICABLE)
        self.assertStatus('ADM-CTRL-ISO', common.FAIL, contains='pair_memory')

    def test_low_dose_fails(self):
        # FM-02 / C02b: realized_dose 0.003 < max(min_realized_dose 0.05, floor 0.05)
        npt = self.f.read('results/receipts/wt1/NPT.json')
        npt['realized_dose'] = {'teaching_withheld_fraction': 0.003}
        self.f.w('results/receipts/wt1/NPT.json', npt)
        self.assertStatus('ADM-CTRL-DOSE', common.FAIL, contains='0.003')

    def test_dose_floor_wins_over_a_lower_lock_value(self):
        lock = self.f.read('ENDPOINT_LOCK.json')
        lock['arms'][1]['min_realized_dose'] = 0.001
        self.f.set_lock(lock)
        npt = self.f.read('results/receipts/wt1/NPT.json')
        npt['realized_dose'] = {'teaching_withheld_fraction': 0.01}
        self.f.w('results/receipts/wt1/NPT.json', npt)
        self.assertStatus('ADM-CTRL-DOSE', common.FAIL, contains='0.05')

    def test_dose_missing_is_fail(self):
        npt = self.f.read('results/receipts/wt1/NPT.json')
        npt['realized_dose'] = {}
        self.f.w('results/receipts/wt1/NPT.json', npt)
        self.assertStatus('ADM-CTRL-DOSE', common.FAIL)


class FixtureChecks(FixtureCase):
    def set_fixtures(self, rows):
        self.f.w('results/admission/FIXTURES.jsonl', jsonl(rows))

    def test_negative_directional_delta_fails(self):
        # FM-17 / C17
        rows = fixture_rows_default()
        rows[0]['target_margin_delta_vs_passive'] = -0.02
        self.set_fixtures(rows)
        self.assertStatus('ADM-DIR', common.FAIL, contains='not > 0')

    def test_zero_replay_applications_fail(self):
        rows = fixture_rows_default()
        rows[1]['zero_replay_applications'] = 32
        rows[1]['zero_coef_digest_equals_passive'] = False
        self.set_fixtures(rows)
        det = self.assertStatus('ADM-DIR', common.FAIL)
        self.assertIn('32', det)
        self.assertIn('zero_coef_digest_equals_passive', det)

    def test_missing_sign_is_missing_and_undeclared_port_fails(self):
        rows = [r for r in fixture_rows_default() if r.get('sign') != '-']
        self.set_fixtures(rows)
        self.assertStatus('ADM-DIR', common.MISSING, contains='sign -')
        rows = fixture_rows_default() + [dict(fixture_rows_default()[0], port='ghost')]
        self.set_fixtures(rows)
        self.assertStatus('ADM-DIR', common.FAIL, contains='undeclared')

    def test_fixture_density_differs_from_params(self):
        # FM-04 / C04a: fixture pair_density 0.02 vs production 0.0354
        rows = fixture_rows_default(dict(PARAMS, pair_density=0.02))
        self.set_fixtures(rows)
        det = self.assertStatus('ADM-FIX', common.FAIL, contains='pair_density')
        self.assertIn('0.0354', det)
        lock = self.f.read('ENDPOINT_LOCK.json')
        lock['admission_thresholds']['fixture_param_tolerances'] = {'pair_density': 0.02}
        self.f.set_lock(lock)
        self.assertStatus('ADM-FIX', common.PASS)

    def test_missing_and_extra_fixture_keys_fail(self):
        p = dict(PARAMS)
        del p['width']
        self.set_fixtures(fixture_rows_default(p))
        self.assertStatus('ADM-FIX', common.FAIL, contains='lacks production param width')
        self.set_fixtures(fixture_rows_default(dict(PARAMS, probe=1)))
        self.assertStatus('ADM-FIX', common.FAIL, contains='not a production param')
        nested = json.loads(json.dumps(PARAMS))
        nested['warmup']['steps'] = 51
        self.set_fixtures(fixture_rows_default(nested))
        self.assertStatus('ADM-FIX', common.FAIL, contains='warmup.steps')

    def test_params_changed_after_lock_build_fail(self):
        self.f.w('params.json', dict(PARAMS, lr=0.2))
        self.assertStatus('ADM-FIX', common.FAIL, contains='SOURCE_LOCK.params_sha256')

    def test_gate_without_negative_fixture_fails(self):
        # FM-04 / C04a: a plumbing gate that cannot fail
        rows = [r for r in fixture_rows_default() if r.get('kind') != 'negative']
        self.set_fixtures(rows)
        self.assertStatus('ADM-GATE', common.FAIL, contains='no negative fixture')

    def test_gate_accepting_its_negative_fixture_fails(self):
        rows = fixture_rows_default()
        rows[2]['result'] = True
        self.set_fixtures(rows)
        self.assertStatus('ADM-GATE', common.FAIL, contains='accepts its negative')
        rows[2]['result'] = 'false'
        self.set_fixtures(rows)
        self.assertStatus('ADM-GATE', common.FAIL, contains='not a boolean')

    def test_no_fixtures_file_is_missing(self):
        (self.f.d / 'results/admission/FIXTURES.jsonl').unlink()
        for rid in ('ADM-DIR', 'ADM-FIX', 'ADM-GATE'):
            self.assertStatus(rid, common.MISSING)

    def test_no_ports_or_gates_is_not_applicable(self):
        lock = self.f.read('ENDPOINT_LOCK.json')
        lock['ports'] = []
        lock['gates'] = []
        self.f.set_lock(lock)
        for rid in ('ADM-DIR', 'ADM-FIX', 'ADM-GATE'):
            self.assertStatus(rid, common.NOT_APPLICABLE)


class InputAcqShuf(FixtureCase):
    def test_low_contrast_ratio_fails(self):
        # FM-16 / C16: contrast ratio 0.014
        self.f.write_admission(input_contrast={'metric_definition': 'L2', 'analytic_snr': 0.027,
                                               'same_input_diff_noise': 1.0, 'diff_input_same_noise': 0.014,
                                               'ratio': 0.014, 'prompt_id_loo_acc': 0.31, 'chance': 0.01})
        self.assertStatus('ADM-INPUT', common.FAIL, contains='INPUT_STATE_NOT_ADMITTED')

    def test_ratio_must_agree_with_its_components(self):
        ic = dict(self.f.read('ADMISSION.json')['input_contrast'], ratio=9.0)
        self.f.write_admission(input_contrast=ic)
        self.assertStatus('ADM-INPUT', common.FAIL, contains='disagrees')

    def test_prompt_id_accuracy_below_chance_plus_margin(self):
        ic = dict(self.f.read('ADMISSION.json')['input_contrast'], prompt_id_loo_acc=0.05)
        self.f.write_admission(input_contrast=ic)
        self.assertStatus('ADM-INPUT', common.FAIL, contains='prompt_id_loo_acc')

    def test_structure_meters_must_be_present(self):
        sm = dict(self.f.read('ADMISSION.json')['structure_meters'])
        del sm['code_overlap_end']
        self.f.write_admission(structure_meters=sm)
        self.assertStatus('ADM-INPUT', common.FAIL, contains='code_overlap_end')

    def test_not_acquiring(self):
        # FM-16 / C16: taught 0% at the prefix checkpoint
        rows = [r for r in output_rows(TECH, 'technical', ARMS + ('CEILING_EXACT_BP', 'FLOOR_FRF'))]
        for r in rows:
            if r['arm'] == 'W' and r['pool'] == 'TAUGHT' and r['checkpoint'] == 'prefix':
                st = stream_of(wrong_of(bytes.fromhex(r['expected_hex'])))
                r['stream_hex'], r['emitted_hex'], r['own_eos'] = st.hex(), st[:1].hex(), True
        self.f.w('results/admission/OUTPUTS.jsonl', jsonl(rows))
        self.assertStatus('ADM-ACQ', common.FAIL, contains='NOT_ACQUIRING')

    def test_acq_margin_floor_applies(self):
        lock = self.f.read('ENDPOINT_LOCK.json')
        del lock['admission_thresholds']['taught_acq_margin_min']
        self.f.set_lock(lock)
        self.assertStatus('ADM-ACQ', common.PASS)

    def test_shuf_binary_derangement_fails(self):
        # FM-04 / C04b: SHUF is a derangement of binary labels (agreement 0)
        pairs = [(x, y) for (x, y) in T_ALL if x + y in (5, 6)]
        rows = [{'world': w_, 'arm': 'SHUF', 'prompt_hex': ph(x, y), 'answer_hex': str(11 - (x + y)).encode().hex(),
                 'phase': 'teach', 'feedback_bearing': True, 'presentations': 12} for (x, y) in pairs for w_ in ['wt1']]
        for w_ in TECH + DEV:
            self.f.w('manifests/TAUGHT/%s/SHUF.jsonl' % w_, jsonl([dict(r, world=w_) for r in rows]))
        det = self.assertStatus('ADM-SHUF', common.FAIL, contains='derangement')
        self.assertIn('outside shuf_band', det)

    def test_truthful_or_inconsistent_shuf_fails(self):
        for w_ in TECH + DEV:
            self.f.w('manifests/TAUGHT/%s/SHUF.jsonl' % w_, jsonl(taught_rows(w_, 'SHUF')))
        self.assertStatus('ADM-SHUF', common.FAIL, contains='1.0000 outside')

        def mixed(x, y):  # not orbit-consistent: (x, y) and (y, x) get different labels
            return truth(x, y) if x <= y else str(x + y + 1).encode('ascii')
        for w_ in TECH + DEV:
            self.f.w('manifests/TAUGHT/%s/SHUF.jsonl' % w_, jsonl(taught_rows(w_, 'SHUF', label=mixed)))
        self.assertStatus('ADM-SHUF', common.FAIL, contains='orbit-consistent')

    def test_missing_shuf_manifest_is_missing(self):
        (self.f.d / 'manifests/TAUGHT/wd2/SHUF.jsonl').unlink()
        self.assertStatus('ADM-SHUF', common.MISSING, contains='wd2')

    def test_shuf_band_is_intersected_with_the_floor(self):
        lock = self.f.read('ENDPOINT_LOCK.json')
        lock['admission_thresholds']['shuf_band'] = [0.0, 1.0]
        self.f.set_lock(lock)
        for w_ in TECH + DEV:
            self.f.w('manifests/TAUGHT/%s/SHUF.jsonl' % w_, jsonl(taught_rows(w_, 'SHUF')))
        self.assertStatus('ADM-SHUF', common.FAIL, contains='[0.35, 0.65]')


class ExposureChecks(FixtureCase):
    def test_commuted_held_item_in_calibration_stream_fails(self):
        # FM-12 / C12: a commuted held item in a feedback-bearing calibration stream
        rows = taught_rows('wt1', 'W')
        a, b = H[3]
        rows.append({'world': 'wt1', 'arm': 'W', 'prompt_hex': ph(b, a), 'answer_hex': truth(a, b).hex(),
                     'phase': 'calibration', 'feedback_bearing': True, 'presentations': 1})
        self.f.w('manifests/TAUGHT/wt1/W.jsonl', jsonl(rows))
        self.assertStatus('ADM-EXPO', common.FAIL, contains='held orbits exposed')

    def test_presentation_only_rows_do_not_count(self):
        rows = taught_rows('wt1', 'W')
        a, b = H[3]
        rows.append({'world': 'wt1', 'arm': 'W', 'prompt_hex': ph(b, a), 'answer_hex': truth(a, b).hex(),
                     'phase': 'presentation', 'feedback_bearing': False, 'presentations': 1})
        self.f.w('manifests/TAUGHT/wt1/W.jsonl', jsonl(rows))
        self.assertStatus('ADM-EXPO', common.PASS)

    def test_undeclared_stream_file_fails(self):
        self.f.w('manifests/calibration.jsonl', jsonl([{'prompt_hex': ph(1, 1), 'answer_hex': '32',
                                                        'feedback_bearing': True}]))
        self.assertStatus('ADM-EXPO', common.FAIL, contains='undeclared TAUGHT/stream file manifests/calibration.jsonl')
        (self.f.d / 'manifests/calibration.jsonl').unlink()
        self.f.w('results/dev/TAUGHT_LOG.jsonl', jsonl([{'prompt_hex': ph(1, 1)}]))
        self.assertStatus('ADM-EXPO', common.FAIL, contains='results/dev/TAUGHT_LOG.jsonl')

    def test_declared_calibration_stream_is_scanned(self):
        lock = self.f.read('ENDPOINT_LOCK.json')
        lock['exposure_streams'].append({'path_glob': 'manifests/calibration/*.jsonl', 'feedback_bearing': True,
                                         'phase': 'calibration'})
        self.f.set_lock(lock)
        a, b = H[0]
        self.f.w('manifests/calibration/c1.jsonl', jsonl([{'prompt_hex': ph(2, 2)}]))
        self.assertStatus('ADM-EXPO', common.PASS)
        self.f.w('manifests/calibration/c1.jsonl', jsonl([{'prompt_hex': ph(b, a)}]))
        self.assertStatus('ADM-EXPO', common.FAIL, contains='held orbits exposed')

    def test_e2_uses_exact_prompts(self):
        lock = self.f.read('ENDPOINT_LOCK.json')
        lock['claim_level_target'] = 'E2'
        self.f.set_lock(lock)
        rows = taught_rows('wt1', 'W')
        a, b = H[3]
        rows.append({'world': 'wt1', 'arm': 'W', 'prompt_hex': ph(b, a), 'answer_hex': truth(a, b).hex(),
                     'phase': 'calibration', 'feedback_bearing': True, 'presentations': 1})
        self.f.w('manifests/TAUGHT/wt1/W.jsonl', jsonl(rows))
        self.assertStatus('ADM-EXPO', common.PASS)
        rows.append(dict(rows[-1], prompt_hex=ph(a, b)))
        self.f.w('manifests/TAUGHT/wt1/W.jsonl', jsonl(rows))
        self.assertStatus('ADM-EXPO', common.FAIL, contains='held prompts exposed')


class TestbedChecks(FixtureCase):
    def rewrite_arm(self, arm, beh):
        rows = output_rows(TECH, 'technical', ARMS + ('CEILING_EXACT_BP', 'FLOOR_FRF'))
        alt = {(r['arm'], r['pool'], r['checkpoint'], r['item_id']): r
               for r in output_rows(TECH, 'technical', (arm,), beh=beh)}
        rows = [alt.get((r['arm'], r['pool'], r['checkpoint'], r['item_id']), r) for r in rows]
        self.f.w('results/admission/OUTPUTS.jsonl', jsonl(rows))

    def test_ceiling_failing_is_cannot_test(self):
        # FM-05 / C05a
        self.rewrite_arm('CEILING_EXACT_BP', lambda w, a, p, i: i % 9 == 0)
        self.assertStatus('ADM-TB-CEIL', common.FAIL, contains='CANNOT_TEST_CLAIM_AS_SPECIFIED')

    def test_floor_passing_is_cannot_test(self):
        # FM-05 / C05b
        self.rewrite_arm('FLOOR_FRF', lambda w, a, p, i: True)
        self.assertStatus('ADM-TB-FLOOR', common.FAIL, contains='CANNOT_TEST_CLAIM_AS_SPECIFIED')

    def test_grid_must_cover_constraints(self):
        lock = self.f.read('ENDPOINT_LOCK.json')
        lock['testbed']['ceiling_grid'] = [{'id': 'EXACT_BP', 'learner': 'SOMETHING_ELSE'}]
        lock['testbed']['floor_grid'] = []
        self.f.set_lock(lock)
        self.assertStatus('ADM-TB-CEIL', common.FAIL, contains='CONSTRAINTS says')
        self.assertStatus('ADM-TB-FLOOR', common.FAIL, contains='omits CONSTRAINTS config FRF')

    def test_declared_floor_without_rows_is_missing(self):
        lock = self.f.read('ENDPOINT_LOCK.json')
        lock['testbed']['floor_grid'].append({'id': 'FRF2', 'learner': 'FROZEN_RANDOM_FEATURES'})
        self.f.set_lock(lock)
        self.assertStatus('ADM-TB-FLOOR', common.MISSING, contains='FLOOR_FRF2')

    def test_no_admission_rows_is_missing(self):
        (self.f.d / 'results/admission/OUTPUTS.jsonl').unlink()
        self.assertStatus('ADM-TB-CEIL', common.MISSING)
        self.assertStatus('ADM-TB-FLOOR', common.MISSING)


class Adm01Budget(FixtureCase):
    def test_lock_sha_mismatch_fails(self):
        self.f.write_admission(endpoint_lock_sha256='c' * 64)
        self.assertStatus('ADM-01', common.FAIL, contains='endpoint_lock_sha256')

    def test_non_technical_world_fails(self):
        self.f.write_admission(worlds=['wt1', 'wd1'])
        self.assertStatus('ADM-01', common.FAIL, contains='wd1')

    def test_receipt_summary_disagreement_fails(self):
        npt = self.f.read('results/receipts/wt1/NPT.json')
        npt['world_role'] = 'dev'
        self.f.w('results/receipts/wt1/NPT.json', npt)
        self.assertStatus('ADM-01', common.FAIL, contains='not technical')
        self.f.w('results/receipts/wt1/NPT.json', dict(npt, world_role='technical'))
        self.f.write_admission(receipts_glob='results/receipts/wt1/W.json')
        self.assertStatus('ADM-01', common.FAIL, contains='not covered by receipts_glob')

    def test_producer_must_match_endpoint_locked_admit(self):
        self.f.write_admission(producer={'script': 'admit.py', 'sha256': 'd' * 64})
        self.assertStatus('ADM-01', common.FAIL, contains='producer.sha256')
        self.f.write_admission(producer={'script': 'other.py', 'sha256': 'd' * 64})
        self.assertStatus('ADM-01', common.FAIL, contains='lock.files.admit')

    def test_admission_committed_with_dev_outputs_fails(self):
        self.f.timeline(extra=[])
        tl = json.loads((self.f.fx / 'timeline.json').read_text())
        adm = NAME + '/ADMISSION.json'
        for c in tl['commits']:
            if adm in c['paths']:
                c['paths'].remove(adm)
        tl['commits'][2]['paths'].append(adm)
        write(self.f.fx / 'timeline.json', tl)
        self.assertStatus('ADM-01', common.FAIL, contains='not a proper ancestor')

    def test_missing_admission_is_missing(self):
        (self.f.d / 'ADMISSION.json').unlink()
        self.assertStatus('ADM-01', common.MISSING)
        self.assertStatus('ADM-INPUT', common.MISSING)

    def test_budget_above_caps_fails(self):
        lock = self.f.read('ENDPOINT_LOCK.json')
        lock['budget'] = {'max_records': 54432, 'max_cpu_h': 1}
        self.f.set_lock(lock)
        self.assertStatus('ADM-BUDGET', common.FAIL, contains='54432')

    def test_paired_atomic_false_fails(self):
        # FM-20 / C20
        lock = self.f.read('ENDPOINT_LOCK.json')
        lock['schedule']['paired_atomic'] = False
        self.f.set_lock(lock)
        self.assertStatus('ADM-BUDGET', common.FAIL, contains='paired_atomic')

    def test_budget_without_authorization_is_missing(self):
        (self.f.d / 'AUTHORIZATION.json').unlink()
        self.assertStatus('ADM-BUDGET', common.MISSING)
        for rid in ('AUT-01', 'AUT-02', 'AUT-03', 'AUT-04'):
            self.assertStatus(rid, common.MISSING)


class ReuseChecks(FixtureCase):
    def test_reuse_citation(self):
        self.f.write_admission(parent_admission_sha='e' * 64)
        self.assertStatus('ADM-REUSE', common.FAIL, contains='lineage.parent is null')
        self.assertStatus('ADM-INPUT', common.PASS)
        self.assertStatus('ADM-REUSE', common.NOT_APPLICABLE, ctx=self.f.ctx(tier=2))

    def test_valid_reuse_skips_input(self):
        parent = 'minifly_wp4par_dev_v1_20261020'
        pd = self.f.root / parent
        shutil.copytree(str(self.f.d), str(pd))
        plock = json.loads((pd / 'ENDPOINT_LOCK.json').read_text())
        plock['delivery'] = parent
        write(pd / 'ENDPOINT_LOCK.json', plock)
        rep_results = [{'id': 'ADM-INPUT', 'status': 'PASS'}]
        write(pd / 'checks/GATE_roster_20261020T000000Z.json', {
            'schema': common.SCHEMA_GATE_REPORT, 'stage': 'roster', 'dry': False, 'delivery': parent,
            'utc': '20261020T000000Z', 'git_mode': 'timeline', 'results': rep_results, 'exit_code': 0,
            'verdict': 'PASS'})
        lock = self.f.read('ENDPOINT_LOCK.json')
        lock['lineage']['parent'] = parent + '@' + 'a' * 40
        self.f.set_lock(lock)
        self.f.write_admission(parent_admission_sha=common.sha256_file(pd / 'ADMISSION.json'))
        self.assertStatus('ADM-REUSE', common.PASS)
        self.assertStatus('ADM-INPUT', common.PASS, contains='reused parent admission')
        # 'not_derivable' analytic SNR makes ADM-INPUT mandatory
        lock['admission_thresholds']['analytic_input_snr'] = 'not_derivable'
        self.f.set_lock(lock)
        self.f.write_admission(parent_admission_sha=common.sha256_file(pd / 'ADMISSION.json'))
        st, det = status(self.f.ctx(), 'ADM-INPUT')
        self.assertNotIn('reused parent admission', det)


class AuthorizationChecks(FixtureCase):
    def set_auth(self, **over):
        self.f.w('AUTHORIZATION.json', authorization(**over))

    def test_recommended_label_fails(self):
        # FM-28 / C28
        opts = authorization()['options_shown']
        opts[0]['label'] = 'A: open the held E3 endpoint (Recommended)'
        self.set_auth(options_shown=opts)
        self.assertStatus('AUT-03', common.FAIL, contains='Recommended')

    def test_caps_absent_from_verbatim_fail(self):
        # FM-28 / C28: caps 54,432 records absent from the owner's words
        self.set_auth(caps={'max_records': 54432, 'max_cpu_h': 2, 'max_cycles': 3})
        self.assertStatus('AUT-02', common.FAIL, contains='max_records = 54432')
        self.set_auth(caps={'max_records': 54432, 'max_cpu_h': 2, 'max_cycles': 3},
                      verbatim=VERBATIM + ' Raise the record cap to 54,432.')
        self.assertStatus('AUT-02', common.PASS)

    def test_default_caps(self):
        self.set_auth(verbatim='Go ahead with the design and the dev cycles.', caps={}, defaults_applied=True)
        self.assertStatus('AUT-02', common.PASS, contains='default')
        self.set_auth(verbatim='Go ahead with the design and the dev cycles.', caps={}, defaults_applied=False)
        self.assertStatus('AUT-02', common.FAIL, contains='defaults_applied')

    def test_stage_must_be_covered(self):
        st = dict(authorization()['stages'], dev_cycles=False)
        self.set_auth(stages=st)
        self.assertStatus('AUT-02', common.FAIL, contains='dev_cycles', ctx=self.f.ctx(stage='roster'))
        self.assertStatus('AUT-02', common.PASS, ctx=self.f.ctx(stage='lock'))
        self.assertStatus('AUT-02', common.FAIL, contains='official', ctx=self.f.ctx(stage='official'))
        self.assertStatus('AUT-02', common.FAIL, ctx=self.f.ctx(stage='roster',
                                                                 options={'launch_stage': 'rehearsal'}))

    def test_official_go(self):
        st = dict(authorization()['stages'])
        st['official'] = {'granted': True, 'verbatim': 'GO for the official run on these locks.',
                          'utc': '20261103T000000Z', 'lock_shas_shown': ['f' * 64]}
        self.set_auth(stages=st)
        self.assertStatus('AUT-02', common.FAIL, contains='lock_shas_shown', ctx=self.f.ctx(stage='official'))
        ctx = self.f.ctx(stage='official')
        st['official']['lock_shas_shown'] = [ctx.endpoint_lock_sha256, ctx.source_lock_sha256]
        self.set_auth(stages=st)
        self.assertStatus('AUT-02', common.PASS, ctx=self.f.ctx(stage='official'))
        self.assertStatus('AUT-01', common.PASS)

    def test_dry_official_needs_dev_cycles_only(self):
        self.assertStatus('AUT-02', common.PASS, ctx=self.f.ctx(stage='official', dry=True))

    def test_unknown_stage_is_missing(self):
        self.assertStatus('AUT-02', common.MISSING, ctx=self.f.ctx(stage=None))

    def test_aut01_fields(self):
        a = authorization()
        del a['recommendation_shown']
        a['stages']['push'] = '__FILL__'
        self.f.w('AUTHORIZATION.json', a)
        det = self.assertStatus('AUT-01', common.FAIL)
        self.assertIn('recommendation_shown', det)
        self.assertIn('__FILL__', det)
        self.set_auth(options_shown=[])
        self.assertStatus('AUT-01', common.FAIL, contains='options_shown')
        st = dict(authorization()['stages'], deploy=True)
        self.set_auth(stages=st)
        self.assertStatus('AUT-01', common.FAIL, contains='unknown stage key')

    def test_conditional_predicates(self):
        self.f.w('results/dev/SCORES.json', {'G': 0.2, 'verdict': 'PASS', 'list': [1, 2]})
        cond = {'predicate': {'artifact': 'results/dev/SCORES.json', 'pointer': '/G', 'op': 'gt', 'value': 0.1},
                'evaluated_by': 'checker', 'result': True, 'utc': '20261102T000000Z'}
        self.set_auth(conditionals=[cond])
        self.assertStatus('AUT-04', common.PASS)
        cond['predicate']['value'] = 0.5
        cond['result'] = None
        self.set_auth(conditionals=[cond])
        self.assertStatus('AUT-04', common.FAIL, contains='not met')
        cond['predicate']['value'] = 0.1
        cond['result'] = False
        self.set_auth(conditionals=[cond])
        self.assertStatus('AUT-04', common.FAIL, contains='recorded result')
        cond = {'predicate': {'artifact': 'results/dev/NOPE.json', 'pointer': '/G', 'op': 'gt', 'value': 0.1}}
        self.set_auth(conditionals=[cond])
        self.assertStatus('AUT-04', common.MISSING)
        cond = {'predicate': {'artifact': 'results/dev/SCORES.json', 'pointer': '/G', 'op': 'approx', 'value': 0.1}}
        self.set_auth(conditionals=[cond])
        self.assertStatus('AUT-04', common.FAIL, contains='invalid')

    def test_self_evaluated_conditional_fails(self):
        # FM-28 / C28: the beneficiary (author family codex) judges its own conditional grant
        cond = {'predicate': 'if dev looks good, continue', 'evaluated_by': 'codex', 'result': True,
                'utc': '20261102T000000Z'}
        self.set_auth(conditionals=[cond])
        self.assertStatus('AUT-04', common.FAIL, contains='beneficiary')
        cond['evaluated_by'] = 'claude:s9'
        self.set_auth(conditionals=[cond])
        self.assertStatus('AUT-04', common.PASS)
        cond['result'] = False
        self.set_auth(conditionals=[cond])
        self.assertStatus('AUT-04', common.FAIL, contains='not met')
        self.set_auth(conditionals=[{'predicate': 'free text only'}])
        self.assertStatus('AUT-04', common.FAIL, contains='self-judged')

    def test_conditional_for_a_later_stage_is_deferred(self):
        cond = {'predicate': {'artifact': 'results/official/VERDICT.json', 'pointer': '/verdict', 'op': 'eq',
                              'value': 'PASS'}, 'stage': 'official'}
        self.set_auth(conditionals=[cond])
        self.assertStatus('AUT-04', common.PASS, contains='deferred', ctx=self.f.ctx(stage='roster'))
        self.assertStatus('AUT-04', common.MISSING, ctx=self.f.ctx(stage='official'))


class PointerAndHelpers(unittest.TestCase):
    def test_json_pointer(self):
        doc = {'a': {'b/c': [10, {'~k': 3}]}, '': 1}
        self.assertEqual(admitcheck.json_pointer(doc, '/a/b~1c/0'), 10)
        self.assertEqual(admitcheck.json_pointer(doc, '/a/b~1c/1/~0k'), 3)
        self.assertEqual(admitcheck.json_pointer(doc, ''), doc)
        self.assertIs(admitcheck.json_pointer(doc, '/a/x'), admitcheck._MISSING_PTR)
        self.assertIs(admitcheck.json_pointer(doc, '/a/b~1c/05'), admitcheck._MISSING_PTR)
        with self.assertRaises(common.InvalidInput):
            admitcheck.json_pointer(doc, 'a')

    def test_family(self):
        self.assertEqual(admitcheck.family('claude:abc'), 'claude')
        self.assertEqual(admitcheck.family('Codex'), 'codex')
        self.assertIsNone(admitcheck.family('owner'))
        self.assertIsNone(admitcheck.family('claudette'))


if __name__ == '__main__':
    unittest.main()
