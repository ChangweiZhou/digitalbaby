"""Static files of the corpus base delivery: ENDPOINT_LOCK, AUTHORIZATION, params, stubs, manifests
and the PROJECT_DOCUMENTS/KB overlay. Field names follow SPEC Section 8 and SCHEMAS.md."""
import copy

import base_items as BI
from mrk_util import canonical_json, dump_json, dump_jsonl, hexs, sha256_bytes, sha256_json

D = 'minifly_corpus_base_dev_v1_20261101'
KB = 'PROJECT_DOCUMENTS/KB/'
CODEX = {'author': 'Pencilbard', 'trailers': {'Acting-Agent': ['codex']}}
CLAUDE = {'author': 'Claude', 'trailers': {'Acting-Agent': ['claude:s-corpus-review']}}
CODEX_BRANCH = 'origin/codex/minifly-corpus-base'
CLAUDE_BRANCH = 'origin/claude/minifly-corpus-review'
PARAMS = {'pair_density': 0.0354, 'readout_lr': 0.005, 'warmup_start_ceiling': '1/16'}

CONSTRAINTS = {
    'schema': 'minifly.constraints.v1', 'owner': 'Mr. Changwei Zhou', 'version': '1',
    'status': 'DRAFT_PENDING_OWNER_DECISION', 'user_decision': None,
    'objective_ref': {'path': 'minifly_b/package/DOCS/MINIFLY_OBJECTIVE_CONTRACT.md',
                      'sha256': '3832a30f19ad168a03ed112d8c60f43528a5c4d4dd47adf1951f8c01b79ef57d'},
    'native_forbidden': {
        'imports': ['torch.autograd', 'torch.optim', 'torch.nn.MultiheadAttention', 'transformers',
                    'jax', 'sklearn.linear_model'],
        'calls': ['.backward(', 'scaled_dot_product_attention(', 'numpy.linalg.lstsq(',
                  'numpy.linalg.pinv(', 'torch.linalg.lstsq('],
        'identifiers': ['transformer', 'attention', 'elm', 'requires_grad=True']},
    'supplied_arithmetic_heuristics': ['int() over decoded operand bytes', 'carry or digit tables',
                                       "'+' applied to parsed operands"],
    'evaluator_only_exemptions': [],
    'io': {'raw_bytes': True, 'own_eos_required': True},
    'warmup': {'start_ceiling': '1/16', 'ramp': 'gradual'},
    'objective_targets': 'TARGET_UNSET',
    'threshold_floors': {
        'margin_abs': 0.05, 'absolute_floor_above_chance': 0.05, 'per_world_floor': -0.05,
        'min_worlds_nonpilot': 8, 'min_items_per_stratum': 8, 'input_contrast_min': 2.0,
        'prompt_id_margin_min': 0.1, 'taught_acq_margin_min': 0.05, 'min_realized_dose': 0.05,
        'shuf_band': [0.35, 0.65], 'old_max_drop': 0.1, 'old_acq_margin': 0.05,
        'analytic_input_snr_min': 0.5, 'oc_min_power': 0.8, 'alpha_max': 0.05},
    'tier_thresholds': {'t0_max_cpu_h': 1, 't1_max_cpu_h': 4, 't1_max_records': 20000,
                        'family_t1_max_cpu_h': 12},
    'default_grant_caps': {'max_cpu_h': 2, 'max_records': 5000, 'max_cycles': 3, 'official': False,
                           'push': False, 'adoption': False, 'pointer_update': False},
    'e3_cadence_max': 2, 'e3_cadence_counts_legacy': False, 'review_debt_max_open': 3,
    'required_arms_by_level': {'E1': ['candidate', 'shuf'],
                               'E2': ['candidate', 'shuf', 'positive_control'],
                               'E3': ['candidate', 'no_prior_teaching', 'shuf', 'oracle_plumbing']},
    'testbed_floor_grid': [{'id': f, 'learner': 'FROZEN_RANDOM_FEATURES',
                            'width': int(f.split('_')[1][1:]), 'readout_lr': float(f.split('LR')[1])}
                           for f in BI.FLOORS],
    'testbed_ceiling_grid': [{'id': 'EXACT_BP', 'learner': 'EXACT_BP'},
                             {'id': 'ORACLE_FEATURES', 'learner': 'ORACLE_FEATURES'}],
    'confirm_alpha_spending': 'geometric_0.05_2^-j',
    'verification_subset_max_mb': 50,
    'notes': 'Corpus copy of the DRAFT values of SPEC 13.1 (pending UD-0001).',
}

FINDINGS_BASE = [
    {'id': 'F-UF-15', 'kind': 'finding', 'refers_to': None,
     'claim': 'STOP channel adds +22.10pp own-EOS exact on OLD over the stop-free arm.',
     'level': 'E1', 'conditions': {'substrate': 'minifly_b', 'input_code': 'raw bytes',
                                   'priors_supplied': 'none', 'feedback': 'gradual_warmup',
                                   'dose': 'full', 'comparator': 'stop-free arm',
                                   'best_shortcut': 'FIXED_1', 'worlds': '32 dev',
                                   'metric_definition': 'exact_own_eos'},
     'effect': {'value': 0.221, 'ci': [0.18, 0.26], 'unit': 'abs'},
     'source': 'PROJECT_DOCUMENTS/CATALOG.md:36@f92365e4', 'status': 'provisional',
     'delivery': 'minifly_stop_only_confirm_v1_20261023', 'author_agent': 'claude',
     'utc': '20261024T060000Z', 'caveat': None},
    {'id': 'F-20261024-claude-1', 'kind': 'narrowing', 'refers_to': 'F-UF-15',
     'claim': 'The +22.10pp gain does not exceed FIXED_1 of the arm\'s own stream.', 'level': 'E0',
     'conditions': {'substrate': 'minifly_b', 'input_code': 'raw bytes', 'priors_supplied': 'none',
                    'feedback': 'gradual_warmup', 'dose': 'full', 'comparator': 'FIXED_1',
                    'best_shortcut': 'FIXED_1', 'worlds': '32 dev',
                    'metric_definition': 'exact_own_eos'},
     'effect': {'value': -0.006, 'ci': [-0.02, 0.01], 'unit': 'abs'},
     'source': 'PROJECT_DOCUMENTS/CATALOG.md:40@f92365e4', 'status': 'narrowed',
     'delivery': 'minifly_stop_only_confirm_v1_20261023', 'author_agent': 'claude',
     'utc': '20261024T061000Z', 'caveat': 'FIXED_1: the gain is at the own-stream shortcut level.'},
]


def failure_classes():
    out = {}
    for i in range(1, 31):
        k = 'FM-%02d' % i
        out[k] = {'title': 'failure class %s (corpus copy)' % k, 'definition': 'See SPEC Section 18.',
                  'check_ids': [], 'fixture_ids': [], 'judgment_question': 'See SPEC Section 18.',
                  'archive_fc': [], 'incidents': []}
    return out


def kb_files(findings=None, incidents=None, ledger=None, ud=None, debt=None, confirm=None,
             canary=None, constraints=None):
    c = CONSTRAINTS if constraints is None else constraints
    return {
        KB + 'CONSTRAINTS.json': dump_json(c),
        KB + 'FAILURE_CLASSES.json': dump_json(failure_classes()),
        KB + 'FINDINGS.jsonl': dump_jsonl(FINDINGS_BASE if findings is None else findings),
        KB + 'INCIDENTS.jsonl': dump_jsonl(incidents or []),
        KB + 'DELIVERY_LEDGER.jsonl': dump_jsonl(ledger or []),
        KB + 'USER_DECISIONS.jsonl': dump_jsonl(ud or []),
        KB + 'REVIEW_DEBT.jsonl': dump_jsonl(debt or []),
        KB + 'CONFIRM_LEDGER.jsonl': dump_jsonl(confirm or []),
        KB + 'CANARY_COMMITMENTS.jsonl': dump_jsonl(canary or []),
    }


LEARNER_PY = '''"""Corpus stub learner for the MRK-1 regression corpus.

Parsed by the checker (closure, purity, LOCK-08); never executed.
"""
import hashlib


class Learner(object):
    def __init__(self, params):
        self.params = dict(params)
        self.pairs = {}
        self.log = {'pair_memory': [], 'readout': []}

    def write_pair(self, prompt, answer, channel):
        """State-mutation site: every write to a channel is logged here (write_counter_site)."""
        self.pairs[prompt] = answer
        self.log[channel].append(prompt)

    def writes_by_channel(self):
        return {name: len(rows) for name, rows in self.log.items()}

    def generate(self, prompt, cap):
        out = self.pairs.get(prompt, b'')
        return out.ljust(cap, b'\\n')[:cap]

    def digest(self, channel):
        data = repr(sorted(self.log[channel])).encode('utf-8')
        return hashlib.sha256(data).hexdigest()
'''

RUN_PY = '''"""Corpus stub runner for the MRK-1 regression corpus (never executed)."""
import json
import os
import sys

import learner


def main(argv):
    attempt = os.environ.get('MINIFLY_ATTEMPT_ID')
    if not attempt:
        sys.stderr.write('run.py: run through MFC launch\\n')
        return 2
    with open('params.json', 'rb') as fh:
        params = json.loads(fh.read().decode('utf-8'))
    model = learner.Learner(params)
    sys.stdout.write(json.dumps({'attempt': attempt, 'arms': sorted(model.log)}))
    sys.stdout.write('\\n')
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
'''

ADMIT_PY = '''"""Corpus stub admit.py for the MRK-1 regression corpus (endpoint-locked; never executed)."""
import argparse
import os
import sys


def main(argv=None):
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--testbed', action='store_true')
    mode.add_argument('--technical', action='store_true')
    ap.add_argument('--only-world')
    ap.add_argument('--only-arm')
    args = ap.parse_args(argv)
    if not os.environ.get('MINIFLY_ATTEMPT_ID'):
        sys.stderr.write('admit.py: run through MFC launch\\n')
        return 2
    sys.stdout.write('testbed\\n' if args.testbed else 'technical\\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
'''


def manifest_rows(pool):
    return [{'item_id': i, 'prompt_hex': hexs(BI.prompt(a, b)), 'expected_hex': hexs(BI.truth(a, b))}
            for (i, a, b) in BI.pool_items()[pool]]


def taught_rows(world, arm):
    rows = []
    for (a, b) in BI.TAUGHT_PAIRS:
        if arm == 'SHUF':
            ans, fb = BI.shuf_label(a, b), True
        elif arm == 'NPT':
            ans, fb = BI.truth(a, b), False   # matched input, teaching (feedback) withheld
        else:
            ans, fb = BI.truth(a, b), True
        rows.append({'world': world, 'arm': arm, 'prompt_hex': hexs(BI.prompt(a, b)),
                     'answer_hex': hexs(ans), 'phase': 'teach', 'feedback_bearing': fb,
                     'presentations': 10})
    return rows


def taught_files(prefix):
    files = {}
    for w in BI.TECH_WORLDS:
        arms = BI.ARMS if w == BI.NATIVE_TECH_WORLD else ['W']
        for arm in arms:
            files['%smanifests/TAUGHT/%s/%s.jsonl' % (prefix, w, arm)] = dump_jsonl(taught_rows(w, arm))
    for w in BI.DEV_WORLDS + BI.REH_WORLDS + BI.OFF_WORLDS:
        for arm in BI.ARMS:
            files['%smanifests/TAUGHT/%s/%s.jsonl' % (prefix, w, arm)] = dump_jsonl(taught_rows(w, arm))
    return files


def split_shas(files, prefix):
    """held_set_sha256 = sha256(HELD.jsonl); split_sha256 = sha256(canonical_json({held_manifest,
    old_manifest, new_manifest: file sha256})). TAUGHT streams do not exist yet at the lock."""
    held = sha256_bytes(files[prefix + 'manifests/HELD.jsonl'])
    split = sha256_json({'held_manifest': held,
                         'old_manifest': sha256_bytes(files[prefix + 'manifests/OLD.jsonl']),
                         'new_manifest': sha256_bytes(files[prefix + 'manifests/NEW.jsonl'])})
    return split, held


def arms_block():
    return [
        {'id': 'W', 'role': 'candidate', 'zero_channels': [], 'declared_changed': [],
         'teaching_withheld': [], 'min_realized_dose': None},
        {'id': 'NPT', 'role': 'no_prior_teaching', 'zero_channels': ['pair_memory'],
         'declared_changed': [], 'teaching_withheld': ['pair_teaching'], 'min_realized_dose': 0.05},
        {'id': 'SHUF', 'role': 'shuf', 'zero_channels': [], 'declared_changed': [],
         'teaching_withheld': [], 'min_realized_dose': None},
        {'id': 'FROZEN', 'role': 'frozen', 'zero_channels': ['pair_memory', 'readout'],
         'declared_changed': [], 'teaching_withheld': [], 'min_realized_dose': 0.05},
        {'id': 'ORACLE', 'role': 'oracle_plumbing', 'zero_channels': [],
         'declared_changed': ['readout'], 'teaching_withheld': [], 'min_realized_dose': None},
    ]


def endpoint_lock(constraints_sha, admit_sha, split_sha, held_sha, delivery=D):
    lock = {
        'schema': 'minifly.endpoint_lock.v1', 'delivery': delivery, 'protocol_id': delivery + '-p1',
        'created_utc': '20261101T085000Z', 'author_agent': 'codex', 'tier_declared': 2,
        'intent': 'diagnostic', 'claim_level_target': 'E3',
        'task': {'id': 'add1', 'family': 'addition_single_digit',
                 'format': {'kind': 'addition', 'sep_hex': '2b', 'terminator_hex': '3d',
                            'eos_hex': '0a', 'gen_cap_bytes': BI.GEN_CAP,
                            'normalize_leading_zeros': True}},
        'objective_distance': {'task_family': 'addition_single_digit', 'output': 'own_bytes_own_eos',
                               'held': 'unseen_composition', 'native_only_path': True,
                               'feedback': 'gradual_warmup', 'old_protection_measured': True},
        'units': {'witness': 'pair_rectangle', 'commutative': True},
        'split': {'taught_manifest': 'manifests/TAUGHT', 'held_manifest': 'manifests/HELD.jsonl',
                  'old_manifest': 'manifests/OLD.jsonl', 'new_manifest': 'manifests/NEW.jsonl',
                  'split_sha256': split_sha, 'held_set_sha256': held_sha},
        'exposure_streams': [{'path_glob': 'manifests/TAUGHT/*/*.jsonl', 'feedback_bearing': True,
                              'phase': 'teach'}],
        'arms': arms_block(), 'causal_control_arm': 'NPT', 'positive_control_arm': 'ORACLE',
        'ports': [{'id': 'pair_write', 'signs': ['+', '-']}],
        'primary': {'task_id': 'add1', 'metric': 'exact_own_eos', 'unconditional': True,
                    'pool': 'HELD', 'checkpoint': 'final', 'strata_counted': ['S2', 'S3'],
                    'candidate_arm': 'W', 'comparator_arms': ['NPT'], 'ladder': 'auto',
                    'chance': 0.05, 'margin_abs': 0.05, 'absolute_floor': 0.25,
                    'per_world_floor': -0.05, 'aggregator': {'type': 'paired_lcb', 'alpha': 0.05},
                    'min_worlds': 8, 'n_items_per_pool': {'TAUGHT': 15, 'OLD': 8, 'NEW': 7, 'HELD': 16},
                    'rule_type': 'declarative'},
        'gates': [{'id': 'stop_gate', 'description': 'EOS is emitted only after a readout commit'}],
        'guards': [{'id': 'old_drop', 'kind': 'old_drop', 'pool': 'OLD', 'max_drop': 0.1},
                   {'id': 'old_acq', 'kind': 'old_acq', 'pool': 'OLD', 'margin': 0.05}],
        'udi': None, 'checkpoints': {'prefix': 40, 'old_end': 80, 'final': 160},
        'stage_chain': False,
        'worlds': {'technical': list(BI.TECH_WORLDS), 'dev': list(BI.DEV_WORLDS),
                   'rehearsal': list(BI.REH_WORLDS),
                   'official': {'mode': 'listed', 'ids': list(BI.OFF_WORLDS), 'n': 8},
                   'official_role_intended': 'MAIN'},
        'schedule': {'paired_atomic': True}, 'budget': {'max_records': 20000, 'max_cpu_h': 4},
        'admission_thresholds': {'analytic_input_snr': 1.7, 'input_contrast_min': 2.0,
                                 'prompt_id_margin_min': 0.1, 'taught_acq_margin_min': 0.05,
                                 'shuf_band': [0.35, 0.65]},
        'testbed': {'ceiling_grid': copy.deepcopy(CONSTRAINTS['testbed_ceiling_grid']),
                    'floor_grid': copy.deepcopy(CONSTRAINTS['testbed_floor_grid'])},
        'files': {'admit': 'admit.py', 'evaluator': [], 'generators': [], 'runner': ['run.py'],
                  'native': ['learner.py'], 'shared': [], 'assets': [], 'params': 'params.json'},
        'endpoint_files_sha256': {'admit.py': admit_sha},
        'lineage': {'parent': None, 'changes_declared': [], 'priors_added': []},
        'predictions': [
            {'id': 'P1', 'target': 'primary', 'point': 0.8, 'interval90': [0.6, 0.95], 'p_pass': 0.7,
             'basis_findings': ['none'],
             'base_rate_forecast': {'value': 0.3, 'source': 'DELIVERY_LEDGER Laplace (no v1 rows)'}},
            {'id': 'P2', 'target': 'guard:old_drop', 'point': 0.03, 'interval90': [0.0, 0.08],
             'p_pass': 0.85, 'basis_findings': ['none'],
             'base_rate_forecast': {'value': 0.7, 'source': 'DELIVERY_LEDGER Laplace (no v1 rows)'}},
            {'id': 'P3', 'target': 'guard:old_acq', 'point': 0.45, 'interval90': [0.3, 0.5],
             'p_pass': 0.9, 'basis_findings': ['none'],
             'base_rate_forecast': {'value': 0.7, 'source': 'DELIVERY_LEDGER Laplace (no v1 rows)'}},
        ],
        'e3_cadence': {'waiver_user_decision': None}, 'constraints_sha256': constraints_sha,
        'user_decisions': [], 'coprimary': [],
    }
    return lock


def authorization(official=True, delivery=D):
    return {
        'schema': 'minifly.authorization.v1',
        'verbatim': ('Go ahead with %s: implementation, technical, dev cycles and one official MAIN '
                     'run, up to 20000 records, 4 CPU-hours and 3 cycles. Push to your own branch.'
                     % delivery),
        'utc': '20261101T084000Z', 'session': 'owner-chat-20261101', 'channel': 'chat',
        'options_shown': [
            {'label': 'A: open held E3 on single-digit addition',
             'text': 'Pair-rectangle units, 8 MAIN worlds, NPT and SHUF controls.',
             'cost': 'about 1 CPU-h', 'objective_distance': 'held unseen_composition'},
            {'label': 'B: stay at component level',
             'text': 'TAUGHT-pool readout only.', 'cost': 'about 0.3 CPU-h',
             'objective_distance': 'held taught_only'}],
        'recommendation_shown': 'A, because it opens held E3 at low cost.',
        'stages': {'implementation': True, 'technical': True, 'dev_cycles': True,
                   'official': bool(official), 'push': True, 'push_lock': True, 'adoption': False,
                   'pointer_update': False},
        'caps': {'max_records': 20000, 'max_cpu_h': 4, 'max_cycles': 3},
        'defaults_applied': False, 'conditionals': [], 'recorded_by': 'codex',
    }


def params_bytes(params=None):
    return dump_json(PARAMS if params is None else params)


__all__ = ['D', 'KB', 'CODEX', 'CLAUDE', 'canonical_json']
