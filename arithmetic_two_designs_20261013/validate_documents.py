"""Static document/manifest/count audit only. No learner or task-runner imports."""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
A_DIR = ROOT / 'arithmetic_feedback_rule_discovery_design_v2_20261013'
B_DIR = ROOT / 'arithmetic_stream_transfer_design_v1_20261013'
A = json.loads((A_DIR / 'DESIGN.json').read_text())
B = json.loads((B_DIR / 'DESIGN.json').read_text())
checks = {}

def check(name, condition):
    checks[name] = bool(condition)

check('A_preserved_design_only', not A['implemented'] and A['scientific_jobs_executed'] == 0
      and A['technical_cycles_executed'] == 0 and not A['science_authorized'])
check('A_roster_unchanged', len(A['configs']) == 8
      and [c['id'] for c in A['configs']] ==
      ['RAW-RLS', 'G-SQ', 'G-CE', 'L-SQ', 'L-CE', 'G-FROZEN-CE', 'L-FROZEN-CE', 'LEGACY-B'])
check('A_counts_unchanged', A['counts']['conditional_max_jobs'] == 352
      and A['counts']['conditional_max_records'] == 743424)
check('B_design_only', not B['implemented'] and not B['science_authorized']
      and not B['github_upload_authorized'] and B['scientific_jobs_executed'] == 0
      and B['technical_cycles_executed'] == 0)
check('B_distinct_scope', not B['four_operation_completion_claim']
      and not B['mixed_expression_completion_claim'] and B['engineered_backprop_reference'])
configs = B['configurations']
check('B_fixed_factorial', len(configs) == 4
      and {c['id'] for c in configs} == {'ALL-0', 'ALL-D', 'ANSWER-0', 'ANSWER-D'}
      and sum(c['selectable'] for c in configs) == 2
      and {c['decay'] for c in configs} == {0, .01})
check('B_generic_IO', B['learner']['embedding'][0] == 256 + 3
      and B['learner']['head'][1] == 256 + 1
      and not B['learner']['task_ID_or_arithmetic_features']
      and not B['learner']['arithmetic_output_mask']
      and B['learner']['own_EOS'] and B['learner']['learning_state_pure_generation'])
encoder_params = 259 * 16 + 3 * (64 * 16 + 64 * 64 + 64)
all_params = encoder_params + 65 * 257
check('B_parameter_ledger', all_params == B['learner']['model_parameters'] == 36401
      and all_params * 2 * 8 == B['learner']['model_and_RMS_bytes_float64'])
check('B_same_core_and_no_replay', B['learner']['same_code_settings_across_tasks']
      and not B['learner']['offline_replay_cache']
      and not B['learner']['task_boundary_parameter_optimizer_reset']
      and not B['learner']['generated_target_learning'])
ar = B['arithmetic']
check('B_orbit_accounting', 10 * 11 // 2 == ar['unordered_orbits_per_task']
      and 8 * 9 // 2 == ar['eligible_held_nontrivial_orbits_per_task']
      and ar['train_orbits_per_task'] + ar['held_orbits_per_task'] == 55
      and ar['held_orbits_per_task'] <= 36)
check('B_exposure_checkpoints', [p * ar['records_per_pass'] for p in ar['checkpoints_passes']]
      == ar['checkpoints_external_records']
      and ar['records_per_pass'] == ar['train_orbits_per_task'] * 2)
co = B['cohort']
nonzero_checkpoints = len(ar['checkpoints_passes']) - 1
check('B_correlated_forks', co['addition_forks_per_job'] == 3 * nonzero_checkpoints
      and co['multiplication_forks_per_job'] == 5 * nonzero_checkpoints
      and co['forks_per_job'] == 40 and not co['count_as_independent_lives'])
ct = B['counts']
task_records = ar['records_per_pass'] * ar['checkpoints_passes'][-1]
check('B_prefix_and_branch_records', ct['counting_records_per_job'] == 128
      and ct['addition_records_per_job'] == 3 * task_records
      and ct['multiplication_records_per_job'] == 5 * task_records
      and ct['cohort_records_per_job'] == co['forks_per_job'] * co['records_per_fork'])
per_job = 128 + 3 * task_records + 5 * task_records + 40 * 32
check('B_per_job_records', per_job == ct['external_record_applications_per_job'] == 144768)
check('B_per_history_vs_executed_ledger', ct['final_W_history_records_before_cohort']
      == 128 + 2 * task_records == 35968
      and ct['final_W_retained_history_records'] == 35968 + 32 == 36000)
dev = B['selection']
conf = B['confirmation']
check('B_stage_counts', dev['dev_jobs'] == dev['dev_worlds'] * dev['dev_configurations'] == 8
      and conf['jobs'] == conf['fresh_worlds'] * conf['configurations'] == 48)
check('B_max_counts', ct['dev_records'] == 8 * per_job
      and ct['conditional_confirm_records'] == 48 * per_job
      and ct['conditional_max_jobs'] == 8 + 48
      and ct['conditional_max_records'] == 56 * per_job == 8107008
      and ct['conditional_max_independent_worlds'] == 2 + 16)
check('B_primary_family', sum(conf['groups'].values()) == conf['primary_tests'] == 12
      and conf['holm_alpha'] == .05 and conf['all_primary_required'])
check('B_resource_admission_only', not B['operations']['authorized']
      and not B['operations']['namespaces_admitted']
      and B['operations']['full_budget_cost_measurement_required']
      and B['operations']['admission_hours'] == 8
      and B['operations']['hard_session_hours'] == 10
      and not B['operations']['auto_retry_restart_resume'])

inv_path = ROOT / B['full_inventory_views']['complete_review']
inv = json.loads(inv_path.read_text())
expected = {'historical108': 108, 'catalogue136': 136,
            'versions254': 254, 'subset71': 71, 'first16': 16}
observed = {key: len(inv['views'][key]['rows']) for key in expected}
check('full_inventory_rows_preserved', expected == observed
      and B['full_inventory_views']['overlapping_not_additive'])

links = []
for document in [HERE / 'README.md', B_DIR / 'EXPERIMENT.md', B_DIR / 'REFERENCE_REVIEW.md']:
    content = document.read_text()
    for target in re.findall(r'\[[^\]]+\]\(([^)]+)\)', content):
        if target.startswith(('https://', 'http://')):
            continue
        path = (document.parent / target.split('#')[0]).resolve()
        links.append({'document': str(document.relative_to(ROOT)),
                      'target': target, 'exists': path.exists()})
check('all_local_document_links', all(link['exists'] for link in links))
check('B_goal_scope_caveats', all(phrase in (B_DIR / 'EXPERIMENT.md').read_text() for phrase in
      ['no task-specific arithmetic structure', 'A marginal multiplication benefit',
       'not a “minutes” promise', "counting's isolated contribution is not identified",
       'fresh worlds whose split is fixed before any of the four operations is taught']))

result = {'schema': 'ARITHMETIC_TWO_DESIGNS_STATIC_REVIEW',
          'scope': 'document links, manifests and count algebra; no learning/science or three-cycle qualification',
          'passed': all(checks.values()), 'checks': checks,
          'inventory_rows': observed, 'local_links': links,
          'A_max_jobs': 352, 'A_max_records': 743424,
          'B_max_jobs': 56, 'B_max_records': 8107008,
          'trajectories_executed': 0, 'qualification_completed': False}
(HERE / 'STATIC_REVIEW.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'passed': result['passed'], 'checks': checks,
                  'trajectories_executed': 0}, indent=2))
raise SystemExit(0 if result['passed'] else 1)
