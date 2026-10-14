"""Static design consistency only: stdlib, no learner imports or execution."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    d = json.loads((ROOT / 'DESIGN.json').read_text())
    checks = []

    def check(name, value):
        if not value:
            raise AssertionError(name)
        checks.append(name)

    check('design only; no implementation, training or launch claim',
          not d['implemented'] and not d['science_launch_authorized']
          and not d['github_upload_authorized']
          and d['technical_cycles_executed'] == d['scientific_jobs_executed']
          == d['training_records_executed'] == d['model_forward_passes_executed'] == 0)
    check('all explicit prohibitions enabled', all(d['prohibitions'].values()))
    check('only specified two non-ELM non-attention cells',
          set(d['models']) == {'KEEP', 'G62'}
          and d['models']['G62']['cell'] == 'standard_LSTM_no_peepholes'
          and not d['models']['G62']['attention']
          and not d['models']['G62']['ELM_source_reuse_allowed']
          and not d['models']['KEEP']['runtime_ELM_imports_allowed'])
    parameters = {
        'KEEP': 259 * 24 + 3 * 64 * (24 + 64 + 1) + 64 * 64 + 65 * 257 + 65,
        'G62': 259 * 24 + 4 * 62 * (24 + 62 + 1) + 63 * 257 + 63,
    }
    for name, count in parameters.items():
        check(name + ' exact parameter count', d['models'][name]['parameters'] == count)
        check(name + ' parameter and RMS bytes',
              d['models'][name]['parameter_and_RMS_bytes'] == 2 * 8 * count)
    check('new core not larger by parameter or inference state count',
          parameters['G62'] < parameters['KEEP']
          and d['models']['G62']['inference_state_scalars'] == 2 * 62
          and d['models']['KEEP']['inference_state_scalars'] == 2 * 64)
    check('all byte payload values and own EOS, no hidden output masking',
          d['byte_IO']['payload_values'] == 256
          and d['byte_IO']['input_embedding_rows'] == 259
          and d['byte_IO']['output_classes'] == 257
          and {d['byte_IO'][k] for k in ('EOS', 'BEGIN', 'REPLY')} == {256, 257, 258}
          and d['byte_IO']['max_output_payload_bytes'] + 1 == d['byte_IO']['termination_attempts'])
    check('exact two by three roster and one fixed candidate',
          d['configurations'] == [m + '_' + k for m in ('KEEP', 'G62')
                                  for k in ('CE_ONLY', 'OLD', 'SAFE')]
          and d['selectable'] == ['G62_SAFE'])
    check('pure answer-only reference really has no PG', not d['objective']['CE_ONLY_PG'])
    check('OLD and SAFE preserve objective and critic isolation',
          not d['objective']['WARM_PG'] and d['objective']['CE_coefficient'] == 1
          and d['objective']['PG_coefficient'] == 1
          and not d['optimizer']['actor_critic_shared_clipping'])
    check('bounded declared SAFE computation and no hidden reward queries',
          d['safe']['candidate_scales'] == [2. ** -k for k in range(8)]
          and d['safe']['max_candidate_evaluations'] == 8
          and d['safe']['KL_limit_nats'] == .01
          and d['safe']['new_reward_queries'] == d['safe']['hidden_answer_queries'] == 0
          and d['safe']['all_internal_recomputation_charged'])
    check('no false exact trajectory KL or global monotonicity guarantee',
          d['safe']['not_expected_trajectory_KL']
          and not d['safe']['global_monotonicity_guarantee'])
    course = d['course']
    warm = course['WARM']['binary'] + course['WARM']['mixed']
    half, sparse, scalar = (course[k]['records'] for k in ('HALF', 'SPARSE', 'SCALAR'))
    check('warm and main history totals', warm == course['WARM']['records'] == 20480
          and warm + half + sparse + scalar == course['main_history_records'] == 28672)
    check('answer visibility and identical late difficulty mix',
          [course[k]['answer_probability'] for k in ('WARM', 'HALF', 'SPARSE', 'SCALAR')]
          == [1, .5, .125, 0] and course['tail_binary_fraction'] == .5
          and not course['teacher_trace_or_numeric_auxiliary_data'])
    check('warm interleave divisibility',
          course['WARM']['binary'] - 3 * course['WARM']['mixed']
          == course['first_binary_only_records'] == 4096)
    check('separate same-start old and safe freezes',
          d['frozen_branches'] == {'FREEZE_OLD': 'OLD after SPARSE',
                                  'FREEZE_SAFE': 'SAFE after SPARSE'})
    check('seven unique related histories and sixteen state archives',
          len(set(d['histories'])) == 7
          and d['counts']['related_terminal_histories_per_job'] == 7
          and d['counts']['full_state_archives_per_job'] == 1 + 3 + 5 + 7 == 16)
    job = 3 * warm + 5 * (half + sparse) + 7 * scalar
    check('shared prefix native execution count',
          job == d['counts']['world_core_job_external_applications'] == 110592
          and d['counts']['world_external_applications'] == 2 * job)
    all_worlds = []
    for stage in ('DEV', 'CONFIRM'):
        n = d['counts'][stage + '_worlds']
        check(stage + ' roster and external application count',
              len(d[stage]['worlds']) == len(set(d[stage]['worlds'])) == n
              and d['counts'][stage + '_jobs'] == 2 * n
              and d['counts'][stage + '_external_applications'] == 2 * n * job
              and d['counts'][stage + '_related_histories'] == 2 * n * 7)
        all_worlds.extend(d[stage]['worlds'])
    for stage in ('cycle2', 'cycle3'):
        s = d['technical'][stage]
        sw = s['binary'] + s['mixed']
        sh, sp, sc = s['stages']
        expected = (3 * sw + 5 * (sh + sp) + 7 * sc) * 2 * len(s['worlds'])
        check(stage + ' separate native technical count',
              s['jobs'] == 2 * len(s['worlds']) and s['external_applications'] == expected)
        all_worlds.extend(s['worlds'])
    check('no overlap between DEV, confirmation and technical rosters',
          len(all_worlds) == len(set(all_worlds)))
    check('conditional maximum science count, not mandatory workload',
          d['counts']['conditional_max_jobs'] == 16 + 64 == 80
          and d['counts']['conditional_max_external_applications'] == 80 * job == 8847360
          and d['counts']['conditional_max_related_histories'] == 80 * 7 == 560)
    comparisons = d['required_contrasts']
    check('twenty fixed necessary contrasts, no candidate substitution',
          len(comparisons) == len({x['id'] for x in comparisons}) == 4 + 12 + 3 + 1 == 20
          and all(x['candidate'] == 'G62_SAFE' for x in comparisons)
          and not d['CONFIRM']['candidate_replacement'] and not d['CONFIRM']['extend_n'])
    check('no falsely qualified E4 or idle time retention',
          not d['scope']['E4'] and not d['scope']['idle_time_decay_tested']
          and d['scope']['not_a_four_operation_mastery_certificate'])
    check('same frozen data algorithm, all sixteen mixed operation pairs',
          not d['data']['algorithm_changed'] and d['data']['mixed_ordered_operation_pairs'] == 16
          and d['data']['mixed_held_groups'] == 16 * 16
          and not d['data']['semantic_fields_may_enter_learner'])
    lineage = json.loads((ROOT / 'SOURCE_LINEAGE.json').read_text())
    check('all lineage sources exist; this is not a runtime source lock',
          all((ROOT.parent / p).is_file() for p in lineage['inputs'])
          and lineage['no_native_implementation_claim'])
    links = json.loads((ROOT / 'MECHANISM_LINKS.json').read_text())
    expected_views = {'historical108': 108, 'catalogue136': 136, 'versions254': 254,
                      'subset71': 71, 'first16': 16}
    for view, count in expected_views.items():
        data = json.loads((ROOT / links['full_views'][view]).read_text())
        rows = data['rows']
        check('full inventory view retained: ' + view, len(rows) == count)
    check('no family-wide exclusions invented by design',
          links['views_overlap_not_additive'] and links['not_a_replacement_for_full_inventory']
          and not links['new_family_exclusions'])
    check('bounded two-worker operations and no automatic restart',
          d['execution']['detached_workers'] == 2
          and d['execution']['admission_stop_hours'] == 8
          and d['execution']['hard_session_hours'] == 10
          and d['execution']['caffeinate_attached_to_supervisor']
          and not d['execution']['automatic_resume_or_retry']
          and not d['execution']['repeat_native_trajectories'])
    result = dict(accepted=True, scope='Static document/manifest/formula/source-reference consistency only; not architecture implementation, numerical tests, three-cycle qualification or learning evidence',
                  checks=checks, checks_count=len(checks), model_forward_passes=0,
                  native_training_records=0, scientific_jobs=0, three_cycles_completed=False,
                  source_lock_created=False, launch_authorized=False)
    (ROOT / 'STATIC_VALIDATION.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'checks'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
