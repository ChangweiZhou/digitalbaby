#!/usr/bin/env python3
"""Independent, read-only audit of saved V43 science or engineering smoke.

No learner is imported or executed. This script checks identities and saved
receipts, independently replays virtual ledger arithmetic, and delegates only
saved probability/control arithmetic to the adjacent audit-only helper.
"""
from __future__ import annotations

import argparse
import ast
import base64
import collections
import hashlib
import json
import math
from pathlib import Path
import sys
import traceback
import zlib

import numpy as np

from audit_offline_controls import audit_events

ARMS = ('SUPER', 'READ-HYBRID', 'SILENT-HARD', 'ANCHOR-ONLY', 'FIXED-50',
        'GLOBAL-BAYES', 'LOCAL-BAYES-0', 'LOCAL-BAYES-E2')
CONTROLS = ('SHUFFLED-ALPHA', 'EVENT-ORACLE', 'SLOT-ORACLE')
LN2 = math.log(2.)


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def sha(obj):
    return hashlib.sha256(obj if isinstance(obj, bytes) else canonical(obj)).hexdigest()


class Checks:
    def __init__(self):
        self.counts = collections.Counter()
        self.failed_counts = collections.Counter()
        self.errors = []

    def check(self, name, condition, detail=None):
        self.counts[name] += 1
        if not condition:
            self.failed_counts[name] += 1
            if len(self.errors) < 100:
                self.errors.append({'check': name, 'detail': detail})
        return bool(condition)

    def close(self, name, a, b, detail=None, absolute=1e-8, relative=2e-10):
        valid = a is not None and b is not None
        if valid:
            a, b = float(a), float(b)
            valid = math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, abs_tol=absolute, rel_tol=relative)
        return self.check(name, valid, detail if detail is not None else {'actual': a, 'expected': b})

    def report(self):
        return {'passed': not self.failed_counts, 'checks': dict(self.counts),
                'failed_checks': dict(self.failed_counts), 'errors': self.errors}


def logistic_bits(e):
    x = float(e)*LN2
    if x >= 0:
        z = math.exp(-x); return 1/(1+z)
    z = math.exp(x); return z/(1+z)


def _optional(value):
    if isinstance(value, np.generic): value = value.item()
    if isinstance(value, float) and not math.isfinite(value): return None
    return value


def ledger_audit(result, cols):
    """Replay the observer ledger from saved births and every saved G outcome.

    Birth E2 likelihood ratios themselves are inputs to this reconstruction:
    probationary candidate probabilities on unlogged route F are unavailable.
    """
    c = Checks(); resident = {}; born = set(); prior_birth = -1; global_e = 0.
    life = result['ledger_lifecycle']; cursor = 0
    counts = collections.Counter(); ages = {}; event_indices = [int(x) for x in cols['event_index']]
    event_lookup = {t: i for i, t in enumerate(event_indices)}
    c.check('lifecycle_time_order', all(a['time'] <= b['time'] for a, b in zip(life, life[1:])))
    c.check('authority_event_time_order', all(a < b for a, b in zip(event_indices, event_indices[1:])))

    def transitions(through):
        nonlocal cursor, prior_birth
        while cursor < len(life) and int(life[cursor]['time']) <= through:
            row = life[cursor]; cursor += 1
            uid, now, kind = int(row['uid']), int(row['time']), row['event']
            counts[kind] += 1
            c.check('lifecycle_within_stream', 1 <= now <= result['byte_events'], row)
            if kind == 'birth':
                c.check('generation_never_reused', uid not in born and uid > prior_birth, row)
                c.check('birth_seed_finite', math.isfinite(float(row['E2_seed'])), row)
                prior_birth = max(prior_birth, uid); born.add(uid)
                resident[uid] = {'E0': 0., 'E2': float(row['E2_seed']), 'uses': 0,
                    'birth': now, 'last_revision': now, 'was_revised': False}
                ages[uid] = now
            elif kind == 'eviction':
                c.check('eviction_has_resident_generation', uid in resident, row)
                resident.pop(uid, None)
            elif kind == 'revision_reset':
                c.check('revision_has_resident_generation', uid in resident, row)
                index = event_lookup.get(now-1)
                c.check('revision_has_exact_pre_event_record', index is not None, row)
                if index is not None:
                    c.check('revision_on_exact_mature_anchor', str(cols['route'][index]) == 'A' and
                            int(cols['slot_uid'][index]) == uid, row)
                    c.check('revision_old_top_agrees', int(cols['slow_top1'][index]) == int(row['old_top']), row)
                    c.check('revision_committed_to_observed_byte', int(row['old_top']) != int(row['new_top']) and
                            int(row['new_top']) == int(cols['true_byte'][index]), row)
                if uid in resident:
                    resident[uid].update(E0=0., E2=0., last_revision=now, was_revised=True)
            else: c.check('known_lifecycle_kind', False, row)

    for i, now in enumerate(event_indices):
        transitions(now)
        route = str(cols['route'][i]); uid = int(cols['slot_uid'][i])
        c.check('logged_authority_route', route in ('A', 'T', 'G'), {'event': now, 'route': route})
        c.check('generation_alias_agrees', uid == int(cols['generation_uid'][i]), now)
        c.close('global_pre_event_evidence', cols['global_E_before'][i], global_e, now)
        c.close('global_pre_event_weight', cols['global_w_before'][i], logistic_bits(global_e), now)
        if route in ('A', 'G'):
            if not c.check('mature_read_has_live_ledger', uid in resident, {'event': now, 'uid': uid}): continue
            ledger = resident[uid]
            c.close('local_E2_is_prior_only', cols['local_E_before'][i], ledger['E2'], now)
            c.close('local_E0_is_prior_only', cols['local_0_E_before'][i], ledger['E0'], now)
            c.close('local_E2_weight_reconstructs', cols['local_w_before'][i], logistic_bits(ledger['E2']), now)
            c.close('local_E0_weight_reconstructs', cols['local_0_w_before'][i], logistic_bits(ledger['E0']), now)
            c.check('prior_G_use_count', int(cols['prior_similarity_use_count'][i]) == ledger['uses'], now)
            c.check('causal_slot_age', int(cols['slot_age'][i]) == now-ledger['birth'], now)
            c.check('revision_flag_causal', bool(cols['was_revised'][i]) == ledger['was_revised'], now)
            recorded_age = _optional(cols['revision_age'][i])
            # Integer NPZ columns use -1 for missing; float columns use NaN.
            if recorded_age == -1: recorded_age = None
            expected_age = now-ledger['last_revision'] if ledger['was_revised'] else None
            c.check('revision_age_causal', recorded_age == expected_age, now)
        else:
            c.close('trace_has_no_local_E2', cols['local_E_before'][i], 0., now)
            c.close('trace_has_no_local_E0', cols['local_0_E_before'][i], 0., now)
            c.check('trace_has_no_mature_uses', int(cols['prior_similarity_use_count'][i]) == 0, now)
        if route == 'G':
            delta = (float(cols['logpS_true'][i])-float(cols['logpF_true'][i]))/LN2
            ledger = resident[uid]
            ledger['E0'] += delta; ledger['E2'] += delta; ledger['uses'] += 1
            global_e += delta; counts['G_update'] += 1
        # No A or T outcome changes either local or global G evidence.
    transitions(int(result['byte_events']))
    c.check('all_lifecycle_records_consumed', cursor == len(life))
    terminal = {int(x['slot_uid']): x for x in result['terminal_evidence']}
    c.check('terminal_generations_exactly_resident', set(terminal) == set(resident))
    for uid in set(terminal) & set(resident):
        for key in ('E0', 'E2'):
            c.close('terminal_'+key, terminal[uid][key], resident[uid][key], uid)
        for key in ('uses', 'birth', 'last_revision', 'was_revised'):
            c.check('terminal_'+key, terminal[uid][key] == resident[uid][key], uid)
    a = result['audits']; pe = result['physical_events']
    for lifecycle, field in (('birth', 'ledger_births'), ('eviction', 'ledger_evictions'), ('revision_reset', 'revision_resets')):
        c.check('lifecycle_count_'+lifecycle, counts[lifecycle] == a[field])
    c.check('births_equal_native_promotions', counts['birth'] == pe.get('promotion', 0))
    c.check('births_equal_pre_E2_capture_receipt', counts['birth'] == a['prepromotion_captures'])
    c.check('evictions_equal_native_slot_events', counts['eviction'] == sum(v for k, v in pe.items() if k.startswith('slot_eviction:')))
    c.check('G_update_count_exact', counts['G_update'] == a['route_counts']['G'])
    c.check('terminal_ledger_occupancy', len(resident) == result['costs']['slow']['slots'] == result['costs']['virtual_ledger_count'])
    if result['checkpoint_memory']:
        c.close('terminal_global_evidence', result['checkpoint_memory'][-1]['global_E'], global_e)
    return {**c.report(), 'counts': dict(counts), 'terminal_generations': len(resident),
        'limitation': 'E2 birth seeds are checked for causal placement and subsequent use, but their pre-promotion '
            'trace-vs-fast probabilities on route F were not logged and are not independently reconstructed. '
            'Revision commits are cross-checked against saved exact-event top1/outcome metadata; full post-write '
            'value vectors are covered by engineering replay, not reconstructed from this event table.'}


def source_audit(plan, source):
    c = Checks(); data = source.read_bytes()
    c.check('source_sha256', sha(data) == plan['source_sha256'], str(source))
    constants = {}
    for node in ast.parse(data).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ('PAYLOADS', 'PROTOCOL', 'ORIGINAL_PROTOCOL', 'VERSION'):
                constants[name] = ast.literal_eval(node.value)
    c.check('embedded_protocol_sha256', sha(constants['PROTOCOL'].encode()) == plan['protocol_sha256'])
    payloads = constants['PAYLOADS']
    c.check('embedded_payload_roster', set(payloads) == set(plan['payload_hashes']))
    for name, payload in payloads.items():
        raw = zlib.decompress(base64.b85decode(payload['data']))
        c.check('embedded_payload_'+name, sha(raw) == payload['sha256'] == plan['payload_hashes'][name])
    c.check('source_version', constants['VERSION'] == plan['version'])
    return {**c.report(), 'source_file': str(source), 'source_sha256': sha(data),
            'source_was_executed': False, 'payloads_verified': len(payloads)}


def engineering_audit(root, plan, smoke):
    c = Checks(); path = root/'ENGINEERING.json'
    if not path.exists(): return {'passed': False, 'missing': True, 'errors': ['ENGINEERING.json is absent']}
    env = json.loads(path.read_text())
    c.check('engineering_source_sha256', env.get('source_sha256') == plan['source_sha256'])
    c.check('engineering_pass_receipt', env.get('passed') is True)
    if smoke:
        c.check('engineering_smoke_explicit', env.get('mode') == 'smoke_only')
        return {**c.report(), 'scope': 'Engineering smoke; no full E0 qualification is claimed.'}
    expected = {(seed, load) for seed in (943001, 943002) for load in (64, 256)}
    identity = {'source_sha256': plan['source_sha256'], 'protocol_sha256': plan['protocol_sha256']}
    c.check('engineering_envelope_hash', env.get('sha256') == sha({k: v for k, v in env.items() if k != 'sha256'}))
    c.check('engineering_identity', env.get('identity') == identity)
    runs = env.get('runs', [])
    c.check('engineering_four_cell_roster', len(runs) == 4 and {(r['seed'], r['load']) for r in runs} == expected)
    c.check('E0_through_E8_receipts', set(env.get('checks', {})) == {f'E{i}' for i in range(9)} and all(env['checks'].values()))
    for r in runs:
        load = r['load']; records = load*8+9*max(1, (load//4)//4)
        c.check('engineering_full_length', r['passed'] and r['full_length'] and r['records'] == records, [r['seed'], load])
        c.check('engineering_all_byte_checks', r['byte_checks'] == 14*records == r['firewall_checks'])
        c.check('engineering_full_hash_every_record', r['full_hash_recomputations'] == records)
        c.check('engineering_terminal_context_count', r['terminal_probe_checks'] == 3*load)
        receipt_path = root/'engineering'/f"{r['seed']}_{load}.json"
        if c.check('engineering_individual_receipt_exists', receipt_path.exists(), str(receipt_path)):
            receipt = json.loads(receipt_path.read_text())
            c.check('engineering_individual_identity', receipt['identity'] == identity)
            c.check('engineering_individual_hash', receipt['sha256'] == sha(receipt['data']))
            c.check('engineering_individual_matches_master', receipt['data'] == r)
    return {**c.report(), 'scope': 'Source-bound saved physical reproduction and E0–E8 receipts; no fresh learner run.'}


def structural_audit(result, job, cols):
    c = Checks(); load = int(job['load']); repeats = int(job['config']['repeats'])
    nrev = max(1, (load//4)//4); records = load*repeats+9*nrev; nbytes = 14*records
    c.check('one_physical_trajectory', result['physical_trajectories'] == 1)
    c.check('cell_identity', result['seed'] == job['seed'] and result['load'] == load)
    c.check('record_and_byte_counts', result['records'] == records and result['target_presentations'] == records and result['byte_events'] == nbytes)
    c.check('unique_example_counts', result['unique_target_contexts'] == 2*load and result['unique_complete_records'] == 2*load+2*nrev)
    c.check('one_base_and_slow_update_per_byte', result['physical_base_updates'] == nbytes == result['physical_slow_updates'])
    c.check('single_fast_update_path', 0 < result['physical_fast_updates'] <= nbytes)
    a = result['audits']; routes = a['route_counts']
    c.check('exclusive_route_counts', set(routes) == set('ATGF') and sum(routes.values()) == nbytes and a['route_exclusivity_pass'])
    c.check('firewall_every_byte', a['firewall_checks'] == nbytes)
    c.check('full_Merkle_every_record', a['full_hash_recomputations'] == records)
    c.check('probe_guard_count', a['probe_checks'] == 2*load+21)
    c.check('no_state_or_numerical_failures', a['policy_state_divergences'] == 0 and a['nonfinite_predictions'] == 0)
    for r in 'ATG':
        c.check('saved_'+r+'_event_count', sum(str(x) == r for x in cols['route']) == routes[r])
    c.check('no_F_dump', all(str(r) in 'ATG' for r in cols['route']))
    c.check('event_clock_is_preupdate', np.array_equal(cols['event_index'], cols['physical_time']) and
            all(0 <= int(x) < nbytes for x in cols['event_index']))
    c.check('event_seed_load', all(int(x) == job['seed'] for x in cols['seed']) and all(int(x) == load for x in cols['load']))
    groups = result['groups']
    c.check('eight_policy_roster', set(groups['whole']) == set(ARMS)|{'SHUFFLED-ALPHA'})
    for arm in ARMS:
        for group, count in (('whole', nbytes), ('target', records), ('nontarget', nbytes-records)):
            g = groups[group][arm]
            c.check('group_sample_count', g['n'] == count, [arm, group])
            c.check('group_correct_range', isinstance(g['correct'], int) and 0 <= g['correct'] <= count, [arm, group])
            c.close('group_bpb_reconstruction', g['bpb'], g['loss_sum']/count, [arm, group])
            c.close('group_accuracy_reconstruction', g['accuracy'], g['correct']/count, [arm, group])
            if group+'_bpb' in result['metrics_by_arm'][arm]:
                c.close('headline_metric_matches_group', result['metrics_by_arm'][arm][group+'_bpb'], g['bpb'], [arm, group])
        components = [groups['route/'+r][arm] for r in 'ATGF' if routes[r]]
        c.check('route_group_counts_reconcile', sum(g['n'] for g in components) == nbytes, arm)
        c.check('route_group_top1_reconciles', sum(g['correct'] for g in components) == groups['whole'][arm]['correct'], arm)
        c.close('route_group_loss_reconciles', math.fsum(g['loss_sum'] for g in components), groups['whole'][arm]['loss_sum'], arm)
    probes = result['probes']; query_groups = collections.defaultdict(list)
    for p in probes: query_groups[(p['checkpoint'], p['kind'], p['key_id'], p['alias'])].append(p)
    expected_queries = 8*load-9*nrev
    c.check('probe_query_count', len(query_groups) == expected_queries and len(probes) == 8*expected_queries)
    for query, rows in query_groups.items():
        c.check('all_policies_share_probe_query', len(rows) == 8 and {x['arm'] for x in rows} == set(ARMS), query)
        c.check('shared_probe_physical_route', len({x['route'] for x in rows}) == 1 and len({x['physical_slot_uid'] for x in rows}) == 1, query)
        c.check('shared_native_SLOTS_probe_score', len({x['physical_slots_accuracy'] for x in rows}) == 1, query)
    for acquisition in a['physical_acquisition_current']:
        rows = [p for p in probes if p['arm'] == 'SILENT-HARD' and p['checkpoint'] == acquisition['checkpoint'] and p['kind'] == 'current']
        mature = [p for p in rows if p['route'] in ('A', 'G')]
        c.check('Q0_intended_current_population', len(rows) == load//2 == acquisition['population_n'], acquisition['checkpoint'])
        c.check('Q0_native_SLOTS_correct_count', sum(p['physical_slots_accuracy'] for p in rows) == acquisition['population_correct'], acquisition['checkpoint'])
        c.close('Q0_native_SLOTS_accuracy', acquisition['population_accuracy'], acquisition['population_correct']/len(rows), acquisition['checkpoint'])
        c.check('mature_coverage_numerator', len(mature) == acquisition['mature_n'], acquisition['checkpoint'])
        c.close('mature_coverage_ratio', acquisition['mature_coverage'], len(mature)/len(rows), acquisition['checkpoint'])
    def probe_mean(arm, checkpoint, kind, ids=None, aliases=None):
        values = [p['accuracy'] for p in probes if p['arm'] == arm and p['checkpoint'] == checkpoint and p['kind'] == kind
                  and (ids is None or p['key_id'] in ids) and (aliases is None or p['alias'] in aliases)]
        return sum(values)/len(values) if values else None
    for arm in ARMS:
        expected_metrics = {
            'E1_all': probe_mean(arm, 'E1', 'emergence'),
            'E2_all': probe_mean(arm, 'E2', 'emergence'),
            'E1_A': probe_mean(arm, 'E1', 'emergence', ids=range(load//4)),
            'E1_later': probe_mean(arm, 'E1', 'emergence', ids=range(load//4, load)),
            'A_after_D_seen': probe_mean(arm, 'after_D', 'untouched_A'),
            'A_after_D_heldout': probe_mean(arm, 'after_D', 'untouched_A_heldout'),
            'after_D_current_seen': probe_mean(arm, 'after_D', 'current'),
            'after_D_current_heldout': probe_mean(arm, 'after_D', 'current_heldout'),
            'revision_after_D_seen': probe_mean(arm, 'after_D', 'revision_retained', aliases=(0, 1)),
            'revision_after_D_heldout': probe_mean(arm, 'after_D', 'revision_retained', aliases=(2,))}
        for exposure in (1, 2, 4, 8):
            for suffix, aliases in (('seen', (0, 1)), ('heldout', (2,)), ('exact', (0,)), ('known', (1,))):
                expected_metrics[f'revision_{exposure}_{suffix}'] = probe_mean(arm, f'revision_{exposure}', 'revision', aliases=aliases)
        untouched = {p['key_id'] for p in probes if p['arm'] == arm and p['checkpoint'] == 'after_D' and p['kind'] == 'untouched_A_heldout'}
        joint = []
        for uid in untouched:
            outcomes = [probe_mean(arm, 'E1', 'emergence', ids={uid}, aliases=(2,))]
            outcomes += [probe_mean(arm, cp, 'untouched_A_heldout', ids={uid}, aliases=(2,)) for cp in ('after_B', 'after_C', 'after_D')]
            joint.append(int(all(x == 1 for x in outcomes)))
        expected_metrics['same_identity_E1_all_delays'] = sum(joint)/len(joint) if joint else None
        for metric, value in expected_metrics.items():
            c.close('probe_endpoint_'+metric, result['metrics_by_arm'][arm][metric], value, arm)
    cpnames = {'after_A', 'after_B', 'after_C', 'after_D', 'revision_1', 'revision_2', 'revision_4', 'revision_8'}
    c.check('physical_checkpoint_roster', {x['checkpoint'] for x in result['checkpoint_memory']} == cpnames)
    for cp in result['checkpoint_memory']:
        c.check('checkpoint_capacity', cp['traces'] <= 512 and cp['slots'] <= 384 and cp['anchors'] <= 768, cp['checkpoint'])
        c.check('checkpoint_resident_ledgers', cp['slots'] == cp['ledger_count'], cp['checkpoint'])
    c.check('terminal_hash_fields', len(result['final_physical_hash']) == 64 and
            set(result['final_physical_components']) == {'encoder', 'base', 'live', 'memory', 'history'})
    if result['checkpoint_memory']:
        last = result['checkpoint_memory'][-1]
        c.check('terminal_checkpoint_hash_agrees', last['physical_hash'] == result['final_physical_hash'] and last['components'] == result['final_physical_components'])
    return {**c.report(), 'byte_events': nbytes, 'records': records, 'probe_queries': len(query_groups), 'route_counts': routes}


def audit(root, source, allow_smoke=False):
    root = Path(root); planpath = root/'PLAN.json'; c = Checks()
    output = {'audit_version': 1, 'results_directory': str(root.resolve()), 'new_learning_updates': 0,
              'source_executed': False, 'jobs': [], 'missing_jobs': [], 'unexpected_jobs': [], 'limitations': [
        'This audit independently reconstructs saved predictive and ledger arithmetic. It verifies source hashes and physical engineering receipts; it does not rerun the learner.',
        'The logged E2 birth evidence is not independently derived from unlogged route-F pre-promotion expert distributions.',
        'Top-1 outcomes are reconciled against saved counts; full expert vectors were not saved and their argmax cannot be independently recovered from true-byte probabilities.']}
    if not planpath.exists():
        output.update(status='partial', complete=False, passed=False, reason='PLAN.json is not present yet.')
        return output
    plan = json.loads(planpath.read_text()); smoke = plan.get('mode') == 'engineering_smoke'
    expected_pairs = ({(943901, 16)} if smoke and allow_smoke else
                      {(s, l) for s in range(43101, 43117) for l in (64, 128, 256, 512)})
    c.check('scientific_or_explicit_smoke_scope', not smoke or allow_smoke)
    c.check('locked_seed_load_roster', {(j['seed'], j['load']) for j in plan['jobs']} == expected_pairs and len(plan['jobs']) == len(expected_pairs))
    c.check('expected_trajectory_count', plan['expected_trajectories'] == len(expected_pairs))
    c.check('locked_eight_policies', tuple(plan['policies']) == ARMS)
    c.check('locked_offline_controls', tuple(plan['offline_controls']) == CONTROLS)
    for job in plan['jobs']:
        cfg = job['config']; c.check('locked_config', cfg['load'] == job['load'] and cfg['keys_per_wave'] == job['load']//4 and
            cfg['trace_cap'] == 512 and cfg['slot_cap'] == 384 and cfg['repeats'] == (2 if smoke else 8), job)
    output.update(mode=plan['mode'], expected_trajectories=len(expected_pairs), source_sha256=plan['source_sha256'], plan_sha256=sha(plan))
    try: output['source'] = source_audit(plan, source)
    except Exception as exc: output['source'] = {'passed': False, 'errors': [repr(exc)]}
    try: output['engineering'] = engineering_audit(root, plan, smoke)
    except Exception as exc: output['engineering'] = {'passed': False, 'errors': [repr(exc)]}
    expected_names = {f"seed_{j['seed']}_load_{j['load']}.json" for j in plan['jobs']}
    output['unexpected_jobs'] = sorted(p.name for p in (root/'completed').glob('*.json') if p.name not in expected_names)
    c.check('no_unplanned_completed_jobs', not output['unexpected_jobs'])
    for job in plan['jobs']:
        name = f"seed_{job['seed']}_load_{job['load']}"; path = root/'completed'/(name+'.json')
        if not path.exists(): output['missing_jobs'].append(name); continue
        cell = {'job': name, 'seed': job['seed'], 'load': job['load']}; cc = Checks()
        try:
            envelope = json.loads(path.read_text()); result = envelope['data']
            cc.check('completed_identity', envelope['identity'] == {'plan_sha256': sha(plan), 'job': job})
            cc.check('completed_result_hash', envelope['sha256'] == sha(result))
            evpath = root/result['authority_events_file']
            cc.check('authority_event_file_hash', sha(evpath.read_bytes()) == result['authority_events_sha256'])
            with np.load(evpath, allow_pickle=False) as saved:
                cc.check('authority_columns_exact', set(saved.files) == set(result['authority_events_columns'])|{'__row_count__'})
                cc.check('authority_row_count', int(saved['__row_count__']) == result['authority_events_count'])
                cols = {key: saved[key] for key in saved.files if key != '__row_count__'}
            cc.check('authority_column_lengths', all(len(v) == result['authority_events_count'] for v in cols.values()))
            cell['integrity'] = cc.report()
            cell['structure'] = structural_audit(result, job, cols)
            cell['virtual_ledger'] = ledger_audit(result, cols)
            cell['readouts_and_controls'] = audit_events(result, cols)
            cell['passed'] = all(cell[key].get('passed', False) for key in ('integrity', 'structure', 'virtual_ledger', 'readouts_and_controls'))
        except Exception as exc:
            cell.update(passed=False, exception=repr(exc), traceback=traceback.format_exc())
        output['jobs'].append(cell)
    output['plan'] = c.report()
    output['completed_trajectories'] = len(output['jobs'])
    failures = not c.report()['passed'] or not output['source']['passed'] or any(not j['passed'] for j in output['jobs'])
    engineering_missing = output['engineering'].get('missing', False)
    if not engineering_missing and not output['engineering']['passed']: failures = True
    complete = len(output['jobs']) == len(expected_pairs) and not output['missing_jobs'] and not engineering_missing
    output['complete'] = complete
    output['passed'] = complete and not failures
    output['status'] = ('failed' if failures else 'partial' if not complete else
                        'engineering_smoke_passed' if smoke else 'passed')
    output['total_physical_byte_events'] = sum(j.get('structure', {}).get('byte_events', 0) for j in output['jobs'])
    return output


def write_report(result, prefix):
    prefix = Path(prefix); prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix.with_suffix('.json').write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False)+'\n')
    rows = [f"# V43 independent saved-results audit\n\nStatus: **{result['status']}**. New learner updates: **0**.\n",
        f"Completed cells: {result.get('completed_trajectories', 0)} / {result.get('expected_trajectories', 64)}. "
        f"Saved physical byte events inspected: {result.get('total_physical_byte_events', 0):,}.\n"]
    if result.get('source_sha256'): rows.append(f"Frozen source SHA256: `{result['source_sha256']}`.\n")
    rows.append('## What was checked\n\nSource and embedded payload identities; complete roster and configuration; '
        'source-bound engineering receipts; completed JSON and NPZ checksums; one physical trajectory per cell; '
        'all-byte route and firewall counts; full-record hash receipts; probe policy sharing; independent causal '
        'ledger reconstruction; saved fast/slow mixture losses; shuffle bins, derangements, exact weight conservation '
        'and matched specificity subsets; event and generation oracles.\n')
    rows.append('## Limits\n\n'+'\n'.join('- '+x for x in result['limitations'])+'\n')
    if result.get('missing_jobs'):
        rows.append('## Missing cells\n\n'+', '.join(result['missing_jobs'])+'\n')
    bad = [x for x in result.get('jobs', []) if not x['passed']]
    if bad:
        rows.append('## Failed cell checks\n\n'+ '\n'.join('- '+x['job']+': '+x.get('exception', 'See JSON for categorized discrepancies.') for x in bad)+'\n')
    elif result.get('jobs'):
        rows.append('All inspected cell-level arithmetic and integrity checks passed. '
            'A partial audit remains incomplete until every planned cell and required engineering receipt is available.\n')
    rows.append('The adjacent JSON contains per-cell check counts, errors and explicit limitations.\n')
    prefix.with_suffix('.md').write_text('\n'.join(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', nargs='?', default='v43_results')
    parser.add_argument('--out-dir', type=Path, help='Results directory to audit (alias for the positional directory).')
    parser.add_argument('--source', type=Path)
    parser.add_argument('--out-prefix', type=Path)
    parser.add_argument('--allow-smoke', action='store_true')
    parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args(); root = args.out_dir or Path(args.results)
    candidates = [root/'elm_v43_fast_slow_authority.py', Path('elm_v43_fast_slow_authority.py'),
                  Path(__file__).resolve().parents[1]/'elm_v43_fast_slow_authority.py']
    source = args.source or next((p for p in candidates if p.exists()), candidates[0])
    result = audit(root, source, args.allow_smoke)
    write_report(result, args.out_prefix or root/'INDEPENDENT_AUDIT')
    print(json.dumps({'status': result['status'], 'completed': result.get('completed_trajectories', 0),
                      'expected': result.get('expected_trajectories', 64), 'new_learning_updates': 0}))
    return 0 if result['passed'] or (args.allow_partial and result['status'] == 'partial') else 2


if __name__ == '__main__':
    raise SystemExit(main())
