"""V42 fixed-history authority evaluator.

This module switches saved predictions on a fixed V41 training history.  It does
not instantiate or update a learner.  Reference records are treated as immutable
and are not copied into mutable execution checkpoints.
"""
import base64
import copy
import hashlib
import json
import math

ARMS = ('READ-HYBRID', 'SLOT-ONLY', 'EXACT-ONLY', 'ANCHOR-ONLY', 'SUPER')
MASKS = {
    'READ-HYBRID': ('exact_slot', 'exact_trace', 'supported_slot'),
    'SLOT-ONLY': ('exact_slot', 'supported_slot'),
    'EXACT-ONLY': ('exact_slot', 'exact_trace'),
    'ANCHOR-ONLY': ('exact_slot',),
    'SUPER': (),
}
AUTHORITATIVE = ('exact_slot', 'exact_trace', 'supported_slot')
STRATA = AUTHORITATIVE + ('no_candidate', 'tentative_trace', 'tentative_slot')
SCOPES = ('whole', 'target', 'nontarget')


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _close(a, b):
    return math.isclose(float(a), float(b), rel_tol=2e-12, abs_tol=2e-10)


def _original(ref):
    return ref['original_completed_envelope']['data']['result']


def _probe_key(row):
    return (str(row['checkpoint']), str(row['kind']), int(row['key_id']),
            int(row['alias']))


def _key_id(key):
    # JSON arrays encode checkpoint, kind, key, and alias without delimiters
    # that could collide with checkpoint names.
    return json.dumps(list(key), separators=(',', ':'))


def _stats(row):
    n = int(row['n'])
    _require(n > 0 and n == row['n'], 'Invalid aggregate count')
    accuracy = float(row['accuracy'])
    loss = float(row['loss_sum'])
    _require(math.isfinite(accuracy) and 0 <= accuracy <= 1,
             'Invalid aggregate accuracy')
    _require(math.isfinite(loss) and loss >= 0, 'Invalid aggregate loss')
    correct = round(n * accuracy)
    _require(abs(n * accuracy - correct) <= 2e-8,
             'Aggregate accuracy does not encode an integer count')
    _require(_close(row['bpb'], loss / n), 'Aggregate loss/bpb mismatch')
    return {'n': n, 'correct': int(correct), 'loss_sum': loss}


def _empty():
    return {'n': 0, 'correct': 0, 'loss_sum': 0., 'authority_reads': 0}


def _add(dest, stats, authority=False):
    dest['n'] += int(stats['n'])
    dest['correct'] += int(stats['correct'])
    dest['loss_sum'] += float(stats['loss_sum'])
    dest['authority_reads'] += int(stats['n']) if authority else 0


def _finish(stats):
    n = stats['n']
    return dict(stats, accuracy=stats['correct'] / n if n else None,
                bpb=stats['loss_sum'] / n if n else None,
                authority_read_rate=stats['authority_reads'] / n if n else None)


def _chosen_source(arm, stratum):
    if arm == 'SUPER':
        return 'SUPER'
    # Preserve the original READ-HYBRID fallback on nonauthoritative packets.
    # It may use BASE when the superposed scores have negligible norm.
    if stratum not in AUTHORITATIVE or stratum in MASKS[arm]:
        return 'READ-HYBRID'
    return 'SUPER'


def _gather_probes(ref, validate=False):
    original = _original(ref)
    paired = {}
    origins = {}
    strata = {}
    contexts = {}
    historical_n = 0
    for row in original['probes']:
        if row['arm'] not in ('READ-HYBRID', 'SUPER') or int(row['alias']) != 2:
            continue
        key = _probe_key(row)
        bucket = paired.setdefault(key, {})
        _require(row['arm'] not in bucket, 'Duplicate historical probe join key')
        bucket[row['arm']] = row
        origins[key] = 'historical_alias2'
        historical_n += 1
    for key, rows in paired.items():
        _require(set(rows) == {'READ-HYBRID', 'SUPER'},
                 'Historical probe is missing a prediction partner')
        authority = rows['READ-HYBRID']['authority']
        _require(type(authority) is bool, 'Historical READ authority must be bool')
        strata[key] = 'supported_slot' if authority else 'nonauthoritative_unresolved'
    final_seen = set()
    for row in ref['final_after_D_probes']:
        _require(row['arm'] in ('READ-HYBRID', 'SUPER'),
                 'Unexpected final probe arm')
        _require(row['checkpoint'] == 'after_D', 'Final probe is not after_D')
        key = _probe_key(row)
        pairkey = (key, row['arm'])
        _require(pairkey not in final_seen, 'Duplicate final probe join key')
        final_seen.add(pairkey)
        packet = row['packet']
        stratum = packet['stratum']
        _require(stratum in STRATA, 'Unknown final packet stratum')
        if validate:
            _require(bool(packet['authority']) == (stratum in AUTHORITATIVE),
                     'Packet authority/stratum mismatch')
            _require(bool(packet['exact']) == (stratum in ('exact_slot', 'exact_trace')),
                     'Packet exact/stratum mismatch')
            if stratum == 'supported_slot':
                _require(packet['type'] == 'slot' and int(packet['support']) >= 2,
                         'Supported-slot authority lacks two-member support')
            if stratum.startswith('exact_'):
                _require(stratum == 'exact_' + packet['type'],
                         'Exact packet type disagrees with stratum')
            if int(row['alias']) == 2:
                _require(not bool(packet['exact']), 'Never-trained alias2 had an exact hit')
            if row['arm'] == 'READ-HYBRID':
                _require(row['authority'] == packet['authority'],
                         'Saved READ authority differs from final packet')
        old = paired.get(key, {}).get(row['arm'])
        if old is not None:
            _require(int(old['accuracy']) == int(row['accuracy']) and
                     _close(old['bpb'], row['bpb']) and
                     old['authority'] == row['authority'],
                     'Enriched final probe differs from its historical prediction')
        if key in strata and origins.get(key) == 'final_frozen_state':
            _require(strata[key] == stratum, 'Paired final packet strata disagree')
            _require(contexts[key] == row['context_hex'],
                     'Paired final probe contexts disagree')
        paired.setdefault(key, {})[row['arm']] = row
        origins[key] = 'final_frozen_state'
        strata[key] = stratum
        contexts[key] = row['context_hex']
    for key, _ in final_seen:
        _require((key, 'READ-HYBRID') in final_seen and (key, 'SUPER') in final_seen,
                 'Enriched final query is missing a newly reproduced partner')
    tasks = []
    for key in sorted(paired):
        rows = paired[key]
        _require(set(rows) == {'READ-HYBRID', 'SUPER'},
                 'Final probe is missing a prediction partner')
        for row in rows.values():
            _require(row['accuracy'] in (0, 1), 'Probe correctness is not binary')
            _require(math.isfinite(float(row['bpb'])) and float(row['bpb']) >= 0,
                     'Nonfinite or negative probe loss')
        if key in contexts:
            _require(len(bytes.fromhex(contexts[key])) == 12,
                     'Final context is not 12 bytes')
        tasks.append({'kind': 'probe', 'key': list(key)})
    return paired, origins, strata, contexts, tasks, historical_n


def _prepare(ref, validate=True):
    original = _original(ref)
    seed = int(ref['seed'])
    _require(seed == int(ref['original_completed_envelope']['data']['seed']),
             'Reference seed mismatch')
    if validate:
        envelope = ref['original_completed_envelope']
        _require(envelope['sha256'] == _digest(envelope['data']),
                 'Original completed-record digest failed')
    groups = original['groups']
    tasks = []
    original_totals = {scope: {arm: _empty() for arm in ('READ-HYBRID', 'SUPER')}
                       for scope in SCOPES}
    for group in sorted(groups):
        if not group.startswith('slots_state/'):
            continue
        fields = group.split('/')
        _require(len(fields) == 3, 'Malformed common-stratum group')
        _, stratum, scope = fields
        _require(stratum in STRATA and scope in ('target', 'nontarget'),
                 'Unknown common stratum or category')
        rows = groups[group]
        r, s = _stats(rows['READ-HYBRID']), _stats(rows['SUPER'])
        _require(r['n'] == s['n'], 'Common stratum has unequal arm counts')
        for arm, stats in (('READ-HYBRID', r), ('SUPER', s)):
            _add(original_totals[scope][arm], stats)
            _add(original_totals['whole'][arm], stats)
        tasks.append({'kind': 'group', 'key': group})
    _require(bool(tasks), 'No common-stratum aggregates')
    for scope in SCOPES:
        for arm in ('READ-HYBRID', 'SUPER'):
            expected = _stats(groups[scope][arm])
            actual = original_totals[scope][arm]
            _require(actual['n'] == expected['n'] and
                     actual['correct'] == expected['correct'] and
                     _close(actual['loss_sum'], expected['loss_sum']),
                     'Disjoint strata do not reconstruct original ' + scope + '/' + arm)
    paired, origins, strata, contexts, probe_tasks, historical_n = _gather_probes(ref, validate)
    tasks.extend(probe_tasks)
    cert = ref['alias2_certification']
    _require(cert['alias2_contexts_present'] == 0,
             'Historical alias2 cannot be classified because a context was trained')
    if validate:
        raw = base64.b64decode(ref['raw_stream_base64'], validate=True)
        _require(hashlib.sha256(raw).hexdigest() == original['stream_sha256'],
                 'Training-stream digest failed')
        _require(len(raw) == int(original['byte_events']),
                 'Training-stream byte count failed')
        alias_rows = ref['alias2_contexts']
        alias_ids = [int(x['key_id']) for x in alias_rows]
        alias_contexts = [bytes.fromhex(x['context_hex']) for x in alias_rows]
        _require(len(set(alias_ids)) == len(alias_ids) == original['keys'],
                 'Alias2 certificate does not uniquely cover every identity')
        _require(len(set(alias_contexts)) == len(alias_contexts),
                 'Alias2 contexts are not identity-unique')
        trained = {raw[i-12:i] for i in range(12, len(raw))}
        _require(all(len(context) == 12 and context not in trained
                     for context in alias_contexts),
                 'Never-trained alias2 certification failed independent scan')
        alias_map = dict(zip(alias_ids, alias_contexts))
        for key, context_hex in contexts.items():
            if key[3] == 2:
                _require(bytes.fromhex(context_hex) == alias_map[key[2]],
                         'Final held-out context differs from alias2 certificate')
    audit = {
        'common_strata_reconstruct_original_counts_correctness_and_loss': True,
        'original_completed_digest_valid': True,
        'historical_alias2_contexts_absent_from_every_full_prebyte_window': True,
        'historical_alias2_rows_before_join': historical_n,
        'paired_probe_contexts_after_join': len(paired),
        'final_rows_unique_by_probe_and_arm': True,
        'historical_final_overlap_predictions_identical': True,
        'new_training_updates': 0,
        'online_switch_uses_preexisting_common_strata': True,
        'non_authoritative_fallback_preserved': True,
    }
    return tasks, (paired, origins, strata, contexts), audit


def init_state(ref):
    tasks, _, audit = _prepare(ref)
    return {
        'schema_version': 1, 'seed': int(ref['seed']),
        'input_digest': _digest(ref), 'cursor': 0, 'processed_units': 0,
        'total_units': len(tasks), 'done': False,
        'online_raw': {scope: {arm: _empty() for arm in ARMS} for scope in SCOPES},
        'stratum_effects': [], 'probes': [], 'audits': audit,
    }


def step(state, ref, max_units=16):
    _require(isinstance(max_units, int) and max_units > 0,
             'max_units must be a positive integer')
    _require(state['input_digest'] == _digest(ref), 'Reference changed after initialization')
    if state['done']:
        return True
    tasks, sources, _ = _prepare(ref, validate=False)
    paired, origins, strata, contexts = sources
    _require(state['total_units'] == len(tasks), 'Reference task count changed')
    original = _original(ref)
    for task in tasks[state['cursor']:state['cursor'] + max_units]:
        if task['kind'] == 'group':
            group = task['key']
            _, stratum, scope = group.split('/')
            rows = original['groups'][group]
            r, s = _stats(rows['READ-HYBRID']), _stats(rows['SUPER'])
            for arm in ARMS:
                chosen = _chosen_source(arm, stratum)
                stats = r if chosen == 'READ-HYBRID' else s
                authority = stratum in MASKS[arm]
                _add(state['online_raw'][scope][arm], stats, authority)
                _add(state['online_raw']['whole'][arm], stats, authority)
            delta = r['loss_sum'] - s['loss_sum']
            state['stratum_effects'].append({
                'scope': scope, 'stratum': stratum, 'n': r['n'],
                'read_loss_sum': r['loss_sum'], 'super_loss_sum': s['loss_sum'],
                'read_minus_super_loss_sum': delta,
                'read_minus_super_scope_bpb': delta / original['groups'][scope]['SUPER']['n'],
                'read_minus_super_whole_bpb': delta / original['groups']['whole']['SUPER']['n'],
                'read_correct': r['correct'], 'super_correct': s['correct'],
            })
        else:
            key = tuple(task['key'])
            rows = paired[key]
            stratum = strata[key]
            for arm in ARMS:
                chosen = _chosen_source(arm, stratum)
                source = rows[chosen]
                out = {'checkpoint': key[0], 'kind': key[1], 'key_id': key[2],
                       'alias': key[3], 'arm': arm, 'accuracy': int(source['accuracy']),
                       'bpb': float(source['bpb']), 'authority': stratum in MASKS[arm],
                       'stratum': stratum, 'source': origins[key],
                       'selected_saved_arm': chosen,
                       'query_origin': source.get('origin', 'historical_saved_alias2')}
                if key in contexts:
                    out['context_hex'] = contexts[key]
                state['probes'].append(out)
        state['cursor'] += 1
        state['processed_units'] += 1
    state['done'] = state['cursor'] == len(tasks)
    return state['done']


def _endpoint(rows):
    if not rows:
        return {'n': 0, 'correct': 0, 'accuracy': None, 'loss_sum': 0., 'bpb': None}
    n = len(rows)
    correct = sum(int(x['accuracy']) for x in rows)
    loss = sum(float(x['bpb']) for x in rows)
    return {'n': n, 'correct': correct, 'accuracy': correct / n,
            'loss_sum': loss, 'bpb': loss / n}


def summarize(state, ref):
    _require(state['done'], 'Cannot summarize an incomplete evaluation')
    _require(state['input_digest'] == _digest(ref), 'Reference changed before summary')
    original = _original(ref)
    online = {scope: {arm: _finish(values) for arm, values in arms.items()}
              for scope, arms in state['online_raw'].items()}
    k = int(original['keys_per_wave'])
    untouched = set(int(x) for x in original['untouched_ids'])
    endpoints, metrics, joint_rows = {}, {}, []
    for arm in ARMS:
        p = [row for row in state['probes'] if row['arm'] == arm]
        selectors = {
            'E1_all': lambda x: x['checkpoint'] == 'E1' and x['kind'] == 'emergence',
            'E1_A': lambda x: x['checkpoint'] == 'E1' and x['kind'] == 'emergence' and x['key_id'] < k,
            'E1_later': lambda x: x['checkpoint'] == 'E1' and x['kind'] == 'emergence' and x['key_id'] >= k,
            'E2_all': lambda x: x['checkpoint'] == 'E2' and x['kind'] == 'emergence',
            'A_after_D_heldout': lambda x: x['checkpoint'] == 'after_D' and x['kind'] == 'untouched_A_heldout' and x['alias'] == 2,
            'A_after_D_seen': lambda x: x['checkpoint'] == 'after_D' and x['kind'] == 'untouched_A' and x['alias'] in (0, 1),
            'after_D_current_seen': lambda x: x['checkpoint'] == 'after_D' and x['kind'] == 'current' and x['alias'] in (0, 1),
            'after_D_current_heldout': lambda x: x['checkpoint'] == 'after_D' and x['kind'] == 'current_heldout' and x['alias'] == 2,
            'revision_4_heldout': lambda x: x['checkpoint'] == 'revision_4' and x['kind'] == 'revision' and x['alias'] == 2,
            'revision_8_heldout': lambda x: x['checkpoint'] == 'revision_8' and x['kind'] == 'revision' and x['alias'] == 2,
            'revision_after_D_seen': lambda x: x['checkpoint'] == 'after_D' and x['kind'] == 'revision_retained' and x['alias'] in (0, 1),
            'revision_after_D_heldout': lambda x: x['checkpoint'] == 'after_D' and x['kind'] == 'revision_retained' and x['alias'] == 2,
        }
        for cp in ('after_A', 'after_B', 'after_C', 'after_D'):
            selectors[cp + '_current_heldout'] = lambda x, cp=cp: x['checkpoint'] == cp and x['kind'] == 'current_heldout' and x['alias'] == 2
            selectors['A_' + cp + '_heldout'] = lambda x, cp=cp: x['checkpoint'] == cp and x['kind'] == 'untouched_A_heldout' and x['alias'] == 2
        endpoints[arm] = {name: _endpoint([x for x in p if choose(x)])
                          for name, choose in selectors.items()}
        index = {}
        for row in p:
            key = _probe_key(row)
            _require(key not in index, 'Duplicate evaluated probe key')
            index[key] = row
        good = []
        for identity in sorted(untouched):
            keys = [('E1', 'emergence', identity, 2)] + [
                (cp, 'untouched_A_heldout', identity, 2)
                for cp in ('after_B', 'after_C', 'after_D')]
            _require(all(key in index for key in keys),
                     'Missing a required observation in the same-identity endpoint')
            correctness = [int(index[key]['accuracy']) for key in keys]
            success = int(all(correctness))
            good.append(success)
            joint_rows.append({'arm': arm, 'key_id': identity,
                               'E1_after_B_after_C_after_D_correct': correctness,
                               'all_correct': success})
        endpoints[arm]['same_identity_E1_all_delays'] = {
            'n': len(good), 'correct': sum(good),
            'accuracy': sum(good) / len(good) if good else None,
            'loss_sum': None, 'bpb': None}
        metrics[arm] = {}
        for scope in SCOPES:
            stats = online[scope][arm]
            metrics[arm][scope + '_bpb'] = stats['bpb']
            metrics[arm][scope + '_accuracy'] = stats['accuracy']
            metrics[arm][scope + '_reads'] = stats['authority_reads']
            metrics[arm][scope + '_read_rate'] = stats['authority_read_rate']
        for name, stats in endpoints[arm].items():
            if stats['accuracy'] is not None:
                metrics[arm][name] = stats['accuracy']
                if stats['bpb'] is not None:
                    metrics[arm][name + '_bpb'] = stats['bpb']
    reconstruction = {}
    matching_names = {'E1_all': 'E1_all', 'E1_A': 'E1_A', 'E1_later': 'E1_later',
                      'E2_all': 'E2_all', 'A_after_D_seen': 'A_after_D_seen',
                      'A_after_D_heldout': 'A_after_D_heldout',
                      'after_D_current_seen': 'after_D_acquisition',
                      'same_identity_E1_all_delays': 'same_identity_E1_all_delays',
                      'revision_4_heldout': 'revision_4_heldout'}
    for arm in ('READ-HYBRID', 'SUPER'):
        checks = {}
        for scope in SCOPES:
            expected = original['groups'][scope][arm]
            actual = online[scope][arm]
            checks[scope] = actual['n'] == expected['n'] and _close(actual['bpb'], expected['bpb']) and _close(actual['accuracy'], expected['accuracy'])
        for name, original_name in matching_names.items():
            if original_name in original.get('metrics_by_arm', {}).get(arm, {}):
                expected = original['metrics_by_arm'][arm][original_name]
                actual = metrics[arm].get(name)
                if expected is not None:
                    checks[name] = actual is not None and _close(actual, expected)
        _require(all(checks.values()), 'Reference reconstruction failed for ' + arm)
        reconstruction[arm] = checks
    # Exact ownership remains available in both masks on every historical row;
    # because held-out contexts never had it, these two results must coincide.
    for a, b in zip([x for x in state['probes'] if x['arm'] == 'EXACT-ONLY'],
                    [x for x in state['probes'] if x['arm'] == 'ANCHOR-ONLY']):
        if a['alias'] == 2:
            _require(a['accuracy'] == b['accuracy'] and a['bpb'] == b['bpb'],
                     'Exact-only and anchor-only held-out predictions differ')
    audits = dict(state['audits'], baseline_reconstruction=reconstruction,
                  historical_heldout_exact_equals_anchor=True,
                  inherited_reference_audits=copy.deepcopy(ref.get('audits', {})))
    return {
        'schema_version': 1, 'experiment': 'fixed_history_authority',
        'seed': state['seed'], 'input_digest': state['input_digest'],
        'training_updates': 0, 'new_stream_presentations': 0,
        'historical_byte_events': original['byte_events'],
        'historical_target_presentations': original['target_presentations'],
        'historical_unique_complete_records': original['unique_complete_records'],
        'masks': {arm: list(MASKS[arm]) for arm in ARMS},
        'online': online, 'stratum_effects': copy.deepcopy(state['stratum_effects']),
        'probes': copy.deepcopy(state['probes']), 'endpoints': endpoints,
        'same_identity_details': joint_rows, 'metrics_by_arm': metrics,
        'metrics': [{'arm': arm, 'metric': name, 'value': value}
                    for arm in ARMS for name, value in metrics[arm].items()],
        'audits': audits, 'provenance': copy.deepcopy(ref.get('provenance', {})),
        'historical_costs': copy.deepcopy(original.get('costs', {})),
        'limitations': [
            'Read policies are evaluated on the same already-trained SLOTS history; changes to admission, learning, eviction, or storage are not tested.',
            'All memory is still stored; suppressing a read establishes no RAM or measured-runtime saving.',
            'Historical intermediate seen-alias packet types were not saved, so their switched-policy endpoints are omitted.',
            'Four reused V41 seeds provide a post hoc mechanism audit rather than an independent confirmatory replication.',
            'Probe rows and byte events are nested observations, not independent experimental units.',
            'Non-authoritative READ fallback is retained exactly; it can differ from the raw SUPER arm.',
            'No global-policy extrapolation beyond the saved histories, fixed workload, and capacities is established.',
        ],
    }


def smoke_reference():
    """Synthetic data only; this fixture never reads a V41 scientific record."""
    seed, k, keys = 942001, 2, 8
    groups = {}
    totals = {scope: {arm: _empty() for arm in ('READ-HYBRID', 'SUPER')}
              for scope in SCOPES}
    for scope in ('target', 'nontarget'):
        for i, stratum in enumerate(STRATA):
            n = 4 + i
            # Unequal no-authority predictions test BASE fallback preservation.
            r = {'n': n, 'correct': n - (i % 3), 'loss_sum': (i + 1.) * n}
            s = {'n': n, 'correct': n - ((i + 1) % 3), 'loss_sum': (i + 1.5) * n}
            group = 'slots_state/' + stratum + '/' + scope
            groups[group] = {}
            for arm, stats in (('READ-HYBRID', r), ('SUPER', s)):
                groups[group][arm] = {'n': n, 'accuracy': stats['correct'] / n,
                                      'loss_sum': stats['loss_sum'], 'bpb': stats['loss_sum'] / n}
                _add(totals[scope][arm], stats)
                _add(totals['whole'][arm], stats)
    for scope in SCOPES:
        groups[scope] = {arm: {'n': stats['n'], 'accuracy': stats['correct'] / stats['n'],
                               'loss_sum': stats['loss_sum'], 'bpb': stats['loss_sum'] / stats['n']}
                         for arm, stats in totals[scope].items()}
    aliases = [{'key_id': i, 'context_hex': (bytes([254]) * 11 + bytes([i])).hex()}
               for i in range(keys)]
    contexts = {row['key_id']: row['context_hex'] for row in aliases}
    probes = []
    final = []
    def add_probe(cp, kind, identity, alias, stratum, final_only=False):
        authority = stratum in AUTHORITATIVE
        context = contexts[identity] if alias == 2 else (bytes([253 - alias]) * 11 + bytes([identity])).hex()
        for arm in ('READ-HYBRID', 'SUPER'):
            row = {'checkpoint': cp, 'kind': kind, 'key_id': identity, 'alias': alias,
                   'arm': arm, 'accuracy': int((identity + alias + (arm == 'SUPER')) % 3 != 0),
                   'bpb': float(1 + identity / 8 + alias / 10 + (arm == 'SUPER') / 4),
                   'authority': authority if arm == 'READ-HYBRID' else None}
            if not final_only:
                probes.append(copy.deepcopy(row))
            if cp == 'after_D':
                packet = {'stratum': stratum, 'exact': stratum in ('exact_slot', 'exact_trace'),
                          'type': 'slot' if 'slot' in stratum else 'trace',
                          'authority': authority, 'support': 2 if stratum == 'supported_slot' else 1,
                          'sim': .9}
                final.append(dict(row, context_hex=context, packet=packet,
                                  origin='new_terminal_read_only_revision_query' if final_only else 'reproduced_original_after_D'))
    for identity in range(keys):
        for cp in ('E1', 'E2'):
            add_probe(cp, 'emergence', identity, 2, 'supported_slot' if identity % 2 else 'no_candidate')
    for i, cp in enumerate(('after_A', 'after_B', 'after_C', 'after_D')):
        for identity in range(2 * i, 2 * i + 2):
            add_probe(cp, 'current_heldout', identity, 2, 'supported_slot')
            if cp == 'after_D':
                for alias in (0, 1):
                    add_probe(cp, 'current', identity, alias, 'exact_slot' if alias == 0 else 'exact_trace')
        for identity in (1,):
            add_probe(cp, 'untouched_A_heldout', identity, 2, 'supported_slot')
            if cp == 'after_D':
                for alias in (0, 1):
                    add_probe(cp, 'untouched_A', identity, alias, 'exact_slot')
    for cp in ('revision_1', 'revision_2', 'revision_4', 'revision_8'):
        add_probe(cp, 'revision', 0, 2, 'supported_slot')
    # Twelve extra synthetic read-only observations create several CLI chunks.
    for j in range(12):
        add_probe('synthetic_delay_' + str(j), 'diagnostic_heldout', j % keys, 2,
                  'supported_slot' if j % 2 else 'no_candidate')
    for alias in (0, 1, 2):
        add_probe('after_D', 'revision_retained', 0, alias,
                  'supported_slot' if alias == 2 else 'exact_slot', final_only=True)
    raw = bytes(range(groups['whole']['SUPER']['n']))
    original = {'groups': groups, 'probes': probes, 'keys': keys, 'keys_per_wave': k,
                'untouched_ids': [1], 'stream_sha256': hashlib.sha256(raw).hexdigest(),
                'byte_events': groups['whole']['SUPER']['n'], 'target_presentations': 16,
                'unique_complete_records': 16, 'metrics_by_arm': {}, 'costs': {}}
    data = {'experiment': 'hybrid', 'seed': seed, 'config': {}, 'result': original}
    return {'seed': seed, 'original_completed_envelope': {'data': data, 'sha256': _digest(data), 'identity': {}},
            'raw_stream_base64': base64.b64encode(raw).decode(), 'alias2_contexts': aliases,
            'alias2_certification': {'alias2_contexts_present': 0},
            'final_after_D_probes': final, 'audits': {'synthetic': True},
            'provenance': {'synthetic_fixture': True}}


def self_test():
    ref = smoke_reference()
    before = _digest(ref)
    state = init_state(ref)
    while not step(state, ref, max_units=7):
        pass
    whole = summarize(state, ref)
    resumed = init_state(ref)
    step(resumed, ref, max_units=3)
    resumed = json.loads(json.dumps(resumed))
    while not step(resumed, ref, max_units=11):
        pass
    _require(state == resumed, 'JSON interruption/resume changes final state')
    _require(_digest(whole) == _digest(summarize(resumed, ref)),
             'JSON interruption/resume changes scientific output')
    _require(_digest(ref) == before, 'Evaluator mutated reference')
    expected_n = sum(_original(ref)['groups']['slots_state/' + x + '/target']['READ-HYBRID']['n']
                     for x in MASKS['SLOT-ONLY'])
    _require(whole['online']['target']['SLOT-ONLY']['authority_reads'] == expected_n,
             'Policy read-count mask is wrong')
    # The fallback stratum contributes its saved READ loss to ANCHOR-ONLY.
    r = _original(ref)['groups']
    expected_loss = sum(r['slots_state/' + x + '/target'][_chosen_source('ANCHOR-ONLY', x)]['loss_sum']
                        for x in STRATA)
    _require(_close(whole['online']['target']['ANCHOR-ONLY']['loss_sum'], expected_loss),
             'Nonauthoritative fallback was replaced')
    before_done = _digest(state)
    _require(step(state, ref, max_units=1) and _digest(state) == before_done,
             'Completed step is not a no-op')
    failures = {}
    bad = copy.deepcopy(ref)
    bad['original_completed_envelope']['sha256'] = '0' * 64
    try:
        init_state(bad)
    except ValueError:
        failures['original_digest_rejection'] = True
    bad = copy.deepcopy(ref)
    bad['final_after_D_probes'].append(copy.deepcopy(bad['final_after_D_probes'][0]))
    try:
        init_state(bad)
    except ValueError:
        failures['duplicate_probe_rejection'] = True
    bad = copy.deepcopy(ref)
    bad['alias2_contexts'][0]['context_hex'] = bytes(range(12)).hex()
    try:
        init_state(bad)
    except ValueError:
        failures['trained_alias2_rejection'] = True
    bad = copy.deepcopy(ref)
    row = bad['original_completed_envelope']['data']['result']['groups']['target']['SUPER']
    row['loss_sum'] += 1
    row['bpb'] = row['loss_sum'] / row['n']
    data = bad['original_completed_envelope']['data']
    bad['original_completed_envelope']['sha256'] = _digest(data)
    try:
        init_state(bad)
    except ValueError:
        failures['stratum_reconstruction_rejection'] = True
    _require(len(failures) == 4, 'A corruption negative control was accepted')
    # Exact and anchor masks are different on seen exact-trace packets, and
    # coincide on held-out packets; no accidental alias-specific mask shortcut.
    seen_exact = [x for x in whole['probes'] if x['stratum'] == 'exact_trace']
    _require(any(x['selected_saved_arm'] == 'READ-HYBRID' for x in seen_exact if x['arm'] == 'EXACT-ONLY') and
             all(x['selected_saved_arm'] == 'SUPER' for x in seen_exact if x['arm'] == 'ANCHOR-ONLY'),
             'Exact-trace mask distinction failed')
    return {'passed': True, 'synthetic_seed': ref['seed'], 'work_units': state['total_units'],
            'checks': ['disjoint_strata_reconstruction', 'all_five_authority_masks',
                       'nonauthoritative_fallback_preserved', 'read_counts',
                       'paired_probe_uniqueness', 'historical_final_overlap',
                       'alias2_independent_raw_scan', 'same_identity_join',
                       'json_checkpoint_resume', 'unchanged_reference',
                       'completed_no_op', 'heldout_exact_anchor_identity'] + list(failures)}
