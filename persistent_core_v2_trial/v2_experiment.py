import dataclasses
import time
import bootstrap
from v2_core import ChoiceOrgan
from v2_fixture import DT, RECORD_SECONDS, permitted
from compact_experiment import instrument


def record(core, event, branch, trace):
    trace.clear()
    began = time.process_time()
    at = event['at']
    for i, byte in enumerate(bytes.fromhex(event['cue_hex'])): core.feed(byte, at + i * DT)
    p = core.predict(at + 12 * DT)
    predicted = time.process_time()
    flag = permitted(branch, event['stage'])
    w = core.observe_outcome(event['outcome'], at + 12 * DT, learn=flag)
    core.feed(10, at + 13 * DT)
    error = core.flush(at + RECORD_SECONDS)
    if len(trace) != 8 or error >= 1e-8: raise AssertionError('store calls/clock')
    return {'index': event['index'], 'stage': event['stage'], 'item': event['item'], 'learn': flag,
            'prediction_precedes_outcome': True, 'predicted_at': at + 12 * DT, 'observed_at': at + 12 * DT,
            'prediction': dataclasses.asdict(p), 'write': w, 'actual_calls': list(trace),
            'flush_error': error, 'cpu_input_predict_s': predicted - began,
            'cpu_observe_flush_s': time.process_time() - predicted}


def probe(core, world, at, name):
    before = core.state_digest(); began = time.process_time(); rows = []
    if world['assay'] == 'lifetime':
        for stage in ('old', 'new', 'revision'):
            if name.startswith('old_') and stage != 'old': continue
            for item, cue in enumerate(world['sets'][stage]):
                c = core.clone()
                for i, byte in enumerate(bytes.fromhex(cue['cue_hex'])): c.feed(byte, at + i * DT)
                p = c.predict(at + 12 * DT)
                rows.append({'stage': stage, 'item': item, 'prediction': dataclasses.asdict(p),
                             'correct': int(p.emitted == cue['outcome'])})
    else:
        for stage in ('old', 'heldout', 'new'):
            if name.startswith('old_') and stage == 'new': continue
            group = world['sets'][stage]
            for pair in range(len(group) // 2):
                # Frozen relation fixture puts outcome 1 then outcome 0.
                a, n = group[2 * pair:2 * pair + 2]
                assert (a['outcome'], n['outcome']) == (49, 48)
                for order in (0, 1):
                    first, second = (a, n) if order == 0 else (n, a)
                    organ = ChoiceOrgan(core)
                    emitted, values, ps = organ.choose(bytes.fromhex(first['cue_hex']), bytes.fromhex(second['cue_hex']), at, DT)
                    correct = ord('L') if order == 0 else ord('R')
                    rows.append({'stage': stage, 'pair': pair, 'order': order, 'first_hex': first['cue_hex'],
                                 'second_hex': second['cue_hex'], 'emitted': emitted, 'target': correct,
                                 'values': values, 'predictions': [dataclasses.asdict(p) for p in ps],
                                 'correct': int(emitted == correct), 'response_at': at + 25 * DT})
    after = core.state_digest()
    if before != after: raise AssertionError('probe changed continuing state')
    return {'name': name, 'at': at, 'rows': rows, 'continuing_state_before': before,
            'continuing_state_after': after, 'cpu_probe_s': time.process_time() - began}
