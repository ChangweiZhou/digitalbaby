"""Full technical jobs: three histories, shared prefixes, six audited forks."""
import argparse
import copy
import gzip
import hashlib
import json
import math
import os
import resource
import time
from pathlib import Path
import numpy as np
import config
import worlds
from assets import canonical
from cores import ByteLearner, features
from reference import Reference, answer


def serialize(state):
    return {key: {'array': v.tolist(), 'dtype': v.dtype.str, 'shape': list(v.shape)}
            if isinstance(v, np.ndarray) else v for key, v in state.items()}


def deserialize(state):
    return {k: np.array(v['array'], dtype=v['dtype']).reshape(v['shape'])
            if isinstance(v, dict) and 'array' in v else v for k, v in state.items()}


def read(path):
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def atomic_json(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.pending')
    with temporary.open('w') as f:
        json.dump(value, f, sort_keys=True, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def commit(receipt, path):
    path = Path(path)
    if path.exists():
        raise ValueError('already committed job; refusing repeat')
    path.parent.mkdir(parents=True, exist_ok=True)
    provisional = path.with_name(path.name+'.provisional')
    with gzip.open(provisional, 'wt') as f:
        json.dump(receipt, f, sort_keys=True, separators=(',', ':'), allow_nan=False)
    # Independent offline reconstruction gates acceptance of the disk artifact.
    loaded = read(provisional)
    result = audit(loaded, expected_source=receipt['source'])
    if result['verdict'] != 'PASS':
        raise AssertionError('audit failed')
    os.replace(provisional, path)
    return result


def own_ids(pair):
    # Auditor does not call production features.
    return (b'01234ABCDEFGHIJ'.index(pair[0]),
            15+b'01234ABCDEFGHIJ'.index(pair[1]), 30)


def consume(model, reference, raw, allowed):
    ids = own_ids(raw[:1]+raw[2:3])
    predicted = reference.predict(ids)
    events = []
    start_cpu = time.process_time()
    production_cpu = 0.
    for b in raw:
        before = time.process_time()
        event = model.feed(int(b), model.ticks+1, allowed)
        production_cpu += time.process_time()-before
        if b == 61:
            if event['prediction'] != {'value': predicted, 'emitted': answer(predicted)+48}:
                raise AssertionError('independent pre-feedback output mismatch')
        elif event['prediction'] is not None:
            raise AssertionError('wrong output timing')
        if len(events) == 4 and b != 10:
            expected = reference.step(ids, b-48, allowed, predicted)
            if event['update'] != expected:
                raise AssertionError('independent update decision mismatch')
        elif event['update'] is not None:
            raise AssertionError('unexpected update')
        events.append(event)
    if len(raw) == 5:
        model.rest(1)
    actual = model.core.digest()
    reference.require_seal(actual)
    return {'raw': raw.hex(), 'allowed': allowed, 'events': events,
            'core_seal': actual, 'end_ticks': model.ticks,
            'bytes_seen': model.bytes_seen,
            'production_cpu_seconds': production_cpu,
            'record_inclusive_cpu_seconds': time.process_time()-start_cpu}


def snapshot(model, reference):
    reference.require_seal(model.core.digest())
    reference.invariants()
    return {'state': serialize(model.core.state()), 'core_seal': model.core.digest(),
            'ticks': model.ticks, 'bytes_seen': model.bytes_seen,
            'payload_bytes': model.core.payload_bytes(),
            'scalar_ops': model.core.scalar_ops,
            'estimated_algorithmic_bytes_moved': model.core.traffic_bytes}


def probe(model, reference, master, phase, taught, erase=False):
    parent = model.core.digest()
    ref_before = reference.digest()
    rows = []
    for query in worlds.queries(master, phase, taught):
        responses = []
        for pair in query['pairs']:
            child = model.clone()
            shadow = reference.clone()
            if erase:
                child.core.erase()
                shadow = Reference(model.core.arm)
            start = child.core.digest()
            before_bytes = child.bytes_seen
            predicted = shadow.predict(own_ids(pair))
            events = []
            for b in worlds.raw(pair):
                events.append(child.feed(int(b), child.ticks+1, False))
            if child.core.digest() != start or shadow.digest() != start:
                raise AssertionError('probe/erasure changed learned state')
            response = events[3]['prediction']
            if response != {'value': predicted, 'emitted': answer(predicted)+48}:
                raise AssertionError('query does not use operative state')
            if any(row['update'] is not None for row in events):
                raise AssertionError('query received teacher')
            responses.append({'pair': pair, **response, 'events': events,
                              'start_core_seal': start, 'end_core_seal': child.core.digest(),
                              'start_ticks': model.ticks, 'end_ticks': child.ticks,
                              'query_bytes': child.bytes_seen-before_bytes,
                              'exact': response['emitted'] == 48+query['truth_private']})
        rows.append({'orbit': query['orbit'], 'truth_private': query['truth_private'],
                     'responses': responses,
                     'accuracy': sum(r['exact'] for r in responses)/len(responses)})
    if model.core.digest() != parent or reference.digest() != ref_before:
        raise AssertionError('query parent changed')
    return {'rows': rows, 'accuracy': sum(r['accuracy'] for r in rows)/len(rows),
            'parent_core_seal': parent, 'erase': erase}


def run(world, family, noise, ordinary, arm):
    config.admit(world, family)
    start = time.monotonic()
    cpu = time.process_time()
    source = config.lock()
    master = worlds.generate(world, family, noise, ordinary)
    receipt = {'schema': 'SAMPLE_GEN_TECH_JOB_V2', 'technical': True, 'science_worlds': 0,
               'world': world, 'family': family, 'noise': noise, 'ordinary': ordinary,
               'arm': arm, 'master': master, 'source': source, 'branches': {},
               'inherited_source': canonical()['inherited_source'],
               'canonical_birth': canonical()['birth']}
    for history in config.HISTORIES:
        parent = ByteLearner(arm)
        reference = Reference(arm)
        reference.require_seal(parent.core.digest())
        branch = {'birth': snapshot(parent, reference), 'old_records': [], 'forks': {}}
        for i, r in enumerate(master['old'], start=1):
            target = r['shuffled_target'] if history == 'SHUFFLED' else r['target']
            branch['old_records'].append(consume(parent, reference, worlds.raw(r['pair'], target),
                                                history != 'N_OLD'))
            if i in config.CHECKPOINTS:
                before = parent.core.digest()
                child, shadow = parent.clone(), reference.clone()
                fork = {'old_checkpoint': snapshot(parent, reference),
                        'immediate_old_held': probe(parent, reference, master, 'old', False),
                        'new_records': []}
                child.rest(40320)
                for new_record in master['new']:
                    fork['new_records'].append(consume(
                        child, shadow, worlds.raw(new_record['pair'], new_record['target']), True))
                child.rest(40320)
                fork['final_checkpoint'] = snapshot(child, shadow)
                for phase in ('old', 'new'):
                    for taught in (False, True):
                        key = 'final_'+phase+('_taught' if taught else '_held')
                        fork[key] = probe(child, shadow, master, phase, taught)
                fork['erased_old_held'] = probe(child, shadow, master, 'old', False, erase=True)
                if parent.core.digest() != before:
                    raise AssertionError('continuation contaminated shared parent')
                branch['forks'][str(i)] = fork
        receipt['branches'][history] = branch
    receipt.update(training_records=2400,
                   training_bytes=sum(len(bytes.fromhex(row['raw']))
                                      for out in receipt['branches'].values()
                                      for rows in [out['old_records']] +
                                      [f['new_records'] for f in out['forks'].values()]
                                      for row in rows),
                   state_trajectories=21, wall_seconds=time.monotonic()-start,
                   inclusive_cpu_seconds=time.process_time()-cpu,
                   peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    if source != config.lock():
        raise AssertionError('source changed during technical job')
    return receipt


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def audit_record(actual, record, history, phase, reference, ticks, seen):
    target = record['shuffled_target'] if history == 'SHUFFLED' and phase == 'old' else record['target']
    raw = worlds.raw(record['pair'], target)
    allowed = not(history == 'N_OLD' and phase == 'old')
    require(actual['raw'] == raw.hex() and actual['allowed'] is allowed, 'raw byte/permission provenance')
    require(len(actual['events']) == len(raw), 'dropped byte ledger')
    ids = own_ids(record['pair'])
    value = reference.predict(ids)
    for j, event in enumerate(actual['events']):
        expected_prediction = {'value': value, 'emitted': answer(value)+48} if j == 3 else None
        require(event['prediction'] == expected_prediction, 'output before teacher')
        expected_update = reference.step(ids, target, allowed, value) if j == 4 and target is not None else None
        require(event['update'] == expected_update, 'independent transition equation')
    ticks += 6
    seen += len(raw)
    require(actual['end_ticks'] == ticks and actual['bytes_seen'] == seen, 'record clock/bytes')
    require(actual['production_cpu_seconds'] >= 0 and
            actual['record_inclusive_cpu_seconds'] >= actual['production_cpu_seconds'], 'CPU accounting')
    reference.require_seal(actual['core_seal'])
    return ticks, seen


def audit_snapshot(actual, reference, ticks, seen):
    state = deserialize(actual['state'])
    require(set(state) == set(reference.state()), 'complete checkpoint state required')
    # The independent transition seal covers every mutable array and auxiliary,
    # not just a few output weights; no numerical tolerance is used.
    from reference import checksum
    require(checksum(state) == reference.digest() == actual['core_seal'], 'checkpoint operative state')
    require(actual['ticks'] == ticks and actual['bytes_seen'] == seen, 'checkpoint time')
    reference.invariants()


def audit_probe(actual, reference, master, phase, taught, ticks, erase=False):
    shadow = Reference(reference.arm) if erase else reference
    expected_queries = worlds.queries(master, phase, taught)
    require(actual['erase'] is erase and len(actual['rows']) == len(expected_queries), 'query scope')
    require(actual['parent_core_seal'] == reference.digest(), 'probe parent seal')
    scores = []
    for row, q in zip(actual['rows'], expected_queries):
        require(row['orbit'] == q['orbit'] and row['truth_private'] == q['truth_private'], 'query truth/roster')
        require(len(row['responses']) == len(q['pairs']), 'orientation query count')
        scores0 = []
        for r, pair in zip(row['responses'], q['pairs']):
            value = shadow.predict(own_ids(pair))
            correct = answer(value)+48 == 48+q['truth_private']
            require(r['pair'] == pair and r['value'] == value and r['emitted'] == answer(value)+48,
                    'native own emitted answer')
            require(r['exact'] is correct, 'exact score arithmetic')
            require(r['start_core_seal'] == r['end_core_seal'] == shadow.digest(), 'readonly query learned state')
            require(r['start_ticks'] == ticks and r['end_ticks'] == ticks+5 and r['query_bytes'] == 5,
                    'query isolation clock')
            require(len(r['events']) == 5 and all(e['update'] is None for e in r['events']), 'query has no teacher')
            require(r['events'][3]['prediction'] == {'value': value, 'emitted': answer(value)+48},
                    'byte trace/output link')
            require(all(e['prediction'] is None for i, e in enumerate(r['events']) if i != 3),
                    'extra query outputs')
            scores0.append(int(correct))
        score = sum(scores0)/len(scores0)
        require(row['accuracy'] == score, 'orbit equal weighting')
        scores.append(score)
    require(actual['accuracy'] == sum(scores)/len(scores), 'query group accuracy')


def audit(receipt, expected_source=None):
    config.admit(receipt['world'], receipt['family'])
    require(receipt['schema'] == 'SAMPLE_GEN_TECH_JOB_V2' and receipt['technical'] is True
            and receipt['science_worlds'] == 0, 'technical-only admission')
    require(receipt['source'] == (config.lock() if expected_source is None else expected_source), 'source lock')
    master = worlds.generate(receipt['world'], receipt['family'], receipt['noise'], receipt['ordinary'])
    require(receipt['master'] == master, 'immutable generated environment')
    require(receipt['inherited_source'] == canonical()['inherited_source'] and
            receipt['canonical_birth'] == canonical()['birth'], 'canonical native ancestry')
    require(set(receipt['branches']) == set(config.HISTORIES), 'three causal histories')
    rows = byte_count = 0
    for history, branch in receipt['branches'].items():
        reference = Reference(receipt['arm'])
        ticks = seen = 0
        audit_snapshot(branch['birth'], reference, ticks, seen)
        require(len(branch['old_records']) == 320 and
                set(branch['forks']) == set(map(str, config.CHECKPOINTS)), 'whole prefix/continuations')
        for i, (record, actual) in enumerate(zip(master['old'], branch['old_records']), start=1):
            ticks, seen = audit_record(actual, record, history, 'old', reference, ticks, seen)
            rows += 1
            byte_count += len(bytes.fromhex(actual['raw']))
            if history == 'N_OLD':
                require(reference.digest() == Reference(receipt['arm']).digest(), 'all OLD learned state forbidden')
            if str(i) in branch['forks']:
                fork = branch['forks'][str(i)]
                audit_snapshot(fork['old_checkpoint'], reference, ticks, seen)
                audit_probe(fork['immediate_old_held'], reference, master, 'old', False, ticks)
                child = reference.clone()
                child_ticks, child_seen = ticks+40320, seen
                require(len(fork['new_records']) == 80, 'exact NEW external sample count')
                for new_record, new_actual in zip(master['new'], fork['new_records']):
                    child_ticks, child_seen = audit_record(new_actual, new_record, history, 'new',
                                                         child, child_ticks, child_seen)
                    rows += 1
                    byte_count += len(bytes.fromhex(new_actual['raw']))
                child_ticks += 40320
                audit_snapshot(fork['final_checkpoint'], child, child_ticks, child_seen)
                for phase in ('old', 'new'):
                    for taught in (False, True):
                        name = 'final_'+phase+('_taught' if taught else '_held')
                        audit_probe(fork[name], child, master, phase, taught, child_ticks)
                audit_probe(fork['erased_old_held'], child, master, 'old', False, child_ticks, erase=True)
    require(receipt['training_records'] == rows == 2400 and receipt['training_bytes'] == byte_count,
            'committed exposure totals')
    require(receipt['state_trajectories'] == 21, 'trajectory definition')
    return {'verdict': 'PASS', 'technical': True, 'science_worlds': 0,
            'independent_record_transitions': rows, 'training_bytes': byte_count,
            'state_trajectories_with_shared_prefix': 21, 'exact_state_seals': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--world', type=int, required=True)
    parser.add_argument('--family', choices=['PATH_LOOP', 'STAR_LOOP'], required=True)
    parser.add_argument('--noise', choices=['STANDARD', 'BLURRY'], required=True)
    parser.add_argument('--ordinary', action='store_true')
    parser.add_argument('--arm', choices=config.ARMS, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.world, args.family, args.noise, args.ordinary, args.arm)
    print(json.dumps(commit(result, args.output), sort_keys=True))
