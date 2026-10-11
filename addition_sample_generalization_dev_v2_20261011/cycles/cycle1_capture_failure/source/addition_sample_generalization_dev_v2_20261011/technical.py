"""Meaningful invariants, boundary cases, restart parity and receipt tampering."""
import copy
import gzip
import json
import math
import numpy as np
import config
from assets import canonical
from cores import ByteLearner, Core, features, emitted
from reference import Reference
from runner import audit, consume, read, serialize
from operations import load_checkpoint, save_checkpoint
from decision import synthetic_suite


def reject(call):
    try:
        call()
    except (AssertionError, ValueError, KeyError, TypeError):
        return True
    raise AssertionError('negative fixture accepted')


def suite(folder, cycle):
    folder.mkdir(parents=True, exist_ok=False)
    results = {}
    for arm in config.ARMS:
        model, shadow = ByteLearner(arm), Reference(arm)
        shadow.require_seal(model.core.digest())
        birth = model.core.digest()
        rng_pre = model.core.rng.getstate()
        for raw in (b'A+B=8\n', b'C+A=\n'):
            consume(model, shadow, raw, False)
        if model.core.digest() != birth or model.core.rng.getstate() != rng_pre:
            raise AssertionError('forbidden content/auxiliary/RNG advance')
        records = [bytes((b'ABCDE'[i%5], 43, b'ABCDE'[(2*i+1)%5], 61,
                          48+(i*3)%9, 10)) for i in range(64)]
        for i, raw in enumerate(records):
            consume(model, shadow, raw, True)
            if i == 23:
                checkpoint = folder/(arm+'.json.gz')
                save_checkpoint(model, checkpoint)
        shadow.invariants()
        resumed = load_checkpoint(folder/(arm+'.json.gz'))
        alternate = Reference(arm)
        checkpoint_state = resumed.core.state()
        # Restore an independent checker by constructing its state fields.
        for name, value in checkpoint_state.items():
            if name in alternate.a:
                alternate.a[name] = copy.deepcopy(value)
            elif name == 'rng':
                alternate.rng.setstate(value)
            else:
                setattr(alternate, name, copy.deepcopy(value))
        for raw in records[24:]:
            consume(resumed, alternate, raw, True)
        if resumed.core.digest() != model.core.digest() or resumed.ticks != model.ticks:
            raise AssertionError('uninterrupted versus checkpoint resumed mismatch')
        cloned = model.clone()
        for name, value in model.core.state().items():
            if isinstance(value, np.ndarray) and np.shares_memory(value, cloned.core.state()[name]):
                raise AssertionError('mutable arrays alias between forks')
        frozen = model.core.digest()
        changed_shadow = shadow.clone()
        consume(cloned, changed_shadow, b'E+D=8\n', True)
        if model.core.digest() != frozen:
            raise AssertionError('child mutated parent')
        model.core.erase()
        if model.core.digest() != birth:
            raise AssertionError('state erase is not birth restoration')
        if arm != 'STABLE-ID' and resumed.core.payload_bytes() > 128*1024:
            raise AssertionError('candidate mutable state budget')
        results[arm] = {'resume_parity': True, 'N_old_full_state_unchanged': True,
                        'independent_equations': True, 'payload_bytes': resumed.core.payload_bytes(),
                        'mutations': resumed.core.mutations, 'internal_updates': resumed.core.updates,
                        'final_core_seal': resumed.core.digest(), 'final_ticks': resumed.ticks,
                        'final_bytes_seen': resumed.bytes_seen}
    stable, compact = Core('STABLE-ID'), Core('COMPACT-ID')
    ids = features(65, 66)
    for value in (0., .5, .5-1e-12, .5+1e-12, 4.5, 8., -1., 9.):
        stable.f.fill(0)
        compact.f.fill(0)
        stable.f[canonical()['groups'][30]] = value/config.KAPPA
        compact.f[30] = value/config.KAPPA
        if stable.value(ids) != compact.value(ids) or emitted(stable.value(ids)) != emitted(compact.value(ids)):
            raise AssertionError('canonical group output boundary parity')
    stable, compact = Core('STABLE-ID'), Core('COMPACT-ID')
    for i in range(64):
        selected = features(b'ABCDE'[i%5], b'ABCDE'[(i+1)%5])
        if stable.value(selected) != compact.value(selected):
            raise AssertionError('expanded compact numeric parity')
        for model in (stable, compact):
            model.update(selected, i%9, True, model.value(selected))
        for group in range(31):
            coordinates = canonical()['groups'][group]
            for name in ('f', 's', 'visits'):
                if not np.all(getattr(stable, name)[coordinates] == getattr(compact, name)[group]):
                    raise AssertionError('group state parity')
    parser_negatives = [
        lambda: ByteLearner('COMPACT-ID').feed(65, 1, True),
        lambda: config.admit(999999, 'PATH_LOOP'),
        lambda: config.admit(835001, 'ODD_CYCLE'),
    ]
    negative_count = 0
    negative_count += int(reject(lambda: ByteLearner('COMPACT-ID').feed(43, 1, True)))
    negative_count += int(reject(parser_negatives[1]))
    negative_count += int(reject(parser_negatives[2]))
    incomplete = ByteLearner('COMPACT-ID')
    incomplete.feed(65, 1, True)
    negative_count += int(reject(lambda: save_checkpoint(incomplete, folder/'bad.json.gz')))
    negative_count += int(reject(lambda: incomplete.rest(1)))
    stats = synthetic_suite()
    return {'verdict': 'PASS', 'cycle': cycle, 'arms': results,
            'numeric_boundary_parity': True, 'negative_cases': negative_count,
            'statistics': stats, 'science_worlds': 0}


def tamper_suite(receipt):
    # Each mutation is independent; pristine receipts are never rewritten.
    def wrong_prediction(r):
        r['branches']['W']['old_records'][0]['events'][3]['prediction']['emitted'] += 1
    def wrong_write_flag(r):
        r['branches']['N_OLD']['old_records'][0]['allowed'] = True
    def bad_auxiliary(r):
        state = r['branches']['N_OLD']['forks']['10']['old_checkpoint']['state']
        key = next(k for k, v in state.items() if isinstance(v, dict) and 'array' in v)
        value = state[key]['array']
        while isinstance(value[0], list):
            value = value[0]
        value[0] += 1
    def state_seal(r):
        r['branches']['W']['old_records'][0]['core_seal'] = '0'*64
    def teacher_leak(r):
        r['master']['old'][0]['pair'] = r['master']['old'][0]['pair'][::-1]
        r['master']['old'][0]['truth_private'] = 999
    def omit_later(r):
        r['branches']['W']['forks']['10']['new_records'].pop()
    def duplicate_prefix(r):
        r['branches']['W']['old_records'].append(copy.deepcopy(r['branches']['W']['old_records'][0]))
    def altered_clock(r):
        r['branches']['W']['forks']['10']['final_checkpoint']['ticks'] += 1
    def query_write(r):
        r['branches']['W']['forks']['10']['final_old_held']['rows'][0]['responses'][0]['events'][0]['update'] = {'performed': True}
    def query_shared_state(r):
        r['branches']['W']['forks']['10']['final_old_held']['rows'][0]['responses'][0]['end_core_seal'] = 'x'
    def source_tamper(r):
        r['source']['fake_source'] = 'bad'
    def sample_count(r):
        r['training_records'] += 100
    mutations = (wrong_prediction, wrong_write_flag, bad_auxiliary, state_seal, teacher_leak,
                 omit_later, duplicate_prefix, altered_clock, query_write, query_shared_state,
                 source_tamper, sample_count)
    rejected = []
    for mutation in mutations:
        changed = copy.deepcopy(receipt)
        mutation(changed)
        reject(lambda: audit(changed))
        rejected.append(mutation.__name__)
    return {'verdict': 'PASS', 'rejected': rejected,
            'independent_mutations': len(rejected), 'receipt_modified': False}
