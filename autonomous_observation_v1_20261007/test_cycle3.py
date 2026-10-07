"""Fresh DEV qualification, checkpoint equality and independent adversarial ledger tests."""
import copy
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

import audit
import checkpoint
import spec
from core import ObservationCore
from runner import consume
from streams import world_inputs, stream

ROOT = Path(__file__).parent


def rejected(fn):
    try:
        fn()
    except (ValueError, AssertionError, KeyError):
        return
    raise AssertionError('tamper control accepted')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--out-dir', type=Path, required=True)
    directory = parser.parse_args().out_dir
    directory.mkdir(parents=True, exist_ok=True)
    checks, tamper = [], []
    raw = stream(world_inputs(810201)['old'], 2, 810291)
    uninterrupted = ObservationCore()
    resumed = uninterrupted.clone()
    consume(uninterrupted, raw)
    consume(resumed, raw[:101])  # mid-block, with no externally signaled answer boundary.
    at = resumed.t + spec.BYTE_SECONDS
    pending = resumed.predict(at)
    checkpoint.save(resumed, directory / 'pending.npz')
    restored = checkpoint.load(directory / 'pending.npz')
    assert restored.state_digest() == resumed.state_digest()
    restored.observe(raw[101], at)
    consume(restored, raw[102:])
    assert restored.state_digest() == uninterrupted.state_digest()
    for a, b in zip(restored.stores, uninterrupted.stores):
        for name in ('fast', 'slow', 'adapt'):
            assert np.array_equal(getattr(a.fly.m, name), getattr(b.fly.m, name))
    checks.append('mid-block pending-prediction checkpoint equals uninterrupted native arrays and full digest')
    # Disabled writes are compared to direct native no-plastic events, not trusted from receipt flags.
    nw = restored.clone()
    nw.plastic = False
    at = nw.t + spec.BYTE_SECONDS
    nw.predict(at)
    reference = []
    for s, code in zip(nw.stores, nw.cached['codes']):
        native = s.fly.clone()
        native.event(at - nw.t, code, 0., False)
        reference.append(native)
    nw.observe(48, at)
    for store, native in zip(nw.stores, reference):
        for name in ('fast', 'slow', 'adapt'):
            assert np.array_equal(getattr(store.fly.m, name), getattr(native.m, name))
    checks.append('disabled write arrays equal independent direct-native no-plastic event')
    corrupted = bytearray((directory / 'pending.npz').read_bytes())
    corrupted[-16] ^= 1
    bad = directory / 'bad_byte.npz'
    bad.write_bytes(corrupted)
    bad.with_suffix('.npz.sha256').write_text((directory / 'pending.npz.sha256').read_text())
    rejected(lambda: checkpoint.load(bad))
    tamper.append('checkpoint_byte_changed')
    arrays = checkpoint.payload(resumed)
    meta = json.loads(arrays['metadata'].tobytes())
    meta['source_identity'] = '0' * 64
    arrays['metadata'] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
    bad_source = directory / 'bad_source.npz'
    with bad_source.open('wb') as f:
        np.savez_compressed(f, **arrays)
    bad_source.with_suffix('.npz.sha256').write_text(hashlib.sha256(bad_source.read_bytes()).hexdigest())
    rejected(lambda: checkpoint.load(bad_source))
    tamper.append('checkpoint_wrong_source_with_recomputed_byte_hash')
    prior = json.loads((ROOT / 'cycles/cycle2/WORLD_810101.json').read_text())
    mutations = {
        'future_byte_in_address': lambda r: r['training']['old']['branches']['W'][0].update(context='30'),
        'wrong_actual_byte': lambda r: r['training']['old']['branches']['W'][0].update(byte=48),
        'oracle_sign': lambda r: r['training']['old']['branches']['W'][0].update(signs=[0.,0.,0.,0.]),
        'nan_sign': lambda r: r['training']['old']['branches']['W'][0].update(signs=[float('nan')]*4),
        'dropped_event': lambda r: r['training']['old']['branches']['W'].pop(),
        'wrong_input_hash': lambda r: r['training']['old'].update(sha256='0'*64),
        'wrong_emitted_byte': lambda r: r['probes']['old']['W']['old']['rows'][0].update(
            emitted=spec.ALPHABET[(spec.ALPHABET.index(r['probes']['old']['W']['old']['rows'][0]['emitted']) + 1) % 4]),
        'no_write_actually_writes': lambda r: r['training']['old']['branches']['N_OLD'][0].update(native_write_l1=[1.,0.,0.,0.]),
        'probe_learns': lambda r: r['probes']['old']['W']['old']['rows'][0].update(plastic=True),
        'wrong_reported_accuracy': lambda r: r['probes']['old']['W']['old']['metrics'].update(focal_accuracy=.123),
        'wrong_byte_clock': lambda r: r['training']['old']['branches']['W'][1].update(t=999.),
    }
    for name, mutate in mutations.items():
        bad_receipt = copy.deepcopy(prior)
        mutate(bad_receipt)
        assert json.dumps(bad_receipt, sort_keys=True) != json.dumps(prior, sort_keys=True), 'tamper must really change a field'
        rejected(lambda: audit.audit_receipt(bad_receipt))
        tamper.append(name)
    checks.append('independent event ledger rejects future context, target, sign, write, clock/report tampering')
    # IID has no learnable successor law: an independent test stream must not be mistaken for a failed solvable fixture.
    rng = np.random.default_rng(810290)
    iid = ObservationCore()
    train = bytes(rng.choice(list(spec.ALPHABET), 512).tolist())
    test = bytes(rng.choice(list(spec.ALPHABET), 512).tolist())
    consume(iid, train)
    iid.plastic = False
    rows = consume(iid, test)
    accuracy = sum(r['emitted'] == r['byte'] for r in rows) / len(rows)
    bpb = sum(r['loss_bits'] for r in rows) / len(rows)
    assert .17 <= accuracy <= .33
    checks.append('independent IID stream remains near chance')
    # Bounded exact-context count reference learns from observed bytes only, without reward or task metadata.
    table = defaultdict(lambda: np.zeros(4, dtype=np.int64))
    history = b''
    for b in raw:
        table[history][spec.ALPHABET.index(b)] += 1
        history = (history + bytes([b]))[-4:]
    mapping = world_inputs(810201)['old']
    table_accuracy = sum(spec.ALPHABET[int(np.argmax(table[key]))] == b for key, b in mapping.items()) / len(mapping)
    assert table_accuracy == 1.
    checks.append('observed-context count countermodel solves focal prediction; no E3 claim')
    result = dict(verdict='PASS', checks=checks, rejected_tampers=tamper,
                  checkpoint_digest=restored.state_digest(), native_arrays_exact=True,
                  iid_accuracy=accuracy, iid_bits_per_byte=bpb, context_count_focal_accuracy=table_accuracy,
                  native_mutable_bytes=iid.mutable_bytes(), no_science_trajectories=True)
    (directory / 'TECHNICAL_TEST.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
