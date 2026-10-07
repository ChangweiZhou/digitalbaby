"""DEV runner only. Predictions are made before observing every byte; phases never enter core."""
import argparse
import json
import time
from pathlib import Path

import bootstrap
import spec
import streams
from core import ObservationCore


def consume(model, raw):
    rows = []
    for b in raw:
        t = model.t + spec.BYTE_SECONDS
        model.predict(t)
        rows.append(model.observe(int(b), t))
    return rows


def metrics(rows, focal):
    r = [row for row in rows if bytes.fromhex(row['context']) in focal]
    return dict(all_byte_bpb=sum(x['loss_bits'] for x in rows) / len(rows),
                all_byte_accuracy=sum(x['byte'] == x['emitted'] for x in rows) / len(rows),
                focal_accuracy=sum(x['byte'] == x['emitted'] for x in r) / len(r),
                focal_bpb=sum(x['loss_bits'] for x in r) / len(r),
                all_byte_count=len(rows), focal_count=len(r))


def probe(model, mapping, seed, erase=False):
    before = model.state_digest()
    clone = model.clone()
    clone.plastic = False
    if erase:
        clone.erase_native_content()
    raw = streams.stream(mapping, spec.PROBE_REPEATS, seed)
    streams.assert_contexts(raw, mapping, clone.history)
    rows = consume(clone, raw)
    if model.state_digest() != before:
        raise AssertionError('probe contaminated training state')
    return dict(raw_hex=raw.hex(), sha256=streams.sha(raw), initial_history=model.history.hex(),
                initial_time=model.t, rows=rows, metrics=metrics(rows, mapping), erase=erase)


def run_world(world, *, compact=False):
    spec.require_dev(world)
    started, cpu = time.monotonic(), time.process_time()
    inputs = streams.world_inputs(world)
    models = {name: ObservationCore(plastic=name == 'W') for name in ('W', 'N_OLD', 'N_ALL')}
    out = dict(world=world, evidence='DEV_ONLY_E0_E1', inputs={name: {k.hex(): v for k, v in m.items()}
               for name, m in inputs.items()}, birth={n: [s.birth_record for s in m.stores]
               for n, m in models.items()}, training={}, probes={}, states={}, training_cost={})
    stages = [('old', inputs['old']), ('day1', None), ('new', inputs['new']),
              ('day2', None), ('revised', inputs['revised'])]
    if compact:
        stages = stages[:1]
    for si, (stage, mapping) in enumerate(stages):
        print(f'DEV {world} {stage}', flush=True)
        if mapping is None:
            for m in models.values():
                m.rest(spec.DAY_SECONDS)
        else:
            raw = streams.stream(mapping, 4 if compact else spec.REPEATS, world + si)
            out['training'][stage] = dict(raw_hex=raw.hex(), sha256=streams.sha(raw), branches={})
            for name, model in models.items():
                model.plastic = name == 'W' or (name == 'N_OLD' and stage != 'old')
                streams.assert_contexts(raw, mapping, model.history)
                start_cpu, start_wall = time.process_time(), time.monotonic()
                rows = consume(model, raw)
                out['training_cost'][f'{stage}/{name}'] = dict(cpu_seconds=time.process_time() - start_cpu,
                    wall_seconds=time.monotonic() - start_wall, actual_bytes=len(raw))
                out['training'][stage]['branches'][name] = rows
        out['probes'][stage] = {}
        for name, model in models.items():
            sets = {'old': inputs['old']}
            if stage in ('new', 'day2', 'revised'):
                sets['new'] = inputs['new']
            if stage == 'revised':
                sets.pop('old')
                sets['current_old'] = {**inputs['old'], **inputs['revised']}
                sets['revised'] = inputs['revised']
                sets['unchanged'] = {k: v for k, v in inputs['old'].items() if k not in inputs['revised']}
            out['probes'][stage][name] = {label: probe(model, mp, world + 100 + si)
                for label, mp in sets.items()}
            if name == 'W':
                erase_label = 'current_old' if stage == 'revised' else 'old'
                erase_mapping = {**inputs['old'], **inputs['revised']} if stage == 'revised' else inputs['old']
                out['probes'][stage]['NATIVE_ERASE'] = {erase_label: probe(model, erase_mapping, world + 100 + si, erase=True)}
            out['states'][f'{stage}/{name}'] = dict(digest=model.state_digest(), clock=model.t,
                bytes_seen=model.bytes_seen, mutable_bytes=model.mutable_bytes(),
                native_elapsed=[float(s.fly.m.elapsed) for s in model.stores],
                elapsed_base=[s.elapsed_base for s in model.stores],
                native_committed_time=[s.brain_t for s in model.stores])
    out['resources'] = dict(cpu_seconds=time.process_time() - cpu, wall_seconds=time.monotonic() - started,
                            training_bytes=sum(len(bytes.fromhex(p['raw_hex'])) for p in out['training'].values()),
                            actual_training_lives=3, readonly_probes_separate=True)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--world', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--compact', action='store_true')
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Refusing to overwrite an existing DEV receipt')
    result = run_world(args.world, compact=args.compact)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, sort_keys=True, allow_nan=False))
    print(json.dumps({stage: {branch: {k: v['metrics'] for k, v in sets.items()}
                    for branch, sets in branches.items()} for stage, branches in result['probes'].items()}, indent=2))


if __name__ == '__main__':
    main()
