"""Final DEV qualification with all locked gates; no scientific adoption verdict."""
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).parent


def focal(receipt, stage, branch, label):
    rows = receipt['probes'][stage][branch][label]['rows']
    if label == 'unchanged':
        keys = set(receipt['inputs']['old']) - set(receipt['inputs']['revised'])
    else:
        keys = set(receipt['inputs'][label])
    selected = [r for r in rows if r['context'] in keys]
    return sum(r['emitted'] == r['byte'] for r in selected) / len(selected)


def bpb(receipt, stage, branch, label):
    import math
    rows = receipt['probes'][stage][branch][label]['rows']
    return sum(-math.log2(r['probabilities'][b'0123'.index(r['byte'])]) for r in rows) / len(rows)


def main():
    rows = []
    for world in (810201, 810202, 810203, 810204):
        r = json.loads((ROOT / f'cycles/cycle3/WORLD_{world}.json').read_text())
        audit = json.loads((ROOT / f'cycles/cycle3/AUDIT_{world}.json').read_text())
        if audit['verdict'] != 'PASS':
            raise ValueError('DEV ledger not accepted')
        old = focal(r, 'old', 'W', 'old')
        retained = focal(r, 'day2', 'W', 'old')
        control = focal(r, 'day2', 'N_OLD', 'old')
        rows.append(dict(world=world, old=old, day2_retained=retained,
            day2_N_OLD=control, causal_old_gain=retained-control, retention_delta=retained-old,
            new=focal(r, 'day2', 'W', 'new'), revised=focal(r, 'revised', 'W', 'revised'),
            unchanged=focal(r, 'revised', 'W', 'unchanged'),
            day2_old_all_byte_bpb=bpb(r, 'day2', 'W', 'old'),
            day2_old_all_byte_bpb_gain=bpb(r, 'day2', 'N_ALL', 'old')-bpb(r, 'day2', 'W', 'old'),
            W_training_cpu=sum(v['cpu_seconds'] for k,v in r['training_cost'].items() if k.endswith('/W')),
            total_world_cpu=r['resources']['cpu_seconds'], total_world_wall=r['resources']['wall_seconds'],
            audited_rows=audit['independently_checked_rows']))
    means = {key: statistics.mean(row[key] for row in rows) for key in rows[0] if key != 'world'}
    gates = dict(old_acquisition=means['old'] >= .90,
                 retention=means['retention_delta'] >= -.05,
                 earlier_learning_causal_gain=means['causal_old_gain'] >= .25,
                 new_acquisition=means['new'] >= .80,
                 revision=means['revised'] >= .80,
                 unchanged_preservation=means['unchanged'] >= .85,
                 whole_byte_loss=means['day2_old_all_byte_bpb_gain'] > 0.)
    technical = json.loads((ROOT / 'cycles/cycle3/technical_attempt2/TECHNICAL_TEST.json').read_text())
    source = json.loads((ROOT / 'cycles/cycle3/SOURCE_AT_FULL_RUN.json').read_text())
    changed = [name for name, expected in source.items() if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected]
    if changed:
        raise ValueError('source changed during DEV full run: ' + repr(changed))
    complete = technical['verdict'] == 'PASS' and all(gates.values())
    result = dict(status='DEV_QUALIFIED_SCIENCE_NOT_STARTED' if complete else 'DEV_BEHAVIOR_BLOCKED',
        technical_verdict=technical['verdict'], behavior_verdict='PASS' if all(gates.values()) else 'BLOCKED',
        cycle3_worlds=4, cycle3_completed_training_lives=12,
        cycle3_committed_training_bytes=4*3*1280, science_worlds=0, science_lives=0,
        worlds=rows, means=means, gates=gates, native_mutable_bytes=technical['native_mutable_bytes'],
        source_unchanged=True, evidence='DEV feasibility only; E0/E1, not E3 or formal adoption',
        conservative_32_world_single_worker_wall_minutes=2*max(r['total_world_wall'] for r in rows)*32/60)
    (ROOT / 'QUALIFICATION.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
