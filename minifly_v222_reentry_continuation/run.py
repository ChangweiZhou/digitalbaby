"""Fixed two-arm development continuation. Reuse the published core and BPTT reference."""
import os
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[variable] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parent
CORE = ROOT.parent / 'minifly-reentry-credit-design'
sys.path.insert(0, str(CORE))
from learner import Learner
from worlds import addition, exposure_digest, fact, make_world, tokens
from probes import diagnose, paired_trace

SEED = 66101
DOSES = (4096, 8192, 16384, 32768, 65536)
REPEAT_DOSES = (1, 4, 16, 64, 256)
ARMS = ('R8', 'R32-BP')


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def save_model(model, path):
    tmp = path.with_suffix('.tmp.npz')
    model.save(tmp)
    tmp.replace(path)


def repeated_history(world):
    panel = [addition(*ab) for ab in world['taught'][:4]]
    panel += [fact(*item) for item in world['assignments'][0][:4]]
    rng = random.Random(6610101)
    records = []
    for _ in range(REPEAT_DOSES[-1]):
        batch = panel.copy()
        rng.shuffle(batch)
        records.extend(batch)
    return records, panel


def layout(study):
    world = make_world(SEED)
    if study == 'curve':
        records = world['phases'][0]
        stops = [len(tokens(records[:n + n // 16])) for n in DOSES]
        panels = {'taught': [addition(*ab) for ab in world['taught']],
                  'held': [addition(*item[2]) for item in world['primary']]}
        doses = DOSES
    else:
        records, panel = repeated_history(world)
        stops = [len(tokens(records[:n * len(panel)])) for n in REPEAT_DOSES]
        panels, doses = {'taught_repeat': panel}, REPEAT_DOSES
    counterfactual = [(addition(*left)[0], addition(*right)[0]) for left, right in world['counterfactual'][:4]]
    return records, doses, stops, panels, counterfactual


def lock():
    runs = {}
    for study in ('curve', 'acquisition'):
        records, doses, stops, panels, _ = layout(study)
        runs[study] = {'doses': list(doses), 'symbol_stops': stops, 'records': len(records),
                       'symbols': stops[-1], 'stream_sha256': exposure_digest(records),
                       'panels': {k: len(v) for k, v in panels.items()}}
    manifest = json.loads((CORE / 'MANIFEST.json').read_text())
    for path, digest in manifest.items():
        assert hashlib.sha256((CORE / path).read_bytes()).hexdigest() == digest, path
    files = ('run.py', 'probes.py')
    value = {'source_commit': '366377ee9d0422abc6ee17e886b7b2c0b4e937a1',
             'core_sha256': manifest, 'continuation_sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files},
             'world_and_birth_seed': SEED, 'arms': list(ARMS), 'probe_modes': ['frozen', 'causal'],
             'study_scope': 'One paired development world. Uniform next-symbol objective. Full A dose curve and independent repeated acquisition.',
             'schedule': runs, 'python': sys.version, 'numpy': np.__version__}
    p = ROOT/'LOCK.json'
    if p.exists():
        if json.loads(p.read_text()) != value:
            raise RuntimeError("Existing lock differs from current source or schedule")
    else:
        write_json(p, value)
    return value


def reduce_rows(rows):
    def mean(items):
        return float(np.mean(items)) if items else None
    return {
        'n': len(rows), 'exact': sum(r['exact'] for r in rows),
        'first_byte_correct': sum(r['first_byte_correct'] for r in rows),
        'mean_first_probability': mean([r['first_probability'] for r in rows]),
        'mean_first_rank': mean([r['first_rank'] for r in rows]),
        'later_answer_bits_per_byte': mean([b for r in rows for b in r['answer_bits'][1:]]),
        'all_answer_bits_per_byte': mean([b for r in rows for b in r['answer_bits']]),
        'operand_bits_per_byte': mean([b for r in rows for b in r['operand_bits']]),
        'delimiter_bits_per_byte': mean([b for r in rows for b in r['delimiter_bits']]),
        'eos_bits': mean([r['eos_bits'] for r in rows]),
        'eos_top1_at_true_boundary': sum(r['eos_rank'] == 1 for r in rows),
        'autonomous_eos': sum(r['own_eos'] for r in rows),
        'immediate_eos': sum(r['immediate_eos'] for r in rows),
        'payload_correct': sum(r['payload_correct'] for r in rows),
    }


def evaluate(model, panels):
    original = model.digest()
    raw, summaries = {}, {}
    for mode in ('frozen', 'causal'):
        raw[mode], summaries[mode] = {}, {}
        for name, panel in panels.items():
            rows = [diagnose(model, p, a, causal=mode == 'causal') for p, a in panel]
            raw[mode][name] = rows
            summaries[mode][name] = reduce_rows(rows)
    assert model.digest() == original
    return raw, summaries


def check():
    world = make_world(SEED)
    model = Learner(SEED)
    for y in tokens(world['phases'][0][:4]):
        model.step(y)
    prompt, ans = addition(*world['taught'][3])
    initial = model.digest()
    for causal in (False, True):
        outcome = diagnose(model, prompt, ans, causal)
        altered = diagnose(model, prompt, bytes([ord('9') if ans[0] != ord('9') else ord('8'), *ans[1:]]), causal)
        assert altered['emitted'] == outcome['emitted']
        if not causal:
            assert outcome['emitted'] == model.probe(prompt)
        assert model.digest() == initial
    alternate = prompt[:2] + bytes([ord('8') if prompt[2] != ord('8') else ord('7')]) + prompt[3:]
    trace = paired_trace(model, prompt, alternate, True)
    assert trace[2]['writer_value'] > 0 and trace[-1]['sampled_kc'] > 0 and trace[-1]['memory'] > 0
    write_json(ROOT/'CHECKS.json', {
        'legacy_frozen_probe_parity': True, 'state_digest_unchanged': True,
        'altering_gold_does_not_change_autonomous_output': True,
        'digit_edit_changes_causal_inputs': trace,
    })


def run(study, arm):
    plan = lock()
    records, doses, stops, panels, pairs = layout(study)
    stream = tokens(records)
    out = ROOT/'results'/study/arm
    out.mkdir(parents=True, exist_ok=True)
    bp = arm == 'R32-BP'
    if bp:
        from reference import train_block
    model = Learner(SEED)
    start_cpu, start_wall = time.process_time(), time.time()
    log = []
    running_loss = 0.0
    for n, stop in zip(doses, stops):
        checkpoint = out/f'{n}.npz'
        receipt = out/f'{n}.json'
        if checkpoint.exists() and receipt.exists():
            row = json.loads(receipt.read_text())
            model = Learner.load(checkpoint)
            assert model.sites == row['committed_symbols']
            assert model.digest() == row['checkpoint_state_sha256']
            running_loss = row['committed_loss_nats']
            log.append(row)
            continue
        t0 = time.process_time()
        if bp:
            while model.sites + 32 <= stop:
                start = model.sites
                running_loss += train_block(model, stream[start:start + 32])
            evaluated = copy.deepcopy(model)
            pending = stream[model.sites:stop]
            partial_loss = train_block(evaluated, pending) if pending else 0.0
        else:
            running_loss += sum(model.step(y) for y in stream[model.sites:stop])
            evaluated, partial_loss = model, 0.0
        training_cpu = time.process_time() - t0
        assert evaluated.sites == stop
        assert np.isfinite(evaluated.theta).all() and np.isfinite(evaluated.head).all()
        save_model(model, checkpoint)
        if evaluated is not model and evaluated.sites != model.sites:
            save_model(evaluated, out/f'{n}_exact_dose.npz')
        print(f'{study} {arm} reached dose {n:,} ({stop:,} symbols), evaluating', flush=True)
        t0 = time.process_time()
        raw, summaries = evaluate(evaluated, panels)
        traces = {mode: [{'left': a.decode(), 'right': b.decode(),
                          'prefix_distances': paired_trace(evaluated, a, b, mode == 'causal')} for a, b in pairs]
                  for mode in ('frozen', 'causal')} if study == 'curve' else {}
        row = {'study': study, 'arm': arm, 'dose': n, 'symbols': stop,
               'committed_symbols': model.sites, 'optimizer_updates': evaluated.updates,
               'committed_loss_nats': running_loss, 'aggregate_train_bits_per_symbol': (running_loss + partial_loss)/(stop*np.log(2)),
               'training_cpu_seconds_this_leg': training_cpu,
               'evaluation_cpu_seconds_this_dose': time.process_time() - t0,
               'checkpoint_state_sha256': model.digest(),
               'scored_state_sha256': evaluated.digest(), 'metrics': summaries}
        write_json(out/f'{n}_outputs.json', {'rows': raw, 'paired_prompt_traces': traces})
        write_json(receipt, row)
        log.append(row)
        write_json(out/'PROGRESS.json', {'completed_doses': [r['dose'] for r in log], 'latest': row})
        print(f'{study} {arm} scored dose {n:,}: ' + ' | '.join(
            f'{m} {p} {r["exact"]}/{r["n"]} first {r["first_byte_correct"]}/{r["n"]}'
            for m, group in summaries.items() for p, r in group.items()), flush=True)
    write_json(out/'COMPLETE.json', {'doses': list(doses), 'worker_cpu_seconds_this_invocation': time.process_time()-start_cpu,
                                    'wall_seconds_this_invocation': time.time()-start_wall})
    print(f'{study} {arm} complete', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('lock','check','curve','acquisition'))
    parser.add_argument('--arm', choices=ARMS)
    args = parser.parse_args()
    if args.mode == 'lock':
        print(json.dumps(lock()['schedule'], indent=2))
    elif args.mode == 'check':
        check()
        print('Probe checks passed')
    elif args.arm:
        run(args.mode, args.arm)
    else:
        parser.error('--arm is required for training')
