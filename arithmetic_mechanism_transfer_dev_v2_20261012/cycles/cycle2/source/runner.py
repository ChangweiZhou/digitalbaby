"""Reserved technical trajectories and authorization-gated scientific jobs."""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from core import MechanismCore, CONFIGS, EOS
from worlds import make_world, OPS, CHECKPOINTS, digest

OUT = Path(__file__).resolve().parent


def working_state(core):
    names = ('_context_len', '_next', '_open', '_generated_end', '_reply_offset',
             '_previous', '_in_reply', '_bank', '_question_total', 'observed_payload_bytes',
             'generated_payload_bytes', 'context_overflow_events', 'elapsed_seconds')
    return {**{key: getattr(core, key) for key in names},
            '_context': core._context.copy(), '_question': bytearray(core._question),
            '_route_h': core._route_h.copy()}


def restore_working(core, state):
    for key, value in state.items():
        setattr(core, key, value)


def probe(core, rows):
    prior = core.learned_digest()
    working = working_state(core)
    answers = []
    for row in rows:
        op, a, b, y = row
        text = f'{a}{op}{b}='.encode()
        target = str(y).encode()
        core.begin_sequence()
        core.observe_bytes(text, learn=False)
        core.start_reply()
        result = core.generate(16)
        answers.append({'prompt_hex': text.hex(), 'target_hex': target.hex(),
                        'payload_hex': result.payload.hex(), 'status': result.status,
                        'selected_events': result.selected_events, 'operation': op,
                        'correct': result.status == 'ENDED' and result.payload == target})
    restore_working(core, working)
    if prior != core.learned_digest():
        raise AssertionError('probe changed persistent learning state')
    return answers


def scores(answers):
    return {op: sum(x['correct'] for x in answers if x['operation'] == op) /
            sum(x['operation'] == op for x in answers) for op in OPS}


def countermodels(plan, prefix, history):
    models = {key: defaultdict(Counter) for key in ('exact', 'modal', 'left', 'right')}
    for rec in plan['records'][:prefix]:
        op, a, b, _ = rec['row']
        target = plan['shuffled_answers'][str(rec['index'])] if history == 'SHUFFLE' else rec['answer']
        for name, key in [('exact',(op,a,b)),('modal',(op,)),('left',(op,a)),('right',(op,b))]:
            models[name][key][target] += 1
    answers = {}
    for name, groups in models.items():
        fitted = {key: min(counts, key=lambda target:(-counts[target],target)) for key, counts in groups.items()}
        hits = {op: [] for op in OPS}
        for op,a,b,y in plan['held_pool']:
            key = {'exact':(op,a,b),'modal':(op,), 'left':(op,a),'right':(op,b)}[name]
            hits[op].append(fitted.get(key)==str(y))
        answers[name] = {op: float(np.mean(v)) for op,v in hits.items()}
    # Family split makes inverse-inclusive exact recitation equally unavailable.
    answers['inverse_family_lookup'] = dict.fromkeys(OPS,0.)
    return answers


def source_lock():
    paths = sorted(OUT.glob('*.py'))
    paths += [OUT.parent/'generic_byte_io_v1_20261012/generic_byte_io/core.py',
              OUT.parent/'arithmetic_mechanism_transfer_design_v2_20261012/audit_design.py',
              OUT.parent/'arithmetic_mechanism_transfer_design_v2_20261012/DESIGN.json',
              OUT.parent/'arithmetic_mechanism_transfer_design_v2_20261012/EXPERIMENT.md']
    return {str(p.relative_to(OUT.parent)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run_job(world, config, path, technical=True, checkpoints=CHECKPOINTS,
            heartbeat_path=None, authorization=None):
    if not technical:
        if authorization is None:
            raise PermissionError('scientific job requires explicit current authorization')
        auth = json.loads(Path(authorization).read_text())
        if auth.get('science_authorized') is not True or auth.get('source_lock') != source_lock():
            raise PermissionError('authorization/source identity mismatch')
    path = Path(path)
    if path.exists():
        raise FileExistsError('immutable full job already exists')
    plan = make_world(world, technical, checkpoints)
    receipt = {'schema': 'ARITHMETIC_TRANSFER_JOB_V2', 'technical': technical,
               'world': world, 'config': config, 'plan': plan, 'source_lock': source_lock(),
               'histories': {}, 'scientific_jobs': 0 if technical else 1}
    wall = time.monotonic()
    with threadpool_limits(limits=1):
        for history in ('W', 'N_ALL', 'SHUFFLE'):
            core = MechanismCore(config, world)
            birth = core.learned_digest()
            birth_fixed = core.birth_digest()
            ledger, checkpoints_out, blocks_out = [], [], []
            cpu = 0.
            seen = {op: set() for op in OPS}
            characters = {op: set() for op in OPS}
            for block in plan['blocks']:
                for index in block['record_indices']:
                    record = plan['records'][index]
                    target = (plan['shuffled_answers'][str(index)] if history == 'SHUFFLE'
                              else record['answer']).encode()
                    text = record['prompt'].encode()
                    allow = history != 'N_ALL'
                    before = core.persistent_components()
                    events = []
                    tick = time.process_time()
                    core.begin_sequence()
                    for byte in text:
                        prior_count = core.learning_events
                        p = core.observe(byte, learn=False)
                        events.append({'event': byte, 'phase': 'prompt', 'prior_choice': p.event,
                                       'actual_update_count': core.learning_events-prior_count})
                    core.start_reply()
                    selected_bank = core._bank
                    for byte in list(target) + [EOS]:
                        prior_count = core.learning_events
                        p = core.observe_end(learn=allow) if byte == EOS else core.observe(byte, learn=allow)
                        events.append({'event': byte, 'phase': 'reply', 'prior_choice': p.event,
                                       'actual_update_count': core.learning_events-prior_count})
                    cpu += time.process_time() - tick
                    after = core.persistent_components()
                    ledger.append({'index': index, 'operation': record['operation'],
                                   'prompt_hex': text.hex(), 'target_hex': target.hex(),
                                   'bank': selected_bank, 'events': events,
                                   'before': before, 'after': after,
                                   'learning_events': core.learning_events,
                                   'bank_counts': core.counts.tolist() if config != 'HASH' else None})
                    seen[record['operation']].add(record['prompt'])
                    characters[record['operation']].update(target)
                    if heartbeat_path and index % 16 == 0:
                        hb = Path(heartbeat_path)
                        temp = hb.with_suffix('.tmp')
                        temp.write_text(json.dumps({'world': world, 'config': config,
                            'history': history, 'partial_records': index+1, 'epoch': time.time()}))
                        os.replace(temp, hb)
                held_answers = probe(core, plan['held_pool'])
                blocks_out.append({'operation': block['operation'], 'checkpoint': block['checkpoint'],
                                   'held': held_answers, 'scores': scores(held_answers), 'cpu': cpu})
                if block['record_indices'][-1] + 1 == 4 * block['checkpoint']:
                    taught_pool = [r for r in plan['training_pool'] if
                        f'{r[1]}{r[0]}{r[2]}=' in seen[r[0]]]
                    taught_answers = probe(core, taught_pool)
                    checkpoints_out.append({'samples_per_operation': block['checkpoint'],
                        'scores': scores(held_answers), 'held': held_answers,
                        'taught': taught_answers, 'cpu': cpu,
                        'countermodels': countermodels(plan, 4*block['checkpoint'], history),
                        'unique_seen_prompts': {op: len(seen[op]) for op in OPS},
                        'observed_answer_bytes': {op: sorted(characters[op]) for op in OPS}})
            before_idle = core.learned_digest()
            core.rest(86400.)
            final = probe(core, plan['held_pool'])
            if before_idle != core.learned_digest():
                raise AssertionError('declared stable bank changed during idle')
            receipt['histories'][history] = {'birth': birth, 'birth_fixed': birth_fixed,
                'ledger': ledger, 'checkpoints': checkpoints_out, 'blocks': blocks_out,
                'final': final, 'learning_events': core.learning_events,
                'final_digest': core.learned_digest(), 'storage': core.storage_accounting(),
                'training_core_cpu': cpu, 'simulated_seconds': core.elapsed_seconds}
    receipt['records_applied'] = 3*len(plan['records'])
    receipt['observed_payload_bytes'] = sum(len(bytes.fromhex(r['prompt_hex'])) +
        len(bytes.fromhex(r['target_hex'])) for h in receipt['histories'].values() for r in h['ledger'])
    receipt['wall_seconds'] = time.monotonic() - wall
    receipt['receipt_digest'] = digest(receipt)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.partial')
    with gzip.open(temp, 'wt') as out:
        json.dump(receipt, out, separators=(',', ':'))
    # Claim path before rename; duplicate attempts must never replace a receipt.
    claim = path.with_suffix('.claim')
    fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    os.replace(temp, path)
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--world', type=int, required=True)
    parser.add_argument('--config', choices=CONFIGS, required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--technical', action='store_true')
    parser.add_argument('--short', action='store_true')
    parser.add_argument('--authorization')
    args = parser.parse_args()
    if not args.technical:
        if not args.authorization:
            parser.error('explicit scientific authorization file required')
        auth = json.loads(Path(args.authorization).read_text())
        if auth.get('science_authorized') is not True or auth.get('source_lock') != source_lock():
            parser.error('missing authorization or frozen source mismatch')
    receipt = run_job(args.world, args.config, args.out, args.technical,
                      (4, 8) if args.short else CHECKPOINTS, authorization=args.authorization)
    print(json.dumps({'config': args.config, 'world': args.world,
                      'records': receipt['records_applied'], 'wall_seconds': receipt['wall_seconds']}))
