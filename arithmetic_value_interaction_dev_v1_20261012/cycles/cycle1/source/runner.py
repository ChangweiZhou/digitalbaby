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
from core import build_core, CONFIGS, EOS
from worlds import make_world, OPS, CHECKPOINTS, digest

OUT = Path(__file__).resolve().parent


def working_state(core):
    names = ('_context_len', '_next', '_open', '_generated_end', '_reply_offset',
             '_previous', '_in_reply', '_bank', '_question_total', 'observed_payload_bytes',
             'generated_payload_bytes', 'context_overflow_events', 'elapsed_seconds')
    state = {key: getattr(core,key) for key in names if hasattr(core,key)}
    state.update(_context=core._context.copy(), _question=bytearray(core._question))
    if hasattr(core,'_num'):
        import copy
        state['_num']=copy.deepcopy(core._num);state['last_event_audit']=copy.deepcopy(core.last_event_audit)
    if hasattr(core,'_route_h'): state['_route_h']=core._route_h.copy()
    return state


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
    inverse=defaultdict(Counter)
    for rec in plan['records'][:prefix]:
        op, a, b, _ = rec['row']
        target = plan['shuffled_answers'][str(rec['index'])] if history == 'SHUFFLE' else rec['answer']
        for name, key in [('exact',(op,a,b)),('modal',(op,)),('left',(op,a)),('right',(op,b))]:
            models[name][key][target] += 1
        # Algebraic recitation uses the OBSERVED answer, not the oracle row[3].
        y=int(target)
        if op=='+':x,z,s=a,b,y;kind='SUM'
        elif op=='-':x,z,s=(b,y,a) if y>=0 else (a,-y,b);kind='SUM'
        elif op=='*':x,z,s=a,b,y;kind='PRODUCT'
        else:x,z,s=b,y,a;kind='PRODUCT'
        derived=[('+' if kind=='SUM' else '*',x,z,s),('+' if kind=='SUM' else '*',z,x,s)]
        if kind=='SUM':derived += [('-',s,x,z),('-',s,z,x),('-',x,s,-z),('-',z,s,-x)]
        else:
            if x:derived.append(('/',s,x,z))
            if z:derived.append(('/',s,z,x))
        for o,left,right,result in set(derived):inverse[o,left,right][str(result)]+=1
    answers = {}
    for name, groups in models.items():
        fitted = {key: min(counts, key=lambda target:(-counts[target],target)) for key, counts in groups.items()}
        hits = {op: [] for op in OPS}
        for op,a,b,y in plan['held_pool']:
            key = {'exact':(op,a,b),'modal':(op,), 'left':(op,a),'right':(op,b)}[name]
            hits[op].append(fitted.get(key)==str(y))
        answers[name] = {op: float(np.mean(v)) for op,v in hits.items()}
    fitted={key:min(counts,key=lambda y:(-counts[y],y)) for key,counts in inverse.items()}
    answers['inverse_family_lookup']={op:float(np.mean([fitted.get((o,a,b))==str(y)
        for o,a,b,y in plan['held_pool'] if o==op])) for op in OPS}
    return answers


def source_lock():
    paths = sorted(OUT.glob('*.py')) + sorted((OUT/'vendor').rglob('*.py'))
    paths += sorted((OUT/'design').glob('*'))
    return {str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths if p.is_file()}


def qualification_ready():
    path=OUT/'QUALIFICATION.json'
    if not path.exists():return False
    q=json.loads(path.read_text())
    return q.get('ready_for_authorized_science') is True and q.get('source_lock')==source_lock()


def run_job(world, config, path, technical=True, checkpoints=CHECKPOINTS,
            heartbeat_path=None, authorization=None):
    if not technical:
        if authorization is None:
            raise PermissionError('scientific job requires explicit current authorization')
        auth = json.loads(Path(authorization).read_text())
        if auth.get('science_authorized') is not True or auth.get('source_lock') != source_lock():
            raise PermissionError('authorization/source identity mismatch')
        if not qualification_ready():
            raise PermissionError('three-cycle qualification/source identity incomplete')
    path = Path(path)
    if path.exists() or path.with_suffix('.claim').exists():
        raise FileExistsError('immutable/claimed full job already exists')
    if config not in CONFIGS:
        raise ValueError('unknown configuration')
    plan = make_world(world, technical, checkpoints)
    receipt = {'schema': 'VALUE_INTERACT_JOB_V1', 'technical': technical,
               'world': world, 'config': config, 'plan': plan, 'source_lock': source_lock(),
               'histories': {}, 'scientific_jobs': 0 if technical else 1}
    wall = time.monotonic()
    job_cpu_started=time.process_time()
    probe_cpu=0.
    def measured_probe(core,rows):
        nonlocal probe_cpu
        started=time.process_time()
        result=probe(core,rows)
        probe_cpu += time.process_time()-started
        return result
    path.parent.mkdir(parents=True,exist_ok=True)
    claim=path.with_suffix('.claim')
    fd=os.open(claim,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    with os.fdopen(fd,'w') as f: json.dump({'pid':os.getpid(),'world':world,'config':config,'state':'STARTED','technical':technical},f)
    with threadpool_limits(limits=1):
        for history in ('W', 'N_ALL', 'SHUFFLE'):
            core = build_core(config, world)
            birth = core.learned_digest()
            birth_fixed = core.birth_digest()
            initial = measured_probe(core, plan['held_pool'])
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
                                       'actual_update_count': core.learning_events-prior_count,
                                       'core_audit': dict(core.last_event_audit) if hasattr(core,'last_event_audit') else None})
                    core.start_reply()
                    selected_bank = getattr(core, '_bank', None)
                    for byte in list(target) + [EOS]:
                        prior_count = core.learning_events
                        p = core.observe_end(learn=allow) if byte == EOS else core.observe(byte, learn=allow)
                        events.append({'event': byte, 'phase': 'reply', 'prior_choice': p.event,
                                       'actual_update_count': core.learning_events-prior_count,
                                       'core_audit': dict(core.last_event_audit) if hasattr(core,'last_event_audit') else None})
                    cpu += time.process_time() - tick
                    after = core.persistent_components()
                    ledger.append({'index': index, 'operation': record['operation'],
                                   'prompt_hex': text.hex(), 'target_hex': target.hex(),
                                   'bank': selected_bank, 'events': events,
                                   'before': before, 'after': after,
                                   'learning_events': core.learning_events,
                                   'bank_counts': core.counts.tolist() if hasattr(core, 'counts') else None,
                                   'optimizer_step': getattr(core,'step',None),
                                   'aux_events': getattr(core,'aux_events',None)})
                    seen[record['operation']].add(record['prompt'])
                    characters[record['operation']].update(target)
                    if heartbeat_path and index % 16 == 0:
                        hb = Path(heartbeat_path)
                        temp = hb.with_suffix('.tmp')
                        temp.write_text(json.dumps({'world': world, 'config': config,
                            'history': history, 'partial_records': index+1, 'epoch': time.time()}))
                        os.replace(temp, hb)
                arrays = core.arrays().values() if hasattr(core,'arrays') else [core._weights,core.V,core.P,core.centers,core.A,core.embedding]
                if not all(np.isfinite(a).all() for a in arrays):
                    raise ArithmeticError('nonfinite actual core state')
                held_answers = measured_probe(core, plan['held_pool'])
                block_taught_pool=[r for r in plan['training_pool'] if f'{r[1]}{r[0]}{r[2]}=' in seen[r[0]]]
                block_taught=measured_probe(core,block_taught_pool)
                blocks_out.append({'operation': block['operation'], 'checkpoint': block['checkpoint'],
                                   'held': held_answers, 'taught':block_taught, 'scores': scores(held_answers), 'cpu': cpu})
                if block['record_indices'][-1] + 1 == 4 * block['checkpoint']:
                    taught_pool = [r for r in plan['training_pool'] if
                        f'{r[1]}{r[0]}{r[2]}=' in seen[r[0]]]
                    taught_answers = block_taught
                    checkpoints_out.append({'samples_per_operation': block['checkpoint'],
                        'scores': scores(held_answers), 'held': held_answers,
                        'taught': taught_answers, 'cpu': cpu,
                        'countermodels': countermodels(plan, 4*block['checkpoint'], history),
                        'unique_seen_prompts': {op: len(seen[op]) for op in OPS},
                        'observed_answer_bytes': {op: sorted(characters[op]) for op in OPS}})
            before_idle = core.learned_digest()
            core.rest(86400.)
            final = measured_probe(core, plan['held_pool'])
            if before_idle != core.learned_digest():
                raise AssertionError('declared stable bank changed during idle')
            receipt['histories'][history] = {'birth': birth, 'birth_fixed': birth_fixed,
                'initial':initial, 'ledger': ledger, 'checkpoints': checkpoints_out, 'blocks': blocks_out,
                'final': final, 'learning_events': core.learning_events,
                'final_digest': core.learned_digest(), 'storage': core.storage_accounting(),
                'training_core_cpu': cpu, 'simulated_seconds': core.elapsed_seconds}
    receipt['records_applied'] = 3*len(plan['records'])
    receipt['observed_payload_bytes'] = sum(len(bytes.fromhex(r['prompt_hex'])) +
        len(bytes.fromhex(r['target_hex'])) for h in receipt['histories'].values() for r in h['ledger'])
    receipt['wall_seconds'] = time.monotonic() - wall
    receipt['resource']={'pre_commit_process_cpu_seconds':time.process_time()-job_cpu_started,
        'probe_cpu_seconds':probe_cpu,
        'training_core_cpu_seconds':sum(h['training_core_cpu'] for h in receipt['histories'].values()),
        'observed_reply_events':sum(len(bytes.fromhex(row['target_hex']))+1
            for h in receipt['histories'].values() for row in h['ledger']),
        'scope':'pre-commit CPU includes state accounting and logging; independent audit/compression measured by worker certificate'}
    receipt['receipt_digest'] = digest(receipt)
    from auditor import audit
    certificate = audit(receipt, check_sources=True)
    temp = path.with_suffix('.partial')
    with gzip.open(temp, 'wt') as out:
        json.dump(receipt, out, separators=(',', ':'))
    with temp.open('rb') as f: os.fsync(f.fileno())
    certificate['receipt_sha256']=hashlib.sha256(temp.read_bytes()).hexdigest()
    certificate['source_lock']=receipt['source_lock']
    from operations import atomic_json
    atomic_json(path.parent.parent/'certificates'/f'{world}_{config}.json',certificate)
    os.replace(temp, path)
    atomic_json(claim, {'pid':os.getpid(),'world':world,'config':config,
                       'state':'ACCEPTED_COMMITTED','technical':technical})
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
