"""Staged V223 main comparison on three fresh paired worlds."""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import copy
import json
import math
import os
from pathlib import Path
import resource
import time

import numpy as np
import torch

from learner import EOS, Learner
from worlds import addition, exposure_digest, fact, make_world, tokens
from screen import OUT, ROOT, put, source_hashes

WORLDS = (66301, 66302, 66303)
DOSES = (4096, 8192, 16384, 32768, 65536)
METHODS = ('forward','bp')
SYMBOL_BOUND = 1226752


def model_for(seed, spec):
    return Learner(seed, spec['width'], spec['topology'], spec['route'])


def train_to(model, stream, stop, method):
    start = time.process_time()
    loss = 0.0
    while model.sites + 32 <= stop:
        loss += model.train(stream[model.sites:model.sites+32],method)
    if not np.isfinite(loss) or not np.isfinite(model.cell).all() or not np.isfinite(model.head).all():
        raise RuntimeError('Nonfinite learner state')
    return time.process_time()-start, loss


def exact_snapshot(model, stream, stop):
    child = copy.deepcopy(model)
    for y in stream[model.sites:stop]:
        child.transition(child.feature(y))
    return child


def evaluate(model, panel):
    before = model.digest()
    rows = []
    for prompt, answer in panel:
        output = model.probe(prompt)
        h, fe = model.h.copy(), model.fe.copy()
        for y in prompt:
            model.transition(model.feature(y))
        losses, ranks = [], []
        for y in (*answer, EOS):
            logits = model.logits()
            logits -= logits.max()
            losses.append(float((np.log(np.exp(logits).sum()) - logits[y])/math.log(2)))
            ranks.append(1 + int((logits > logits[y]).sum()) + int((logits[:y] == logits[y]).sum()))
            model.transition(model.feature(y))
        model.h, model.fe = h, fe
        rows.append({'prompt':prompt.decode('ascii'),'expected':[*answer,EOS],'output':output,
                     'exact':output==[*answer,EOS],'own_eos':output[-1]==EOS,
                     'first_correct':output[0]==answer[0],'answer_bits':losses[:-1],
                     'first_rank':ranks[0],'eos_bits':losses[-1],'eos_rank':ranks[-1]})
    assert before == model.digest()
    return rows


def counts(rows):
    return {'n':len(rows),'exact':sum(r['exact'] for r in rows),'first_correct':sum(r['first_correct'] for r in rows),
            'answer_bits_per_byte':float(np.mean([v for r in rows for v in r['answer_bits']])),
            'eos_top1':sum(r['eos_rank']==1 for r in rows)}


def parsed(prompt):
    return tuple(sorted(map(int,prompt[:-1].split(b'+'))))


def shortcuts(records, world):
    widths, output_by_column = defaultdict(Counter), defaultdict(Counter)
    pairs, labels = defaultdict(list), defaultdict(list)
    global_answers, digit_marginals, all_digits = Counter(), defaultdict(Counter), Counter()
    def mode(counts):
        return min(counts, key=lambda v:(-counts[v],v))
    for prompt, answer in records:
        if b'+' not in prompt:
            continue
        a,b = parsed(prompt)
        width = len(str(a))
        pairs[width].append((a,b))
        labels[width].append(answer)
        widths[width][answer] += 1
        global_answers[answer] += 1
        for i,(x,y,z) in enumerate(zip(str(a)[::-1],str(b)[::-1],answer.decode()[::-1])):
            role = 'units' if i==0 else 'leading' if i==width-1 else 'interior'
            output_by_column[width,role,*sorted((x,y))][z] += 1
            digit_marginals[width,role][z] += 1
            all_digits[z] += 1
    arrays = {k:np.asarray(v,np.int32) for k,v in pairs.items()}
    modes = {k:mode(v) for k,v in widths.items()}
    column = {k:mode(v) for k,v in output_by_column.items()}
    fallback = mode(global_answers)
    digit_modes = {k:mode(v) for k,v in digit_marginals.items()}
    digit_default = mode(all_digits)
    def one(pair):
        a,b = sorted(pair); width = len(str(a))
        true = str(a+b).encode()
        freq = modes.get(width,fallback)
        nearest = labels[width][int(np.abs(arrays[width]-(a,b)).sum(1).argmin())] if width in arrays else fallback
        digits = []
        for i,(x,y) in enumerate(zip(str(a)[::-1],str(b)[::-1])):
            role = 'units' if i==0 else 'leading' if i==width-1 else 'interior'
            digits.append(column.get((width,role,*sorted((x,y))),digit_modes.get((width,role),digit_default)))
        nocarry = (''.join(reversed(digits)).lstrip('0') or '0').encode()
        return {name: value==true for name,value in [('frequency',freq),('nearest',nearest),('no_carry',nocarry)]}
    panels = {name:[one(r[2]) for r in world[name]] for name in ('primary','extrapolation')}
    paired = [tuple(map(one,p)) for p in world['counterfactual']]
    panels['both_correct'] = [{k:a[k] and b[k] for k in a} for a,b in paired]
    return {k:{name:sum(r[name] for r in rows) for name in rows[0]} for k,rows in panels.items()}


def evaluation(model, world, phase, arithmetic):
    panels = {}
    if arithmetic:
        panels = {k:[addition(*r[2]) for r in world[k]] for k in ('primary','extrapolation')}
        panels['taught'] = [addition(*p) for p in world['taught']]
        panels['counterfactual'] = [addition(*pair) for pairset in world['counterfactual'] for pair in pairset]
    original,new,revised = world['assignments']
    panels['old_all' if phase!='A_prime' else 'unchanged'] = [fact(*p) for p in (original if phase!='A_prime' else original[32:])]
    if phase!='A':
        panels['new'] = [fact(*p) for p in new]
    if phase=='A_prime':
        panels['corrected'] = [fact(*p) for p in revised]
    rows = {name:evaluate(model,panel) for name,panel in panels.items()}
    summary = {name:counts(group) for name,group in rows.items()}
    if arithmetic:
        group = rows['counterfactual']
        summary['both_correct'] = {'exact':sum(a['exact'] and b['exact'] for a,b in zip(group[::2],group[1::2],strict=True)), 'n':len(group)//2}
    if phase=='A_prime':
        summary['obsolete'] = {'exact':sum(r['output']==[*fact(*old)[1],EOS] for r,old in zip(rows['corrected'],original[:32],strict=True)), 'n':32}
    return summary,rows


def worker(seed, method, stage, spec):
    torch.set_num_threads(1)
    folder = OUT/f'{seed}_{method}'
    folder.mkdir(exist_ok=True)
    completion = folder/f'{stage}.json'
    if completion.exists():
        return json.loads(completion.read_text())
    world = make_world(seed)
    phases = world['phases']
    stream = tokens([r for phase in phases for r in phase])
    a,b = len(tokens(phases[0])),len(tokens(phases[1]))
    assert len(stream)<=SYMBOL_BOUND
    if stage=='A':
        model = model_for(seed,spec)
        roster = [('A',n,len(tokens(phases[0][:n+n//16])),True) for n in DOSES]
    else:
        model = Learner.load(folder/'A_end.npz')
        roster = [('B',8192,a+b,True)]+[('A_prime',n,a+b+len(tokens(phases[2][:n*32*5])),n==32) for n in (1,4,16,32)]
    rows = []
    began = time.process_time()
    for phase,dose,stop,arithmetic in roster:
        cpu,loss = train_to(model,stream,stop,method)
        child = exact_snapshot(model,stream,stop)
        summary,raw = evaluation(child,world,phase,arithmetic)
        row = {'phase':phase,'dose':dose,'symbols':stop,'committed_symbols':model.sites,'updates':model.updates,
               'training_cpu_this_leg':cpu,'train_loss_nats_this_leg':loss,'metrics':summary}
        rows.append(row)
        put(folder/f'outputs_{phase}_{dose}.json',raw)
        put(folder/'PROGRESS.json',rows)
        print(seed,method,phase,dose,{k:(v['exact'],v['n']) for k,v in summary.items()},flush=True)
    if stage=='A':
        model.save(folder/'A_end.npz')
    else:
        model.save(folder/'final.npz')
    result = {'seed':seed,'method':method,'stage':stage,'rows':rows,'cpu_seconds':time.process_time()-began,
              'peak_rss_bytes_mac':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'stream_sha256':exposure_digest([r for phase in phases for r in phase])}
    if stage=='A' and method=='bp':
        result['arithmetic_shortcuts_at_A_end'] = shortcuts(phases[0],world)
    put(completion,result)
    return result


def profile(spec):
    torch.set_num_threads(1)
    world = make_world(66101)
    stream = tokens(world['phases'][0])[:8192]
    rows = {}
    for method in METHODS:
        model = model_for(66101,spec)
        cpu,_ = train_to(model,stream,len(stream),method)
        start = time.process_time()
        evaluate(model,[addition(*r[2]) for r in world['primary'][:64]])
        probe_cpu = time.process_time()-start
        rows[method] = {'probe_cpu_seconds_64':probe_cpu,'sites':len(stream),'cpu_seconds':cpu,'sites_per_cpu_second':len(stream)/cpu,
                        'array_bytes':sum(a.nbytes for a in model.arrays().values())}
    estimate = 3*SYMBOL_BOUND*sum(1/r['sites_per_cpu_second'] for r in rows.values())
    # Bound the number of scored records per life across all planned checkpoints.
    probe_estimate = 3*10000*sum(r['probe_cpu_seconds_64']/64 for r in rows.values())
    return {'rows':rows,'main_training_cpu_seconds_estimate':estimate,
            'probe_cpu_seconds_estimate':probe_estimate,
            'training_and_probe_cpu_seconds_with_reserve':1.25*(estimate+probe_estimate),
            'training_cpu_seconds_planned_with_reserve':estimate*1.25,
            'single_process_peak_rss_bytes_mac':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def main():
    selection = json.loads((OUT/'SELECTION.json').read_text())
    if selection['status']!='SELECTED':
        put(OUT/'SUMMARY.json',{'status':selection['status']})
        return
    spec = {k:selection[k] for k in ('width','topology','route')}
    lock = OUT/'MAIN_LOCK.json'
    if not lock.exists():
        timing = profile(spec)
        workers = min(6,max(1,os.cpu_count() or 1))
        put(lock,{'worlds':WORLDS,'methods':METHODS,'spec':spec,'doses':DOSES,'workers':workers,
                  'profile':timing,'source':source_hashes(['learner.py','reference.py','official.py','screen.py']),
                  'scope':'Three fresh paired development worlds. Exact forward and BPTT32.',
                  'A_fact_floor': '52/64 in at least 2 of 3 BPTT references'})
    config = json.loads(lock.read_text())
    assert config['source']==source_hashes(['learner.py','reference.py','official.py','screen.py'])
    with ProcessPoolExecutor(max_workers=config['workers']) as pool:
        a = [pool.submit(worker,s,m,'A',spec) for s in WORLDS for m in METHODS]
        phase_a = [f.result() for f in a]
        qualifying = sum(r['rows'][-1]['metrics']['old_all']['exact']>=52 for r in phase_a if r['method']=='bp')
        put(OUT/'A_DECISION.json',{'reference_worlds_passing':qualifying,'continue':qualifying>=2})
        rest = [pool.submit(worker,s,m,'B_Aprime',spec) for s in WORLDS for m in METHODS] if qualifying>=2 else []
        phase_rest = [f.result() for f in rest]
    put(OUT/'SUMMARY.json',{'status':'COMPLETE_LIFETIME' if phase_rest else 'COMPLETE_A_REFERENCE_FLOOR_MISSED',
                            'spec':spec,'screen':selection,'config':config,'A':phase_a,'later':phase_rest})
    print('Completed',qualifying,'reference worlds cleared A',flush=True)


if __name__=='__main__':
    main()
