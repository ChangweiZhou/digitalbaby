"""V197 factual-memory isolation test using the recovered Round 7 byte schedule."""
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
from statistics import mean, stdev
import time

import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parent
SEEDS = (19701, 19702, 19703, 19704)
CELLS = [(8, 65536, False, True), (8, 65536, True, True)]

def substrate(waves, g4):
    # Only the reviewed stream builder and address functions are loaded.
    source = (ROOT/'round7_recovered.py').read_text()
    code = source[source.index('WORDS='):source.index('def run(')]
    code = code.replace("'/tmp/alice.txt'", "ALICE_PATH")
    env = dict(np=np, os=os, G4=g4, WAVES=waves, PROSE=True, ALICE_PATH=os.environ['V197_ALICE'])
    exec(compile(code, 'round7_stream_and_address', 'exec'), env)
    return env['build'], env['bag'], env['HO']

def run(job):
    seed, waves, units, g4, importance = job
    start = time.perf_counter()
    build, address, heldout = substrate(waves, g4)
    stream, teach, keys, initial, revised, corrections, _, _, checkpoints, facts_per_wave = build(seed, nf=128)
    held_prefixes = {form(k) for form in heldout for k in keys}
    assert all(prefix not in stream for prefix in held_prefixes)
    weights = np.zeros((units, 256), np.float32)
    imp = np.zeros(units)
    encode = lru_cache(maxsize=8192)(lambda context: address(context, units))
    updates = 0
    for index, marked in enumerate(teach):
        if not marked:
            continue
        ctx = stream[max(0, index-16):index]
        if ctx[-7:-3] == b'cmp ':
            continue
        assert 48 <= stream[index] <= 57
        ids, values = encode(ctx)
        error = -(values @ weights[ids])
        error[stream[index]] += 1
        scale = 1/(1+imp[ids]/5) if importance else 1
        weights[ids] += (.6*np.outer(scale*values/(values@values), error)).astype(np.float32)
        imp[ids] += np.abs(values)
        updates += 1
    assert updates == (waves*128+(waves-1)*32)*12
    fixed = set(corrections)
    panels = []
    for wave in range(waves):
        panel = dict(wave=wave, still_valid=0, corrected=0, obsolete=0, valid_n=0, corrected_n=0)
        # Match Round 7's alternating held-out templates within each cohort.
        ids = range(wave*facts_per_wave,(wave+1)*facts_per_wave)
        for corrected in (False, True):
            for j, i in enumerate(i for i in ids if (i in fixed)==corrected):
                addr, values = encode(heldout[j%2](keys[i]))
                choice = int(np.argmax(values@weights[addr]))
                if corrected:
                    panel['corrected_n'] += 1
                    panel['corrected'] += int(choice==revised[i][0])
                    panel['obsolete'] += int(choice==initial[i][0])
                else:
                    panel['valid_n'] += 1
                    panel['still_valid'] += int(choice==initial[i][0])
        panels.append(panel)
    def rate(metric,n):
        return sum(p[metric] for p in panels)/sum(p[n] for p in panels)
    row=dict(seed=seed,waves=waves,units=units,g4=g4,importance=importance,
             old=panels[0]['still_valid']/panels[0]['valid_n'], kept=rate('still_valid','valid_n'),
             corrected=rate('corrected','corrected_n'), obsolete=rate('obsolete','corrected_n'),
             stream_sha256=hashlib.sha256(stream).hexdigest(), bytes=len(stream), updates=updates,
             readout_bytes=weights.nbytes, importance_bytes=imp.nbytes, panels=panels,
             seconds=time.perf_counter()-start)
    name='real-filler-'+'-'.join(map(str,job))
    tmp=ROOT/(name+'.tmp')
    tmp.write_text(json.dumps(row,indent=2)+'\n')
    tmp.replace(ROOT/(name+'.json'))
    print(name,round(row['kept'],4),round(row['old'],4),flush=True)
    return row

def reduce(rows):
    metrics=('kept','old','corrected','obsolete')
    cells=[]
    groups={}
    for cell in CELLS:
        group=[r for r in rows if (r['waves'],r['units'],r['g4'],r['importance'])==cell]
        assert len(group)==len(SEEDS)
        groups[cell]=group
        cells.append(dict(cell=cell,means={m:mean(r[m] for r in group) for m in metrics}))
    contrasts=[]
    for a,b in ((CELLS[1],CELLS[0]),):
        ar,br=groups[a],groups[b]
        assert all(x['seed']==y['seed'] and x['stream_sha256']==y['stream_sha256'] for x,y in zip(ar,br))
        deltas={}
        for m in metrics:
            paired=[x[m]-y[m] for x,y in zip(ar,br)]
            avg=mean(paired)
            half=float(t.ppf(.975,3))*stdev(paired)/2
            deltas[m]=[avg,avg-half,avg+half]
        contrasts.append(dict(a=a,b=b,paired_mean_and_95pct_interval=deltas))
    return dict(cells=cells,contrasts=contrasts)

if __name__=='__main__':
    jobs=[(seed,*cell) for cell in CELLS for seed in SEEDS]
    plan=dict(version='V197',seeds=SEEDS,cells=CELLS,workload='recovered Round 7 real-prose filler stream, isolated factual pathway',source_hashes={
        name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ('filler.py','round7_recovered.py')})
    plan['corpus_sha256']=hashlib.sha256(Path(os.environ['V197_ALICE']).read_bytes()).hexdigest()
    (ROOT/'FILLER_PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    with ProcessPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(run,jobs))
    result=reduce(rows)
    (ROOT/'FILLER_RESULTS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['cells'],indent=2))
