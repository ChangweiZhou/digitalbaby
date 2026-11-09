"""Reconstruct the old blocked uniform input stream from pure data functions.

No model, training or historical raw batch transcript. This reference predicts
the exact prompt digest the new observed-target run should produce.
"""
import ast
import hashlib
import json
from random import Random
from e16wr2_common import ROOT,SEEDS,save,sha

def load_data_functions(path,names,ns):
    nodes=[x for x in ast.parse(path.read_text()).body if isinstance(x,ast.FunctionDef) and x.name in names]
    assert {x.name for x in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)

def main():
    e=ROOT/'exp8_e16wr.py';x=ROOT/'exp45.py'
    ns=dict(Random=Random,BATCH=64,EOS=256,SYM={'add':'+','sub':'-','max':'>','min':'<'})
    ns['OP_OF']={v:k for k,v in ns['SYM'].items()}
    load_data_functions(e,['apply','render','split','render_pair','triples'],ns)
    load_data_functions(x,['stream','batches'],ns)
    tasks=[(o,m,bytes([c])) for m,codes in ((97,b'rtxn'),(73,b'suyz')) for o,c in zip(ns['SYM'],codes)]
    held={('+','-'),('-','+'),('>','<'),('<','>')}
    pairs=[(a,b) for a in ns['SYM'].values() for b in ns['SYM'].values() if (a,b) not in held]
    out=[]
    for seed in SEEDS:
        schedule=[(o+str(m),[ns['render'](t,a,b) for a,b in ns['split'](t,.25,seed)[0]],6144) for t in tasks for o,m,_ in [t]]
        schedule.append(('compose',[ns['render_pair'](*p,*abc) for p in pairs for abc in ns['triples'](p,seed,2000)],8192))
        rng=Random(seed+9);buffer=[];h=hashlib.sha256();rows=[]
        for si,(task,items,dose) in enumerate(schedule,1):
            cur=ns['batches'](items,seed+si);seen=set();stage=hashlib.sha256()
            for _ in range(dose):
                fresh=next(cur);fresh=fresh[:32] if buffer else fresh
                replay=rng.sample(buffer,32) if buffer else []
                p=[p for p,t in fresh]+replay
                seen.update(tuple(p) for p,t in fresh)
                raw=json.dumps(p,separators=(',',':')).encode();frame=len(raw).to_bytes(8,'little')+raw
                h.update(frame);stage.update(frame)
            if task!='compose':
                add=[p for p,t in rng.sample(items,1024)]
                assert all(tuple(p) in seen for p in add)
                buffer+=add
            rows.append(dict(stage=si,task=task,updates=dose,prompt_sha256=stage.hexdigest(),distinct_current_prompts=len(seen),eligible_items=len(items)))
        out.append(dict(seed=seed,prompt_sha256=h.hexdigest(),stages=rows))
    save(ROOT/'E16WR2_B_INPUT_REFERENCE.json',dict(status='SOURCE_RECONSTRUCTION_PASS',model_executions=0,
        old_default_source_sha256=sha(e),shared_stream_source_sha256=sha(x),seeds=out,
        boundary='Source-derived expected input hashes, not an attestation from historical E13-A1 batch logs'))
    print('Three complete blocked input streams reconstructed; every sampled legacy admission had been observed.')

if __name__=='__main__':main()
