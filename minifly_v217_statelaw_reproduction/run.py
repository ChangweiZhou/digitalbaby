"""Fresh-seed reproduction of Claude Round 23 and Round 24. No learner tuning."""
import argparse, hashlib, importlib.util, json, os, random, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
HASH_SEED='241105'
ROUND23_SEEDS=list(range(23011,23021))
ROUND24_SEEDS=list(range(24011,24021))

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def generator():
    spec=importlib.util.spec_from_file_location('generator',ROOT/'recovered/miniworld_generator.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def generate():
    G=generator(); data=ROOT/'data';data.mkdir(exist_ok=True); specs=[];manifest=[]
    func={w.lower() for t in G.MOVE_TR+G.MOVE_TE+G.GET+G.DROP for w in t.replace('{N}','').replace('{L}','').replace('{O}','').replace('.','').split()}|{'where','is'}
    def select(r,pool,condition,used):
        for attempt in range(100000):
            w=G.world(r,pool); toks=[x.lower() for x in w[0]+w[1]+w[2]]
            if condition=='A': return w
            unique=len(set(toks))==len(toks) and list(map(len,w[:3]))==[4,6,3]
            if unique and (condition=='B' or not set(toks)&used):
                used.update(toks); return w
        raise RuntimeError('generator rejection limit')
    def render(r,w,nq):
        return [x if kind=='s' else x[0]+' '+x[1] for kind,x in G.stream(r,w,nq)]
    def save(name,lines,worlds):
        path=data/(name+'.txt');path.write_text('\n'.join(lines)+'\n')
        tokens=[[x.lower() for x in w[0]+w[1]+w[2]] for w in worlds]
        seen=set();repeats=0
        for row in tokens: repeats+=len(set(row)&seen);seen.update(row)
        manifest.append({'stream':str(path.relative_to(ROOT)),'sha256':sha(path),'lines':len(lines),
            'questions':sum('?' in s for s in lines),'cross_world_reused_content_types':repeats,
            'worlds':worlds})
        return str(path.relative_to(ROOT))
    def add(name,**kwargs):
        spec={'id':name,**kwargs};sp=ROOT/'specs'/f'{name}.json';sp.parent.mkdir(exist_ok=True);sp.write_text(json.dumps(spec,indent=2)+'\n');specs.append(spec)
    for sd in ROUND23_SEEDS:
        r=random.Random(sd);worlds=[];lines=[]
        for _ in range(25):
            w=G.world(r,G.MOVE_TR);worlds.append(w);lines+=render(r,w,60)
        stream=save(f'r23_{sd}',lines,worlds)
        for mode in ['full','ptronly','snaponly','shufsig','noupd']:
            add(f'r23_{sd}_{mode}',round=23,seed=sd,condition='lifetime',mode=mode,stream=stream,source='recovered/statelaw.py')
    for sd in ROUND24_SEEDS:
        for condition in ['A','B','C']:
            used=set(func) if condition=='C' else set();r=random.Random(100+sd);hist=[];worlds=[]
            for _ in range(24):
                w=select(r,G.MOVE_TR,condition,used);worlds.append(w);hist+=render(r,w,60)
            tw=select(random.Random(600+sd),G.MOVE_TE,condition,used)
            test=render(random.Random(500+sd),tw,80)
            for arm,h in [('k0',[]),('k24',hist)]:
                stream=save(f'r24_{sd}_{condition}_{arm}',h+test,([*worlds,tw] if h else [tw]))
                variants=[('dispersion','recovered/statelaw_t_dispersion.py',1500)] if condition=='A' else ([('dispersion','recovered/statelaw_t_dispersion.py',1500),('span','recovered/statelaw_t.py',1500)] if condition=='B' else [('span','recovered/statelaw_t.py',1500)])
                if condition=='C' and arm=='k24': variants += [('span','recovered/statelaw_t.py',v) for v in [800,3000]]
                for detector,source,span in variants:
                    add(f'r24_{sd}_{condition}_{arm}_{detector}_{span}',round=24,seed=sd,condition=condition,arm=arm,detector=detector,span=span,mode='noprint',stream=stream,source=source,testfrom=len(h))
    (ROOT/'DATA_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (ROOT/'PLAN.json').write_text(json.dumps({'python_hash_seed':HASH_SEED,'round23_seeds':ROUND23_SEEDS,'round24_seeds':ROUND24_SEEDS,'jobs':specs,'note':'Original learner shuffling RNG remains 0. All world seeds are fresh; hash seed pinned.'},indent=2)+'\n')
    return specs

def main():
    p=argparse.ArgumentParser();p.add_argument('--generate-only',action='store_true');p.add_argument('--limit',type=int);a=p.parse_args()
    if os.environ.get('PYTHONHASHSEED')!=HASH_SEED:
        env=dict(os.environ,PYTHONHASHSEED=HASH_SEED);raise SystemExit(subprocess.call([sys.executable,str(Path(__file__).resolve()),*sys.argv[1:]],env=env))
    specs=generate()
    if a.generate_only: print('Generated',len(specs),'jobs');return
    results=ROOT/'results';results.mkdir(exist_ok=True);start=time.time()
    for i,spec in enumerate(specs[:a.limit] if a.limit else specs):
        out=results/(spec['id']+'.json')
        expected={'source_sha256':sha(ROOT/spec['source']),'stream_sha256':sha(ROOT/spec['stream']),'python_hash_seed':HASH_SEED,'python':sys.version}
        valid=False
        if out.exists():
            cached=json.loads(out.read_text());valid=cached.get('spec')==spec and cached.get('receipts')==expected and out.with_suffix('.trace.jsonl').exists()
        if not valid:
            subprocess.run([sys.executable,'-B',str(ROOT/'worker.py'),'--spec',str(ROOT/'specs'/(spec['id']+'.json')),'--out',str(out)],check=True)
        if (i+1)%10==0: print(f'{i+1}/{len(specs)} complete ({time.time()-start:.1f}s)',flush=True)
    print('Finished',flush=True)
if __name__=='__main__': main()
