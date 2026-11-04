import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
CORPUS=Path(os.environ['ROUND8_ALICE'])
HASH_SEEDS=(0,1,2)
PREDICTORS=[
    ('pred2.py','prob','0','0.3'), ('pred2.py','prob','0','0.1'),
    ('pred2.py','soft','0.1','0.3'), ('pred2.py','soft','0.2','0.1'),
    ('pred2.py','prob','0','0.05'), ('pred3.py','prob','0','0.1')]

def run(job):
    kind, hash_seed, spec=job
    env=dict(os.environ,PYTHONHASHSEED=str(hash_seed),OPENBLAS_NUM_THREADS='1')
    if kind=='hop':
        cmd=[sys.executable,'-B',str(ROOT/'hop_eval.py'),str(spec)]
    else:
        filename,*args=spec
        code=(ROOT/filename).read_text().replace("'/tmp/alice.txt'",repr(str(CORPUS)))
        if kind=='frozen':
            code=code.replace('W[ix]+=(rate*e/len(ix)).astype(np.float32)',
                'W[ix]+=(rate*e/len(ix)).astype(np.float32) if split=="train" else 0')
        cmd=[sys.executable,'-B','-c',code,*args]
    rows=[json.loads(line) for line in subprocess.check_output(cmd,env=env,text=True).splitlines()]
    for row in rows:
        row.update(hash_seed=hash_seed,kind=kind)
        if kind!='hop':
            row['orders']=7 if spec[0]=='pred3.py' else 5
    return rows

if __name__=='__main__':
    source=(ROOT/'hop.py').read_text()
    source=source.replace("print(json.dumps(run(int(sys.argv[1]))))","")
    source=source.replace("    Q=lambda k: b'tell me, what is '+k+b'? '","    Q=QQ")
    source=source.replace("    ix=np.unique(","    gs+=[hash(b'SUF'+h[-k:])+100*k for k in (1,2) if len(h)>=k]\n    ix=np.unique(")
    source=source.replace('    two=[]','    two=[]\n    strict=[]\n    raw_key=[]\n    eos=[]')
    source=source.replace('mid=gen(Q(f))[:-1]',"generated=gen(Q(f))\n        mid=generated[:-1]")
    needle="        two.append(gen(Q(mid))==dig[s]+b'.')"
    source=source.replace(needle,needle+"""
        complete=generated==s+b'.'
        strict.append(complete and two[-1])
        raw_key.append(generated[:len(s)]==s)
        eos.append(generated.endswith(b'.'))""")
    source=source.replace("return {'seed':seed,", "return {'strict_two_hop':sum(strict)/len(strict),'key_prefix':sum(raw_key)/len(raw_key),'hop1_eos_rate':sum(eos)/len(eos),'seed':seed,")
    source+="""
for name, QQ in (
    ('trained',lambda k:b'the code of '+k+b' is '),
    ('tell_me_what_is',lambda k:b'tell me, what is '+k+b'? '),
    ('remember',lambda k:b'remember '+k+b': ')):
    print(json.dumps(dict(template=name,**run(int(sys.argv[1])))))
"""
    (ROOT/'hop_eval.py').write_text(source)
    jobs=[('online',h,p) for h in HASH_SEEDS for p in PREDICTORS]
    jobs += [('frozen',h,PREDICTORS[1]) for h in HASH_SEEDS]
    jobs += [('hop',h,seed) for h in HASH_SEEDS for seed in (1,2,3)]
    plan=dict(corpus_sha256=hashlib.sha256(CORPUS.read_bytes()).hexdigest(),
              python=sys.version, jobs=jobs, source_sha256={
                  f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
                  for f in ('pred2.py','pred3.py','hop.py','hop_eval.py','reproduce.py')})
    (ROOT/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        rows=[row for result in pool.map(run,jobs) for row in result]
    (ROOT/'RESULTS.json').write_text(json.dumps(rows,indent=2)+'\n')
    print('Completed',len(jobs),'jobs and',len(rows),'scored rows')
