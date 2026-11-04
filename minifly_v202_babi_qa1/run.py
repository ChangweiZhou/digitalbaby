"""V202 exact-protocol QA1 reproduction with pinned data and hash seeds."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'data'
DIGESTS={
 'train':'c64ba9b1a49bbb111ffc4cd2ae4f129d25c012399dd7ade9679eea464b9afb05',
 'test':'55acf66cef2f6d798e2aa1d056e1ec8f24e910ea9215b639450574321132959b'}
CONFIG=dict(GORD='3,4,5',QNOSTORE='1',STEPN='2',MATCHN='3',GTH='0.15',
            UNIT='1',SEG='1',WIN='32',HABT='0',CREDIT='match',RATE='1.0',OSC='1',RATEP='0.033')
FILES={split:DATA/f'qa1_single-supporting-fact_{split}.txt' for split in DIGESTS}
ARCHIVE={'gated':[.854,.846,.857],'oracle':[.999,1.,.998],'rand':[.039,.033,.038],
         'all':[0.,0.,0.],'none':[.210,.210,.213]}

def evaluate(job):
    mode,seed,valid,threshold=job
    env=dict(os.environ,**CONFIG)
    env.update(STORE=mode,PYTHONHASHSEED=str(seed),VALID=str(int(valid)),
               GTH=str(threshold),OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    cmd=[sys.executable,'-B',str(ROOT/'babi13.py'),str(FILES['train']),str(FILES['test'])]
    row=json.loads(subprocess.check_output(cmd,env=env,text=True))
    row.update(hash_seed=seed)
    if not valid:
        assert row['n']==1000
        row['archived_exact']=ARCHIVE[mode][seed-1]
        row['difference']=round(row['exact_answer_with_own_newline']-row['archived_exact'],6)
    print(mode,seed,'validation' if valid else 'test',threshold,
          row['exact_answer_with_own_newline'],row['write_frac'],flush=True)
    return row

def shortcut():
    last=None
    correct=total=0
    for line in FILES['test'].read_text().splitlines():
        index,text=line.split(' ',1)
        if index=='1':
            last=None
        if '\t' in text:
            _,answer,_=text.split('\t')
            correct+=int(last==answer)
            total+=1
        else:
            last=re.search(r'(\w+)\.$',text)[1]
    assert total==1000
    return correct/total

if __name__=='__main__':
    for split,path in FILES.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest()==DIGESTS[split]
        assert sum('\t' in line for line in path.read_text().splitlines())==1000
    # The test settings reproduce the memo and do not depend on this rerun's validation scores.
    jobs=[(mode,seed,False,.15) for mode in ARCHIVE for seed in (1,2,3)]
    jobs += [('gated',1,True,t) for t in (.10,.15,.20,.30)]
    plan=dict(version='V202',protocol='online QA1 reproduction',data_sha256=DIGESTS,
              source_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ('babi13.py','run.py')},
              config=CONFIG,roster=jobs,python=sys.version)
    (ROOT/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(evaluate,jobs))
    result=dict(version='V202',last_location_shortcut=shortcut(),
                exact_match_to_archived=all(r['difference']==0 for r in rows if not r['valid']),
                rows=rows)
    tmp=ROOT/'RESULTS.tmp'
    tmp.write_text(json.dumps(result,indent=2)+'\n')
    tmp.replace(ROOT/'RESULTS.json')
