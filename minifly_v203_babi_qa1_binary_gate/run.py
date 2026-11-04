"""V203 reproduction of Round 11 QA1 and frozen-state audits."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parent
DATA=ROOT.parent/'minifly-v202'/'data'
DIGESTS={'train':'c64ba9b1a49bbb111ffc4cd2ae4f129d25c012399dd7ade9679eea464b9afb05',
         'test':'55acf66cef2f6d798e2aa1d056e1ec8f24e910ea9215b639450574321132959b'}
CONFIG=dict(STORE='gated',QNOSTORE='1',STEPN='2',MATCHN='3',UNIT='1',SEG='1',
            WIN='32',HABT='0',CREDIT='match',RATE='1.0',OSC='1',RATEP='0.033')
SOURCES=('babi14.py','babi15.py','babi16.py','run.py')

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def run(job):
    mode,seed,epochs,orders,threshold,expected=job
    frozen=mode!='binary_online' and mode!='scaled_validation_online'
    valid='validation' in mode
    binary=mode.startswith('binary')
    env=dict(os.environ,**CONFIG)
    env.update(PYTHONHASHSEED=str(seed),FREEZE=str(int(frozen)),VALID=str(int(valid)),
               BIN=str(int(binary)),EPOCHS=str(epochs),GORD=orders,GTH=str(threshold),
               OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    source='babi16.py' if binary else 'babi15.py' if valid else 'babi14.py'
    code=(ROOT/source).read_text()
    if frozen:
        code='import hashlib\n'+code
        code=code.replace('        if FROZ[0]: TAGS.clear()', '''        if FROZ[0]:
            TAGS.clear()
            def learned_digest():
                h=hashlib.sha256(P.W.tobytes()+G.tobytes())
                for key,value in sorted(A.items()):
                    h.update(repr(key).encode())
                    h.update(value.tobytes())
                return h.hexdigest()
            pre=learned_digest()
            store_pre=hashlib.sha256(M.tobytes()).hexdigest()
            diag=[0,0,0,0,0.]
''')
        needle="gw=float(np.clip(G[gix].mean(),0,1));"
        code=code.replace(needle,needle+''' AUDIT_DIAG(diag, hist, gw) if FROZ[0] else None;''')
        code=code.replace('    acc=np.mean', '''    assert pre==learned_digest()
    store_changed=store_pre!=hashlib.sha256(M.tobytes()).hexdigest()
    assert store_changed == bool(diag[1]+diag[3])
    acc=np.mean''')
        code=code.replace("'n':len(res)", "'frozen_digest':pre,'store_changed':store_changed,'gate_audit':diag,'n':len(res)")
        code='''def AUDIT_DIAG(diag, hist, gate):
    if hist.endswith(b'the '):
        diag[0]+=1
        diag[1]+=int(gate>GTH)
        diag[4]+=gate
    else:
        diag[2]+=1
        diag[3]+=int(gate>GTH)

'''+code
    args=[str(DATA/f'qa1_single-supporting-fact_{s}.txt') for s in ('train','test')]
    result=json.loads(subprocess.check_output([sys.executable,'-B','-c',code,*args],env=env,text=True))
    actual=result['exact_answer_with_own_newline']
    assert result['n']==(200 if valid else 1000)
    result.update(mode=mode,hash_seed=seed,expected_exact=expected,difference=round(actual-expected,6))
    print(mode,seed,epochs,orders,threshold,actual,flush=True)
    return result

if __name__=='__main__':
    for split,digest in DIGESTS.items():
        assert sha((DATA/f'qa1_single-supporting-fact_{split}.txt').read_bytes())==digest
    jobs=[]
    for mode,values in (('scaled_frozen',(.713,.730,.741)),('binary_frozen',(.999,1.,.998)),
                        ('binary_online',(.995,.996,.994))):
        jobs.extend((mode,seed,1,'3,4,5',.15,expected) for seed,expected in enumerate(values,1))
    jobs += [('scaled_validation',1,ep,orders,t,.905)
             for ep,orders,t in itertools.product((1,3),('3,4,5','2,3,4,5,6'),(.15,.30))]
    jobs += [('binary_validation',1,1,'3,4,5',.15,.990),
             ('scaled_validation',1,1,'3,4,5',.90,.165),
             ('scaled_validation_online',1,1,'3,4,5',.15,.900)]
    # Archived scaled-frozen rows printed concurrently and did not carry their seed.
    plan=dict(version='V203',data_sha256=DIGESTS,source_sha256={n:sha((ROOT/n).read_bytes()) for n in SOURCES},
              python=sys.version,config=CONFIG,jobs=jobs)
    (ROOT/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows=list(pool.map(run,jobs))
    exact=all(r['difference']==0 for r in rows if r['mode']!='scaled_frozen')
    scaled=sorted(r['exact_answer_with_own_newline'] for r in rows if r['mode']=='scaled_frozen')
    out=dict(version='V203',replicated=exact and scaled==[.713,.73,.741],rows=rows)
    tmp=ROOT/'RESULTS.tmp'
    tmp.write_text(json.dumps(out,indent=2)+'\n')
    tmp.replace(ROOT/'RESULTS.json')
