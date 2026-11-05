import os
for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[name]='1'
import hashlib,json,resource,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from worlds import world_text
import inner2
ROOT=Path(__file__).resolve().parent
SEEDS=list(range(280101,280109))
def make_life(seed):
 hist=b''
 for i in range(2):d,_=world_text(seed*100+i);hist+=d
 t,ans=world_text(seed*100+99);off=len(hist)
 return hist+t,[(a+off,l) for a,l in ans],off

def job(spec):
 name,seed,z,fast=spec;d,ans,off=make_life(seed);start=time.process_time()
 b,a=inner2.life(np.array(z),d,ans,off,seed=seed,fast=fast,n=256)
 return {'arm':name,'seed':seed,'bpb':float(b),'answer_bits_including_newline':float(a),'cpu_seconds':time.process_time()-start,'bytes':len(d),'scored_bytes':len(d)-off,'answers':len(ans),'stream_sha256':hashlib.sha256(d).hexdigest(),'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}

def main():
 cp=json.loads((ROOT/'RECOVERED_CHECKPOINT.json').read_text());z=cp['mu'];zero=[0.]*15
 plan=[(arm,sd,zz,fast) for sd in SEEDS for arm,zz,fast in [('gen10_fast_on',z,True),('gen10_fast_off',z,False),('initial_fast_on',zero,True)]]
 plan += [(arm,sd,zz,True) for sd in [900001,900002,900003,900004] for arm,zz in [('parity_gen10',z),('parity_initial',zero)]]
 (ROOT/'PLAN.json').write_text(json.dumps({'fresh_seeds':SEEDS,'jobs':plan,'scope':'Saved generation-10 evaluation only. No new ES search or final-genome test.'},indent=2)+'\n')
 start=time.time();results=[]
 with ProcessPoolExecutor(max_workers=2) as pool:
  for r in pool.map(job,plan):
   results.append(r);(ROOT/'RESULTS.json').write_text(json.dumps(results,indent=2)+'\n');print(r['arm'],r['seed'],round(r['bpb'],4),round(r['answer_bits_including_newline'],3),flush=True)
 def ci(v):
  x=np.array(v);rng=np.random.default_rng(280110);b=x[rng.integers(0,len(x),(20000,len(x)))].mean(1);return {'mean':float(x.mean()),'ci95':np.quantile(b,[.025,.975]).tolist()}
 groups={arm:{key:ci([r[key] for r in results if r['arm']==arm]) for key in ['bpb','answer_bits_including_newline']} for arm in sorted({r['arm'] for r in results})}
 paired={key:ci([next(r[key] for r in results if r['seed']==sd and r['arm']=='gen10_fast_off')-next(r[key] for r in results if r['seed']==sd and r['arm']=='gen10_fast_on') for sd in SEEDS]) for key in ['bpb','answer_bits_including_newline']}
 summary={'groups':groups,'fast_write_benefit_off_minus_on':paired,'wall_seconds':time.time()-start,'worker_cpu_seconds':sum(r['cpu_seconds'] for r in results),'jobs_completed':len(results),'numpy':np.__version__,'scope':'Run A generation 10 only; original validation is parity, not fresh evidence. Positive paired effect favors fast writing.','caveats':['Final genomes absent','Original answer spans include newline','Original score floors probability at 1e-6','Fast-off control disables fast readout and recurrent writes from birth, while slow readout remains trainable','No ES rerun, GLN rerun, scale test or autonomous generation']}
 (ROOT/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
