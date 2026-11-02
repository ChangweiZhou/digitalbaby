"""Bounded engineering profile. No natural-text training or scientific checkpoint."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import sys,time,cProfile,pstats,json,resource
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parent
sys.path.insert(0,str(ROOT/'minifly_v176_feature_deployment_dev_20261101'))
import native_stage as N
N.W.H=H
import numpy as np
def main():
 t=time.process_time();wh=N.rng('V176|optimization|H').normal(0,1/np.sqrt(184),(5177,512)).astype(np.float32)
 m=N.Model((wh,np.zeros(512,np.float32)))
 rows=[dict(id=f'engineering:{k}',cohort='engineering',prompt=bytes((i*37+k)%256 for i in range(35)).hex(),answer=bytes([17+k]).hex()) for k in range(3)]
 prof=cProfile.Profile();prof.enable()
 for r in rows:m.train_row(r);N.W.pair(m,r)
 prof.disable();stats=pstats.Stats(prof);entries=[]
 for (path,line,name),(cc,nc,tt,ct,callers) in stats.stats.items():
  entries.append(dict(file=path,line=line,function=name,primitive_calls=cc,calls=nc,self_CPU=tt,cumulative_CPU=ct))
 entries.sort(key=lambda r:r['cumulative_CPU'],reverse=True)
 (H/'PROFILE.json').write_text(json.dumps(dict(CPU=time.process_time()-t,engineering_training_records=3,own_TF_pairs=3,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,functions=entries[:70]),indent=2)+'\n')
 print(json.dumps(entries[:18],indent=2))
if __name__=='__main__':main()
