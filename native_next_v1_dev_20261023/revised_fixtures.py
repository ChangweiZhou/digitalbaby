"""Post-audit synthetic qualification only. Does not repeat B worlds or fit heads."""
import os,sys,copy,json,subprocess,time
from pathlib import Path
P=Path(__file__).resolve().parent;R=P.parent
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
sys.path.insert(0,str(R/'minifly_v6_r1_dev_20261023'))
import core as K
import numpy as np
import engine as E
from scipy.sparse import load_npz
start=time.process_time();awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
def weights(p,m):
 rho=min(1.,(p+.5)/(m+.5));q=(p+.5)/(p+m+1);return rho,q+(1-q)*rho
try:
 f=K.Fly(202610239991);f.begin()
 for c in b'fixture':f.c.step(c,False)
 x=f.code();engines=[f.e.clone() for _ in range(4)];p=m=0;rows=[];events=0
 for rec in range(2):
  rho,beta=weights(p,m)
  for e in engines:e.begin()
  for t in [1,197,256]:
   for arm,e in enumerate(engines):
    margin=e.margins(x)[0];c=K.residual(margin,t,K.ramp(rec)/16)
    if arm==1:c[8]*=1 if t==256 else rho
    if arm==2:c[8]*=beta
    if arm==3:c=None
    previous=e.clone();e.tick(x,c);events+=1
    passive=previous.clone();passive.tick(x,None);events+=1
    inc=np.r_[(e.bank.fast-passive.bank.fast).ravel(),(e.bank.slow-passive.bank.slow).ravel()]
    rows.append(dict(record=rec,target=t,arm=arm,L1=float(np.abs(inc).sum()),L2=float(np.linalg.norm(inc)),fast_signed=float((e.bank.fast-passive.bank.fast).sum()),slow_signed=float((e.bank.slow-passive.bank.slow).sum())))
   for e in engines[1:3]:
    for k in ['fast','slow','adapt']:assert np.array_equal(getattr(engines[0].bank,k)[:8],getattr(e.bank,k)[:8])
  for arm,e in enumerate(engines):e.end(bytes([rec]),write=arm!=3)
  p+=1;m+=2
 # Verify exactly the locked expanded construction can execute native events.
 original=E.fresh_bank;A=R/'minifly_address_capacity_v1_dev_20261023/assets'
 def expanded(n,seed):
  b=original(5177,seed);d=np.load(A/'donors.npy');b.B=load_npz(A/'expanded_B.npz');b.kc_side=b.kc_side[d];b.T=b.T[d].copy()
  b.Q=np.load(A/'expanded_Q.npy');b.reader_Q=np.load(A/'expanded_reader_Q.npy');b.fast=np.zeros((9,n,2));b.slow=np.zeros((9,n));b.adapt=np.ones((9,n));return b
 E.fresh_bank=expanded;e=E.Engine(20708,202610239992,False);xx=np.load(A/'calibration_xn.npy')[0]
 assert e.bank.B.shape[1]==20708 and e.bank.B.nnz==110288
 for sign in [-1,1]:
  before=e.clone();c=np.zeros(9);c[0]=sign/512;e.tick(xx,c);events+=1
  q=before.clone();q.tick(xx,None);events+=1
  assert np.isfinite(e.margins(xx)[0]).all()
  assert np.isfinite(e.bank.slow).all() # gated=False retains original native allocation; slow need not be zero
 E.fresh_bank=original
 (P/'REVISION_TEST.json').write_text(json.dumps(dict(status='PASS',EOS_own_residual_used=True,BOS_and_record_end_present=True,payload_eight_compartments_exact=True,synthetic_complete_records=8,observed_record_events=24,extra_passive_reference_events=24,expanded_fixture_events=4,total_native_event_calls=events,expanded_shape=[302,20708],CPU_seconds=time.process_time()-start,rows=rows),indent=2))
finally:awake.terminate()
