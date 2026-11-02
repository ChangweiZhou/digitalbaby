"""Native-side exact frozen feature export and head-removal qualification."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import sys,json,pickle,importlib.util
import runtime as C
sys.path.insert(0,str(C.ROOT/'execution_optimization_20261101'));import optimized_native as O
import numpy as np
from native_checkpoint import NativeModel,Store
W=O.N.W;W.H=C.R;W.COUNTS=C.COUNTS;W.budget=C.budget;W.cpu=C.usage;W.TIER='qualification'
old_prob=W.prob
def prob(v):C.COUNTS['probability_evaluations']+=1;return old_prob(v)
W.prob=prob
ACTIVE=None
def hcode(model,idx):
 wh,b=model.asset;idx=np.asarray(idx);idx=idx[idx>=0];h=np.maximum(0,wh[idx].sum(0)+b);x,u=model.rawencode(O.F.BC._norm(model.bridge@h));return h,np.flatnonzero(x).astype(np.int16)
def qualify():
 global ACTIVE
 a=np.load(C.R/'local/qualification_H.npz');asset=(a['W'],a['b']);m=NativeModel(asset);ACTIVE=m
 synthetic=np.load(C.R/'local/qualification_features.npz');actual=np.array([hcode(m,ix)[0] for ix in synthetic['idx']]);np.testing.assert_allclose(actual,synthetic['h'],**C.AUTH['H_float32_formula_tolerance'])
 assert importlib.util.find_spec('torch') is None and 'torch' not in sys.modules
 for rec in (bytes(range(128)),bytes(range(128,256))):
  cached=O.N.sparse_codes(m,rec);view=m.clone();view._control(True);view._control(False)
  for i,idx in enumerate(cached):
   f=view.fe.clone();f.advance(view.time+O.F.DELTA);live=view.encode(f.read())[0];h,ix=hcode(m,idx)
   assert np.array_equal(np.flatnonzero(live),ix)
   if i<len(rec):view._event(rec[i])
 literal=O.N.Model(asset)
 for write in (True,False):
  row=dict(id='driver-qualified:'+str(write),cohort='qualification',prompt='00ff117f',answer='ab')
  literal.train_row(row,write=write);m.train_row(row,write=write)
  for ba,bb in zip(literal.bank,m.bank):
   for k in ('fast','slow','adapt'):assert np.array_equal(getattr(ba.fly.m,k),getattr(bb.fly.m,k))
  assert W.pair(literal,row)==O.pair_shared(m,row)
 store=Store();path=store.save(m,'qualification_native.pkl');restored=store.load(path)
 assert W.digest(m)==W.digest(restored)
 assert all(np.shares_memory(s.fly.m.fast,restored._packed['fast']) for s in restored.bank)
 row=dict(id='restore-next-target',cohort='qualification',prompt='817f',answer='33');reference=literal.clone();reference.train_row(row);restored.train_row(row)
 for a,b in zip(reference.bank,restored.bank):
  for k in ('fast','slow','adapt'):assert np.array_equal(getattr(a.fly.m,k),getattr(b.fly.m,k))
 C.save('NATIVE_DRIVER_QUALIFICATION.json',dict(status='PASS',all_bytes_live_cached_code_parity=True,H_float32_max_error=float(np.max(np.abs(actual-synthetic['h']))),native_output_without_aux_heads=True,torch_not_available=True,write_no_write_parity=True,own_TF_complete_endpoint_parity=True,restore=True))
 C.done()
def export():
 global ACTIVE
 assert json.loads((C.R/'SOURCE_TRAINING.json').read_text())['status']=='COMPLETE';index=[]
 for pair in ('P1','P2'):
  data=np.load(C.DEV/f'local/{pair}_calib.npz');idx=data['idx'];n=len(idx)
  for arm in ('TRUE-H','FROZEN-H','SHUFFLED-H'):
   a=np.load(C.R/f'local/{pair}_{arm}_H.npz');m=O.Model((a['W'],a['b']));ACTIVE=m
   dp=C.R/f'local/{pair}_{arm}_direct.npy';bp=C.R/f'local/{pair}_{arm}_bridged.npy'
   direct=np.lib.format.open_memmap(dp,mode='w+',dtype=np.float32,shape=(n,512));bridged=np.lib.format.open_memmap(bp,mode='w+',dtype=np.int16,shape=(n,184));bridged[:]=-1
   for j,ix in enumerate(idx):
    h,x=hcode(m,ix);direct[j]=h;bridged[j,:len(x)]=x
    if j%512==0:C.pulse('BRIDGE_EXPORT',pair=pair,arm=arm,positions=j)
   direct.flush();bridged.flush();del direct,bridged
   index.append(dict(pair=pair,arm=arm,positions=n,direct_sha256=C.sha(dp),bridged_sha256=C.sha(bp)));C.save('BRIDGE_EXPORT.json',dict(status='RUNNING',files=index));ACTIVE=None
 C.save('BRIDGE_EXPORT.json',dict(status='COMPLETE',files=index));C.done()
if __name__=='__main__':
 try:globals()[sys.argv[1]]()
 except BaseException as e:
  p=C.R/'local/FAILED_bridge_state.pkl'
  if ACTIVE is not None:
   with p.open('wb') as f:pickle.dump(ACTIVE,f,protocol=5)
  C.fail(e,str(p) if p.exists() else None);raise
