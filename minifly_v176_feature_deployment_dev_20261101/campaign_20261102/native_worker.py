"""Twelve fixed native lives, only after both source/bridge gates pass."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import sys,json,pickle,gc,hashlib
import runtime as C
from native_checkpoint import NativeModel,Store,O
import numpy as np
from collections import Counter
W=O.N.W;W.H=C.R;W.COUNTS=C.COUNTS;W.TIER='native';W.budget=C.budget;W.cpu=C.usage
old_prob=W.prob
def prob(v):C.COUNTS['probability_evaluations']+=1;return old_prob(v)
W.prob=prob
ACTIVE=None
ARMS=('TRUE-H','FROZEN-H','SHUFFLED-H','RAW','TRUE-H-NOWRITE','FROZEN-H-NOWRITE')
def rng(s):return np.random.default_rng(int.from_bytes(hashlib.sha256(s.encode()).digest()[:16],'big'))
def panel(m,rows,name):
 before=W.digest(m);items=[]
 for j,row in enumerate(rows):
  items.append(O.pair_shared(m,row))
  if j%32==0:C.pulse('NATIVE_PANEL',panel=name,items=j)
 assert W.digest(m)==before
 path=C.R/f'local/panels/{name}.json';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(items)+'\n')
 out={}
 for cohort in sorted(set(r['cohort'] for r in rows)):
  rr=[x for x in items if x['cohort']==cohort];answers={r['id']:r['answer'] for r in rows}
  out[cohort]=dict(items=len(rr),correct=sum(x['correct'] for x in rr),accuracy=sum(x['correct'] for x in rr)/len(rr),payload_NLL=float(np.mean([-x['tf'][0]['lp'] for x in rr])),EOS_NLL=float(np.mean([-x['tf'][1]['lp'] for x in rr])),own_payload_correct=sum(bool(x['output_hex']) and x['output_hex'][:2]==answers[x['id']] for x in rr),tf_payload_correct=sum(x['tf'][0]['argmax']==x['tf'][0]['y'] for x in rr),tf_EOS_correct=sum(x['tf'][1]['argmax']==256 for x in rr),own_EOS=sum(x['EOS'] for x in rr),empty_EOS=sum(x['EOS'] and not x['output_hex'] for x in rr),CAP=sum(x['CAP'] for x in rr),correct_ids=[x['id'] for x in rr if x['correct']])
 return dict(path=str(path.relative_to(C.R)),sha256=C.sha(path),cohorts=out)
def main():
 global ACTIVE
 admission=json.loads((C.R/'ADMISSION.json').read_text());assert admission['native_B_admitted']
 assert 'torch' not in sys.modules
 manifest=json.loads((C.DEV/'local/DATA_MANIFEST.json').read_text());orders=json.loads((C.R/'B_ORDERS.json').read_text());store=Store();allresults=[]
 for pair in ('P1','P2'):
  train=manifest[pair]['rows']['train'];held=manifest[pair]['rows']['held']
  for arm in ARMS:
   source=arm.replace('-NOWRITE','');asset=None
   if source!='RAW':
    a=np.load(C.R/f'local/{pair}_{source}_H.npz');asset=(a['W'],a['b'])
   m=NativeModel(asset);ACTIVE=m;assert m.n==0 and m.records==0
   tag=pair+'_'+arm;metrics={};checkpoints=[];metrics['B0']=panel(m,train+held,tag+'_B0');checkpoints.append(store.save(m,tag+'_B0.pkl'))
   for ep in range(1,9):
    for i in orders[pair][str(ep)]:
     m.train_row(train[i],write=not arm.endswith('NOWRITE'));C.COUNTS['native_prompt_byte_observations']+=len(bytes.fromhex(train[i]['prompt']));C.pulse('NATIVE_TRAINING',pair=pair,arm=arm,pass_index=ep,records=m.records,targets=m.n)
    if ep in (4,8):
     metrics['B'+str(ep)]=panel(m,train+held,tag+'_B'+str(ep));checkpoints.append(store.save(m,tag+f'_B{ep}.pkl'))
   assert m.n==2048 and m.records==1024
   m.rest(128*O.F.DELTA);metrics['REST']=panel(m,train,tag+'_REST');checkpoints.append(store.save(m,tag+'_REST.pkl'))
   # Frozen feature arrays must match the inherited source asset after every life.
   if asset is not None:
    original=np.load(C.R/f'local/{pair}_{source}_H.npz');assert np.array_equal(asset[0],original['W']) and np.array_equal(asset[1],original['b'])
   result=dict(pair=pair,arm=arm,records=1024,targets=2048,native_source_head_absent=True,metrics=metrics)
   C.receipt('NATIVE_LIFE',tag,1024,2048,checkpoints,metrics=result);allresults.append(result);C.save('NATIVE_RESULTS.json',dict(status='RUNNING',results=allresults));ACTIVE=None;del m,asset;gc.collect()
 C.save('NATIVE_RESULTS.json',dict(status='COMPLETE',results=allresults));C.done()
if __name__=='__main__':
 try:main()
 except BaseException as e:
  p=C.R/'local/FAILED_native_state.pkl'
  if ACTIVE is not None:
   with p.open('wb') as f:pickle.dump(ACTIVE,f,protocol=5)
  C.fail(e,str(p) if p.exists() else None);raise
