"""Frozen thresholds and training-only recitation countermodels."""
import json,math
from collections import Counter,defaultdict
import numpy as np
import core as C
def references(w):
 x,y,_=C.data(w,'train',64);hx,hy,_=C.data(w,'held',64);uni=Counter(y.ravel().tolist());mark=defaultdict(Counter);ng=defaultdict(Counter)
 for row,target in zip(x,y):
  for t,v in enumerate(target):
   mark[int(row[t])][int(v)]+=1
   if t>=7:ng[tuple(row[t-7:t+1])][int(v)]+=1
 losses={'UNI':[],'MARKOV1':[],'NGRAM8':[],'GEN_ORACLE':[]};coverage=0
 for row,target in zip(hx,hy):
  for t in range(7,95):
   yy=int(target[t]);key=tuple(row[t-7:t+1]);covered=key in ng;coverage+=covered
   for name,c in [('UNI',uni),('MARKOV1',mark[int(row[t])]),('NGRAM8',ng[key] if covered else uni)]:losses[name].append(-math.log2((c[yy]+1)/(sum(c.values())+257)))
   losses['GEN_ORACLE'].append(-math.log2(57/64 if yy==row[t-7] else 1/64))
 return dict(world=w,lag_bits={k:float(np.mean(v)) for k,v in losses.items()},NGRAM8_coverage=coverage/(64*88),training_reference_records=64,reference_target_updates=64*97,oracle_is_not_learner=True)
def bp_gate():
 rr=json.loads((C.H/'BP_RESULTS.json').read_text())['results'];out=[]
 for r in rr:
  refs=references(r['world']);gain=refs['lag_bits']['UNI']-r['held']['lag_bits'];out.append(dict(world=r['world'],gain_bits=gain,accuracy=r['held']['lag_accuracy'],passed=gain>=.75 and r['held']['lag_accuracy']>=.35,EOS_PROTOCOL_WARNING=r['held']['early_EOS']>.05 or r['held']['EOS_accuracy']<.9,reference=refs))
 C.save('BP_GATE.json',dict(passed=all(r['passed'] for r in out),worlds=out,verdict='REFERENCE_QUALIFIED' if all(r['passed'] for r in out) else 'REFERENCE_NOT_QUALIFIED'));print(json.dumps(out))
if __name__=='__main__':bp_gate()
