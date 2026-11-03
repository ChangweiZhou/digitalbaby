"""Streamed read-only full-vocabulary evaluation and incremental own-byte rollout."""
import learner as L
import numpy as np,hashlib
T=L.T
def state_hash(net):
 h=hashlib.sha256()
 for d in (net.p,net.m,net.v):
  for v in d.values():h.update(v.numpy().tobytes())
 h.update(net.rng.get_state().numpy().tobytes());h.update(str(net.clock).encode());return h.hexdigest()
def pack(sensor,records):
 return [[np.flatnonzero(c).astype(np.uint16) for c in sensor.codes(r)] for r in records]
def score(net,records,packed):
 before=state_hash(net);rows=[];labels=[];vals=[]
 def flush():
  if not labels:return
  x=T.zeros((len(labels),5177))
  for i,row in enumerate(rows):x[i,T.tensor(row.astype(np.int64))]=1
  logits,_=net.forward(x);lp=logits.double().log_softmax(1);y=T.tensor(labels);p=lp.exp();bits=-lp[T.arange(len(y)),y]/np.log(2);is_eos=y==256
  vals.extend(zip(bits.tolist(),is_eos.tolist(),p[:,256].tolist(),(p.argmax(1)==256).tolist()));rows.clear();labels.clear()
 for r,codes in zip(records,packed):
  for y,row in zip([*r,256],codes):
   rows.append(row);labels.append(y)
   if len(rows)==256:flush()
 flush();a=np.array(vals);assert state_hash(net)==before
 return {'targets':len(a),'bits':float(a[:,0].mean()),'EOS_NLL':float(a[a[:,1]==1,0].mean()),'EOS_Brier':float(((a[:,2]-a[:,1])**2).mean()),'premature_EOS':float(a[a[:,1]==0,3].mean())}
def generate(net,sensor,records):
 before=state_hash(net);out=[]
 for r in records:
  p=np.zeros(88)
  for b in r[:3]:p=(p*.95)[sensor.perm]+sensor.R[b]
  result=[];stop=False
  for j in range(64):
   x=T.from_numpy(sensor.encode(p*.95)[None,:]);logits,_=net.forward(x);y=int(logits.argmax(1))
   if y==256:stop=True;break
   result.append(y);p=(p*.95)[sensor.perm]+sensor.R[y]
  out.append({'own_EOS':stop,'truncated':not stop,'outputs':len(result)+int(stop),'generated_payload_length':len(result),'observed_prefix_length':min(3,len(r))})
 assert state_hash(net)==before
 return out
