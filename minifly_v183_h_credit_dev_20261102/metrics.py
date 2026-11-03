"""Site-reconstructable own-output metrics and log-domain mean forecast."""
import bootstrap as B
from pathlib import Path
import math
T=B.T;np=B.np

def pack(sensor,records):return [[np.flatnonzero(c).astype(np.uint16) for c in sensor.codes(r)] for r in records]
def score(net,records,codes,alternate_records=None,record_log_sums=False):
 before=B.statehash(net);fields={k:[] for k in ('logp_true','logp_EOS','prob_EOS','y','pred_y','record_id','position')};margins=[];logs=[];lengths=[]
 for ri,(r,code) in enumerate(zip(records,codes)):
  x=T.zeros((len(code),5177));y=T.tensor([*r,256]);assert len(code)==len(y)
  for i,row in enumerate(code):x[i,T.tensor(row.astype(np.int64))]=1
  logits,_=net.forward(x);lp=logits.double().log_softmax(1);p=lp.exp();n=len(y)
  values={'logp_true':lp[T.arange(n),y].numpy(),'logp_EOS':lp[:,256].numpy(),'prob_EOS':p[:,256].numpy(),'y':y.numpy().astype(np.uint16),'pred_y':logits.argmax(1).numpy().astype(np.uint16),'record_id':np.full(n,ri,np.uint16),'position':np.arange(n,dtype=np.uint8)}
  for k,v in values.items():fields[k].append(v)
  if alternate_records is not None:
   oy=T.tensor([*alternate_records[ri],256]);margins.extend(((lp[T.arange(n),y]-lp[T.arange(n),oy])/math.log(2)).tolist())
  if record_log_sums:logs.append(T.logsumexp(lp,dim=0).numpy());lengths.append(n)
 sites={k:np.concatenate(v) for k,v in fields.items()};sites['new_old_log2_odds']=np.array(margins);b=-sites['logp_true']/math.log(2);y=sites['y'];eos=y==256;p=sites['prob_EOS']
 out={'targets':len(y),'bits':float(b.mean()),'payload_bits':float(b[~eos].mean()),'accuracy':float((sites['pred_y']==y).mean()),'EOS_NLL':float((-sites['logp_EOS'][eos]/math.log(2)).mean()),'EOS_Brier':float(((p-eos)**2).mean()),'premature_EOS_rate':float((sites['pred_y'][~eos]==256).mean()),'record_bit_sums':[float(b[sites['record_id']==ri].sum()) for ri in range(len(records))],'record_target_counts':[len(r)+1 for r in records]}
 assert B.statehash(net)==before
 if record_log_sums:
  sites['record_class_log_sums']=np.array(logs);sites['record_class_site_counts']=np.array(lengths);sites['record_input_sha256']=np.array([B.hashlib.sha256(r).hexdigest() for r in records]);assert sites['record_class_log_sums'].shape==(len(records),257)
 return out,sites

def partitions(sites,oldrecords,cnew,cold):
 old=np.array([b for r in oldrecords for b in [*r,256]],np.uint16);same=np.array([np.array_equal(a,b) for ra,rb in zip(cnew,cold) for a,b in zip(ra,rb)]);changed=sites['y']!=old;sites['original_y']=old;sites['same_KC']=same;out={}
 for name,m in [('SAME_KC',~changed&same),('ALTERED_KC',~changed&~same),('CHANGED',changed)]:out[name]={'n':int(m.sum()),'bits':float((-sites['logp_true'][m]/math.log(2)).mean()) if m.any() else None,'accuracy':float((sites['pred_y'][m]==sites['y'][m]).mean()) if m.any() else None,'new_old_log2_odds':float(sites['new_old_log2_odds'][m].mean()) if m.any() else None}
 return out

def forecast_from_summaries(sites):
 logs=sites['record_class_log_sums'];counts=sites['record_class_site_counts'];return np.logaddexp.reduce(logs,axis=0)-math.log(int(counts.sum()))
def forecast_loss(logq,held_sites):return float((-logq[held_sites['y']]/math.log(2)).mean())

def generate(net,sensor,records):
 before=B.statehash(net);out=[]
 for r in records:
  prefix=r[:3];p=np.zeros(88)
  for b in prefix:p=(p*.95)[sensor.perm]+sensor.R[b]
  generated=[];stopped=False
  for _ in range(64):
   logits,_=net.forward(T.from_numpy(sensor.encode(p*.95)[None,:]));y=int(logits.argmax(1))
   if y==256:stopped=True;break
   generated.append(y);p=(p*.95)[sensor.perm]+sensor.R[y]
  full=prefix+bytes(generated);feasible=len(r)-len(prefix)+1<=64;out.append({'prefix_hex':prefix.hex(),'generated_hex':bytes(generated).hex(),'own_EOS':stopped,'truncated':not stopped,'payload_length':len(full),'model_outputs':len(generated)+int(stopped),'length_feasible':feasible,'exact_complete':stopped and full==r})
 assert B.statehash(net)==before;return out
