"""Own 257-way outcomes; counts remain evaluator-only and calibration-selected."""
from pathlib import Path
import time
ROOT=Path(__file__).resolve().parent.parent
def check(): return None
import learner as L
import evaluator as E
import numpy as np,ast,math
from collections import defaultdict
T=L.T
def score(net,records,codes,alternate_records=None):
 before=E.state_hash(net);rows=[];ys=[];bits=[];correct=[];eosp=[];eos_pred=[];truep=[];margins=[]
 alternate=np.array([b for r in alternate_records for b in [*r,256]]) if alternate_records is not None else None
 def flush():
  if not ys:return
  x=T.zeros((len(ys),5177))
  for i,row in enumerate(rows):x[i,T.tensor(row.astype(np.int64))]=1
  logit,_=net.forward(x);lp=logit.double().log_softmax(1);p=lp.exp();y=T.tensor(ys)
  if alternate is not None:
   start=len(bits);other=T.tensor(alternate[start:start+len(y)]);margins.extend(((lp[T.arange(len(y)),y]-lp[T.arange(len(y)),other])/math.log(2)).tolist())
  bits.extend((-lp[T.arange(len(y)),y]/math.log(2)).tolist());correct.extend((logit.argmax(1)==y).tolist());eosp.extend(p[:,256].tolist());eos_pred.extend((logit.argmax(1)==256).tolist());truep.extend(p[T.arange(len(y)),y].tolist());rows.clear();ys.clear()
 for r,code in zip(records,codes):
  assert len(code)==len(r)+1
  for y,row in zip([*r,256],code):
   ys.append(y);rows.append(row)
   if len(ys)==256:flush()
 flush();assert before==E.state_hash(net)
 y=np.array([b for r in records for b in [*r,256]]);b=np.array(bits);p=np.array(eosp);eos=y==256;cal=[]
 for i in range(10):
  mask=(p>=i/10)&(p<(i+1)/10 if i<9 else p<=1);cal.append({'n':int(mask.sum()),'predicted':float(p[mask].mean()) if mask.any() else None,'observed':float(eos[mask].mean()) if mask.any() else None})
 offsets=np.cumsum([0]+[len(r)+1 for r in records]);out={'targets':len(y),'bits':float(b.mean()),'accuracy':float(np.mean(correct)),'payload_bits':float(b[~eos].mean()),'EOS_NLL':float(b[eos].mean()),'EOS_Brier':float(((p-eos)**2).mean()),'premature_EOS_rate':float(np.array(eos_pred)[~eos].mean()),'EOS_calibration':cal,'record_target_counts':[len(r)+1 for r in records],'record_bit_sums':[float(b[a:z].sum()) for a,z in zip(offsets[:-1],offsets[1:])]}
 return out,{'bits':b,'correct':np.array(correct),'truep':np.array(truep),'y':y,'new_old_margin':np.array(margins)}
def corrected_partitions(new,old,cnew,cold,site):
 oy=np.array([b for r in old for b in [*r,256]]);same=np.array([np.array_equal(a,b) for ca,cb in zip(cnew,cold) for a,b in zip(ca,cb)]);changed=site['y']!=oy
 masks={'SAME_KC':~changed&same,'ALTERED_KC':~changed&~same,'CHANGED':changed};out={}
 for k,m in masks.items():out[k]={'n':int(m.sum()),'bits':float(site['bits'][m].mean()) if m.any() else None,'accuracy':float(site['correct'][m].mean()) if m.any() else None,'correct_label_probability':float(site['truep'][m].mean()) if m.any() else None,'new_old_log2_odds':float(site['new_old_margin'][m].mean()) if m.any() and len(site['new_old_margin']) else None}
 return out

def generate(net,sensor,records):
 before=E.state_hash(net);out=[]
 for r in records:
  prefix=r[:3];p=np.zeros(88)
  for b in prefix:p=(p*.95)[sensor.perm]+sensor.R[b]
  result=[];stop=False
  for _ in range(64):
   logits,_=net.forward(T.from_numpy(sensor.encode(p*.95)[None,:]));y=int(logits.argmax(1))
   if y==256:stop=True;break
   result.append(y);p=(p*.95)[sensor.perm]+sensor.R[y]
  complete=bytes(prefix)+bytes(result);feasible=len(r)-len(prefix)+1<=64
  out.append({'prefix_hex':prefix.hex(),'generated_hex':bytes(result).hex(),'own_EOS':stop,'truncated':not stop,'payload_length':len(complete),'model_outputs':len(result)+int(stop),'length_feasible':feasible,'exact_complete':stop and complete==r})
 assert before==E.state_hash(net);return out

def counts(data):
 src=ROOT/'minifly_v179_continuous_local_life_dev_20261102/local/source'
 def defs(p,names):
  tree=ast.parse(p.read_text());return compile(ast.Module(body=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name in names],type_ignores=[]),str(p),'exec')
 env=dict(EOS=256,V=257,np=np,math=math,defaultdict=defaultdict);exec(defs(src/'minifly_v148_pretrain_probe_dev_20261025/ladder.py',['stream','Unigram']),env);exec(defs(src/'minifly_v169_native_dev_v1_20261027/v169_count.py',['IKN','line_bits']),env)
 result={};fitrecords=fittargets=0;t=time.process_time()
 for w,sp in data.items():
  result[w]={}
  for role in ('A','B'):
   tr,cal,held=[sp[role+'_'+k] for k in ('train','calib','held')];best=None;grid=[]
   uni=env['Unigram']().fit(tr);ub=env['line_bits'](uni,held);result[w][role+'_unigram']={'bits':float(ub.sum()/sum(len(r)+1 for r in held)),'record_bit_sums':ub.tolist()};fitrecords+=len(tr);fittargets+=sum(len(r)+1 for r in tr)
   for k in range(1,6):
    for d in (.5,.75,.9):
     m=env['IKN'](k,d).fit(tr);cb=float(env['line_bits'](m,cal).sum()/sum(len(r)+1 for r in cal));grid.append({'order':k,'discount':d,'calibration_bits':cb});fitrecords+=len(tr);fittargets+=sum(len(r)+1 for r in tr)
     if best is None or cb<best[0]:best=(cb,k,d,m)
   cb,k,d,m=best;b=env['line_bits'](m,held);result[w][role+'_IKN']={'order':k,'discount':d,'calibration_bits':cb,'bits':float(b.sum()/sum(len(r)+1 for r in held)),'record_bit_sums':b.tolist(),'calibration_grid':grid};check()
 return {'worlds':result,'fit_records':fitrecords,'fit_targets':fittargets,'CPU_seconds':time.process_time()-t,'never_neural_output':True}

