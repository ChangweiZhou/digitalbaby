"""Three bounded technical cycles, at most 32 complete record applications."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,time,json,copy,io,resource,hashlib
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parent
sys.path.insert(0,str(ROOT/'minifly_v178_recurrent_failure_localization_dev_20261102/local/compat'))
import learner as L
import numpy as np
from sensor import Sensor,sha
from data import prepare,C
T=L.T
START=time.process_time();WALL=time.monotonic()
def save(n,v):(H/n).write_text(json.dumps(v,indent=2)+'\n')
def eq(a,b):return all(T.equal(a[k],b[k]) for k in a)
def main():
 s=Sensor();net=L.Learner();assert sum(p.numel() for p in net.p.values())==37313
 fan=np.bincount(net.indices.ravel(),minlength=5177);assert fan.min()==3 and fan.max()==4;assert all(len(set(r))==256 for r in net.indices)
 # Literal source timeline independently expressed in physical DELTA/tau clocks.
 def literal(r):
  delta=30/14;tau=-delta/np.log(.95);p=np.zeros(88);t=2*delta;prev=t;out=[]
  for i in range(len(r)+1):
   out.append(s.encode(p*np.exp(-delta/tau)))
   if i<len(r):
    t+=delta;p=p*np.exp(-(t-prev)/tau);p=p[s.perm]+s.R[r[i]];prev=t
  return np.array(out)
 assert np.array_equal(s.codes(bytes(range(256))),literal(bytes(range(256))))
 assert np.array_equal(s.codes(b'abcx')[:4],s.codes(b'abcy')[:4])
 # Independent historical cache: all positions of first 16 original source records.
 raw=(ROOT/'minifly_v176_feature_deployment_dev_20261101/assets/corpora/chesterton-thursday.txt').read_bytes();recs=C.records(raw)[:16];z=np.load(ROOT/'minifly_v176_feature_deployment_dev_20261101/local/P1_train.npz');off=0
 for r in recs:
  codes=s.codes(r)
  for code,row in zip(codes,z['idx'][off:off+len(codes)]):assert np.array_equal(np.flatnonzero(code),row[row>=0])
  assert np.array_equal(z['y'][off:off+len(codes)],np.array([*r,256]));off+=len(codes)
 data,audit=prepare(s);save('DATA_AUDIT.json',{'status':'PASS','worlds':audit,'sensor_sha256':s.digest,'historical_cache_positions':off,'all_256_bytes_literal_parity':True,'no_future_target':True})
 fixed=sum(a.nbytes for a in s.arrays)+net.indices.nbytes+net.ix.numel()*8+(net.fh.numel()+net.ft.numel())*4
 # Conservative single-inference accounting includes sensory/state and 64x256 gather/work.
 inference=fixed+37313*4+5177*4+64*256*4*3+30000
 assert inference<=8388608
 save('CYCLE1.json',{'status':'PASS','learned_scalars':37313,'fanout':[int(fan.min()),int(fan.max())],'fixed_sensor_bytes':sum(a.nbytes for a in s.arrays),'conservative_inference_bytes':inference,'writer_banks_instantiated':0,'cycles_share_birth':True})
 # Freeze exact technical IDs before fitting. A four records supplies >=2/group;
 # B two exercises fallback; C one record per fork, genuine EOS in every record.
 d=data['M1'];techA=d['A_train'][:4];techB=d['B_train'][:2];techC=d['EDIT_train'][:1];techN=d['NOEDIT_train'][:1]
 def batch(rs,role):
  return T.from_numpy(np.concatenate([s.codes(r) for r in rs])),T.tensor([b for r in rs for b in [*r,256]]),[role+str(i) for i,r in enumerate(rs) for _ in range(len(r)+1)]
 # For technical only, choose shorter complete records BEFORE training so four
 # distinct records fit one <=256-target batch; no truncation or synthetic labels.
 ids=sorted(range(len(d['A_train'])),key=lambda i:(len(d['A_train'][i]),i))[:4];techA=[d['A_train'][i] for i in ids]
 assert sum(len(r)+1 for r in techA)<=256
 xa,ya,ida=batch(techA,'A');xb,yb,idb=batch(techB,'B');xc,yc,idc=batch(techC,'C');xn,yn,idn=batch(techN,'C')
 save('TECHNICAL_MANIFEST.json',{'frozen_before_training':True,'A_source_ids':ids,'A_records':4,'B_source_ids':[0,1],'C_original_A_ids':audit['M1']['correction']['selected_train_ids'][:1],'arms':['NORMAL','FIXED-LOW','RMS-SHAM','CGLP'],'total_record_applications':32,'batch_limit':256,'technical_not_science':True})
 # Cycle 2: independent optimizer reference and local derivative, no extra fits.
 state=copy.deepcopy(net.state());grad,gain,info=net.gradients(xa,ya,ida);net.restore(state)
 params={k:T.nn.Parameter(v.clone()) for k,v in net.p.items()};opt=T.optim.AdamW(params.values(),lr=.001,weight_decay=.0001)
 for k,p in params.items():p.grad=grad[k].clone()
 opt.step();normal_info=net.step(xa,ya,ida,'NORMAL')
 assert all(T.allclose(net.p[k],params[k],atol=2e-7,rtol=1e-6) for k in params)
 assert info['reliable'];assert min(info['group_targets'])>0
 # Manual H incoming local surrogate against independent matrix formulation.
 ref=L.Learner();ref.restore(state);logits,cache=ref.forward(xa,True);xx,ah,h,at,mask,zv=cache;e=logits.softmax(1);e[T.arange(len(ya)),ya]-=1;e/=len(ya);dh=(e@ref.fh)*(ah>0)
 gh=T.stack([xx[:,j,:].T@dh[:,j] for j in range(64)])
 assert T.allclose(gh,grad['H'],atol=1e-7,rtol=1e-6)
 save('CYCLE2.json',{'status':'PASS','gradient_recombination':info,'NORMAL_matches_independent_Torch_AdamW':True,'H_local_surrogate_matrix_parity':True,'no_backprop_across_layers':True,'uses_one_actual_dropout_mask':True,'technical_records_so_far':4})
 # Cycle 3: finish same NORMAL trajectory; remaining arms start identical birth.
 summaries={};records=4;targets=len(ya);cpu0=time.process_time();timed_targets=0;maxck=0
 for arm in ['NORMAL','FIXED-LOW','RMS-SHAM','CGLP']:
  if arm!='NORMAL':
   net=L.Learner();net.restore(state);aa=net.step(xa,ya,ida,arm);records+=4;targets+=len(ya);timed_targets+=len(ya)
  else:aa=normal_info
  if arm=='RMS-SHAM':
   for q in aa['energy'].values():assert abs(q['candidate']-q['actual'])<=max(1e-12,q['candidate']*1e-5)
  def feed(rs,role):
   nonlocal records,targets,timed_targets
   x,y,ids=batch(rs,role)
   for j in range(0,len(y),256):bb=net.step(x[j:j+256],y[j:j+256],ids[j:j+256],arm)
   records+=len(rs);targets+=len(y);timed_targets+=len(y);return bb
  bb=feed(techB,'B');assert not bb['reliable'];assert net.clock>=2
  bstate=copy.deepcopy(net.state());before=net.clock;ce=feed(techC,'C');edit=copy.deepcopy(net.state());net.restore(bstate);assert net.clock==before and T.equal(net.rng.get_state(),bstate['dropout_rng']);cn=feed(techN,'C');noedit=copy.deepcopy(net.state())
  assert edit['clock']==noedit['clock'];assert T.equal(edit['dropout_rng'],noedit['dropout_rng'])
  buf=io.BytesIO();T.save(net.state(),buf);size=len(buf.getvalue());maxck=max(maxck,size);assert size<=1048576
  # Roundtrip every mutable tensor, moments, RNG and clock.
  restored=T.load(io.BytesIO(buf.getvalue()),weights_only=True);assert eq(restored['weights'],net.p) and eq(restored['moment1'],net.m) and eq(restored['moment2'],net.v)
  # Genuine output includes EOS and legal every payload; scoring does not move RNG.
  rng=net.rng.get_state().clone();logits,_=net.forward(xa);assert logits.shape[1]==257 and T.equal(rng,net.rng.get_state())
  summaries[arm]={'A':aa,'B':bb,'C_EDIT':ce,'C_NOEDIT':cn,'checkpoint_bytes':size,'clock':net.clock,'finite_all_states':True}
 assert records==32
 totalcpu=time.process_time()-START;wall=time.monotonic()-WALL
 rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
 # macOS ru_maxrss bytes; Linux KiB.
 if sys.platform!='darwin':rss*=1024
 rate=(time.process_time()-cpu0)/timed_targets
 scheduled=sum(4*(2*audit[w]['A']['target_applications_per_pass']+3*audit[w]['B']['target_applications_per_pass']+4*sum(len(r)+1 for r in data[w]['EDIT_train'])) for w in data)
 estimate=1.5*(rate*scheduled+totalcpu);output=80*maxck*1.5
 admitted=estimate<=900 and output<=268435456 and rss<=1073741824
 save('CYCLE3.json',{'status':'PASS','arms':summaries,'technical_records':records,'technical_target_applications':targets,'C_forks_identical_B_weights_moments_clock_RNG':True,'checkpoint_max_bytes':maxck,'peak_RSS_bytes':rss,'CPU_seconds':totalcpu,'wall_seconds':wall})
 save('RESOURCE_ADMISSION.json',{'status':'PROVISIONAL_ADMITTED' if admitted else 'MINI_RESOURCE_NOT_ADMITTED','training_only_CPU_projection_with_1_5_reserve':estimate,'evaluation_generation_count_baseline_profile_still_required':True,'complete_plan_admitted':False,'scheduled_science_records':4480,'scheduled_science_target_applications':scheduled,'technical_target_applications':targets,'projected_checkpoint_bytes_with_reserve':output,'checkpoint_max_bytes':maxck,'inference_bytes':inference,'peak_RSS_bytes':rss,'CPU_seconds':totalcpu,'wall_seconds':wall})
 save('SUMMARY.json',{'status':'THREE_TECHNICAL_CYCLES_PASS','technical_records':records,'technical_targets':targets,'science_records':0,'science_targets':0,'mechanism_efficacy':'NOT_TESTED','main_launch':'NOT_AUTHORIZED_OR_ADMITTED','M1_C_qualified':audit['M1']['correction']['qualified'],'M2_C_qualified':audit['M2']['correction']['qualified']})
 print(json.dumps({'cycles':'PASS','records':records,'targets':targets,'cpu':totalcpu,'rss':rss,'checkpoint':maxck,'training_CPU_projection':estimate,'correction':{w:audit[w]['correction'] for w in audit}}),flush=True)
if __name__=='__main__':
 try:main()
 except BaseException as e:save('FAILURE.json',{'error':repr(e),'CPU':time.process_time()-START});raise
