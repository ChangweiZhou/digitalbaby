"""Bounded algebra on saved pre-update reads. No native imports or model calls."""
import json
import numpy as np
import support171 as s
s.STAGE='saved_response_reduction'
def load_tape(p):
 with np.load(p,allow_pickle=False) as z:
  meta=json.loads(z['metadata'].tobytes());return meta,{k:z[k].copy() for k in z.files if k!='metadata'}
def main():
 result=json.loads((s.H/'CALIBRATION_RESULTS.json').read_text());words=json.loads((s.ROOT/'minifly_cls_dev_20261017/OUTPUT_CODEBOOK.json').read_text())['words_by_class'];C=np.array([[1. if w&(1<<j) else -1. for j in range(16)] for w in words]);out={};sites=0
 for world in s.SPEC['data']['calibration_worlds']:
  meta,z=load_tape(s.H/'local'/f'cal_{world}_NEW_pass16_NATIVE.npz');recs=[]
  for md,v in zip(meta,z['v']):
   y=md['y'];D=C[y]-C;D-=D.mean(axis=1,keepdims=True);a=D*v[None,:]/4;bound=a.sum(1)+.5*abs(a).sum(1);bound[y]=np.inf;r=int(bound.argmin());tol=1e-12*max(1,float(abs(v).sum()));bad=bool(bound[r]<-tol)
   recs.append(dict(y=y,certified_unreachable_at_saved_read=bad,unbeatable_rival=r,max_gap_against_rival=float(bound[r]),native_correct=md['argmax']==y));sites+=1
  roles={}
  for role in ['EXACT3','EXACT_RECORD']:
   md,x=load_tape(s.H/'local'/f'cal_{world}_NEW_pass16_{role}.npz');roles[role]=dict(gate_max_abs_departure=float(abs(x['g']-1).max()),gate_mean_abs_departure=float(abs(x['g']-1).mean()),direct_signal_max_L2=float(np.linalg.norm(x['direct_error'],axis=1).max()),carried_signal_max_L2=float(np.linalg.norm(x['signal_norms'][:,1:],axis=1).max()))
  out[str(world)]=dict(sites=len(recs),certified_unreachable=sum(x['certified_unreachable_at_saved_read'] for x in recs),native_value_L2_min=float(np.linalg.norm(z['v'],axis=1).min()),native_value_L2_median=float(np.median(np.linalg.norm(z['v'],axis=1))),native_value_L2_max=float(np.linalg.norm(z['v'],axis=1).max()),by_role=roles)
  s.save(f'local/cal_{world}_BOX_BOUND.json',recs)
 assert sites<=128;s.COUNTS['saved_response_target_reuses']+=sites
 s.save('SAVED_RESPONSE_AUDIT.json',dict(status='BOUNDED_READONLY_ALGEBRA_COMPLETE',worlds=out,sites=sites,formula='max_g a.g = sum(a)+0.5 L1(a), g in [0.5,1.5]',model_queries=0,training_records=0,assessment_labels_used=False,qualification_changed=False,limits='Pre-update teacher-forced reads only; native writes can change subsequent reads. No-negative-bound does not establish joint gate feasibility or learned-policy reachability.'))
 s.finish('CALIBRATION_ANALYZED');print(json.dumps(out))
if __name__=='__main__':main()
