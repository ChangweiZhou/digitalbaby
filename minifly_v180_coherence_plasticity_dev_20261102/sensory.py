"""Literal sensory export, strict corrected-state cohorts, no neural efficacy scores."""
import util as U
import sys,pickle,time
import numpy as np
sys.path.insert(0,str(U.ROOT/'minifly_v176_feature_deployment_dev_20261101'));import native_stage as N

def padded(m,r):
 codes=N.sparse_codes(m,r);out=np.full((len(codes),185),5178,np.int16)
 for i,x in enumerate(codes):out[i,:len(x) if len(x) else 1]=x if len(x) else 5177
 return out

def main():
 rec=pickle.loads((U.H/'local/records.pkl').read_bytes());m=N.Model();out={};man={};strict={};technical=[bytes(range(128)),bytes(range(128,256))]
 for r in technical:
  a=N.sparse_codes(m,r);b=N.literal_codes(m,r);assert all(np.array_equal(x,y) for x,y in zip(a,b))
 for w,sp in rec.items():
  # Strict cohort verification on the held streams is data-only; no model/guard scores.
  old=[];new=[];y=[];yp=[]
  for a,b in zip(sp['A_held'],sp['EDIT_held']):old.append(padded(m,a));new.append(padded(m,b));y.extend([*a,256]);yp.extend([*b,256])
  old=np.concatenate(old);new=np.concatenate(new);y=np.array(y);yp=np.array(yp);same=np.all(old==new,axis=1);unchanged=y==yp;coh=dict(same_KC=unchanged&same,altered_KC=unchanged&~same,changed=~unchanged)
  strict[w]={k:int(v.sum()) for k,v in coh.items()};strict[w]['admitted']=strict[w]['same_KC']>=128 and strict[w]['changed']>=24 and int(unchanged.sum())>=128
  np.savez(U.H/f'local/{w}_cohorts.npz',**coh,old_y=y,new_y=yp);assert sum(v.sum() for v in coh.values())==len(y)
  U.save('SENSORY_PROGRESS.json',dict(world=w,cohorts=strict[w],CPU=time.process_time()))
 # Full-main input export forecast from actually measured fixed sensory function.
 t=time.process_time();total=0
 for r in rec['D1']['A_train'][:8]:padded(m,r);total+=len(r)+1
 rate=(time.process_time()-t)/total
 # Technical raw-data rehearsal, never taken from guard streams or used for efficacy.
 raw=[bytes(range(31)),bytes(range(31,78)),bytes(range(78,157)),bytes(range(157,252))];fixtures={}
 for phase in ('A','B','EDIT','NOEDIT'):
  rr=raw if phase=='A' else ([bytes((b+17)%256 for b in r) for r in raw] if phase=='B' else [bytes((b+17)%256 for b in r) for r in raw[:3]])
  if phase=='EDIT':rr=[r[:-1]+bytes([(r[-1]+1)%256]) for r in rr]
  idx=[];ys=[];ids=[]
  for i,r in enumerate(rr):idx.append(padded(m,r));ys.extend([*r,256]);ids.extend([f'technical:record:{i}']*(len(r)+1))
  fixtures[phase]=dict(idx=np.concatenate(idx),y=np.array(ys,np.int64),ids=ids,record_n=np.array([len(r)+1 for r in rr]))
 with (U.H/'local/fixtures.pkl').open('wb') as f:pickle.dump(fixtures,f,protocol=5)
 U.save('SENSORY_AUDIT.json',dict(status='PASS' if all(x['admitted'] for x in strict.values()) else 'CORRECTION_FIXTURE_INSUFFICIENT',world_cohorts=strict,full256_payload_literal_code_parity=True,per_target_export_CPU=rate,no_guard_model_scores=True,technical_fixture_targets={k:len(v['y']) for k,v in fixtures.items()},technical_fixture_records={k:len(v['record_n']) for k,v in fixtures.items()},original_exporter_sha256=U.sha(U.ROOT/'minifly_v176_feature_deployment_dev_20261101/native_stage.py')));U.finish_process('strict_sensory_and_fixture_export');print('SENSORY_AUDIT_COMPLETE',strict,flush=True)
if __name__=='__main__':main()
