"""One independent saved-state response check per native fit; no teaching or refitting."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import sys,time,pickle,json,hashlib
from pathlib import Path
import numpy as np
from util import H,R,save,load
import native
sys.path.insert(0,str(H/'vendor'));import corpus
begin=time.process_time();rows=[];probe_events=0
for text in ['bryant','buster']:
 sp=corpus.split(corpus.records(corpus.load(text)),2,1,2)
 for arm in ['A','B','C']:
  ident=f'{text}_{arm}';path=H/f'NATIVE_{ident}.pkl';before_bytes=hashlib.sha256(path.read_bytes()).hexdigest();m=pickle.loads(path.read_bytes());before=native.state_hash(m);z=load(f'NATIVE_{ident}.json');assert m.target_n==z['targets'] and m.records==z['records'] and not m.homeo and np.all(m.theta==0)
  q=m.clone();q._prompt(b'');errs=[];rms=[];span=[]
  for i,y in enumerate(list(sp['calib'][0])+[256]):
   v=q.values();u=(v-v.mean())/4;logit=q.C@u;shift=logit-logit.max();lp=shift-np.log(np.exp(shift).sum());saved=z['calibration']['rows'][0][i];assert saved[0]==y and saved[2]==int(np.argmax(lp));errs.append(abs(float(lp[y])-saved[1]));rms.append(float(np.sqrt(np.mean((v-v.mean())**2))));span.append(float(logit.max()-logit.min()));probe_events+=1
   if y!=256:q._event(y)
  assert native.state_hash(m)==before and hashlib.sha256(path.read_bytes()).hexdigest()==before_bytes
  for g,rec in zip(z['generations'],sp['held'][:16]):
   out=bytes.fromhex(g['output_hex']);assert g['status'] in ['EOS','CAP'];assert g['decisions']==len(out)+(g['status']=='EOS');assert g['decisions']<=64
   if g['status']=='CAP':assert len(out)==64
   assert g['prefix_hex']==rec[:8].hex() and g['expected_hex']==rec[8:].hex()
  rows.append({'id':ident,'input_checkpoint_sha256':before_bytes,'first_calibration_line_targets_checked':len(errs),'max_log_probability_error':max(errs),'mean_centered_native_value_RMS':float(np.mean(rms)),'mean_logit_span':float(np.mean(span)),'training_mean_coefficient_L2_squared':m.coef_l2/m.target_n,'native_targets':m.target_n,'records':m.records,'own_EOS_count':sum(g['status']=='EOS' for g in z['generations']),'own_exact_continuations':sum(g['status']=='EOS' and g['output_hex']==g['expected_hex'] for g in z['generations']),'state_unchanged':True})
save('NATIVE_RESPONSE_AUDIT.json',{'status':'PASS','source':'One first calibration record from each saved final native state, manually computed full257logits; no learned writes','rows':rows,'extra_passive_target_probes':probe_events,'new_target_feedback_events':0,'scientific_histories_replayed':0,'CPU':time.process_time()-begin});print(json.dumps(rows,indent=2))
