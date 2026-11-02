"""Post-run descriptive exposure/countermodel audit; never changes learner/gates."""
import util as U
import sys,pickle,collections,math
sys.path.insert(0,str(U.ROOT/'minifly_v178_recurrent_failure_localization_dev_20261102/local/compat'))
import numpy as np

def main():
 rs=pickle.loads((U.H/'local/records.pkl').read_bytes());out={};fit_records=targets=0
 for w,sp in rs.items():
  out[w]={}
  for role in ('A','B'):
   tr,held=sp[role+'_train'],sp[role+'_held'];uni=collections.Counter(y for r in tr for y in [*r,256]);total=sum(uni.values());models={name:collections.defaultdict(collections.Counter) for name in ('exact_prefix','canonical_prefix','suffix16')};key=lambda p:dict(exact_prefix=p,canonical_prefix=b' '.join(p.lower().split()),suffix16=p[-16:])
   for r in tr:
    for t,y in enumerate([*r,256]):
     for name,k in key(r[:t]).items():models[name][k][y]+=1
   hits=collections.Counter();seen=collections.Counter();loss=collections.Counter();n=0
   for r in held:
    for t,y in enumerate([*r,256]):
     n+=1
     for name,k in key(r[:t]).items():
      d=models[name].get(k,{});prior=(uni[y]+1)/(total+257);p=(d.get(y,0)+prior)/(sum(d.values())+1);loss[name]-=math.log2(p);hits[name]+=y in d;seen[name]+=bool(d)
   out[w][role]=dict(train_unique_records=len(set(tr)),held_records=len(held),held_exact_record_duplicates=sum(r in set(tr) for r in held),held_targets=n,models={name:dict(bits=loss[name]/n,seen_key_fraction=seen[name]/n,seen_key_and_label_fraction=hits[name]/n,learned_keys=len(models[name])) for name in models});fit_records+=3*len(tr);targets+=3*total
  orig=np.load(U.H/f'local/{w}_A_held.npz');edit=np.load(U.H/f'local/{w}_EDITED_held.npz');stable=np.load(U.H/f'local/{w}_subsets.npz')['stable'];equal=np.all(orig['idx']==edit['idx'],axis=1);out[w]['stable_context_code_audit']=dict(stable_positions=int(stable.sum()),exact_same_KC_code=int((equal&stable).sum()),interpretation='16-byte context criterion does not assert all older decaying sensory trace is identical; retention scores original held contexts')
 U.save('EXPOSURE_AUDIT.json',dict(status='COMPLETE',worlds=out,countermodel_fit_record_exposures=fit_records,countermodel_fit_target_exposures=targets,posthoc_descriptive_only=True,not_neural_features=True,canonicalization='countermodel only: lower-case bytes and collapse whitespace',evidence='E0-qualified interface and bounded predictive-learning proxy; no complete-response E1 battery, E2/E3 or E4 claim',prior_teaching_control='B-COLD has no A training; RESET-OPT matched B optimizer birth; exact/conditional frequency transfer remains a countermodel',held_feedback='no evaluator labels passed to learner; held data not used for retuning',CPU=__import__('time').process_time()));U.finish_process('descriptive_exposure_audit')
if __name__=='__main__':main()
