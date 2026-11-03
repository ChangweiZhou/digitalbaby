"""Future saved-only re-reduction from retained site arrays and all full states."""
import bootstrap as B
import metrics as M
from reduce import rebuild,decide
from pathlib import Path
import json,zipfile,math
np=B.np

def finish(out,budget):
 read=lambda p:json.loads(Path(p).read_text());save=lambda p,v:Path(p).write_text(json.dumps(v,indent=2)+'\n');all_m=read(out/'METRICS.json');nc=nsites=0;forecast=None
 for arm in ('NORMAL','HPC','SHUFFLED'):
  job=out/arm;receipt=read(job/'RECEIPT.json');assert receipt['records']==560 and receipt['targets']==34364
  for stage,cp in receipt['checkpoints'].items():
   assert B.digest(job/(stage+'.pt'))==cp['checkpoint_sha256'];assert B.digest(job/(stage+'_sites.npz'))==cp['sites_sha256'];assert B.digest(job/(stage+'.json'))==cp['metrics_sha256'];sites=np.load(job/(stage+'_sites.npz'));metrics=all_m[arm][stage]
   for name,values in metrics.items():
    if name.startswith('_'):continue
    panel={key[len(name)+2:]:sites[key] for key in sites.files if key.startswith(name+'__')};rebuilt=rebuild(panel)
    assert np.allclose(panel['logp_true'][panel['y']==256],panel['logp_EOS'][panel['y']==256],atol=1e-12)
    assert np.allclose(np.exp(panel['logp_EOS']),panel['prob_EOS'],rtol=1e-9,atol=1e-15)
    for k,v in rebuilt.items():assert abs(v-values[k])<1e-9,(arm,stage,name,k)
    if name=='EDIT_held':
     ch=panel['y']!=panel['original_y'];masks={'SAME_KC':~ch&panel['same_KC'],'ALTERED_KC':~ch&~panel['same_KC'],'CHANGED':ch}
     for key,mask in masks.items():assert int(mask.sum())==values['partitions'][key]['n'] and abs(float((-panel['logp_true'][mask]/math.log(2)).mean())-values['partitions'][key]['bits'])<1e-9
    nsites+=len(panel['y'])
   if stage=='A2':
    train={key[len('A_train__'):]:sites[key] for key in sites.files if key.startswith('A_train__')};held={'y':sites['A_held__y']};logq=M.forecast_from_summaries(train);assert np.max(abs(logq-sites['A2_mean_logq']))<1e-10;value=M.forecast_loss(logq,held);assert abs(value-metrics['_own_mean_forecast_A_held_bits'])<1e-10
    if arm=='HPC':forecast=value
   nc+=1
 assert nc==30 and nsites==999210 and forecast is not None
 for rel,h in read(out/'SOURCE_LOCK.json')['files'].items():assert B.digest(B.ROOT/rel)==h,rel
 decision=decide(all_m,forecast);save(out/'DECISION.json',decision);save(out/'SAVED_SITE_AUDIT.json',{'status':'ACCEPTED','mode':'RUN_LOCAL_SAVED_ONLY_NOT_INDEPENDENT_CERTIFICATION','checkpoints':nc,'sites_reduced':nsites,'new_learning_updates':0,'EOS_loss_from_saved_sites':True,'mean_forecast_from160x257_record_log_sums':True})
 save(out/'SUMMARY.json',{'verdict':decision['verdict'],'counts':read(out/'RUN_COMPLETE.json')['counts'],'state_checkpoints':30,'text_ecologies':1,'scientific_lives':3,'correlated_C_branches':6,'cost':budget(),'source_lock_sha256':B.digest(out/'SOURCE_LOCK.json'),'not_E3_or_adopted':True})
 (out/'REPORT.md').write_text('# V183 bounded development\n\n'+decision['verdict']+'\n\nAll3 fixed M1 A→B→corrected-A lives completed, including matched NOEDIT forks,30 full checkpoints and999,210 saved own-output target-site evaluations. Read DECISION.json for every signed conditional-prediction, still-valid-retention, correction and EOS gate. No composite overrides a failure. Source/technical instrument revision is in the parent REVISED_DESIGN.md. No arithmetic/E3, sealed confirmation or core adoption is claimed. Raw checkpoint and site arrays remain in the separate hash-pinned owner archive.\n')
 entries=[p for p in out.rglob('*') if p.is_file() and p.suffix in ('.pt','.npz','.json','.md') and p.name not in ('SOURCE_LOCK.json','FINAL_AUDIT.json','PACKAGE.json')];states=out/'STATE_BUNDLE.zip'
 with zipfile.ZipFile(states,'w',zipfile.ZIP_DEFLATED) as z:
  for p in entries:z.write(p,str(p.relative_to(out)))
 save(out/'STATE_BUNDLE.json',{'path':states.name,'sha256':B.digest(states),'bytes':states.stat().st_size,'all30_states_and_site_arrays':True})
 save(out/'FINAL_AUDIT.json',{'status':'ACCEPTED_RUN_LOCAL','full_scientific_run_complete':True,'checkpoints':30,'sites':nsites,'verdict':decision['verdict'],'source_lock_sha256':B.digest(out/'SOURCE_LOCK.json'),'owner_state_archive_sha256':B.digest(states),'cross_family_review':'PENDING','formal_MRK1':'UNAVAILABLE','cost':budget()});save(out/'STATUS.json',{'status':'COMPLETED','verdict':decision['verdict'],'audit':'ACCEPTED_RUN_LOCAL','state_archive':'COMPLETE'})
