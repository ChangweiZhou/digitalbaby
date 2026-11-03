"""Future source and saved-site reduction; no model inference or optimizer step."""
import bootstrap as B
import evaluate as E
import protocol as Q
from pathlib import Path
import json,math,zipfile
np=B.np

def reduced(site):
 y=site['y'];eos=y==256;bits=-site['logp_true']/math.log(2)
 return {'targets':len(y),'bits':float(bits.mean()),'payload_bits':float(bits[~eos].mean()),'accuracy':float((site['pred_y']==y).mean()),'EOS_NLL':float((-site['logp_EOS'][eos]/math.log(2)).mean()),'EOS_Brier':float(((site['prob_EOS']-eos)**2).mean()),'premature_EOS_rate':float((site['pred_y'][~eos]==256).mean())}
def finish(out,budget):
 read=lambda p:json.loads(Path(p).read_text());save=lambda p,v:Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n');metrics=read(out/'METRICS.json');checkpoint_count=site_count=0;forecast=None
 for rel,sha in read(out/'SOURCE_LOCK.json')['files'].items():assert B.digest(B.ROOT/rel)==sha,rel
 for arm in ('NORMAL','HRES','NORM_SHAM'):
  job=out/arm;r=read(job/'RECEIPT.json');assert r['records']==560 and r['targets']==34364;norm=read(job/'NORM_CENSUS.json')
  if arm=='NORM_SHAM':assert norm['material_norm_mismatch_sites']==0,'CONTROL_NOT_QUALIFIED'
  for stage,cp in r['checkpoints'].items():
   assert B.digest(job/(stage+'.pt'))==cp['checkpoint_sha256'];assert B.digest(job/(stage+'_sites.npz'))==cp['sites_sha256'];assert B.digest(job/(stage+'.json'))==cp['metrics_sha256'];z=np.load(job/(stage+'_sites.npz'));checkpoint_count+=1
   for name,v in metrics[arm][stage].items():
    if name.startswith('_'):continue
    s={k[len(name)+2:]:z[k] for k in z.files if k.startswith(name+'__')};rr=reduced(s);eos=s['y']==256;assert np.allclose(s['logp_true'][eos],s['logp_EOS'][eos],atol=1e-12);assert np.allclose(np.exp(s['logp_EOS']),s['prob_EOS'],atol=1e-15,rtol=1e-9)
    for k,value in rr.items():assert abs(value-v[k])<1e-9,(arm,stage,name,k)
    b=-s['logp_true']/math.log(2)
    for i,n in enumerate(v['record_target_counts']):
     mask=s['record_id']==i;assert mask.sum()==n and abs(float(b[mask].sum())-v['record_bit_sums'][i])<1e-9
     pmask=mask&(s['y']!=256);assert int(pmask.sum())==v['payload_record_target_counts'][i] and abs(float(b[pmask].sum())-v['payload_record_bit_sums'][i])<1e-9
    if name=='EDIT_held':
     changed=s['y']!=s['original_y']
     for k,mask in [('SAME_KC',~changed&s['same_KC']),('ALTERED_KC',~changed&~s['same_KC']),('CHANGED',changed)]:
      assert int(mask.sum())==v['partitions'][k]['n'];assert abs(float(b[mask].mean())-v['partitions'][k]['bits'])<1e-9;assert abs(float(s['new_old_log2_odds'][mask].mean())-v['partitions'][k]['new_old_log2_odds'])<1e-9
    site_count+=len(s['y'])
   if stage=='A2':
    tr={k[len('A_train__'):]:z[k] for k in z.files if k.startswith('A_train__')};q=E.forecast_from_summaries(tr);assert np.max(abs(q-z['A2_mean_logq']))<1e-10;value=E.forecast_loss(q,{'y':z['A_held__y']});assert abs(value-metrics[arm][stage]['_own_mean_forecast_A_held_bits'])<1e-10
    if arm=='HRES':forecast=value
 assert checkpoint_count==15 and site_count==499605 and forecast is not None
 # Bootstrap and payload summaries derive from the exact same saved A2 sites.
 cis={};payload_CIs={};c=metrics['HRES']['A2']['A_held']
 for arm in ('NORMAL','NORM_SHAM'):
  control=metrics[arm]['A2']['A_held'];assert c['record_target_counts']==control['record_target_counts'];cis[arm]=Q.paired_bootstrap(c['record_bit_sums'],control['record_bit_sums'],c['record_target_counts']);payload_CIs[arm]=Q.paired_bootstrap(c['payload_record_bit_sums'],control['payload_record_bit_sums'],c['payload_record_target_counts'])
 alignment=read(out/'HRES/A2_ALIGNMENT.json');rows=alignment['site_evaluations'];manifest=read(out/'MANIFEST.json')['A_held'];expected=Q.derangement(manifest['targets'],manifest['record_sha256']);assert {(r['record'],r['position']) for r in rows}==set(expected) and len(rows)==len(expected)
 source=np.load(out/'HRES/A2_sites.npz');source_sites={(int(r),int(p)):(int(y),float(lp)) for r,p,y,lp in zip(source['A_held__record_id'],source['A_held__position'],source['A_held__y'],source['A_held__logp_true'])};counts=np.zeros(64,np.int64);a=np.zeros(64);b=np.zeros(64)
 for row in rows:
  assert row['record']!=row['other_record'] and expected[(row['record'],row['position'])]==row['other_record'];observed_y,observed_lp=source_sites[(row['record'],row['position'])];assert row['y']==observed_y and abs(row['aligned_logp']-observed_lp)<1e-5;counts[row['record']]+=1;a[row['record']]-=row['aligned_logp']/math.log(2);b[row['record']]-=row['exchanged_logp']/math.log(2)
 assert int(counts.sum())==alignment['eligible_sites'];assert abs(float(a.sum()/counts.sum())-alignment['aligned_bits'])<1e-10 and abs(float(b.sum()/counts.sum())-alignment['exchanged_bits'])<1e-10
 decision=Q.decide(metrics,cis,forecast,4.403066543210908,alignment);decision.update(total_primary_CIs=cis,payload_secondary_CIs=payload_CIs);save(out/'DECISION.json',decision)
 save(out/'SAVED_SITE_AUDIT.json',{'status':'ACCEPTED_RUN_LOCAL','full_checkpoints':15,'saved_sites_reduced':site_count,'all_summary_sites':999210,'EOS_and_mean_forecast_and_alignment_reconstructed':True,'new_optimizer_updates':0,'independent_cross_family':'PENDING','formal_MRK1':'UNAVAILABLE'})
 save(out/'SUMMARY.json',{'verdict':decision['verdict'],'counts':read(out/'RUN_COMPLETE.json')['counts'],'cost':budget(),'learned_parameters':37313,'scientific_lives':3,'text_ecologies':1,'correlated_C_branches':6,'no_arithmetic_E3_or_adoption':True})
 (out/'REPORT.md').write_text('# V184 H-residual deployment\n\n'+decision['verdict']+'\n\nThree fixed complete M1 lives,1,680 optimizer-record applications and103,092 observed byte/EOS targets. All15 full states and30 read-only summary checkpoints were retained. The norm sham must remain inside the registered validity domain throughout; no gain-cap rescue is permitted. DECISION.json contains the signed full-vector gates, paired-record bootstrap CIs and descriptive payload CIs. Every truncation and own EOS is retained. No independent ecological confirmation, deterministic E1, semantic E2, arithmetic E3 or core adoption is claimed.\n')
 archive=out/'STATE_BUNDLE.zip';files=[p for p in out.rglob('*') if p.is_file() and p.suffix in ('.pt','.npz','.json','.md') and p.name not in ('FINAL_AUDIT.json','PACKAGE.json')]
 with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
  for p in files:z.write(p,str(p.relative_to(out)))
 save(out/'STATE_BUNDLE.json',{'path':archive.name,'sha256':B.digest(archive),'bytes':archive.stat().st_size,'all15_full_states_and_required_site_arrays':True});save(out/'FINAL_AUDIT.json',{'status':'ACCEPTED_RUN_LOCAL','verdict':decision['verdict'],'source_lock_sha256':B.digest(out/'SOURCE_LOCK.json'),'state_archive_sha256':B.digest(archive),'cost':budget(),'formal_MRK1':'UNAVAILABLE','cross_family_review':'PENDING'});save(out/'STATUS.json',{'status':'COMPLETED','verdict':decision['verdict'],'state_archive':'COMPLETE','audit':'ACCEPTED_RUN_LOCAL'})
