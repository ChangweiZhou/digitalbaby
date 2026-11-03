"""Saved-only checks and unchanged four-domain decision. No learner calls."""
from common import *
import math,statistics,zipfile
ARMS=['NORMAL','FIXED-LOW','RMS-SHAM','CGLP'];STAGES=['birth','A1','A2','B1','B2','B3','EDIT1','EDIT2','NOEDIT1','NOEDIT2']
def components(a):return [a['B3']['B_held']['bits'],a['EDIT2']['B_held']['bits'],a['EDIT2']['EDIT_held']['partitions']['SAME_KC']['bits'],a['EDIT2']['EDIT_held']['partitions']['CHANGED']['bits']]
def decide(metrics,counts):
 screen={};all_pass=True;competent=True
 for w,arms in metrics.items():
  scores={arm:sum(components(m))/4 for arm,m in arms.items()};c=arms['CGLP'];norm=arms['NORMAL'];chg=lambda m,stage:m[stage]['EDIT_held']['partitions']['CHANGED']['bits'];same=lambda m:m['EDIT2']['EDIT_held']['partitions']['SAME_KC']['bits']
  checks={'competence':min(norm['A2']['A_held']['bits'],c['A2']['A_held']['bits'])<=counts['worlds'][w]['A_unigram']['bits']-.05,'beats_FIXED_LOW':scores['CGLP']<=scores['FIXED-LOW']-.03,'beats_RMS_SHAM':scores['CGLP']<=scores['RMS-SHAM']-.03,'A2_guard':c['A2']['A_held']['bits']<=norm['A2']['A_held']['bits']+.10,'changed_guard':chg(c,'EDIT2')<=chg(norm,'EDIT2')+.10,'feedback_specific_gain':chg(c,'NOEDIT2')-chg(c,'EDIT2')>=.10}
  for arm in ('FIXED-LOW','RMS-SHAM'):
   checks['B_retention_guard_'+arm]=c['EDIT2']['B_held']['bits']<=arms[arm]['EDIT2']['B_held']['bits']+.10;checks['stable_guard_'+arm]=same(c)<=same(arms[arm])+.10
  screen[w]={'components':{arm:components(m) for arm,m in arms.items()},'composite_bits':scores,'checks':checks,'feedback_gain_bits':chg(c,'NOEDIT2')-chg(c,'EDIT2')};competent &= checks['competence'];all_pass &= all(checks.values())
 if not competent:verdict='MINI_CAPACITY_OR_LEARNING_GAP'
 elif all_pass:verdict='SELECTIVE_PLASTICITY_MINI_DEV'
 elif all(v['composite_bits']['FIXED-LOW']<min(v['composite_bits'][a] for a in ('NORMAL','CGLP')) and v['composite_bits']['FIXED-LOW']<=v['composite_bits']['RMS-SHAM'] for v in screen.values()):verdict='LOW_STEP_SUFFICIENT'
 elif all(v['composite_bits']['RMS-SHAM']<min(v['composite_bits'][a] for a in ('NORMAL','CGLP')) and v['composite_bits']['RMS-SHAM']<=v['composite_bits']['FIXED-LOW'] for v in screen.values()):verdict='UNIFORM_STRENGTH_SUFFICIENT'
 else:verdict='CGLP_NOT_SUPPORTED_MINI'
 return {'verdict':verdict,'worlds':screen,'two_opened_development_worlds_only':True,'no_arithmetic_E3_or_adoption_claim':True}
def finish():
 source_check();index=load(CAM/'COMMIT_INDEX.json');counter=load(CAM/'COUNTERS.json');metrics=load(CAM/'METRICS.json');cost=check();accepted=[];totrec=tottar=checkpoints=0
 for entry in index['receipts']:
  p=CAM/entry['path'];assert sha(p)==entry['sha256'];r=load(p);assert r['status']=='ACCEPTED_FULL_LIFE' and r['records']==560;assert r['source_lock_sha256']==sha(CAM/'SOURCE_LOCK.json');assert set(r['checkpoints'])==set(STAGES)
  for stage,cp in r['checkpoints'].items():
   path=CAM/cp['checkpoint'];assert sha(path)==cp['checkpoint_sha256'] and path.stat().st_size==cp['checkpoint_bytes']<=1048576
   snap=load(p.parent/(stage+'.json'));assert snap['checkpoint_sha256']==cp['checkpoint_sha256'];assert snap['state_sha256']==cp['state_sha256'];assert snap['metrics']==metrics[r['world']][r['arm']][stage]
   if stage!='birth':
    pr=load(p.parent/(stage+'_PASS_RECEIPT.json'));assert pr['status']=='ACCEPTED_COMPLETE_PASS' and pr['checkpoint_sha256']==cp['checkpoint_sha256'] and pr['metrics_sha256']==sha(p.parent/(stage+'.json'))
   checkpoints+=1
  for stage in ('EDIT1','NOEDIT1','EDIT2','NOEDIT2'):assert r['checkpoints'][stage]['clock']==r['checkpoints']['EDIT1' if stage.endswith('1') else 'EDIT2']['clock']
  totrec+=r['records'];tottar+=r['targets'];accepted.append({'receipt':entry['path'],'sha256':entry['sha256'],'records':r['records'],'targets':r['targets']})
 assert len(accepted)==8 and checkpoints==80 and totrec==4480 and tottar==227148
 assert counter['science_records']==totrec and counter['science_targets']==tottar and counter['committed_jobs']==8
 # Births identical within a world, both corrected/unchanged branch clocks equal.
 for w in ('M1','M2'):
  births=[load(CAM/'jobs'/f'{w}_{a}'/'RECEIPT.json')['same_birth_hash'] for a in ARMS];assert len(set(births))==1
 decision=decide(metrics,load(CAM/'COUNT_CONTROLS.json'));save(CAM/'DECISION.json',decision)
 paired={}
 for w in ('M1','M2'):
  paired[w]={}
  for stage,panel in [('A2','A_held'),('B3','B_held'),('EDIT2','B_held'),('EDIT2','EDIT_held')]:
   c=metrics[w]['CGLP'][stage][panel];paired[w][stage+'_'+panel]={}
   for arm in ('NORMAL','FIXED-LOW','RMS-SHAM'):
    r=metrics[w][arm][stage][panel];assert c['record_target_counts']==r['record_target_counts'];diff=[(x-y)/n for x,y,n in zip(c['record_bit_sums'],r['record_bit_sums'],c['record_target_counts'])];paired[w][stage+'_'+panel][arm]={'CGLP_minus_control_record_mean_bits':diff,'mean':statistics.mean(diff),'median':statistics.median(diff),'correlated_records_not_independent_worlds':True}
 save(CAM/'PAIRED_RECORD_EFFECTS.json',paired)
 summary={'status':'COMPLETED','verdict':decision['verdict'],'committed_jobs':8,'independent_text_ecologies':2,'paired_arm_lives':8,'correlated_C_child_branches':16,'scientific_records':totrec,'scientific_target_applications':tottar,'technical_records':32,'technical_target_applications':1064,'total_neural_records':totrec+32,'total_neural_target_applications':tottar+1064,'checkpoint_count':checkpoints,'trainable_parameters':37313,'cost':cost,'countermodel_fit_records':load(CAM/'COUNT_CONTROLS.json')['fit_records'],'countermodel_fit_targets':load(CAM/'COUNT_CONTROLS.json')['fit_targets'],'not_E3_or_arithmetic_or_adoption':True}
 save(CAM/'SUMMARY.json',summary)
 lines=['# V181 official compact local-plasticity experiment','',f"**Completed: {decision['verdict']}.**",'', 'Two opened text ecologies, four fixed arms, 8 paired arm lives; corrected/unchanged C forks are correlated continuations. No E3 arithmetic, semantic revision or adoption claim.','',f'Committed: {totrec:,} complete neural records / {tottar:,} byte/EOS target applications. Including prior technical qualification: 4,512 records / 228,212 targets. All 80 full mutable checkpoints were saved and hash-audited.','', '| World | NORMAL | FIXED-LOW | RMS-SHAM | CGLP |','|---|---:|---:|---:|---:|']
 for w in ('M1','M2'):lines.append('| '+w+' | '+' | '.join(f"{decision['worlds'][w]['composite_bits'][a]:.6f}" for a in ARMS)+' |')
 lines+=['','Scores are the frozen four-domain mean in bits; lower is better. Every domain and gate is in DECISION.json. Train/calibration/held curves, own EOS calibration and 64-step own generation are saved at each declared stage. No result changed training duration or selected a checkpoint.','',f"Main-process charged cost at reduction: {cost['CPU_seconds']:.1f} CPU seconds, {cost['wall_seconds']:.1f} wall seconds, peak RSS {cost['peak_RSS_bytes']:,} bytes. Original 900 CPU/1,800 wall seconds and 1 GiB/256 MiB limits remained fixed.",'','The circuit has 37,313 local-update learned parameters, no attention/Transformer/ELM, and no arithmetic-specific structure. Authentic literary next-byte/EOS training and a generic train-selected byte substitution are bounded diagnostics, not evidence of arithmetic reasoning. Gradual exposure and the separate arithmetic track were not changed.','', 'Unigram and calibration-selected interpolated Kneser–Ney are evaluator-only countermodels. Their separately charged fits never generate neural answers. Per-record paired differences are descriptive; tokens, records, checkpoints and C branches are not independent worlds.','', 'Source/corpus hashes, counts and gates were audited from saved receipts without retraining. Raw corpora, sparse data caches and model checkpoints remain local; the review zip contains sources and compact results. This package audit is not independent external raw-data reproduction.']
 (CAM/'REPORT.md').write_text('\n'.join(lines)+'\n')
 save(CAM/'RECEIPT_AUDITS.json',{'status':'ACCEPTED','saved_only':True,'new_learning_or_science_reruns':0,'receipts':accepted,'checkpoints':checkpoints,'source_hashes_match':True})
 files={str(p.relative_to(CAM)):sha(p) for p in CAM.rglob('*.json') if p.name not in ('FINAL_AUDIT.json','PACKAGE.json','STATUS.json','ACTIVE.json','LAUNCH.json','PROCESS.json')};files['REPORT.md']=sha(CAM/'REPORT.md')
 save(CAM/'FINAL_AUDIT.json',{'status':'ACCEPTED','scientific_records':totrec,'scientific_targets':tottar,'source_lock_sha256':sha(CAM/'SOURCE_LOCK.json'),'files':files,'verdict':decision['verdict'],'external_raw_reproduced':False})
 zpath=CAM/'RESULT_BUNDLE.zip'
 with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
  for rel in [*files,'FINAL_AUDIT.json']:z.write(CAM/rel,rel)
  for p in OPS.glob('*.py'):z.write(p,'source/official_ops/'+p.name)
  for name in ('learner.py','sensor.py','data.py','evaluator.py','REVISED_SPEC.json','REVISED_DESIGN.md'):z.write(DEV/name,'source/'+name)
 save(CAM/'PACKAGE.json',{'status':'COMPLETE','path':'RESULT_BUNDLE.zip','sha256':sha(zpath),'bytes':zpath.stat().st_size,'raw_data_or_weights_included':False})
 cost=check();save(CAM/'STATUS.json',{'status':'COMPLETED','verdict':decision['verdict'],'committed_jobs':8,'scientific_records':totrec,'scientific_targets':tottar,'audit':'ACCEPTED','package':'COMPLETE',**cost});print('V181_COMPLETED',decision['verdict'],totrec,tottar,flush=True)
