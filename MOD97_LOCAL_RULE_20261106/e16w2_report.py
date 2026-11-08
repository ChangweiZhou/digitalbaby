"""Read-only live/final amended E16W scorecard. Does not run training."""
import json,hashlib,csv,gzip,pickle,statistics,datetime
from pathlib import Path
import e16w2_campaign as C
R,O=C.R,C.O
PAIRS=('+−','−+','><','<>') # Display labels below use the actual ASCII keys.
KEYS=('+-','-+','><','<>');OLD=('add97','sub97','max97','min97');NEW=('add73','sub73','max73','min73')
def pct(x,k='exact'):return 100*x[k]/x['n']
def held(sc,k='final'):return statistics.mean(pct(sc['compose'][x],k) for x in KEYS)
def meanops(sc,ops):return statistics.mean(pct(sc[x]) for x in ops)
def table(h,rs):return ['|'+'|'.join(map(str,h))+'|','|'+'|'.join(['---']*len(h))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rs]
def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items() if k!='cpu'}
 if isinstance(x,list):return [clean(v) for v in x]
 return x
def stream_audit(seed):
 path=O/f'e16w2_batches_{seed}.pkl.gz';h=hashlib.sha256();count=0;offset=0;rows=[]
 names=['add97','sub97','max97','min97','compose','add73','sub73','max73','min73'];doses=[6144]*4+[8192]+[6144]*4
 with gzip.open(path,'rb') as f:
  for name,n in zip(names,doses):
   totals=[0,0,0];presented=set()
   for _ in range(n):
    items,flag=pickle.load(f);assert flag is True and len(items)==64
    presented.update(tuple(p) for p,t in items[:64 if count==0 or name=='add97' else 32])
    encoded=json.dumps([items,flag],separators=(',',':')).encode();h.update(len(encoded).to_bytes(8,'little'));h.update(encoded);count+=1
    for prompt,target in items:
     assert len(target)==(6 if prompt[0]==99 else 3)
     category=0 if prompt[0]==99 else (1 if prompt[0] in (114,116,120,110) else 2);totals[category]+=1
   for role in ('C_uni','D_matched'):
    job=next(j for j in C.plan() if j['role']==role and j['seed']==seed)
    admission=C.read((O/job['receipt']).with_suffix('.buffer.json'))[len(rows)]
    assert admission['task']==name and len(set(map(tuple,admission['prompts'])))==1024
    assert all(tuple(prompt) in presented for prompt in admission['prompts']),('Unpresented admission',seed,role,name)
   rows.append(dict(task=name,updates=n,composition=totals[0],mod97=totals[1],mod73=totals[2]))
  try:pickle.load(f)
  except EOFError:pass
  else:raise AssertionError('Extra batch')
 a=C.read(O/f'E16W2_BATCHES_C_uni_{seed}.json');b=C.read(O/f'E16W2_BATCHES_D_matched_{seed}.json')
 assert count==57344 and h.hexdigest()==a['batch_sha256']==b['batch_sha256']
 assert rows==a['exposures']==b['exposures']
 file_sha=C.sha(path);assert file_sha==a['teacher_stream_sha256']==b['teacher_stream_sha256']
 return dict(seed=seed,batches=count,batch_sha256=h.hexdigest(),file_sha256=file_sha,bytes=path.stat().st_size,exposures=rows)
def main():
 state=C.read(O/'E16W2_STATUS.json');lock=C.read(R/'E16W2_LOCK.json');done=state['status']=='complete'
 for n,h in lock['files'].items():assert C.sha(R/n)==h,n
 assert clean(C.read(O/'e16w_profile_C_73129_e8_life_C_73129.json'))==clean(C.read(O/'e13_A1_e8_life_C_73129.json'))
 metrics={};opsrows=[];curve=[];validation=[];costs=[];sections=[]
 for j in C.plan():
  jobfile=O/f'E16W2_JOB_{j["key"]}.json';row=C.read(jobfile) if jobfile.exists() else None
  path=O/j['receipt']
  if not path.exists():continue
  d=C.read(path);stages=d['stages'];scored=[]
  for st in stages:
   sc=st['scores'];m=dict(stage=st['stage'],task=st['task'],held_final=held(sc),held_full=held(sc,'exact'),mod97_mean=meanops(sc,OLD),mod73_mean=meanops(sc,NEW),scores=sc);scored.append(m)
   curve.append(dict(role=j['role'],seed=j['seed'],stage=st['stage'],task=st['task'],held_final=m['held_final'],held_full=m['held_full'],mod97_mean=m['mod97_mean'],mod73_mean=m['mod73_mean']))
   for op in OLD+NEW:opsrows.append(dict(role=j['role'],seed=j['seed'],stage=st['stage'],task=st['task'],operation=op,n=sc[op]['n'],exact=sc[op]['exact'],percent=pct(sc[op])))
  metrics[j['key']]=dict(role=j['role'],seed=j['seed'],status=row['status'] if row else 'UNTRACKED',receipt=j['receipt'],stages=scored)
  sections+=['## '+j['key'],'',f"Job status: {row['status'] if row else 'UNTRACKED'}. Values in pair columns are final-answer / full-scratchpad percent.",'']
  sections+=table(['Stage','Held mean','Held full','Old97 mean','New73 mean',*KEYS,'within'],[[s['task'] or 'birth',f"{s['held_final']:.2f}",f"{s['held_full']:.2f}",f"{s['mod97_mean']:.2f}",f"{s['mod73_mean']:.2f}",*[f"{pct(s['scores']['compose'][k],'final'):.2f}/{pct(s['scores']['compose'][k]):.2f}" for k in KEYS+('within',)]] for s in scored])+['']
  sections+=table(['Stage',*OLD,*NEW],[[s['task'] or 'birth',*[f"{pct(s['scores'][k]):.2f}" for k in OLD+NEW]] for s in scored])+['']
  if row and row['status']=='complete':
   assert row['receipt_sha256']==C.validate(j)
   assert (row['returncode'] is None and j['adopt']) or (row['returncode']==0 and not j['adopt'])
   for k,v in j.items():assert row[k]==v,(j['key'],k)
   validation.append(dict(key=j['key'],sha256=C.sha(path),returncode=row['returncode'],adopted=j['adopt']))
   costs.append(dict(key=j['key'],kind='REUSED_CEILING' if j['adopt'] else 'NEW_LIFE',recorded_cpu_seconds_lower_bound=stages[-1]['cpu'],wall_seconds=row.get('wall_seconds'),adoption_wait_seconds=row.get('adoption_wait_seconds'),numeric_exit_status_captured=not j['adopt']))
 verdicts={};audits=[]
 if done:
  assert len(validation)==12 and not state['failed_parts']
  assert len(state['completed'])==12
  for seed in (73129,93683,116089):
   a,b,d=[metrics[f'{role}_{seed}']['stages'] for role in ('C_uni','C_share','D_matched')]
   loss=a[5]['held_final']-a[9]['held_final'];share=b[5]['held_final']-b[9]['held_final'];mloss=d[5]['held_final']-d[9]['held_final']
   low=min(pct(s['scores'][op]) for series in (a,b) for s in series[6:] for op in OLD)
   verdicts[str(seed)]=dict(uniform_loss_pp=loss,share_loss_pp=share,matched_BP_loss_pp=mloss,P1_uniform_loss_ge15=loss>=15,P2_literal_share_loss_le_two_thirds=share<=loss*2/3,P2_positive_loss_interpretation=loss>0,P3_matched_loss_le10=mloss<=10,matched_absolute_change_le10=abs(mloss)<=10,P4_each_old97_operation_ge90=low>=90,minimum_old_operation=low)
   audits.append(stream_audit(seed))
   # Validate admitted samples and their hashes independently.
   for role in ('C_uni','C_share','D_matched'):
    j=next(j for j in C.plan() if j['role']==role and j['seed']==seed);ad=C.read((O/j['receipt']).with_suffix('.buffer.json'))
    assert len(ad)==9
    for idx,x in enumerate(ad):
     assert x['stage']==idx+1 and x['n']==1024 and len(x['prompts'])==1024 and x['all_presented']
     assert hashlib.sha256(json.dumps(x['prompts'],separators=(',',':')).encode()).hexdigest()==x['sha256']
    if role=='C_share':
     meta=C.read(O/f'E16W2_BATCHES_{role}_{seed}.json');assert all(x['composition']==16*x['updates'] for x in meta['exposures'][5:])
  C.save(R/'E16W2_STREAM_AUDIT.json',dict(status='PASS',streams=audits))
 # Costs retain stopped-work and parity overhead, not only successful candidate CPU.
 profile=C.read(O/'e16w_profile_C_73129_e8_life_C_73129.json')
 profile_job=C.read(O/'E16W_JOB_profile_C_73129.json')
 costs.append(dict(key='default_parity_profile',kind='REUSED_FULL_PROFILE',recorded_cpu_seconds_lower_bound=profile['stages'][-1]['cpu'],wall_seconds=profile_job['wall_seconds']))
 for key in ('C_uni_73129','C_bal_73129'):
  p=O/('e16w_'+key+'_e8_life_C_rp1of2_late_family'+('_balanced' if 'bal' in key else '')+'_73129.json')
  old=C.read(p);start=(O/'e16w_original_attempt'/f'E16W_JOB_{key}.json').stat().st_mtime
  stopped=datetime.datetime.fromisoformat(C.read(R/'E16W_AMENDMENT_HANDOVER.json')['timestamp']).timestamp()
  costs.append(dict(key='original_'+key,kind='INTERRUPTED_NOT_SCIENTIFIC_FAILURE',recorded_cpu_seconds_lower_bound=old['stages'][-1]['cpu'],wall_seconds_estimate_from_job_file_timestamp=max(0,stopped-start),wall_scope='Artifact timestamp estimate, not captured subprocess duration; CPU beyond last recorded stage unmeasured.'))
 technical=[]
 for label,root in [('original','e16w_technical'),('revised','e16w2_technical')]:
  for p in (O/root).glob('*life*.json'):
   if p.is_file():
    d=C.read(p)
    if isinstance(d,dict) and 'stages' in d:technical.append(dict(label=label,path=str(p.relative_to(R)),recorded_cpu_seconds_lower_bound=d['stages'][-1].get('cpu',0)))
 # Original technical fixtures are nested by arm.
 for p in (O/'e16w_technical').glob('*/*life*.json'):
  if p.parent.name=='validator':continue
  d=C.read(p)
  if isinstance(d,dict) and 'stages' in d:technical.append(dict(label='original',path=str(p.relative_to(R)),recorded_cpu_seconds_lower_bound=d['stages'][-1].get('cpu',0)))
 costsummary=dict(records=costs,technical_receipt_counters=technical,successful_and_interrupted_recorded_cpu_seconds_lower_bound=sum(x['recorded_cpu_seconds_lower_bound'] for x in costs),short_technical_recorded_cpu_seconds_lower_bound=sum(x['recorded_cpu_seconds_lower_bound'] for x in technical),campaign_elapsed_seconds=state['elapsed_seconds'],peak_RSS='NOT_MEASURED',total_process_tree_CPU='NOT_MEASURED',timing_scope='Receipt CPU excludes setup/final evaluation/serialization/worker postprocessing. Overlapping wall times are not elapsed campaign time. Adopted ceiling numeric exit status not captured; wall not measured. Technical CPU counters and interrupted last-stage counters are lower bounds, not full billing.')
 C.save(R/'E16W2_COSTS.json',costsummary);C.save(R/'E16W2_METRICS.json',dict(complete=done,arms=metrics,verdicts=verdicts))
 for filename,rows in [('E16W2_OPERATIONS.csv',opsrows),('E16W2_HELD_CURVES.csv',curve)]:
  if rows:
   with (R/filename).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 lines=['# E16W — amended window-net cross-form retention','',f"Status: {state['status']}; {len(validation)}/12 scientific lives verified. Original controller is superseded; no original balanced run is a revised result.",'','Four arms: local uniform, local explicit 25% composition share, BP ceiling, BP on the exact local-generated batches. Three standing seeds. Nine training stages and 57,344 updates each. Full default parity and revised technical checks passed. Amendment was authorized after original runs started, before terminal retention results; original predictions and interrupted provenance remain archived.','']
 if done:
  lines+=['## Amended predictions','']+table(['Seed','Uniform loss pp','Share loss pp','Matched BP loss pp','P1','P2 literal','P3','P4 each op'],[[s,f"{v['uniform_loss_pp']:.2f}",f"{v['share_loss_pp']:.2f}",f"{v['matched_BP_loss_pp']:.2f}",v['P1_uniform_loss_ge15'],v['P2_literal_share_loss_le_two_thirds'],v['P3_matched_loss_le10'],v['P4_each_old97_operation_ge90']] for s,v in verdicts.items()])+['']
 if done:
  summary=[]
  for seed in (73129,93683,116089):
   for role in ('C_uni','C_share','D_matched','D_ceiling'):
    a=metrics[f'{role}_{seed}']['stages'];summary.append([seed,role,f"{a[5]['held_final']:.2f}",f"{a[9]['held_final']:.2f}",f"{a[5]['held_final']-a[9]['held_final']:.2f}",f"{a[9]['held_full']:.2f}",f"{a[9]['mod97_mean']:.2f}"])
  findings=['## Findings','',
   'The window net also forgets compositions across a later input family. Uniform local replay loses 45.58/46.42/48.08 points of held final-answer accuracy, ending 35.33/31.25/37.29%. This does not identify a cell-specific cause of E16 forgetting.','',
   'Fixing composition exposure at 25% improves final accuracy by 6.25/10.88/8.17 points but reduces the loss by only 13.7/23.4/17.0%, missing the promised one-third reduction in every seed. Carry-pair finals remain 8.83–17.67%; increased exposure under this fixed buffer/self-target protocol is insufficient.','',
   'BP trained on the exact input/target batches recorded from the local learner also forgets: losses 44.17/42.25/38.92 points, ending 43.42/45.29/49.88%. This refutes the matched-BP retention prediction in every seed and establishes that the loss is not exclusive to the local update. BP still finishes 8.08/14.04/12.58 points above local uniform: the rule is not proved irrelevant. Data coverage, generated-target quality, store/FIFO dynamics and their interactions are not separated by this study. The joint-ground-truth BP ceiling holds 90.50/90.75/91.88%, under much richer old-family exposure; it is not a rule-only comparison.','',
   'Old mod97 mean accuracy already falls from 93.33/94.92/95.04% to 76.00/78.88/85.33% during the composition stage, before the new mod73 tasks. Final old-operation means are 77.08/79.25/84.62% for uniform and 76.54/79.58/83.79% for share. Both per-operation and mean readings miss 90%. P1 holds in all seeds; P2/P3/P4 miss in all seeds. No setting was changed to rescue a scientific negative.','']
  lines+=findings+table(['Seed','Arm','Post-compose final %','Final %','Loss pp','Final full %','Final old97 mean %'],summary)+['']
 lines+=sections+['## Exposure and matching','', 'New-job realized family exposures and batch digests are in exp45/E16W2_BATCHES_ROLE_SEED.json. Each matched BP consumes the exact completed C_uni inputs, targets, order and store-write flags at the same birth, learning rate, decay 0.25 and optimizer settings. Its own generated answers do not become training targets. It is a fixed-experience update comparison. Adopted ceiling 73129 has no realized exposure log and unknown numeric exit code; do not infer these from the newly instrumented ceilings. Full matched streams are independently re-counted at final verification.','', 'Explicit share allocates 16/64 late-batch items to compositions and reduces operation replay; report both retention outcomes. The uniform/share difference is this allocation package, not an unlimited-replay comparison. If both local and matched BP forget, this shows a limit under this recorded stream; it does not rule out all replay. Window/cell comparisons also differ in inputs, schedule and store and do not isolate a cell cause. Comparison pairs have high shortcut ceilings, so report carry pairs and full scratchpad separately.','', '## Costs and provenance','', 'See E16W2_COSTS.json for separate new, reused, aborted and technical counters. Receipt CPU is a lower bound; peak RSS and total process-tree CPU were not measured. Original 108MB technical state checkpoints remain local with hashes; only their technical JSON and manifests are included in publication. Revised batch streams are retained for the matched control. No tuning or external independent replication is claimed.','']
 (R/'E16W_REPORT.md').write_text('\n'.join(lines)+'\n')
 if done:C.save(R/'E16W2_VERIFICATION.json',dict(status='COMPLETE_VERIFIED',jobs=validation,locked_files=lock['files'],stream_audits=audits,verdicts=verdicts,costs=costsummary,source='Original frozen snapshots plus revised source snapshot; reporter is post-lock read-only tooling.'))
 print(state['status'],len(validation),'verified scientific lives')
if __name__=='__main__':main()
