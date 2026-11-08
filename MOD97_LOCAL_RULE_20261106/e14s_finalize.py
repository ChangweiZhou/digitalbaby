"""Read-only scientific recount and final E14S report; no training or tuning."""
import csv,hashlib,json,statistics,shutil
from pathlib import Path
from random import Random
import e14s_campaign as C
R,O=C.R,C.O
OLD=('add97','sub97','max97','min97','add73','sub73','max73','min73')
PAIRS=('+-','-+','><','<>')
def pct(v,k='exact'):return 100*v[k]/v['n']
def avg(s,k='exact',names=OLD):return statistics.mean(pct(s[n],k) for n in names)
def table(header,rows):return ['|'+'|'.join(header)+'|','|'+'|'.join('---' for _ in header)+'|']+['|'+'|'.join(map(str,x))+'|' for x in rows]
def load(name):return C.read(O/name)
def summarize(name):
 d=load(name);pre,end=d['stages'][-2]['scores'],d['stages'][-1]['scores']
 return dict(receipt=name,seed=d['seed'],stage8_retention=avg(pre),final_retention=avg(end),composition=avg(end['compose'],'final',PAIRS),full_scratchpad=avg(end['compose'],'exact',PAIRS),pairs=end['compose'])
def main():
 state=C.read(O/'E14S_STATUS.json');lock=C.read(R/'E14S_LOCK.json')
 assert state['status']=='complete' and len(state['completed'])==13 and not state['failed_parts']
 for n,h in lock['files'].items():assert C.sha(R/n)==h,n
 plan={j['key']:j for j in lock['jobs']};verified=[];births={};cpu_total=0.;wall_total=0.
 for row in state['completed']:
  j=plan[row['key']];canonical=C.read(O/f'E14S_JOB_{j["key"]}.json');assert row==canonical
  assert row['status']=='complete' and row['returncode']==0
  for k,v in j.items():assert row[k]==v,(j['key'],k)
  h=C.validate(j);assert h==row['receipt_sha256']
  if j['kind']=='birth':births[str(j['seed'])]={j['checkpoint']:C.sha(O/j['checkpoint']),j['receipt']:h}
  d=load(j['receipt']);cpu=(d.get('stages') or d['curve'])[-1]['cpu'];cpu_total+=cpu;wall_total+=row['wall_seconds']
  verified.append(dict(key=j['key'],receipt=j['receipt'],sha256=h,recorded_cpu_seconds_lower_bound=cpu,wall_seconds=row['wall_seconds'],returncode=0))
 # Verify reused controls against their previously published checksums.
 oldlock=C.read(R/'E14_LOCK.json');oldverification=C.read(R/'E14_VERIFICATION.json')
 reused_names=[]
 A=[]
 for seed in (73129,93683,116089):
  names=[('BP wd1 NEW',f'e14s_A_{seed}_e8_life_D_{seed}.json'),('BP wd0.3 REUSED',f'e8_life_D_{seed}.json'),('Local wd1 REUSED',f'e8_life_C_{seed}.json'),('Local wd0.25 REUSED',f'e13_A1_e8_life_C_{seed}.json')]
  for arm,name in names:
   a=summarize(name);a['label']=arm;A.append(a)
   if 'REUSED' in arm:assert C.sha(O/name)==oldlock['files']['exp45/'+name];reused_names.append(name)
 cohort=[]
 for seed in (73129,93683,116089):cohort.append({**summarize(f'e13_A1_e8_life_C_{seed}.json'),'origin':'E13-A1 reused'})
 for seed in (200003,300007,400009):
  n=f'e14_A_C_e8_life_C_{seed}.json';ref=next(x for x in oldverification['receipts'] if x['receipt']==n);assert C.sha(O/n)==ref['sha256'];reused_names.append(n);cohort.append({**summarize(n),'origin':'E14-A reused'})
 fresh=[];bp=[]
 for seed in (500011,600013,700027):
  x={**summarize(f'e14s_B_C_{seed}_e8_life_C_{seed}.json'),'origin':'E14S new'};fresh.append(x);cohort.append(x);bp.append({**summarize(f'e14s_B_D_{seed}_e8_life_D_{seed}.json'),'origin':'E14S new BP'})
 logged={}
 for key in ('C_400009','B_C_500011','B_C_600013','B_C_700027'):
  j=plan[key];d=load(j['receipt']);assert [x['update'] for x in d['compose_checkpoints']]==list(range(1024,8193,1024));logged[key]=d['compose_checkpoints']
  for cp in d['compose_checkpoints']:
   assert set(cp['scores'])==set(PAIRS)|{'within'}
   for v in cp['scores'].values():assert v['n']==600 and all(0<=v[k]<=600 for k in ('exact','final','intermediate'))
 source=load('e14s_C_400009_e8_life_C_400009.json');reference=load('e14_A_C_e8_life_C_400009.json');assert C.clean(source)==C.clean(reference)
 curve=logged['C_400009'];first60=next((r['update'] for r in curve if pct(r['scores']['-+'],'final')>60),None)
 maxplus=max(pct(r['scores']['+-'],'final') for r in curve)
 a_new=[x for x in A if x['label']=='BP wd1 NEW'];bad_new=[x['seed'] for x in fresh if min(pct(x['pairs'][k],'final') for k in PAIRS)<50];good=[x['seed'] for x in cohort if x['composition']>=75]
 verdicts=dict(A_all_seeds_final_retention_le85=all(x['final_retention']<=85 for x in a_new),A_per_seed={str(x['seed']):x['final_retention']<=85 for x in a_new},A_seeds_final_retention_ge92=[x['seed'] for x in a_new if x['final_retention']>=92],B_at_most_one_new_seed_any_pair_lt50=len(bad_new)<=1,B_new_seeds_any_pair_lt50=bad_new,B_at_least_seven_of_nine_means_ge75=len(good)>=7,B_nine_seed_pass_count=len(good),B_nine_seed_pass_seeds=good,C_plus_minus_all_sampled_le40=maxplus<=40,C_plus_minus_max_sampled=maxplus,C_minus_plus_exceeds60_by2048=any(r['update']<=2048 and pct(r['scores']['-+'],'final')>60 for r in curve),C_minus_plus_first_sampled_above60=first60)
 csvrows=[]
 for key,curve2 in logged.items():
  for cp in curve2:
   for pair,v in cp['scores'].items():csvrows.append(dict(life=key,update=cp['update'],pair=pair,n=v['n'],intermediate_count=v['intermediate'],final_count=v['final'],full_scratchpad_count=v['exact'],intermediate_percent=pct(v,'intermediate'),final_percent=pct(v,'final'),full_scratchpad_percent=pct(v)))
 with (R/'E14S_COMPOSITION_CURVES.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
 # Preserve technical fixture evidence without copying symlinked reference models.
 checks=C.read(R/'E14S_CHECKS.json');root=Path(checks['fixture_path']);technical_cpu=0.;archive=O/'e14s_technical_archive';technical_files=[]
 if root.exists():
  for f in root.rglob('*'):
   if f.is_file() and not f.is_symlink():
    target=archive/f.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,target);technical_files.append(str(target.relative_to(R)))
    if f.suffix=='.json' and 'life' in f.name:
     d=C.read(f);technical_cpu+=d['stages'][-1].get('cpu',0)
 costs=dict(scientific_subprocesses=13,births=3,lives=10,recorded_scientific_cpu_seconds_lower_bound=cpu_total,scientific_subprocess_wall_seconds_sum=wall_total,campaign_wall_seconds=state['elapsed_seconds'],technical_trajectories=3,technical_optimizer_steps_per_trajectory=20,technical_receipt_cpu_seconds_lower_bound=technical_cpu,total_technical_cpu='NOT_MEASURED',peak_RSS='NOT_MEASURED',new_birth_checkpoint_bytes=sum((O/n).stat().st_size for x in births.values() for n in x if n.endswith('.pt')),timing_limit='Receipt CPU excludes startup, serialization and final evaluations; it is a lower bound. Summed overlapping subprocess wall times are not campaign elapsed time. Technical receipt CPU is also a lower bound; initial restricted-shell attempt launched no scientific job.')
 verification=dict(status='COMPLETE_VERIFIED',locked_files=lock['files'],receipts=verified,new_birth_hashes=births,logged_lives=4,checkpoints_per_logged_life=8,full_C400009_parity=True,parity_exclusions=['cpu','compose_checkpoints'],short_default_logged_parity=checks,replay_RNG_evidence='Source inspection finds no Random draws in added scoring; short model/optimizer/Torch-RNG/store/loss/score trajectories and full400009 receipt match. Python replay Random state was not independently serialized in the fixture.',reused_controls=[dict(receipt=n,sha256=C.sha(O/n)) for n in sorted(set(reused_names))],verdicts=verdicts,costs=costs,technical_archive=technical_files,review_scope='Same-lineage source/receipt verification and same-family separate read-only recount; no external replication or formal checker certification.',scientific_tuning=False,seed_substitution=False)
 C.save(R/'E14S_VERIFICATION.json',verification);C.save(R/'E14S_COSTS.json',costs);C.save(R/'E14S_METRICS.json',dict(A=A,local_nine_seed_cohort=cohort,new_BP=bp,verdicts=verdicts))
 # Fingerprints of original, unchanged deterministic input recipes, reconstructed after execution.
 import os
 os.environ['EXP45_HIDDEN']='512'
 import exp8_life as E
 def fp(v):return hashlib.sha256(json.dumps(v,separators=(',',':')).encode()).hexdigest()
 inputs={}
 for seed in (500011,600013,700027):
  panels={};train={}
  for task in E.LIFE:
   tr,te=E.split(task,.25,seed);assert not set(tr)&set(te);train[E.NAME(*task)]=dict(n=len(tr),sha256=fp(tr));panels[E.NAME(*task)]=dict(n=600,sha256=fp(E.panel(task,seed)))
  for task in E.DEV:
   tr,te=E.split(task,.5,seed);train['birth_'+E.NAME(*task)]=dict(n=len(tr),sha256=fp(tr))
   pp=[E.render(task,a,b) for a,b in Random(seed+2).sample(te,600)]
   panels['birth_'+E.NAME(*task)]=dict(n=len(pp),sha256=fp(pp))
   pp=E.panel(task,seed,300);panels['life_birth_probe_'+E.NAME(*task)]=dict(n=len(pp),sha256=fp(pp))
  taught={pair:E.triples(pair,seed,2000) for pair in E.TAUGHT_PAIRS}
  for pair,tr in taught.items():train['compose_'+''.join(pair)]=dict(n=len(tr),sha256=fp(tr))
  for pair in E.HELD_PAIRS:
   assert pair not in taught;pp=[E.render_pair(*pair,*tr) for tr in E.triples(pair,seed+1,600)];panels['compose_'+''.join(pair)]=dict(n=len(pp),sha256=fp(pp))
  pp=[]
  for pair in (('+','+'),('-','-'),('>','>'),('<','<')):
   tr=E.triples(pair,seed+1,150,exclude=frozenset(taught[pair]));assert not set(tr)&set(taught[pair]);pp.extend(E.render_pair(*pair,*t) for t in tr)
  panels['compose_within']=dict(n=len(pp),sha256=fp(pp))
  inputs[str(seed)]=dict(train=train,panels=panels)
 C.save(R/'E14S_INPUT_AUDIT.json',dict(method='Post-run fingerprints of locked source-defined recipes, not a retroactive prospective seal. Ordered single-operation split and whole operator-pair holdout.',seeds=inputs))
 lines=['# E14S decay controls and nine-seed composition confirmation','',
 'Completed all **13 scientific jobs**, with observed exit code zero and exact command/environment/receipt validation for each. No tuning or seed substitution. The default-off checkpoint knob passes the short old/default/logged trajectory check and the full seed-400009 logged/unlogged receipt comparison.','',
 '## Main findings','',
 'Increasing decay to1.0 hurts the BP arm too: final old-operation retention is85.40%,83.44%,81.15%, versus94.17%,95.75%,93.94% at0.3. The literal <=85% prediction misses by0.40 points in one seed; none reaches the alternate92% threshold. This removes the basis for treating the damage as exclusive to the local rule, but does not establish equal decay sensitivity or identify a unique mechanism. BP still outperforms the local arm at matched decay1.0, and their replay/teaching procedures differ.','',
 'The composition gate passes on the enlarged cohort: **8/9 local0.25 seeds have held-pair means>=75%**. New means are80.92%,80.92%,83.88%; every new seed has every held pair above50%. Seed400009 remains the one low mean,72.04%. These are nine seeds of one source implementation, with new births and seed-derived splits for the six fresh seeds, not independent-code replication.','',
 'The failing seed shows no sampled high-accuracy acquisition followed by collapse of `+-`: all eight recorded points are<=18.67%. However, `-+` is only1.17% at2048 and first exceeds60% at6144, refuting the predicted early acquisition. Both curves rise late in the fixed dose. “Never learned” is too absolute:1024-update sampling cannot exclude intervening transients or determine what another budget would do. The low final intermediate accuracy19.83% also prevents localizing the failure solely to the second step.','',
 '## A Matched-decay BP control','',
 'All retention means cover the eight life operations. Composition means cover four held pair types; full scratchpad exactness is separate. Existing controls are reused and not billed as new training.','']
 lines+=table(['Seed','Arm and provenance','After stage8 %','After composition %','Held final %','Held full scratchpad %'],[[x['seed'],x['label'],f"{x['stage8_retention']:.2f}",f"{x['final_retention']:.2f}",f"{x['composition']:.2f}",f"{x['full_scratchpad']:.2f}"] for x in A])+['']
 lines+=['## B All nine local seeds and new BP controls','']
 lines+=table(['Seed','Provenance','Final retention %','Held final mean %','Held full mean %','Mean>=75'],[[x['seed'],x['origin'],f"{x['final_retention']:.2f}",f"{x['composition']:.2f}",f"{x['full_scratchpad']:.2f}",str(x['composition']>=75)] for x in cohort])+['']
 for label,group in [('Local0.25 all nine seeds',cohort),('New BP0.3 references',bp)]:
  lines+=['### '+label,'','Each pair cell is **final-answer percent / full-scratchpad percent**, including own EOS in the final-answer score. `within` is separate and excluded from the held-pair mean.','']
  lines+=table(['Seed',*PAIRS,'within'],[[x['seed'],*[f"{pct(x['pairs'][k],'final'):.2f} / {pct(x['pairs'][k]):.2f}" for k in (*PAIRS,'within')]] for x in group])+['']
 lines+=['## C Composition-stage trajectories','',
 'Eight additional read-only scoring points at updates1024 through8192. All five panels are the original panels. Full underlying intermediate/final/full counts and all four logged lives are in E14S_COMPOSITION_CURVES.csv and raw receipts. A correct suffix is not automatically a correct complete scratchpad.','']
 for key,data in logged.items():
  lines+=['### '+key,'']+table(['Update',*PAIRS,'within'],[[cp['update'],*[f"{pct(cp['scores'][k],'final'):.2f}" for k in (*PAIRS,'within')]] for cp in data])+['']
 lines+=['### Seed400009 step-level scores','']+table(['Update','+- intermediate %','+- final %','+- full %','-+ intermediate %','-+ final %','-+ full %'],[[x['update'],*[f"{pct(x['scores'][k],field):.2f}" for k in ('+-','-+') for field in ('intermediate','final','exact')]] for x in curve])+['']
 lines+=['## Prospective prediction verdicts','']+table(['Prediction','Result'],[['A BP final retention<=85 in every seed','MISS:73129 is85.3958%; other two pass'],['B at most one new seed has any pair<50','PASS:zero'],['B at least7/9 local means>=75','PASS:8/9'],['C +- never above40 at recorded checkpoints','PASS:maximum18.67'],['C -+ above60 by2048','MISS:first recorded crossing6144']])+['']
 lines+=['## Per-operation scores','']
 unique={x['receipt']:x for x in A+cohort+bp}
 for name,x in unique.items():
  d=load(name);pre,end=d['stages'][-2]['scores'],d['stages'][-1]['scores'];lines+=['### '+name,'']+table(['Operation','n','Stage8 exact %','Final exact %'],[[k,v['n'],f"{pct(pre[k]):.2f}",f"{pct(v):.2f}"] for k,v in end.items() if k!='compose'])+['']
 lines+=['## Costs and provenance','']+table(['Quantity','Value'],[[k,str(v)] for k,v in costs.items()])+['',
 'All13 jobs used the previously qualified inference wrapper, with original forward/backward/update settings unchanged. The full logged C400009 life reproduces E14 exactly after removing only `cpu` and the added `compose_checkpoints` key. The other three logged lives are new seeds and have no unlogged full-life comparator. Logging affects elapsed CPU, not training data or the scoring panels. No life-weight checkpoints were requested; the three new birth checkpoints and curves are retained.','',
 'Seventeen prospective source/input hashes match. New birth hashes, exact per-job checks and reused-control checksums are in E14S_VERIFICATION.json; source-defined new-seed data fingerprints are in E14S_INPUT_AUDIT.json. The short technical fixture verifies model/optimizer/Torch-RNG/store/loss/score trajectory equality. Python replay Random state was not separately serialized; its noninterference evidence is source inspection plus exact trajectories and the full400009 receipt. PeakRSS and exact total process-tree CPU were not instrumented. No external independent replication or formal checker is claimed. E16 was untouched.','']
 (R/'E14S_REPORT.md').write_text('\n'.join(lines)+'\n')
 paragraph='''## 9.28 Codex E14S decay controls and nine-seed composition confirmation (8 November 2026)

All13 jobs finished with no tuning; the full checkpointed400009 life matches E14 exactly apart from timing and the added read-only trajectory key. BP at decay1.0 retains85.40/83.44/81.15% after composition, versus94.17/95.75/93.94% at0.3: higher decay harms BP too. The literal <=85% gate misses by0.40 points in one seed, and none reaches92%; the experiment does not establish identical decay sensitivity, since BP and local replay/teaching differ. The three new local0.25 seeds500011/600013/700027 have held composition means80.92/80.92/83.88%, with every pair above50%; combined with E13/E14,8of9 seeds reach75%, passing the enlarged gate. The failing400009 `+-` pair stays at3.33/2.83/5.00/1.17/3.83/9.17/16.33/18.67% across1024-update samples, so no sampled strong acquisition then collapse is observed. `-+` first exceeds60% at6144, not by2048 as predicted;1024-step sampling and late rising scores do not prove it could never be learned. Report, all per-pair/operation tables, costs and parity provenance: `MOD97_LOCAL_RULE_20261106/E14S_REPORT.md`. Same-source seed confirmation, not an independent-code replication; E16 was untouched.
'''
 (R/'E14S_REVIEW_SECTION_9_28.md').write_text(paragraph)
 base=(R/'E14S_REVIEW_BASE.md').read_text();assert '## 9.28 ' not in base
 (R.parent/'MINIFLY_PAST_WORK_REVIEW_FOR_MR_ZHOU_20261106.md').write_text(base.rstrip()+'\n\n'+paragraph)
 print(json.dumps(dict(status='finalized',verdicts=verdicts,costs=costs),indent=2))
if __name__=='__main__':main()
