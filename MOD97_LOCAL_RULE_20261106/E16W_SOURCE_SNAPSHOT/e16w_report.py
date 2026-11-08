"""Read-only live/final E16W scorecard. No training or tuning."""
import statistics,csv
from pathlib import Path
import e16w_campaign as C
R,O=C.R,C.O
PAIRS=['+-','-+','><','<>'];OLD=['add97','sub97','max97','min97'];ALL=OLD+['add73','sub73','max73','min73']
def p(v,k='exact'):return 100*v[k]/v['n']
def mean(s):return statistics.mean(p(s['compose'][k],'final') for k in PAIRS)
def tbl(h,rs):return ['|'+'|'.join(h)+'|','|'+'|'.join(['---']*len(h))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rs]
def main():
 status=C.read(O/'E16W_STATUS.json');lock=C.read(R/'E16W_LOCK.json');complete=status['status']=='complete'
 jobs=C.plan();records=[];metrics={};cost=0.;wall=0.;rows=[]
 lines=['# E16W window-net cross-form retention','',f"Campaign status: {status['status']}. {len(status['completed'])}/10 subprocesses returned (one parity profile, nine scientific lives).",'', 'Prospective predictions: E16W_PREDICTIONS.md. New schedule: four mod97 stages, composition, four mod73 stages (nine training stages, 57,344 updates). No tuning or seed substitution.','']
 for j in jobs:
  f=O/j['receipt']
  if not f.exists():continue
  d=C.read(f)
  if j['profile']:continue
  stages=d['stages'];key=j['key'];metrics[key]=[]
  lines+=['## '+key,'']
  rr=[]
  for st in stages:
   sc=st['scores'];v=mean(sc)
   rr.append([st['task'] or 'birth',f'{v:.2f}',*[f'{p(sc["compose"][x],"final"):.2f}/{p(sc["compose"][x]):.2f}' for x in PAIRS+['within']]])
   metrics[key].append(dict(stage=st['stage'],task=st['task'],held_mean=v,scores=sc))
   for n in ALL:rows.append(dict(arm=key,stage=st['stage'],task=st['task'],operation=n,n=sc[n]['n'],exact=sc[n]['exact'],percent=p(sc[n])))
  lines+=tbl(['Stage','Held final mean',*[x+' final/full' for x in PAIRS+['within']]],rr)+['']
  lines+=tbl(['Stage',*ALL],[[st['task'] or 'birth',*[f'{p(st["scores"][x]):.2f}' for x in ALL]] for st in stages])+['']
  if len(stages)==10:records.append(j['key'])
 if rows:
  with (R/'E16W_OPERATIONS.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 verdicts={}
 if complete:
  assert len(records)==9 and len(status['completed'])==10
  for n,h in lock['files'].items():assert C.sha(R/n)==h,n
  verify=[]
  for j in jobs:
   h=C.validate(j);row=C.read(O/f'E16W_JOB_{j["key"]}.json');assert row['status']=='complete' and row['returncode']==0 and h==row['receipt_sha256'];cost+=C.read(O/j['receipt'])['stages'][-1]['cpu'];wall+=row['wall_seconds'];verify.append(dict(key=j['key'],sha256=h))
  for seed in (73129,93683,116089):
   un,ba,bp=[metrics[f'{x}_{seed}'] for x in ('C_uni','C_bal','D')]
   ul=un[5]['held_mean']-un[9]['held_mean'];bl=ba[5]['held_mean']-ba[9]['held_mean']
   old_min=min(p(st['scores'][op]) for a in (un,ba) for st in a[6:] for op in OLD)
   verdicts[str(seed)]=dict(uniform_loss=ul,balanced_loss=bl,P1_loss_ge15=ul>=15,P2_literal_balanced_loss_le_half=bl<=ul/2,P2_informative_uniform_loss_positive=ul>0,P3_all_mod97_ops_ge90=old_min>=90,minimum_mod97_operation=old_min,P4_D_held_mean_change_le3=abs(bp[9]['held_mean']-bp[5]['held_mean'])<=3)
  C.save(R/'E16W_VERIFICATION.json',dict(status='COMPLETE_VERIFIED',hashes=verify,locked_files=lock['files'],verdicts=verdicts,recorded_cpu_lower_bound_seconds=cost,subprocess_wall_sum_seconds=wall,campaign_wall_seconds=status['elapsed_seconds'],peak_RSS='NOT_MEASURED'))
  lines+=['## Prediction verdicts','']+tbl(['Seed','Uniform loss pp','Balanced loss pp','P1','P2 literal','P3 each old op','P4 BP'],[[s,f"{v['uniform_loss']:.2f}",f"{v['balanced_loss']:.2f}",v['P1_loss_ge15'],v['P2_literal_balanced_loss_le_half'],v['P3_all_mod97_ops_ge90'],v['P4_D_held_mean_change_le3']] for s,v in verdicts.items()])+['']
  lines += ['## Costs and integrity','',f'Recorded receipt CPU lower bound (nine new lives plus one parity profile): {cost:.1f} seconds; overlapping subprocess wall sum {wall:.1f} seconds; campaign wall {status["elapsed_seconds"]:.1f} seconds. Receipt CPU excludes final scoring/startup/serialization. Technical CPU total and peak RSS NOT_MEASURED. Standing births are reused and not billed as new training. Three short technical cycles are separately archived. Qualified inference-only acceleration used for all ten subprocesses; full default C73129 scientific receipt matches archived E13-A1 exactly excluding cpu only.','']
 C.save(R/'E16W_METRICS.json',dict(complete=complete,arms=metrics,verdicts=verdicts))
 lines+=['## Interpretation limits','', 'Uniform and per-stage balanced sampling have identical marginal stage probabilities with equal 1024-entry buffers. They differ in replacement/correlation; no intended composition-share increase is present. Report buffer admission files and replay counts. A negative balanced result cannot exclude increasing composition replay. BP uses joint ground truth and differs in optimizer decay and store use; it is a ceiling, not a rule-only control. A window/cell contrast also differs in window, distractors and schedule. Arithmetic carry pairs have much lower shortcut ceilings than comparison pairs; final-answer accuracy and full-scratchpad accuracy stay separate. No broader autonomous-learning or language claim follows.','']
 (R/'E16W_REPORT.md').write_text('\n'.join(lines))
 print(status['status'],len(records),'complete scientific receipts')
if __name__=='__main__':main()
