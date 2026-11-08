"""Finalize E14 only after the fixed campaign finishes. No training or tuning."""
import os
os.environ['EXP45_HIDDEN']='512'
import csv, hashlib, json, math, statistics, subprocess, sys
from collections import Counter
from pathlib import Path
import e14_campaign as C

R,O=C.R,C.O
OLD=('add97','sub97','max97','min97','add73','sub73','max73','min73')
PAIRS=('+-','-+','><','<>')
ARMS=('C_wd1','C_wd025','C_BP')
def read(p):return json.loads(p.read_text())
def save(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def mean_score(scores,key='exact',names=OLD):return statistics.mean(100*scores[n][key]/scores[n]['n'] for n in names)
def comp(scores,key='final'):return mean_score(scores,key,PAIRS)
def pct(v,key):return 100*v[key]/v['n']
def md_table(header,rows):
 return ['|'+'|'.join(map(str,header))+'|','|'+'|'.join('---' for _ in header)+'|']+['|'+'|'.join(map(str,row))+'|' for row in rows]
def f(v):return 'null' if v is None else f'{v:.6f}'

def inputs():
 """Hash deterministic data definitions without fitting or scoring a model."""
 import exp8_life as E8, exp7_decon as E7
 from random import Random
 def fp(x):return hashlib.sha256(json.dumps(x,separators=(',',':')).encode()).hexdigest()
 out={}
 for seed in (*C.FRESH,*C.STANDING):
  record={}
  for harness,E in (('E8',E8),('E7',E7)):
   train={};panels={}
   if harness=='E8':
    for t in E.LIFE:
     tr,te=E.split(t,.25,seed);assert not set(tr)&set(te)
     train[E.NAME(*t)]=dict(n=len(tr),sha256=fp(tr))
     panel=E.panel(t,seed);panels[E.NAME(*t)]=dict(n=len(panel),sha256=fp(panel))
    for t in E.DEV:
     tr,_=E.split(t,.5,seed);train['birth_'+E.NAME(*t)]=dict(n=len(tr),sha256=fp(tr))
     panels['life_birth_probe_'+E.NAME(*t)]=dict(n=300,sha256=fp(E.panel(t,seed,300)))
   else:
    for op in E.OPS:
     tr,te=E.single_split(op,.5,seed);assert not set(tr)&set(te)
     train['birth_'+op]=dict(n=len(tr),sha256=fp(tr))
     pp=[E.render_single(op,a,b) for a,b in Random(seed+2).sample(te,600)]
     panels['single_'+op]=dict(n=len(pp),sha256=fp(pp))
   taught={pair:E.triples(pair,seed,2000) for pair in E.TAUGHT_PAIRS}
   for pair,items in taught.items():train['compose_'+''.join(pair)]=dict(n=len(items),sha256=fp(items))
   for pair in E.HELD_PAIRS:
    assert pair not in taught
    triples=E.triples(pair,seed+1,600)
    pp=[E.render_pair(*pair,*tr,False) if harness=='E7' else E.render_pair(*pair,*tr) for tr in triples]
    panels['compose_'+''.join(pair)]=dict(n=len(pp),sha256=fp(pp))
   pp=[]
   for pair in (('+','+'),('-','-'),('>','>'),('<','<')):
    tr=E.triples(pair,seed+1,150,exclude=frozenset(taught[pair]));assert not set(tr)&set(taught[pair])
    pp.extend([E.render_pair(*pair,*v,False) if harness=='E7' else E.render_pair(*pair,*v) for v in tr])
   panels['compose_within']=dict(n=len(pp),sha256=fp(pp))
   record[harness]=dict(train=train,panels=panels)
  out[str(seed)]=record
 save(R/'E14_INPUT_AUDIT.json',dict(method='Recomputed from locked original data-generation functions; no model training/scoring; canonical compact JSON SHA256. Hashes were produced after the campaign, not prospective sealed-panel hashes.',
  seeds=out,novelty='Four whole operator-pair templates held out; ordered single-operation operand splits. Symmetric operand swaps were not globally held out. E8 life probes of development operations use the original life-panel recipe, not a newly independent split.'))

def plots(logs):
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 colors=['#bb3d32','#2276b4','#33805c'];names=['local wd1','local wd0.25','BP wd0.3']
 specs=[('raw_gradient_cosine_to_true','Raw gradient alignment','E14_RAW_ALIGNMENT.png'),
 ('adam_descent_cosine_to_true','Adaptive Adam descent alignment','E14_ADAM_ALIGNMENT.png'),
 ('total_descent_cosine_to_true','Total applied descent alignment','E14_TOTAL_ALIGNMENT.png'),
 ('weight_norm_after','Weight norm relative to first logged pre-step norm','E14_WEIGHT_NORMS.png'),
 ('adam_step_over_weight_norm','Adaptive Adam step / pre-step weight norm','E14_STEP_WEIGHT_RATIO.png')]
 for field,title,name in specs:
  bounds=[r['parameters'][layer+'.weight'][field] for data in logs.values() for r in data for layer in ('l1','l2','l3') if r['parameters'][layer+'.weight'][field] is not None]
  fig,axes=plt.subplots(3,3,figsize=(13,9),sharex=True,constrained_layout=True)
  for si,seed in enumerate(C.STANDING):
   for li,layer in enumerate(('l1','l2','l3')):
    ax=axes[si,li]
    for arm,color,label in zip(ARMS,colors,names):
     data=logs[(seed,arm)];par=[x['parameters'][layer+'.weight'] for x in data]
     yy=[x[field] for x in par]
     if field=='weight_norm_after':yy=[y/par[0]['weight_norm_before'] for y in yy]
     ax.plot([x['global_update'] for x in data],yy,color=color,lw=1,label=label)
    ax.set_title(f'seed {seed}, {layer}.weight',fontsize=10);ax.grid(alpha=.2)
    if 'cosine' in field:ax.set_ylim(min(-.05,min(bounds)-.04),max(1.04,max(bounds)+.04))
    if field=='weight_norm_after':ax.axhline(1,color='#777777',lw=.6,ls='--')
    if si==2:ax.set_xlabel('Lifetime update')
  axes[0,0].legend(fontsize=8);fig.suptitle(title);fig.savefig(R/name,dpi=150);plt.close(fig)

def main():
 s=read(O/'E14_STATUS.json')
 if s['status']!='complete' or len(s['completed'])!=46 or s.get('failed_parts'):
  raise RuntimeError(f'NOT_READY: {s["status"]}, {len(s["completed"])}/46. No output changed.')
 lock=read(R/'E14_LOCK.json');dep=read(R/'E14_ACCELERATION_DEPLOYMENT_LOCK.json')
 verified={}
 for group in ('files','technical_artifacts'):
  for name,h in lock[group].items():assert C.sha(R/name)==h,(group,name);verified[name]=h
 assert C.sha(R/'E14_LOCK.json')==dep['original_lock_sha256']
 for name,h in dep['runtime_source_sha256'].items():assert C.sha(R/name)==h,name;verified[name]=h
 for files in read(R/'E14_BIRTHS.json')['files'].values():
  for name,h in files.items():assert C.sha(O/name)==h,name
 births,jobs,extra=C.plan();roster=births+jobs+[extra];plan={j['key']:j for j in roster}
 assert len(plan)==46 and set(plan)=={j['key'] for j in s['completed']}
 parities=[];receipts=[]
 for row in s['completed']:
  j=plan[row['key']];h=C.validate(j)
  assert row['status']=='complete'
  if j['kind']=='birth':
   canonical=read(O/f'E14_JOB_{j["key"]}.json')
   assert canonical['returncode'] is None and canonical['exit_code_observed'] is False
   row.update(canonical) # Replace cached inferred exit0; original metadata is archived.
  else:assert row['returncode']==0
  if 'original_command' in row:assert row['original_command']==j['command'] and row['command']==['e14_acceleration_worker.py',*j['command']]
  else:assert row['command']==j['command']
  assert row['env']==j['env'] and row['seed']==j['seed']
  row['receipt_sha256']=h
  receipts.append(dict(key=j['key'],kind=j['kind'],receipt=j['receipt'],sha256=h,reference=j.get('reference'),
   cpu_seconds_recorded_lower_bound=cpu(j),wall_seconds=row['wall_seconds'],wall_is_lower_bound=j['kind']=='birth',
   returncode=row['returncode'],exit_code_observed=j['kind']!='birth'))
  if 'reference' in j:parities.append(dict(key=j['key'],reference=j['reference'],scientific_exact_excluding_cpu_only=True,diagnostic_points=112 if 'diagnostic' in j else None))
 assert len(parities)==10 # Nine logged E8 histories and one E7 profile.
 s['finalization']=dict(birth_rows_mirrored_from_canonical_jobs=True,numeric_exit_codes_observed=40,numeric_exit_codes_unknown=6,all_receipt_sha256_added=True)
 save(O/'E14_STATUS.json',s)
 fast=read(R/'E14_ACCELERATION_STATUS.json');ref=read(O/'e8_life_C_73129.json');new=read(R/fast['receipt'])
 assert C.scientific(ref)==C.scientific(new)
 assert (O/'e14_C_wd1_e8_life_C_73129.diagnostics.jsonl').read_bytes()==(O/'e14_fast_qualification_e8_life_C_73129.diagnostics.jsonl').read_bytes()
 inputs()
 metrics={};predictions=[]
 def pred(name,passed,details):predictions.append(dict(name=name,passed=passed,details=details))
 for seed in C.FRESH:
  for arm in ('C','D'):
   d=read(O/f'e14_A_{arm}_e8_life_{arm}_{seed}.json');a,b=d['stages'][-2]['scores'],d['stages'][-1]['scores'];v=dict(stage8=mean_score(a),final=mean_score(b),held=comp(b['compose']))
   metrics[f'A_E8_{arm}_{seed}']=v
   if arm=='C':pred(f'A E8 seed {seed}',v['final']>=90 and v['held']>=75,v)
  bp=comp(read(O/f'e14_A_BP_e7_life_SCRbp_h512_d16384_{seed}.json')['curve'][-1]['compose'])
  for key,tag in (('batch256','_b256'),('decay025','')):
   d=read(O/f'e14_A_{key}_e7_life_SCR_h512_d16384{tag}_{seed}.json')['curve'][-1]['compose'];v=dict(held=comp(d),full=comp(d,'exact'),within=pct(d['within'],'final'),BP=bp,signed_BP_minus_local=bp-comp(d));metrics[f'A_E7_{key}_{seed}']=v
   pred(f'A E7 {key} seed {seed}',v['held']>=80 and abs(v['signed_BP_minus_local'])<=10,v)
 for seed in C.STANDING:
  a8={};a7={}
  for wd,tag in ((.1,'010'),(.25,'025'),(.5,'050'),(1.,'100')):
   path=O/(f'e13_A1_e8_life_C_{seed}.json' if wd==.25 else f'e8_life_C_{seed}.json' if wd==1 else f'e14_B_E8_wd{tag}_e8_life_C_{seed}.json')
   d=read(path);a,b=d['stages'][-2]['scores'],d['stages'][-1]['scores'];a8[wd]=[mean_score(a),mean_score(b),comp(b['compose'])]
   path=O/(f'e7_life_SCR_h512_d16384_{seed}.json' if wd==1 else f'e14_B_E7_wd{tag}_e7_life_SCR_h512_d16384_{seed}.json')
   d=read(path)['curve'][-1]['compose'];a7[wd]=[comp(d),pct(d['within'],'final'),comp(d,'exact')]
  metrics[f'B_E8_{seed}']=a8;metrics[f'B_E7_{seed}']=a7
  pred(f'B E8 plateau seed {seed}',all(abs(a-b)<=3 for a,b in zip(a8[.1],a8[.25])),dict(delta_010_minus_025=[a-b for a,b in zip(a8[.1],a8[.25])]))
  pred(f'B E7 plateau seed {seed}',all(abs(a-b)<=3 for a,b in zip(a7[.1],a7[.25])),dict(delta_010_minus_025=[a-b for a,b in zip(a7[.1],a7[.25])]))
  pred(f'B E8 wd0.5 composition drop seed {seed}',a8[.25][2]-a8[.5][2]>=5,dict(drop_points=a8[.25][2]-a8[.5][2]))
 logs={};summary=[];csvrows=[]
 for seed in C.STANDING:
  for arm in ARMS:
   letter='D' if arm=='C_BP' else 'C';data=[json.loads(x) for x in (O/f'e14_{arm}_e8_life_{letter}_{seed}.diagnostics.jsonl').read_text().splitlines()];assert len(data)==112;logs[(seed,arm)]=data
   for par in data[0]['parameters']:
    vals=[x['parameters'][par] for x in data]
    def avg(field):
     values=[x[field] for x in vals if x[field] is not None]
     return statistics.mean(values) if values else None
    v=dict(seed=seed,arm=arm,parameter=par,n=112,raw=avg('raw_gradient_cosine_to_true'),adam=avg('adam_descent_cosine_to_true'),total=avg('total_descent_cosine_to_true'),step_weight=avg('adam_step_over_weight_norm'),first_norm=vals[0]['weight_norm_before'],last_norm=vals[-1]['weight_norm_after'],min_logged_norm=min(x['weight_norm_after'] for x in vals),max_logged_norm=max(x['weight_norm_after'] for x in vals));v['endpoint_ratio']=v['last_norm']/v['first_norm'];summary.append(v)
    for x in data:csvrows.append(dict(seed=seed,arm=arm,parameter=par,global_update=x['global_update'],stage=x['stage'],stage_update=x['stage_update'],**x['parameters'][par]))
    if arm!='C_BP' and par.endswith('.weight'):
     pred(f'C raw alignment {seed} {arm} {par}',v['raw']>.8 if par.startswith('l3') else v['raw']<.5,dict(mean_raw_cosine=v['raw']))
    if arm=='C_wd1' and par in ('l1.weight','l2.weight'):pred(f'C hidden norm falls {seed} {par}',v['last_norm']<v['first_norm'],dict(endpoint_ratio=v['endpoint_ratio']))
 for seed in C.STANDING:
  for par in ('l1.weight','l2.weight'):
   vals={x['arm']:x for x in summary if x['seed']==seed and x['parameter']==par};ratio=vals['C_wd1']['step_weight']/vals['C_wd025']['step_weight']
   pred(f'C adaptive-step/weight ratio {seed} {par}',ratio>=2,dict(wd1_over_wd025=ratio))
 with (R/'E14_DIAGNOSTIC_CURVES.csv').open('w',newline='') as h:
  w=csv.DictWriter(h,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
 save(R/'E14_DIAGNOSTIC_SUMMARY.json',summary);plots(logs)
 perf=read(R/'E14_SPEED_BENCHMARK.json');bench=sum(perf[k][a] for k in ('none','episodic','matched_128_update_history') for a in ('old_seconds','new_seconds'))
 technical=read(R/'E14_TECHNICAL_ARCHIVE.json')
 costs=dict(planned_subprocesses=46,recorded_cpu_seconds_lower_bound=sum(r['cpu_seconds_recorded_lower_bound'] for r in receipts),
  per_kind_recorded_cpu_seconds_lower_bound={kind:sum(r['cpu_seconds_recorded_lower_bound'] for r in receipts if r['kind']==kind) for kind in ('birth','E8','E7')},
  subprocess_wall_sum_seconds_with_six_lower_bounds=sum(r['wall_seconds'] for r in receipts),controller_elapsed_seconds=s['elapsed_seconds'],
  extra_qualification_recorded_cpu_seconds_lower_bound=new['stages'][-1]['cpu'],extra_qualification_wall_seconds=fast['wall_seconds'],
  measured_microbenchmark_sections_wall_sum_seconds=bench,technical_short_life_receipt_cpu_seconds_lower_bound=technical['receipt_cpu_seconds_lower_bound'],technical_short_life_receipts=technical['technical_life_receipts'],technical_checks_total_cpu_seconds='NOT_MEASURED',technical_checks_total_wall_seconds='NOT_MEASURED',
  peak_RSS='NOT_MEASURED',scientific_workers_interrupted=0,scientific_trajectories_aborted=0,
  limitations='Receipt CPU timers stop before the final evaluation and omit startup/serialization; totals are recorded lower bounds, not exact process-tree CPU. Birth wall values are artifact timestamp lower bounds. Microbenchmark measured sections exclude setup/untimed checks. Controller elapsed is distinct from sums of overlapping subprocess durations. Exact total technical/controller CPU and peakRSS were not instrumented.')
 costs['recorded_cpu_with_extra_qualification_lower_bound']=costs['recorded_cpu_seconds_lower_bound']+costs['extra_qualification_recorded_cpu_seconds_lower_bound']
 costs['new_birth_checkpoint_bytes']=sum((O/j['receipt']).stat().st_size for j in births)
 save(R/'E14_COSTS.json',costs)
 verification=dict(status='COMPLETE_VERIFIED',planned_subprocesses=46,completed=46,numeric_exit_codes_observed_zero=40,numeric_birth_exit_codes_unknown=6,
  birth_completeness='Six finite checkpoints and 24576-update curves verified; process exit observed; exact numeric return code unavailable. Original inferred-zero metadata preserved.',
  immutable_input_hashes=verified,fresh_birth_hashes=read(R/'E14_BIRTHS.json')['files'],logged_history_parities=parities,
  exact_parity_exclusions=['cpu'],extra_full_qualification_receipt_exact=True,extra_full_qualification_112_diagnostic_checkpoints_exact=True,
  original_three_cycles=read(R/'E14_CYCLES.json')['all_passed'],guard_checks=read(R/'E14_GUARD_CHECK.json')['passed'],
  receipts=receipts,predictions=predictions,metrics=metrics,acceleration_scope='Inference-only qualified tested paths; logit deltas up to1.907e-5. Not a universal floating-point equivalence proof.',
  independent_replication=False,formal_checker=False,peak_RSS='NOT_MEASURED',scientific_tuning_or_seed_substitution=False)
 save(R/'E14_VERIFICATION.json',verification)
 # Start from the locked live table generator, then improve the final report without changing it.
 import e14_report
 e14_report.main();report=(R/'E14_REPORT.md').read_text()
 start=report.index('## A: fresh E8 seeds');end=report.index('## C: standing-seed diagnostic summaries')
 main_tables=report[start:end]
 lines=['# E14 decay confirmation final results','',
 '**Complete: 46 planned subprocesses.** Six fresh births, 39 lives (including the logged C baseline profile), and one additional E7 default profile. Forty numeric zero exit codes were observed; the six adopted births have verified complete artifacts and observed process exits, but their numeric exit codes are unknown. All ten required history comparisons match exactly after excluding `cpu` timing only.','',
 '## Outcome','',
 'Lower decay transfers strongly to fresh seeds, but the full E8 prediction does not pass every seed. E8 final retention is 93.46-94.54%; held composition is 84.58%, 76.67% and 72.04% (the last misses the predeclared 75% threshold). E7 low-decay held composition is 88.67-90.04%, only 2.17-3.21 points below matched fresh BP; both E7 packages pass their 80% and 10-point-gap gates in every seed.','',
 'The standing-seed curves support a broad 0.1-0.25 plateau on the declared mean endpoints. The predicted sharp five-point composition drop at 0.5 is absent in all three E8 seeds. No operating point was selected by rerunning or replacing a seed.','',
 'The diagnostic predictions do not establish a missing-feedback mechanism: hidden raw-gradient alignment is not generally below 0.5, and low Adam-descent cosine must be compared with the BP reference, which also has much lower applied-step than raw-gradient alignment. These observations are on actual training batches, including replay pseudo-labels, not an oracle for their truth. Detailed per-seed/layer verdicts follow.','',
 'Fresh-seed confirmation uses the same implementation and source recipes; it is not an independent-code replication. The operative learner is the 18-byte-window, hidden 512, 470,801-parameter E7/E8 network. Fitting at birth uses BP; local lifetime sign-feedback learning and the joint-ground-truth BP ceiling are separate evidence conditions.','',main_tables,
 '## C diagnostic summaries','',
 'Means are across 112 checkpoints per life. Raw gradient, adaptive Adam descent excluding decay, and total applied descent including decay have separate columns. Norm endpoints begin at update 512 and end at 57,344; they are not birth-to-death norms. Biases remain separate. Low-decay stability had no numeric tolerance in the lock, so no new stability pass threshold is invented.','']
 for tensor in ('weight','bias'):
  lines+=['### '+tensor.title()+' tensors','']+md_table(['Seed','Arm','Tensor','Raw cos','Adam cos','Total cos','Adam step/weight','First norm','Last norm','Last/first'],[[x['seed'],x['arm'],x['parameter'],f(x['raw']),f(x['adam']),f(x['total']),f(x['step_weight']),f(x['first_norm']),f(x['last_norm']),f(x['endpoint_ratio'])] for x in summary if x['parameter'].endswith(tensor)])+['']
 lines+=['Full first/last/min/max norms are in [E14_DIAGNOSTIC_SUMMARY.json](E14_DIAGNOSTIC_SUMMARY.json); all 6,048 parameter/checkpoint rows and stages are in [E14_DIAGNOSTIC_CURVES.csv](E14_DIAGNOSTIC_CURVES.csv).','']
 for title,name in [('Raw gradient','E14_RAW_ALIGNMENT.png'),('Adaptive Adam','E14_ADAM_ALIGNMENT.png'),('Total applied descent','E14_TOTAL_ALIGNMENT.png'),('Weight norms','E14_WEIGHT_NORMS.png'),('Adaptive step to weight ratio','E14_STEP_WEIGHT_RATIO.png')]:lines += [f'![{title}]({name})','']
 lines+=['## Prediction verdicts','']+md_table(['Prediction','Result','Observed detail'],[[x['name'],'PASS' if x['passed'] else 'MISS',json.dumps(x['details'],sort_keys=True)] for x in predictions])+['',
 '## Per-operation and per-pair scores','',
 'All percentages below have explicit denominators in the raw receipts. E8 retention uses the original eight life operations, not the four development operations. E7 standalone exact and final-suffix scores are both shown: a correct suffix after extra generated scratchpad bytes is not a complete standalone response. Held-composition means are averages over four pair types; they are not minimum-pair accuracy.','']
 def details(label,path,kind):
  d=read(path);ret=['### '+label,'']
  if kind=='E8':
   pre,last=d['stages'][-2]['scores'],d['stages'][-1]['scores']
   ret+=md_table(['Operation','n','Stage8 exact %','Final exact %'],[[k,last[k]['n'],f(pct(pre[k],'exact')),f(pct(last[k],'exact'))] for k in last if k!='compose'])+[''];pairs=last['compose']
  else:
   last=d['curve'][-1];ret+=md_table(['Standalone operation','n','Complete exact %','Final suffix %'],[[k,v['n'],f(pct(v,'exact')),f(pct(v,'final'))] for k,v in last['single'].items()])+[''];pairs=last['compose']
  ret+=md_table(['Pair','n','Intermediate %','Final answer %','Full scratchpad %'],[[k,v['n'],f(pct(v,'intermediate')),f(pct(v,'final')),f(pct(v,'exact'))] for k,v in pairs.items()])+['']
  return ret
 for j in jobs:
  if j['part']!='C':lines+=details(j['key'],O/j['receipt'],j['kind'])
 for seed in C.STANDING:
  for name,kind in [(f'e13_A1_e8_life_C_{seed}.json','E8'),(f'e8_life_C_{seed}.json','E8'),(f'e7_life_SCR_h512_d16384_{seed}.json','E7')]:lines+=details('REUSED '+name,O/name,kind)
 lines+=['## Compute cost and execution history','']+md_table(['Quantity','Observed value'],[[k,json.dumps(v)] for k,v in costs.items() if not isinstance(v,dict)])+['',
 'The two original full profiles count within the 46 planned jobs. The acceleration qualification is one additional full logged C73129 life and is charged separately. Technical cycles and measured benchmark sections are also additional; exact total CPU for setup/technical checks was not captured. Recorded CPU totals are lower bounds for the timing reasons above, not a complete process-tree bill.','',
 'An inference-only optimization evaluates the final window when generating a next byte and avoids redundant prefix store lookups. Before deployment,1,120-prompt generation,128-update model/optimizer/RNG/store/loss/reply trajectories, a complete C73129 scientific receipt and all 112 diagnostic checkpoints were exact on tested paths. Raw logits can differ by 1.907e-5; this is not universal bitwise equivalence. Microbenchmarks measured 7.40x replay inference and 1.48x replay-training speedups, not a guaranteed campaign-wide multiplier.','',
 'The old scheduler was suspended while all six running births finished, then replaced. No training worker was interrupted or birth retrained. Canonical birth JOB files mark numeric return codes unknown; the initially inferred-zero metadata and pre-handover controller snapshots are retained. Final E14_STATUS mirrors the corrected canonical birth rows. Historical PASS_NOT_DEPLOYED qualification-time status remains unchanged; the deployment lock and handover record describe the later deployment.','',
 '## Provenance and limits','',
 'E14_LOCK.json and E14_PREDICTIONS.md predate fitting. All original source, technical, deployment and birth hashes were verified. [E14_INPUT_AUDIT.json](E14_INPUT_AUDIT.json) hashes reconstructed source-defined data/panels after execution; these are provenance checks, not retrospectively sealed prospective panels. [E14_VERIFICATION.json](E14_VERIFICATION.json) records every seed, receipt, exact comparison and prediction. Reused E13/E8/E9 controls are labeled in the curve tables; fresh births and all new lives are charged. No candidate accuracy failure triggered tuning. No new mechanism, E11 change, or PI refit was performed. PeakRSS and exact total process-tree CPU are NOT_MEASURED. No formal checker seal or independent replication is claimed.','']
 (R/'E14_REPORT.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps(dict(status='finalized',completed=46,predictions=len(predictions),costs=costs),indent=2))

def cpu(j):
 d=read(O/(j['metadata'] if j['kind']=='birth' else j['receipt']))
 return (d.get('stages') or d['curve'])[-1]['cpu']

if __name__=='__main__':main()
