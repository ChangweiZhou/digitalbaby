import json,hashlib,pathlib,datetime
R=pathlib.Path(__file__).resolve().parent;O=R/'exp45'
def load(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def clean(v):
 if isinstance(v,dict):return {k:clean(x) for k,x in v.items() if k!='cpu'}
 if isinstance(v,list):return [clean(x) for x in v]
 return v
lock=load(R/'E12_LOCK.json');assert all(sha(R/n)==h for n,h in lock['files'].items())
res=load(R/'E12_RESUME_LOCK.json');assert sha(R/'e12_resume.py')==res['resume_source_sha256']
assert all(sha(O/'e12_interrupted_attempt_20261107'/n)==h for n,h in res['archived_hashes'].items())
status=load(O/'E12_STATUS.json');assert status['status']=='complete' and not status['failed_parts'] and len(status['completed'])==49
assert len({j['key'] for j in status['completed']})==49
from e12_resume import receipt_path
jobs=res['jobs'];assert len(jobs)==48
names=('add97','sub97','max97','min97','add73','sub73','max73','min73');carry=('add97','sub97','add73','sub73')
def mean(s,ks=names):return sum(100*s[k]['exact']/s[k]['n'] for k in ks)/len(ks)
def metrics(p):
 d=load(p);assert d['stages'][-1]['task']=='compose'; pre=d['stages'][-2]['scores'];end=d['stages'][-1]['scores'];comp=end['compose'];c=sum(100*comp[k]['final']/comp[k]['n'] for k in ('+-','-+','><','<>'))/4
 return dict(pre=mean(pre),final=mean(end),comp=c,pre_carry=mean(pre,carry),scores_pre=pre,scores_final=end,cpu=d['stages'][-1]['cpu'])
rows=[];cpu=load(O/'e12_profile_e8_life_C_73129.json')['stages'][-1]['cpu'];hashes={}
for j in jobs:
 item=(j['key'],j['command'],j['env'],j['part']);jp=O/f'E12_JOB_{j["key"]}.json';row=load(jp);assert row['status']=='complete' and row['returncode']==0
 p=receipt_path(item);d=load(p);hashes[str(p.relative_to(R))]=sha(p)
 if j['command'][0]=='exp12_reminder.py':
  assert d['updates']==int(j['command'][-1]) and len(d['after'])==8;cpu+=d['cpu']
  for sc in (d['before'],d['after']):
   for v in sc.values():assert 0<=v['exact']<=v['n'] and 0<=v['final']<=v['n']
 else:
  assert len(d['stages'])==(3 if j['part']=='A' else 10)
  for stage in d['stages']:
   for k,v in stage['scores'].items():
    if k=='compose':
     assert set(v)=={'+-','-+','><','<>','within'}
     for w in v.values():assert 0<=w['exact']<=w['n'] and 0<=w['final']<=w['n']
    else:assert 0<=v['exact']<=v['n'] and 0<=v['final']<=v['n']
  cpu+=d['stages'][-1]['cpu'];rows.append(dict(key=j['key'],part=j['part'],seed=d['seed'],**metrics(p)))
parity={}
a=load(O/'e12_profile_e8_life_C_73129.json');b=load(O/'e9_archived_C_73129.json');assert clean(a)==clean(b);parity['default_C73129']=True
for s in (73129,93683,116089):
 assert clean(load(O/f'e12_A_e8_life_A_rp0of1_interleaved_{s}.json'))==clean(load(O/f'e8_life_A_rp0of1_interleaved_{s}.json'));parity[f'A{s}']=True
 for stage in (0,1,2):assert (O/f'e12_A_e8_state_A_rp0of1_interleaved_{s}_stage{stage}.pt').is_file()
archive=O/'e12_interrupted_attempt_20261107';aborted=[load(p) for p in (archive/'exp45').glob('E12_JOB_*.json') if load(p)['status']=='failed']
assert len(aborted)==6
aborted_cpu_floor=0
for j in aborted:
 lines=(archive/j['log']).read_text().splitlines()
 import re
 values=[float(m.group(1)) for line in lines if (m:=re.search(r'cpu ([0-9.]+)s',line))]
 aborted_cpu_floor+=max(values,default=0)
successwall=sum(j.get('wall_seconds',0) for j in status['completed']);abortwall=sum(j['wall_seconds'] for j in aborted)
ver=dict(verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),status='ALL49_COMPLETE_VERIFIED',method='Same-agent source/hash and raw-receipt recount; not independent implementation confirmation',original_lock_unchanged=True,resume_lock_unchanged=True,archived_attempt_hashes_unchanged=True,scientific_parity_excluding_cpu_only=parity,unique_successful_jobs=49,new_full_lives=21,baseline_profile=1,reminders=27,successful_cpu_seconds=cpu,successful_summed_process_wall_seconds=successwall,interrupted_jobs=6,interrupted_process_wall_seconds=abortwall,interrupted_cpu_seconds_lower_bound=aborted_cpu_floor,total_summed_process_wall_seconds=successwall+abortwall,measured_cpu_seconds_lower_bound_including_interruptions=cpu+aborted_cpu_floor,elapsed_recorded_execution_seconds=status['elapsed_seconds'],peak_RSS='NOT_MEASURED',receipt_sha256=hashes)
(R/'E12_VERIFICATION.json').write_text(json.dumps(ver,indent=2)+'\n')
print(json.dumps({k:v for k,v in ver.items() if k!='receipt_sha256'},indent=2))
# Preserve the generated detailed scorecard, adding an outcome and explicit prediction checks.
p=R/'E12_REPORT.md';text=p.read_text();intro='''# E12 replay mechanics — final results

All49 planned jobs completed:21 new full lives,1 default reproduction,27 independent reminders. Three seeds in every new condition. These are development results on the existing E8 panels, not sealed independent confirmation. All original locked scientific source/settings remained unchanged through an orchestration-only repair.

The clearest positive is batch256 at fixed example exposure: final old-operation mean91.50–92.42%, held composition68.62–73.08%, versus76.17–79.79% and48.29–54.92% at batch64. Every seed improves both. Batch16 instead ends near26.5–27.1% retention and12.2–13.3% composition. This refutes the prediction that smaller batches would help and larger batches would harm.

The intervention changes optimizer steps as well as batch size. At the unchanged learning rate and weight decay1.0, batch16/64/256 take229376/57344/14336 total updates. AdamW's pure multiplicative decay factor over those updates is approximately exp(-229.5), exp(-57.4), exp(-14.3), respectively (this factor is not the actual trained weight magnitude). The results cannot isolate gradient averaging from cumulative decay or optimizer-state effects. Batch256 is a promising three-seed package, not a diagnosed new learning mechanism.

A: the predicted large savings is absent in all seeds. At reminder256, the erased learner reaches61.92/63.29/63.42%, but the developed birth reaches71.69/70.38/73.52%. The birth already carries general training on mod89; it is not a random untrained model. Earlier experience with these mod97/mod73 tasks confers no net mean recovery advantage at this reminder dose. At1024 updates one seed has+2.58points savings; the others remain slightly below birth. Neither the proposed positive-savings criterion nor the literal within10points-at-all-doses erasure criterion passes. Do not claim all internal information was erased or uniquely blame the output layer.

C: every seed and both replay shares lose more than10points on the earliest four operations before composition, while gains on the latest four remain below5. Recency weighting hurts old skills in this deliberately nonstationary stream.

D: buffer128 loses14.77–21.19points at stage8 relative to1024. Buffer8192 remains within3points at stage8 and at the final old-operation mean in every seed. Thus retention saturates around1024 prompts per task in this experiment. This is buffer coverage, not a demonstration of Raviv's25–30-exposure saturation. Composition can still change; report its table rather than treating all outcomes as saturated.

## Execution correction and provenance

The first controller wrongly validated unfinished sibling receipts using a broad glob after a different job finished. It stopped six running jobs in partsB/C/D and skipped others. This was a controller failure, not a scientific failure. The stopped attempt is preserved under exp45/e12_interrupted_attempt_20261107 with hashes in E12_RESUME_LOCK.json. The repair validates only the exact job receipt.34 completed jobs were retained;15 unfinished lives restarted from the identical birth, seeds and scientific settings. Original E12_LOCK.json and all its source/checkpoint hashes still match. No outcome-dependent tuning or extra candidate conditions occurred.

Reused controls: batch64/uniform1024 at share1/2 and uniform1024 at share1/4 from E8/E9. New conditions: batch16/256, recency shares1/2 and1/4, capacity128/8192, saved interleaved no-replay reruns and reminders. A full C73129 baseline and all three interleaved A reruns match archived scientific receipts exactly after removing cpu fields only. Nine stage checkpoints exist; model/optimizer/RNG are stored, while reminders intentionally fork weights with fresh matched optimizers. E12_VERIFICATION.json records the recount and receipt hashes. No formal checker or independent replication is claimed.

'''
compute=f'''## Final compute accounting

Successful jobs recorded {cpu:.2f} training/evaluation CPU seconds ({cpu/3600:.2f}CPU hours), plus at least {aborted_cpu_floor:.0f}CPU seconds from interrupted jobs' last printed checkpoints. The latter excludes work since the last printed stage, so the combined {cpu+aborted_cpu_floor:.2f}CPU seconds is a lower bound. Existing clocks exclude some setup/initial scoring, as in E9.

Summed process wall time: successful {successwall:.2f}s; six interrupted jobs {abortwall:.2f}s; total {successwall+abortwall:.2f}s ({(successwall+abortwall)/3600:.2f}process hours). These are parallel-process sums, not elapsed time. Recorded execution elapsed is {status['elapsed_seconds']:.2f}s ({status['elapsed_seconds']/3600:.2f}h), excluding downtime between the stopped controller and resumption. Peak RSS: NOT_MEASURED; the preparation's prospective RSS wording did not yield measurements. No new development checkpoints were trained; three E8 births reused. CPU contention and the controller restart made the initial forecast optimistic.

E11 was not altered. Path-integration follow-up remains NOT_RUN and is not fulfilled by E12.

'''
# Keep detailed per-seed tables and generated checks, removing misleading provisional cost paragraph.
text=text.replace('# E12 replay mechanics\n','',1)
a=text.index('## Actual compute');b=text.index('## Limits',a)
text=text[:a]+compute+text[b:]
p.write_text(intro+text)
