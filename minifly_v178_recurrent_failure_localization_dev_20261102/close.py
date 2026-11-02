"""Artifact-only reduction; imports no learner and executes no trajectories."""
import json,hashlib,zipfile,time
from pathlib import Path
P=Path(__file__).resolve().parent;ROOT=P.parent
load=lambda n:json.loads((P/n).read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(P/n).write_text(json.dumps(v,indent=2)+'\n')
r=load('RESULTS.json');a,b,e=r['stages'];d=[load('A_DIAGNOSTICS.json'),load('B_EXTENSION_DIAGNOSTICS.json')]
checks={}
checks['source_lock']=all(sha(ROOT/p)==h for p,h in load('SOURCE_LOCK.json')['files'].items())
checks['checkpoint_hashes']=all(sha(P/s['checkpoint'])==s['checkpoint_sha256'] for s in r['stages'])
checks['qualification']=load('NUMERICAL_QUALIFICATION.json').get('status')=='PASS'
checks['fixed_route']=[s['key'] for s in r['stages']]==['A','B','B_EXTENSION']
checks['reference_gate']=a['final_calibration']['bits']<=1.35 and a['test']['bits']<=1.35 and abs(a['test']['bits']-1.1516320951135754)<=.05
checks['failure_gate']=b['final_calibration']['bits']>1.5 and b['test'] is None and e['final_calibration']['bits']>1.5
checks['extension_prefix']=e['curve'][:10]==b['curve']
checks['counts']=sum(s['record_applications'] for s in r['stages'])+64==load('COUNTS.json')['training_records']==3064
checks['targets']=load('COUNTS.json')['scored_target_applications']==3064*112
for s in r['stages']:
 for split in ['final_train','final_calibration','test']:
  v=s.get(split)
  if v:checks[s['key']+'_'+split+'_reduction']=abs(sum(v['line_sums'])/sum(v['counts'])-v['bits'])<1e-12
checks['no_execution_failure']=not (P/'FAILURE.json').exists()
assert all(checks.values()),checks
st=load('STATUS.json')
summary=dict(verdict=r['verdict'],reference_numeric_difference=a['test']['bits']-1.1516320951135754,
 main_training_record_applications=3000,technical_training_record_applications=64,total_training_record_applications=3064,total_scored_training_target_applications=343168,
 derivative_only_qualification_exposures=32,derivative_only_qualification_targets=3584,
 probe_training_record_exposures=sum(x['extra_probe_record_exposures'] for x in d),probe_training_target_applications=sum(x['extra_probe_target_applications'] for x in d),
 CPU_execution_seconds=st['CPU'],CPU_preflight_allowance_seconds=120,CPU_conservative_accounted_seconds=st['CPU']+120,peak_execution_RSS_bytes=st['peak_RSS_bytes'],
 unexecuted=['C','fresh guard A','fresh guard C'],core_adopted=False,arithmetic_claim=False,local_learning_claim=False,replication_worlds=1)
save('SUMMARY.json',summary)
report='''# V178 three-cycle result

The original dense V164 reference reproduced. The first failure occurred in the sparse package, before introducing a trainable gate. The prescribed extension improved it but did not pass; downstream stages were not executed.

| Checkpoint | Training bits/target | Calibration | Held test |
|---|---:|---:|---:|
'''
for s in r['stages']:
 t=s.get('test');report+=f"| {s['key']} epoch {s['epochs']} | {s['final_train']['bits']:.6f} | {s['final_calibration']['bits']:.6f} | {t['bits'] if t else 'unopened'} |\n"
report+='''
Lower is better; an uninformed eight-symbol predictor costs 3 bits/target. These are delayed-copy scores, not addition accuracy or complete own-byte/EOS performance.

The dense test differs from the historical 1.151632095 by −0.000002751 bits. This is strict numerical reproduction under the predeclared tolerance, with recovered source hashes and forward/gradient/update qualification. It is not bitwise identity across historical GPU and current CPU environments.

The sparse epoch-10 calibration loss 2.478026 fails the 1.50 gate. Its same-state, same-optimizer extension reached 1.839156 at epoch 20 (test 1.909482), still above the gate. Its original ten-epoch verdict remains **SPARSE_PACKAGE_FAIL_10**; the extension is **UNRESOLVED_OR_ARCHITECTURE_LIMITED_AT_20**. There was no reset, replacement initialization, or rescue of the ladder. C and the fresh guard remain untested.

## What the diagnostics support

The frozen-state linear probe converged in both executed endpoint diagnoses. Dense probe test loss was 0.871584 (78.25% token accuracy); sparse-at-20 probe test loss was 1.662473 (57.84%). A better fitted head recovers some information from sparse states, but it does not erase the representation gap. Probe answers are evaluator outputs, not model outputs, and probe fitting cost is separate.

Both endpoints have a slightly negative measured train-only Lyapunov estimate (dense −0.02439; sparse −0.02905). The sampled eight-step maximum singular values are 7.79 and 8.11. This argues against diagnosing the result from a single global “vanishing” or “exploding” number. These summaries do not establish causation or describe every direction/sequence. Sparse recurrence-off and cross-edge-off lesions raise calibration loss to 3.042 and 3.068; the trained model uses its recurrent and inter-module paths. Lesions also shift the state distribution.

Sparsification changes topology and reduces effective recurrent weights from 65,536 to 4,096. Matching the initial operator norm does not match rank, spectral shape, optimization geometry, or trainable capacity. Thus this isolates the first failing **package**, not the unique causal feature. A single historical data/initialization setting cannot establish population reliability.

## Three cycles actually performed

1. **Design → audit → technical test → revise execution:** recovered literal V164 source and data semantics, checked source hashes, independent forward calculation, gradients and optimizer behavior, then benchmarked execution. Kept the faster plain recurrence and cached indices/masks only. No learned states were cached. Compiled derivative timing was 0.06336 CPU seconds versus 0.04049 plain; no GPU conversion was justified for this tiny CPU workload.
2. **Design → audit → reference test → retain:** executed the original dense ten-epoch schedule. Same-checkpoint train/calibration and held test reproduced the reference; retained the frozen equations and gates.
3. **Design → audit → contrast test → revise interpretation:** sparse ten-epoch gate failed; executed only its predeclared ten-epoch continuation with the saved model, Adam state and ordering generator. Stopped at the first failed package and revised the next-step document. This was a gated three-cycle exercise, not three independent successful replications; no new architecture or post-score tuning was run.

## Cost and scope

Main learner training: 3,000 record applications / 336,000 scored targets. Technical optimizer qualification: 64 / 7,168. Total: **3,064 / 343,168**. Repeated records and positions are not independent worlds. Additional derivative-only checks: 32 record exposures / 3,584 targets, no optimizer updates. Frozen readout fitting: **14,200 repeated record exposures / 1,590,400 target applications** (83 + 59 objective evaluations), not part of the learner's training count. Training/calibration/held evaluations and dynamics diagnostics add read-only work and are included in worker CPU, rather than being hidden as free compute.

The sequential science worker used 10.23 CPU seconds and peaked at 714 MB RSS. A conservative 120-second preflight allowance gives 130.23 accounted CPU seconds; it is an allowance, not an exact per-process preflight measurement. Resource admission's largest-route forecast was intentionally conservative. This does not establish a measured end-to-end speedup against an unoptimized full campaign. No trajectories were repeated for this report.

This reference uses BPTT and the literal V164 schedule, with no added gradual warmup. It is explicitly an instrumental reference, not a local-rule or warmup candidate. No native integration, arithmetic result, deployment, or adoption is claimed.
'''
(P/'REPORT.md').write_text(report)
(P/'REVISED_DESIGN.md').write_text('''# V178-R1: evidence-constrained revision

Status: documentation revision after the authorized V178 exercise; no additional run authorized or launched. Original source, D1 gates and results remain immutable. No new version is allocated here.

## Revised question

The dense recurrence can learn this delayed-copy task under the exact recovered schedule. The next uncertainty is why the fixed sparse package learns substantially slower and remains worse after twice the exposure. Gate design and local credit assignment are downstream uncertainties, not explanations already tested by this result.

## Recommended next controlled design

Before any native/local transfer, specify a bounded, prospectively locked sparse-reference comparison. Keep the dense reference, data stream, initial input/output draws, optimizer, evaluated checkpoints and matched exposure/CPU reporting. Separate recurrent edge budget from modular topology using a degree-matched non-modular mask control, with identical initialization and effective parameter count. Keep the current modular mask as the comparison and retain operator-norm matching while recording the broader singular spectrum. Do not silently tune gain or claim that norm matching equalizes dynamics.

Use a prespecified small set of fresh paired seeds, calibration-only decisions, and an untouched held guard. Freeze the exact budget and comparison before collecting those scores. Report learning curves and cost to a fixed calibration criterion, not only an endpoint winner. Treat inability to meet the criterion as a censored result. The observed twenty-epoch extension does not prove that still more training cannot work.

Only after a sparse reference passes its prospective gate should the trainable-gate arm be evaluated. Only after reference feasibility should local credit and gradual-warmup candidates be compared under matched total data and compute. Preserve end-to-end outputs and prohibit attention, arithmetic-specific representations/rules, or a decoder that supplies model answers. Unseen addition remains a later goal; this delayed-copy diagnostic does not substitute for it.

## Limits and release rules

No arithmetic, E3 reasoning, biological mechanism or adoption claim follows. The present sparse contrast changes both capacity and topology; it is a failure-localization result. The proposed degree-matched control would discriminate topology at one fixed capacity, and would still not by itself isolate every initialization or optimization effect. Any follow-up needs a separate predeclared design and authorization; no extra stages, warmup, optimizer or epochs are appended to V178.
''')
save('THREE_CYCLES.json',dict(status='COMPLETE_GATED_EXERCISE',cycles=[dict(cycle=1,test='qualification and speed benchmark',revision='literal faster execution; no learned-state cache'),dict(cycle=2,test='dense original reference epoch10',revision='reference retained; strict numeric reproduction'),dict(cycle=3,test='sparse epoch10 and predeclared same-state extension20',revision='stop ladder; document capacity/topology confound')],independent_replications=False))
save('FINAL_AUDIT.json',dict(status='ACCEPTED',scope='artifact-only reducer in a separate process; not an external independent scientific audit',checks=checks,no_trajectory_rerun=True,limitations=['single historical setting','sparse capacity and topology confounded','no local-rule or arithmetic result','preflight compute conservatively allocated']))
# Compact release excludes local checkpoints, recovered source copies and raw data.
files=sorted(f for f in P.iterdir() if f.is_file() and f.suffix in ['.json','.md','.py'] and f.name not in ['PACKAGE.json','STATUS.json','ACTIVE.json'])
archive=P/'RESULT_BUNDLE.zip'
with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
 for f in files:z.write(f,f.name)
save('PACKAGE.json',dict(status='COMPLETE',archive=archive.name,sha256=sha(archive),files={f.name:sha(f) for f in files},excludes=['raw data','checkpoints','local dependencies','recovered historical source copies']))
save('STATUS.json',dict(status='COMPLETED',verdict=r['verdict'],final_audit='ACCEPTED',package='COMPLETE'))
print(json.dumps(summary,indent=2))
