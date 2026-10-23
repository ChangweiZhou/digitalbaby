"""One bounded completion controller; no retries, native calls or fitting."""
from pathlib import Path
import json,time,subprocess,sys,os,resource,hashlib,zipfile
P=Path(__file__).resolve().parent;R=P.parent

def finish():
 start=time.monotonic()
 while not (P/'EXECUTION_RESULT.json').exists():
  if (P/'FAILURE.json').exists():return
  if time.monotonic()-start>2700:raise RuntimeError('COMPLETION_WAIT_LIMIT; inspect execution log; no restart')
  time.sleep(15)
 a=time.monotonic();r=subprocess.run([sys.executable,str(P/'audit.py')],capture_output=True,text=True)
 (P/'ops/analysis.log').write_text(r.stdout+r.stderr);r.check_returncode()
 s=json.loads((P/'SUMMARY.json').read_text());execution=json.loads((P/'EXECUTION_RESULT.json').read_text());primary=s['primary'];table=[]
 for st in (False,True):
  for head in ('DIRECT','STOP'):
   rr=[x for x in s['final_rows'] if x['STD']==st and x['arm']=='W' and x['lesion']=='NONE' and (x['head']=='DIRECT')==(head=='DIRECT')]
   values=[]
   for pool in ('OLD_final','NEW_final'):
    q=[x['nonempty_exact'] for x in rr if x['pool']==pool];values.append(100*sum(q)/len(q))
   table.append(f"| {'STD' if st else 'RAW'} | {head} | {values[0]:.2f}% | {values[1]:.2f}% |")
 report=f'''# Official STOP-ONLY component confirmation

**STD confirmation gate: {primary['passes']}.** The complete locked roster finished: 32 fresh counterfactual pairs, two separate value lives each, RAW/STD, W/N/SH and passive NEW continuations. **225,792 native record applications / 1,128,960 main events**, plus four partial qualification events. No new head fits, CREDIT run, arithmetic experiment, attention, Transformer or ELM was introduced.

The six EOS heads and RMS scales were frozen development artifacts. Gradual native warmup was unchanged. All 147 pinned source/birth-package files, imported local runtime modules, head files and the arithmetic pointer were verified. Qualification covered canonical birth, all 257 byte/EOS symbols, payload sign invariance in all six heads, read-only state identity and restoration of all mutable checkpoint fields. Rest1 checkpoints were added to the operations wrapper; native learning equations did not change.

## Primary result

STD is the sole confirmatory storage condition. The pair-level STOP-minus-direct conditional stopping gain averages the three predeclared usable checkpoints, head seeds and two value lives. Mean gain: **{primary['conditional_gain']}**; two-sided paired 95% Student-t interval: **{primary['paired_95_interval']}**. Positive pair component decisions: **{primary['positive_pairs']}/32**. Defined denominators: **{primary['all_denominators_defined']}**. Confirmation requires the interval lower endpoint above zero and positive conditional plus unconditional complete-recall gains in every pair. Raw pair rows and denominators remain in SUMMARY.json; undefined subsets fail rather than being dropped. RAW is descriptive, not a second unadjusted confirmation claim.

## Whole-output scores

| Storage | Readout | Final OLD complete | Final NEW complete |
|---|---|---:|---:|
{chr(10).join(table)}

Complete scores require nonempty exact payload and the learner's own EOS, averaging all frozen seeds rather than selecting the best one. Conditional stopping is an evaluator diagnostic on payload-correct native responses, never an inference feature or replacement for whole-output performance. Better stopping is not simultaneous reliable OLD/NEW memory.

## Scope and controls

This is **E1**, recovery of explicitly taught random associations at fixed surrogate key addresses, not E3 unseen-addition transfer or a learned byte-key interface. Native W/N/within-length SH controls, ZERO/FOREIGN lesions, absent keys and passive NEW branches are retained in the raw emitted-output files. A strict/canonicalized taught-key table can solve these facts after observation; the balanced-length frequency prior emits empty and gets 11.11% over all values but zero nonempty exact. These are evaluator countermodels, not candidate backends. Payload bits are native signs; the head can change EOS only. No native protection or general-core adoption follows from this component gate.

The four development pairs and the earlier V1 evaluation facts are excluded. New keys were collision-checked before new values. The analysis unit is one pair, not a head seed, fact, or value life. Results are conditional on the one canonical anatomy and original calibration. The paired interval makes that limited population claim; it supplies no deployment accuracy guarantee or broad biological conclusion.

## Accounting and audit

Execution: {execution['resources']}. Reused historical training exposures, calibration and prior head-fitting supervision remain chargeable information/compute; operational reuse does not make them free learning. The new roster contains 384 base histories and 128 correlated passive continuations, 32,256 shared-OLD logical equivalents and 258,048 total main logical record equivalents. All direct/STOP comparisons reuse the same native states.

FINAL_AUDIT.json records independent artifact reconstruction in a separate process, not independent human review. It verifies all saved outputs and primary reductions without replaying native learning or fitting. SOURCE_LOCK.json and PARENT_CLOSURE.json predate the roster. RUNTIME_MODULE_CLOSURE.json was saved after birth qualification and before fresh values. Full native checkpoints and raw tensors remain local; compact archive parts omit checkpoints, which remain indexed by hash. This bounded dense prototype uses global encoding/native operations and copies for read-only audits; no sparse-compute or scaling claim is established.

Stopping-only confirmation does not fix write-dependent OLD damage. Preserve the separate arithmetic track and failed CREDIT recipe. Any new native intervention requires its own prospective definition and qualification; do not automatically enlarge a decoder or launch a host.
'''
 (P/'REPORT.md').write_text(report)
 files=[f for f in P.rglob('*') if f.is_file() and not {'scratch','__pycache__'}.intersection(f.relative_to(P).parts) and f.name not in ('MANIFEST.json','PACKAGE.json','COMPLETION.json') and not f.name.startswith('RESULT_BUNDLE') and f.name!='completion.log']
 manifest={str(f.relative_to(P)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
 (P/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
 # Several small independent ZIPs avoid an oversized single publication artifact.
 selected=[f for f in files if 'checkpoints' not in f.relative_to(P).parts]+[P/'MANIFEST.json'];parts=[];current=[];size=0
 for f in selected:
  if current and size+f.stat().st_size>40*1024**2:parts.append(current);current=[];size=0
  current.append(f);size+=f.stat().st_size
 if current:parts.append(current)
 receipts=[]
 for i,items in enumerate(parts,1):
  path=P/f'RESULT_BUNDLE_PART_{i:02d}.zip'
  with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
   for f in items:z.write(f,str(f.relative_to(P)))
  with zipfile.ZipFile(path) as z:assert z.testzip() is None
  receipts.append(dict(file=path.name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
 package=dict(status='COMPLETE',parts=receipts,native_checkpoints='Retained individually, indexed in MANIFEST.json; omitted from compact archive',standalone=False,audit_requires='parent repository sources and frozen evidence assets')
 (P/'PACKAGE.json').write_text(json.dumps(package,indent=2)+'\n')
 u=resource.getrusage(resource.RUSAGE_CHILDREN)
 (P/'COMPLETION.json').write_text(json.dumps(dict(status='COMPLETE',confirmation_passes=primary['passes'],report=str(P/'REPORT.md'),package=str(P/'PACKAGE.json'),audit_packaging_wall_seconds=time.monotonic()-a,analysis_children_CPU_seconds=u.ru_utime+u.ru_stime,controller_CPU_seconds=time.process_time()),indent=2)+'\n')
 ptr=R/'PROJECT_DOCUMENTS/CURRENT_EXTERNAL_MEMORY_EXPERIMENT_DESIGN.json';d=json.loads(ptr.read_text())
 if d.get('version')=='STOP_ONLY_CONFIRM_V1':
  d.update(status='OFFICIAL_CONFIRMATION_COMPLETE',report=str(P/'REPORT.md'),package=str(P/'PACKAGE.json'),confirmation_passes=primary['passes'],last_completed_exercise=dict(version='STOP_ONLY_CONFIRM_V1',records=225792,pairs=32,head_fits=0,report=str(P/'REPORT.md'),package=str(P/'PACKAGE.json'),evidence='E1 stopping component only',host_qualified=False));ptr.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
 print(json.dumps(dict(status='COMPLETE',primary=primary,package_parts=len(receipts))),flush=True)

if __name__=='__main__':
 awake=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())])
 try:finish()
 except BaseException as ex:(P/'ANALYSIS_FAILURE.json').write_text(json.dumps(dict(error=repr(ex),retry=False),indent=2)+'\n');raise
 finally:awake.terminate();awake.wait()
