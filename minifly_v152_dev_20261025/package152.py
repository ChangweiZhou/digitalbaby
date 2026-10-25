"""Compact delivery manifest. Raw checkpoints, datasets and response rows stay local."""
import json,hashlib,zipfile,time
from pathlib import Path
H=Path(__file__).resolve().parent;R=H.parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
load=lambda p:json.loads(Path(p).read_text())
def save(name,x):(H/name).write_text(json.dumps(x,indent=2)+'\n')
assert load(H/'FINAL_AUDIT.json')['status']=='ACCEPTED' and load(H/'STATE_AUDIT.json')['status']=='ACCEPTED'
counts=load(H/'COUNTS.json');save('POSTRUN_COST.json',dict(primary_CPU_including_conservative_resume_inspection_reserve=counts['CPU'],postrun_CPU_charged_conservative_reserve=10.,all_charged_CPU_upper_bound=counts['CPU']+10.,note='10 CPU seconds reserved for all saved-response reductions, state/mutation audit, source checks and compact packaging, including repeat metadata generation. No additional learner inference or teaching.',postrun_state_audit_CPU=1.164274,postrun_peak_RSS_bytes=223100928))
# Validate original arithmetic pointer and all operative source files remain unmodified.
lock=load(H/'SOURCE_LOCK.json');assert all(sha(R/p)==d for p,d in lock.items())
prov=load(R/'minifly_v152_design_20261025/PROVENANCE.json');p='PROJECT_DOCUMENTS/CURRENT_ARITHMETIC_EXPERIMENT_DESIGN.json';assert sha(R/p)==prov['local_sources'][p]
# Keep the old pointer snapshot and name actual execution, without adopting a core.
pointer=R/'PROJECT_DOCUMENTS/CURRENT_OUTPUT_DIAGNOSTIC.json';prior=R/'PROJECT_DOCUMENTS/OUTPUT_DIAGNOSTIC_POINTER_HISTORY/pre_V152_COMPLETION_20261025.json';prior.parent.mkdir(exist_ok=True)
if not prior.exists():prior.write_bytes(pointer.read_bytes())
pointer.write_text(json.dumps(dict(version='V152',status='THREE_CYCLES_COMPLETE_REFERENCE_NOT_READY',scope='Independent recurrent addition calibration; native technical parity only; native efficacy and fresh assessment not run',parent_version='V151',design='minifly_v152_design_20261025/DESIGN.md',report='minifly_v152_dev_20261025/REPORT.md',revision='minifly_v152_dev_20261025/REVISED_DESIGN.md',audit='minifly_v152_dev_20261025/FINAL_AUDIT.json',state_audit='minifly_v152_dev_20261025/STATE_AUDIT.json',new_core_adopted=False,arithmetic_pointer_replaced=False,assessment_scores_opened=False,training_record_applications=15413,main_calibration_record_applications=15360,external_training_technical_target_applications=43450,previous_pointer=str(prior.relative_to(R)),next_design=dict(version='V152-R1',status='PROPOSAL_ONLY_NOT_RUN',implementation_authorized=False,training_authorized=False)),indent=2)+'\n')
post={str(p.relative_to(R)):sha(p) for p in H.glob('*.py') if p.name in ['summarize152.py','closure152.py','package152.py']};save('POSTRUN_SOURCE_LOCK.json',post)
status=load(H/'STATUS.json');status.update(package_complete=True,final_audit='ACCEPTED',saved_state_audit='ACCEPTED',github_pushed=False,new_run_authorized=False);save('STATUS.json',status)
# Compact summaries and executable source only; never include groups' raw responses or datasets.
paths=[p for p in H.iterdir() if p.is_file() and p.suffix in ['.md','.py','.json'] and p.name not in ['PACKAGE.json','PACKAGE_VERIFICATION.json','ACTIVE.json']]
paths += [p for p in (H/'cycles').iterdir() if p.is_file() and p.suffix in ['.md','.json']]
paths += [p for p in (R/'minifly_v152_design_20261025').iterdir() if p.is_file() and p.suffix in ['.md','.json']]
paths += list((H/'groups').glob('*/RECEIPT.json'))
# Historical attempts retain diagnostic sources/traces, not the raw startup DATA.json.
for folder in ['cycle1_attempt1','cycle2_startup_attempt']:
 paths += [p for p in (H/'cycles'/folder).iterdir() if p.is_file() and p.suffix in ['.py','.json']]
paths=sorted(set(paths));entries={str(p.relative_to(R)):sha(p) for p in paths}
with zipfile.ZipFile(H/'RESULT_BUNDLE.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
 for p in paths:z.write(p,str(p.relative_to(R)))
with zipfile.ZipFile(H/'RESULT_BUNDLE.zip') as z:
 assert z.testzip() is None and set(z.namelist())==set(entries)
 for name,digest in entries.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest
package=dict(status='COMPLETE',raw_data_included=False,raw_checkpoints_included=False,raw_prediction_rows_included=False,files=entries,zip_sha256=sha(H/'RESULT_BUNDLE.zip'),zip_bytes=(H/'RESULT_BUNDLE.zip').stat().st_size)
save('PACKAGE.json',package);save('PACKAGE_VERIFICATION.json',dict(status='PASS',checked_entries=len(entries),zip_CRC_and_hashes=True,raw_data_and_checkpoints_excluded=True))
assert all(sha(R/p)==d for p,d in entries.items())
print(json.dumps({'status':status['status'],'package_entries':len(entries),'zip_bytes':package['zip_bytes'],'raw_data_included':False}))
