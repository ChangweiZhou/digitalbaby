from pathlib import Path
import hashlib,json,zipfile
root=Path(__file__).resolve().parent.parent
sha=lambda b:hashlib.sha256(b).hexdigest()
source=root/'elm_v42_read_authority.py'
assert sha(source.read_bytes())=='7f50146e0e8980db8dc7ff461a4185b12cdfd9f02111de7e171a51bb340b8dce'
raw=root/'v42_results_results.zip'
with zipfile.ZipFile(raw) as z:
 assert z.testzip() is None
 h=json.loads(z.read('ARTIFACT_HASHES.json'))
 assert set(z.namelist())==set(h)|{'ARTIFACT_HASHES.json'}
 for n,d in h.items():assert sha(z.read(n))==d,n
 assert z.read(source.name)==source.read_bytes()
completed=list((root/'v42_results/completed').glob('*.json'))
assert len(completed)==4
runs=[json.loads(p.read_text()) for p in (root/'v42_results/execution').glob('*.json')]
assert len(runs)==2
assert sum(r['new_training_updates'] for r in runs)==0
assert sum(r['new_evaluation_units'] for r in runs)==2436
assert any(r['reused_jobs']==4 and r['new_evaluation_units']==0 for r in runs)
files=[source,raw,root/'v42_README.md',root/'v42_analysis_report.md',root/'V42_results.png',root/'v42_run.log',root/'v42_completed_rerun.log']
for n in ['DESIGN.md','prepare_inputs.py','reference_inputs.json','preparation_audit.md','evaluator.py','reporting.py','runner_template.py','build_runner.py','verify_cli.py','verification.json','cli_verification.json','cli_verification_run.log','independent_engineering_audit.md','final_results_audit.md','interpretation_review.md','assemble_report.py','package_handoff.py']:
 files.append(root/'v42_build'/n)
manifest={str(p.relative_to(root)):sha(p.read_bytes()) for p in files}
output=root/'v42_complete_handoff.zip'
with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED,compresslevel=9) as z:
 for p in files:z.write(p,str(p.relative_to(root)))
 z.writestr('HANDOFF_SHA256.json',json.dumps(manifest,indent=2,sort_keys=True)+'\n')
with zipfile.ZipFile(output) as z:
 assert z.testzip() is None
 assert set(z.namelist())==set(manifest)|{'HANDOFF_SHA256.json'}
 for n,d in manifest.items():assert sha(z.read(n))==d,n
report={'status':'passed','source_sha256':sha(source.read_bytes()),'results_zip_sha256':sha(raw.read_bytes()),'results_zip_bytes':raw.stat().st_size,'handoff_zip_sha256':sha(output.read_bytes()),'handoff_zip_bytes':output.stat().st_size,'handoff_members':len(manifest)+1,'completed_histories':4,'new_training_updates':0,'evaluation_units_first_run':2436,'evaluation_units_completed_rerun':0,'archive_crc_exact_manifest_and_hashes':True}
(root/'v42_delivery_verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
