"""Artifact integrity only; does not instantiate or train any core."""
import hashlib,json,zipfile
from pathlib import Path
root=Path(__file__).parent
package=json.loads((root/'PACKAGE.json').read_text())
qualification=json.loads((root/'QUALIFICATION.json').read_text())
lock=json.loads((root/'SOURCE_LOCK.json').read_text())
assert qualification['completed_cycles']==[1,2,3]
assert qualification['technical_correctness_passed'] and not qualification['ready_for_formal_science']
assert qualification['scientific_jobs_executed']==0
for path,expected in lock['files'].items():assert hashlib.sha256((root/path).read_bytes()).hexdigest()==expected,path
bundle=root/package['bundle'];assert hashlib.sha256(bundle.read_bytes()).hexdigest()==package['sha256']
with zipfile.ZipFile(bundle) as z:
 assert set(z.namelist())==set(package['files'])
 for path,expected in package['files'].items():
  assert hashlib.sha256(z.read(path)).hexdigest()==expected,path
  assert hashlib.sha256((root/path).read_bytes()).hexdigest()==expected,path
report={'accepted':True,'archive_files_verified':len(package['files']),
        'source_files_verified':len(lock['files']),'scientific_jobs':0,'trajectories_executed_by_verifier':0}
(root/'PACKAGE_VERIFICATION.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
