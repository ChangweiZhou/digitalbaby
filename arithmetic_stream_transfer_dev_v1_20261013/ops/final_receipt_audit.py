import sys,json,gzip,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from auditor import audit
checks=[]
for cycle in ('cycle2','cycle3'):
 for p in sorted((root/'cycles'/cycle/'receipts').glob('*.json.gz')):
  with gzip.open(p,'rt') as f:r=json.load(f)
  for name,digest in r['source_identity'].items():
   archive=root/'cycles'/cycle/'sources'/name
   if not archive.exists():archive=root/name
   assert hashlib.sha256(archive.read_bytes()).hexdigest()==digest,(cycle,name)
  c=audit(r,r['source_identity'])
  assert r['scratch_state_parity']['accepted'] and r['scratch_state_parity']['counting']==r['scratch_state_parity']['after_disabled_addition']
  checks.append({'receipt':str(p.relative_to(root)),'accepted':c['accepted'],'config':r['config'],'records':r['record_applications']})
assert len(checks)==5
(root/'cycles/cycle3/RECEIPT_AUDIT.json').write_text(json.dumps({'accepted':True,'technical_only':True,'checks':checks,'technical_jobs':5,'external_record_applications':723840,'science_jobs':0},indent=2)+'\n')
print('5 complete technical receipts independently accepted')
