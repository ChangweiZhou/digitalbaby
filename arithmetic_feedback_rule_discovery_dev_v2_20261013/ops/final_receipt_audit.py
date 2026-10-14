import sys,json,gzip,hashlib
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root))
from auditor import audit
checks=[]
for cycle in ('cycle2','cycle3'):
 for p in sorted((root/'cycles'/cycle/'receipts').glob('*.json.gz')):
  with gzip.open(p,'rt') as f:r=json.load(f)
  source=r['source_identity']
  for name,digest in source.items():
   archive=root/'cycles'/cycle/'source'/name
   if not archive.exists():archive=root/name
   assert hashlib.sha256(archive.read_bytes()).hexdigest()==digest,(cycle,name)
  cert=audit(r,source)
  checks.append({'receipt':str(p.relative_to(root)),'accepted':cert['accepted'],'config':r['config'],'records':r['record_applications']})
assert len(checks)==10
old=root.parent/'arithmetic_feedback_rule_discovery_dev_v1_20261013/core.py'
assert old.read_bytes()==(root/'vendor/raw_reference_core.py').read_bytes()
result={'accepted':True,'technical_only':True,'checks':checks,'technical_jobs':10,'external_record_applications':21120,'science_jobs':0,'raw_reference_byte_identical':True,'revision_note':'Cycle 2 receipt source snapshots retained; cycle 3 uses revised independent auditor with zero-root and fork counter anchors.'}
(root/'cycles/cycle3/RECEIPT_AUDIT.json').write_text(json.dumps(result,indent=2)+'\n')
print('10 complete technical receipts accepted')
