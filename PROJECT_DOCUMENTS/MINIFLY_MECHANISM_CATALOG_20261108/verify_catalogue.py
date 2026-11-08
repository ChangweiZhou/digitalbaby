"""Source-fidelity checks only; not a biological or repository evidence audit."""
import json,hashlib,re
from pathlib import Path
r=Path(__file__).resolve().parent
load=lambda p:json.loads(p.read_text())
s=r/'minifly_fly_mechanisms_audit_ledger_2026-11-08.json';a=load(r/'IDENTITY_LOCK.json');b=load(s)
assert hashlib.sha256(s.read_bytes()).hexdigest()==a['source']['sha256']
assert a==json.loads(re.search(r'```json\s*\n(.*?)\n```',(r/'IDENTITY_LOCK.md').read_text(),re.S).group(1))
assert [x['old_id'] for x in a['old_rows']]==list(range(1,71))
assert [x['id'] for x in a['operational_records']]==[f'F{i:02}' for i in range(1,21)]
for x,y in zip(a['old_rows'],b['old_reconstructed_crosswalk']):
 for k in ['old_id','old_title']:assert x[k]==y[k]
 for nk,ok in [('source_disposition','disposition'),('source_destination','destination'),('source_reason','reason')]:assert x[nk]==y[ok]
for x,y in zip(a['operational_records'],b['entries']):
 for k in ['id','name']:assert x[k]==y[k]
 for nk,ok in [('source_section','section'),('source_paper_keys','sources'),('source_evidence_claim','evidence'),('source_computational_proposal','algorithm'),('source_overlap_claim','overlap')]:assert x[nk]==y[ok]
for x in a['old_rows']+a['operational_records']:
 assert x['audit']['biological_evidence']['status']=='NOT_YET_VERIFIED'
 assert x['audit']['minifly_implementation']['status']==x['audit']['minifly_test']['status']=='UNKNOWN'
for p,h in load(r/'MANIFEST.json')['sha256'].items():assert hashlib.sha256((r/p).read_bytes()).hexdigest()==h
print('PASS:70 legacy identities,20 operational identities,retained source fields and all file hashes. Evidence audit remains pending.')
