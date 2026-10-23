"""Independent saved-receipt validator. Does not import or execute the learner."""
from pathlib import Path
import hashlib,json,math
R=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256((R/p).read_bytes()).hexdigest()
a=json.loads((R/'audits/CYCLE1_SOURCE_REVIEW.json').read_text())
assert a['accepted'] is True
assert sha(a['report_file'])==a['report_sha256']
for p,h in a['source_hashes'].items():assert sha(p)==h,p
source_digest=hashlib.sha256(json.dumps(a['source_hashes'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
assert source_digest==a['source_digest']
rp='receipts/cycle1/smoke.json';rh=sha(rp);d=json.loads((R/rp).read_text())
assert rh=='09beebb6ad7bd26444ffcfba9427471d1962d30f0ee4d9e859f2a4854c171dc9'
assert d['passed'] is True and d['schema']=='RC-SURVIVOR-CYCLE1-TEST-v1'
assert d['fixture']=='476101747327d8d842fc37d363328c95ce7cfa0de5153b12acf6f72472e43715'
assert d['identifiability']['consistent_tables']==1 and d['identifiability']['heldout_unique']
assert [r['branch'] for r in d['records']]==['W','N_old_relation']*2
for i,r in enumerate(d['records']):
 n=i//2+1;p=r['prediction'];s=[x/1.4911274663291492 for x in p['shared']];v=[x/1.3452365735750882 for x in p['private']]
 assert len(s)==len(v)==4 and all(math.isfinite(x) for x in s+v)
 expected={'alpha1':[x+y for x,y in zip(s,v)],'alpha_half':[x+.5*y for x,y in zip(s,v)],'alpha0':s,'private_only':v}
 for name,u in expected.items():
  q=r['policies'][name];assert q['scores']==u
  ties=[k for k,x in enumerate(u) if x==max(u)]
  assert q['ties']==ties and q['emitted']==48+ties[0]
 assert p['combined']==expected['alpha1'] and p['emitted']==r['policies']['alpha1']['emitted']
 assert len(r['clock'])==8 and r['error']==0
 for c in r['clock']:
  assert c['brain_t']==c['elapsed']==c['fe_t']==165*n
  assert c['elapsed_base']==0 and c['pending_t'] is None
  assert c['bytes_seen']==14*n and c['teach_seen']==n
  assert c['last_byte_t']==165*(n-1)+13*(30/14)
 assert len(r['state'])==64 and all(c in '0123456789abcdef' for c in r['state'])
assert d['records'][0]['prediction']==d['records'][1]['prediction']
for i in (0,2):
 assert d['records'][i]['clock']==d['records'][i+1]['clock']
 assert d['records'][i]['state']!=d['records'][i+1]['state']
ledger=json.loads((R/'operations/RUN_LEDGER.json').read_text());assert len(ledger['attempts'])==1
j=ledger['attempts'][0];assert j['key']=='cycle1/test' and j['status']=='completed' and j['error'] is None
assert j['source_digest']==source_digest and j['receipt_sha256']==rh
assert j['runtime']=={'python':'3.11.15','numpy':'2.2.6','scipy':'1.14.1','numba':'0.61.2'}
assert ledger['worker_s']==ledger['active_wall_s']==j['charged_s']
assert 0<d['resources']['wall_s']<=j['charged_s']<=900
assert d['resources']['peak_rss_bytes']==j['peak_rss_bytes']<=512*1024**2
assert ledger['worker_s']<12*3600 and ledger['active_wall_s']<18*3600
paths=sorted(str(p.relative_to(R)) for p in (R/'receipts').rglob('*.json'))
assert paths==[rp],paths
out={'schema':'rcenter-cycle1-independent-receipt-check-v1','passed':True,'native_execution_performed':False,'source_files_rechecked':len(a['source_hashes']),'source_digest':source_digest,'receipt_sha256':rh,'record_rows':4,'native_records_per_branch':2,'store_clock_rows_verified':32,'max_flush_error':0.0,'completed_recorded_attempts':1,'charged_s':j['charged_s'],'peak_rss_bytes':j['peak_rss_bytes'],'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
print(json.dumps(out,indent=2,sort_keys=True))
