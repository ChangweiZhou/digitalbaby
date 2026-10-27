"""Complete the source/artifact audit using saved artifacts only. No native import/run."""
import json,subprocess,hashlib
from pathlib import Path
import numpy as np
import support171 as s
s.STAGE='cycle3_saved_artifact_audit'
def main():
 # Verify complete on-disk source/fixed-asset closure against pinned Git objects.
 pin='ec38e8be63e0a448d387e4698fd619c7368ddf12'
 tree=subprocess.check_output(['git','ls-tree','-r',pin,'--','minifly_a_v3','minifly_cls_dev_20261017','minifly_v152_dev_20261025','minifly_v154_dev_20261025'],cwd=s.ROOT,text=True)
 blobs={line.split('\t',1)[1]:line.split()[2] for line in tree.splitlines()}
 anchor_path='minifly_a_v3/SOURCE_LOCK.json';anchor_raw=(s.ROOT/anchor_path).read_bytes();assert hashlib.sha1(b'blob '+str(len(anchor_raw)).encode()+b'\0'+anchor_raw).hexdigest()==blobs[anchor_path]
 anchored=json.loads(anchor_raw)['files'];previous=json.loads((s.ROOT/'minifly_v157_dev_20261026/CONTINUATION_SOURCE_LOCK.json').read_text())
 lock=s.source_lock();native_checked=0
 for path in lock:
  if path.startswith(('minifly_a_v3/','minifly_cls_dev_20261017/','minifly_v152_dev_20261025/','minifly_v154_dev_20261025/')):
   raw=(s.ROOT/path).read_bytes();git_sha=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
   if path in blobs:assert git_sha==blobs[path],path
   else:
    expected=previous.get(path) or anchored.get(path.removeprefix('minifly_a_v3/'));assert expected and hashlib.sha256(raw).hexdigest()==expected,path
   native_checked+=1
 log=[json.loads(x) for x in (s.H/'LEDGER.jsonl').read_text().splitlines()]
 commits=[x for x in log if x['event']=='REVISED_TECHNICAL_RECORD_COMMIT'];assert len(commits)==2
 before=next(x for x in log if x.get('tag')=='cycle1_roundtrip_before');after=next(x for x in log if x.get('tag')=='cycle1_roundtrip_after')
 for role in before['results']:
  a=json.loads((s.H/before['results'][role]['path']).read_text())['rows'];b=json.loads((s.H/after['results'][role]['path']).read_text())['rows'];assert a==b
 recs=[]
 for x in log:
  if x['event']=='TAPE_COMMIT':
   p=s.H/x['path'];assert s.sha(p)==x['sha256']
   with np.load(p,allow_pickle=False) as z:
    md=json.loads(z['metadata'].tobytes());assert len(md)==len(z['p']) and all('margin' in a and 'rival' in a for a in md)
    if 'EXACT' not in x['path']:assert not z['signal_norms'][:,3:].any()
   recs.append(x)
 assert s.COUNTS['technical_records']==31
 numerical=json.loads((s.H/'cycles/cycle2/NUMERICAL_CHECKS.json').read_text())
 s.save('QUALIFICATION.json',dict(status='PASS_TECHNICAL_WITH_PRESERVED_REVISIONS',native_source_files_verified=native_checked,source_closure=len(lock),numerical=numerical,record_roundtrips=len(before['results']),source_closure_completed_post_import=True,additional_imports=['fixture.py','common_platform.py'],native_sources_modified=False,records_replayed=False,model_queries_in_this_audit=0,technical_records=31,failed_first_9_record_arrays_unavailable=True,full_protocol_without_technical_deviation=False,record_benchmark_conservative_interval_CPU=before['CPU']-commits[-1]['CPU'],pair_benchmark_conservative_CPU=(after['CPU']-before['CPU'])/8,artifacts=recs,source_fix='enumerate canonical package Python as well as fixed assets; no model change',working_state_clear='explicitly reset at EOS before calibration; next-record reset was already present'))
 s.save('QUALIFIED_SOURCE_LOCK.json',lock);s.log('TECHNICAL_AUDIT_COMPLETED_WITHOUT_MODEL_RERUN',files=len(lock));s.finish('TECHNICAL_QUALIFIED_AWAITING_CALIBRATION')
if __name__=='__main__':main()
