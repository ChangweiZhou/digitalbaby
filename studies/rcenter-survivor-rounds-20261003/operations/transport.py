# SPDX-License-Identifier: GPL-3.0-or-later
"""Content identities and acknowledgements for the authenticated transport adapter.

No credentials, private archives, or administrative state are in the public
projection. This helper never calls an external service or changes a remote ref.
"""
import argparse,base64,hashlib,json,os,sys,zipfile,fcntl,time,shutil,subprocess,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'source/runtime'))
from durable import replace,sha,journal,validate_journal,create,fsync_dir
from integrity import hashes,digest_map
PREFIX='studies/rcenter-survivor-rounds-20261003/'

def public_files(include_code):
 out=[]
 for group in ('protocol','audits','receipts','results'):
  out.extend(p for p in (ROOT/group).rglob('*') if p.is_file() and p.suffix in ('.json','.md','.py') and '__pycache__' not in p.parts)
 if include_code:
  out.extend(p for p in (ROOT/'source/runtime').rglob('*') if p.is_file() and p.suffix in ('.py','.json','.md','.txt') and '__pycache__' not in p.parts)
  out.extend(p for p in (ROOT/'source/baseline/vendor').rglob('*') if p.is_file() and '__pycache__' not in p.parts and (p.suffix in ('.py','.json','.npz','.txt','.md') or p.name=='COPYING'))
  out.extend(ROOT/'source/baseline/src'/n for n in ('learner.py','bootstrap.py'))
  out.append(ROOT/'source/BASELINE_PROJECTION_MANIFEST.json')
  out.extend(p for p in (ROOT/'tests').glob('*.py'))
  out.extend(ROOT/'operations'/n for n in ('supervise.py','checkpoint.py','transport.py','controller.js','runner.js','host_session.py'))
  if (ROOT/'README.md').exists():out.append(ROOT/'README.md')
 # Exclude audit copies that are private operational ledgers, connection data or source-history admin.
 out=[p for p in out if not any(x in p.name for x in ('LEDGER','CONNECTION','PRIVATE','PERSISTENCE_STATE'))]
 for p in out:assert p.resolve().is_relative_to(ROOT.resolve()),'symlink outside study'
 return sorted(set(out))
def manifest(include_code):
 files={}
 for p in public_files(include_code):
  b=p.read_bytes();rel=str(p.relative_to(ROOT));files[rel]={'sha256':hashlib.sha256(b).hexdigest(),'git_sha':hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest(),'bytes':len(b),'encoding':'base64' if p.suffix=='.npz' else 'utf-8'}
 identity=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 state_path=ROOT/'operations/PUBLICATION_STATE.json';old=json.loads(state_path.read_text()) if state_path.exists() else {'files':{}}
 changed=[p for p in files if old.get('files',{}).get(p)!=files[p]]
 return {'identity':identity,'files':files,'changed':changed,'prefix':PREFIX,'state':old,'include_code':include_code}
def assert_host():
 state=json.loads((ROOT/'operations/HOST_WALL.json').read_text());token=os.environ.get('SURVIVOR_HOST_TOKEN')
 assert token and state.get('active') is True and state.get('host_token')==token,'active serialized tool host required'
 assert state['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()
 assert 0<=time.monotonic()-state['updated_monotonic_s']<=5 and state['effective_s']<=18*3600,'stale host heartbeat or wall cap'
 with (ROOT/'operations/HOST_COORDINATOR.lock').open('a') as f:
  try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
  except BlockingIOError:pass
  else:fcntl.flock(f,fcntl.LOCK_UN);raise AssertionError('host keeper lock is not held')
 return {'active':True,'effective_s':state['effective_s']}
def limits():
 assert shutil.disk_usage(ROOT).free>=3*1024**3,'disk reserve'
 backup=sum(p.stat().st_size for p in (ROOT/'backups').rglob('*') if p.is_file())
 output=sum(p.stat().st_size for group in ('receipts','results','operations/attempts','operations/mock-tests','operations/journal','operations/transport_journal','operations/host_events','operations/numba') for p in (ROOT/group).rglob('*') if p.is_file())
 assert backup<=2*1024**3 and output<=1024**3,'backup/output cap'
 return {'backup_bytes':backup,'output_bytes':output,'free_bytes':shutil.disk_usage(ROOT).free}
def locate(name):
 p=Path(name)
 if not p.is_absolute():p=ROOT/p
 assert p.resolve().is_relative_to(ROOT.resolve()),'stored path must be consumer-local'
 return p
def rel(name):return str(Path(name).resolve().relative_to(ROOT.resolve()))
def cleanup_readbacks():
 # Only materialized copies with a byte-identical preserved local original qualify.
 from checkpoint import sha as stream_sha
 candidates=[]
 for p in [* (ROOT/'backups/readback').glob('*.zip'),* (ROOT/'backups/reconcile-readback').glob('*.zip')]:
  try:version=int(os.getxattr(p,'user.library-file-version'))
  except (OSError,ValueError):continue
  candidates.append((version,p))
 candidates.sort(key=lambda x:(x[0],x[1].name),reverse=True);removed=[]
 newest=set(sorted({v for v,p in candidates},reverse=True)[:2])
 for version,p in candidates:
  if version in newest:continue
  original=ROOT/'backups'/p.name
  if original.is_file() and stream_sha(original)==stream_sha(p):
   h=stream_sha(p);p.unlink();fsync_dir(p.parent);removed.append({'copy':rel(p),'preserved_original':rel(original),'sha256':h,'version':version})
 if removed:journal(ROOT/'operations/transport_journal',{'event':'verified_readback_duplicates_removed','files':removed})
 return removed
def checkpoint():
 limits();sys.path.insert(0,str(ROOT/'operations'));from checkpoint import build
 meta=build();assert meta.get('bytes',(Path(meta['path']).stat().st_size))<=48*1024**2,'prepared-upload review required above fixed48MiB checkpoint cap'
 assert sum(p.stat().st_size for p in (ROOT/'backups').rglob('*') if p.is_file())<=2*1024**3,'local backup cap'
 state=ROOT/'operations/CHECKPOINT_STATE.json';old=json.loads(state.read_text()) if state.exists() else None
 limits();return {'meta':meta,'prior':old,'upload_needed':old is None or old.get('content_identity')!=meta['identity']}
def private_ack(d):
 from checkpoint import sha as stream_sha,verify_zip
 meta=d['meta'];r=d['library'];p=Path(d['readback_path']);assert p.is_file() and stream_sha(p)==meta['sha256']
 m=verify_zip(p);assert m['content_identity']==meta['identity']
 state={'status':'private_backup_readback_verified','library_file_id':r['library_file_id'],'file_id':r['file_id'],'file_name':r['file_name'],
        'version':r['current_version_number'],'local_path':rel(meta['path']),'readback_path':rel(p),'sha256':meta['sha256'],'content_identity':meta['identity'],'payload_files':len(m['all_payload_sha256'])}
 replace(ROOT/'operations/CHECKPOINT_STATE.json',state)
 journal(ROOT/'operations/transport_journal',{'event':'private_readback_verified',**state})
 cleanup_readbacks();limits();return state

def restore_verified(d):
 from checkpoint import verify_zip,sha as stream_sha
 state=json.loads((ROOT/'operations/CHECKPOINT_STATE.json').read_text());archive=locate(state['readback_path']);assert stream_sha(archive)==state['sha256'];m=verify_zip(archive)
 target=ROOT/'backups'/('restore-'+state['content_identity']+'-'+uuid.uuid4().hex[:8])
 if not target.exists():
  with zipfile.ZipFile(archive) as z:needed=sum(i.file_size for i in z.infolist())
  main('preflight_io',{'extra_bytes':needed+1024*1024})
  target.mkdir();fsync_dir(target.parent)
  with zipfile.ZipFile(archive) as z:
   for name in m['all_payload_sha256']:
    q=Path(name);assert not q.is_absolute() and '..' not in q.parts
    dest=target/q;dest.parent.mkdir(parents=True,exist_ok=True)
    with z.open(name) as src,dest.open('xb') as out:shutil.copyfileobj(src,out,1024*1024);out.flush();os.fsync(out.fileno())
    fsync_dir(dest.parent)
 else:assert target.is_dir()
 assert all(stream_sha(target/name)==h for name,h in m['all_payload_sha256'].items())
 # Separate process imports the restored pure validator; no engine/learner is imported.
 code="""import sys,json;from pathlib import Path
root=Path(sys.argv[1]);sys.path.insert(0,str(root/'source/runtime'))
from validate_receipt import load
from integrity import hashes,require_runtime
from recovery import exclusive_lock,reconcile_and_plan,read
from durable import sha
out=[];source=hashes()
for p in sorted((root/'receipts').rglob('manifest.json')):
 v=load(p.parent,expected_source=source);out.append({'path':str(p.parent.relative_to(root)),'manifest_sha256':v['manifest_sha256'],'scientific_digest':v['scientific_digest']})
ledger_path=root/'operations/RUN_LEDGER.json';before=read(ledger_path);before_sha=sha(ledger_path)
mode='science' if (root/'protocol/OFFICIAL_LOCK.json').exists() else 'cycle3'
with exclusive_lock(root) as lock:plan=reconcile_and_plan(root,source,require_runtime(),mode,lock=lock)
assert plan['ledger']['worker_s']>=before['worker_s']
for a in before['attempts']:
 if a['status'] in ('running','reserved'):
  b=next(x for x in plan['ledger']['attempts'] if x.get('reservation_id')==a['reservation_id']);assert b['charged_s']==900 and b['status'] in ('interrupted','receipt_complete_pending_review')
assert 'engine' not in sys.modules and 'survivor_frozen_learner' not in sys.modules
print(json.dumps({'receipts':out,'production_plan':{k:plan[k] for k in ('missing','completed','pending_barriers','blocked','next_key')},'worker_s_before':before['worker_s'],'worker_s_after':plan['ledger']['worker_s'],'ledger_before_sha256':before_sha,'ledger_after_sha256':sha(ledger_path),'historical_validation':True,'native_events':0}))"""
 rr=subprocess.run([str(ROOT/'operations/venv/bin/python'),'-c',code,str(target)],capture_output=True,text=True);assert rr.returncode==0,rr.stderr
 result={'archive_sha256':state['sha256'],'content_identity':state['content_identity'],'restored_payload_count':len(m['all_payload_sha256']),'restored_recovery':json.loads(rr.stdout),'native_events':0,'target':rel(target)}
 replace(ROOT/'operations/RESTORE_PROOF.json',result);limits();return result

def main(action,d):
 if action not in ('manifest','data','pending','info'):assert_host()
 if action=='assert_host':return assert_host()
 if action=='stop_host':
  replace(ROOT/'operations/HOST_STOP.json',{'host_token':os.environ['SURVIVOR_HOST_TOKEN']});return {'stop_requested':True}
 if action=='info':
  return {n:json.loads((ROOT/'operations'/n).read_text()) if (ROOT/'operations'/n).exists() else None for n in ('GITHUB_STATE.json','CHECKPOINT_STATE.json','TRANSPORT_PENDING.json','HOST_WALL.json')}
 if action=='restore':return restore_verified(d)
 if action=='limits':return limits()
 if action=='preflight_io':
  current=limits();extra=int(d['extra_bytes']);assert 0<=extra<=2*1024**3
  assert current['backup_bytes']+extra<=2*1024**3 and current['free_bytes']-extra>=3*1024**3,'projected temporary/readback space exceeds reserve'
  return current
 if action=='reservation_ack':
  a=d['attempt'];state=json.loads((ROOT/'operations/CHECKPOINT_STATE.json').read_text())
  from checkpoint import verify_zip
  m=verify_zip(locate(state['readback_path']));assert m['all_payload_sha256']['operations/RUN_LEDGER.json']==d['ledger_sha256']
  ack={'reservation_id':a['reservation_id'],'key':a['key'],'source_digest':a['source_digest'],'charged_s':900.,'private_readback_verified':True,'ledger_sha256':d['ledger_sha256'],'checkpoint_sha256':state['sha256'],'content_identity':state['content_identity'],'library_file_id':state['library_file_id'],'version':state['version']}
  replace(ROOT/'operations/RESERVATION_ACK.json',ack);return ack
 if action=='pending':
  p=ROOT/'operations/TRANSPORT_PENDING.json';return json.loads(p.read_text()) if p.exists() else None
 if action=='phase':
  validate_journal(ROOT/'operations/transport_journal');replace(ROOT/'operations/TRANSPORT_PENDING.json',None if d.get('status')=='complete' else d);journal(ROOT/'operations/transport_journal',d);return {'saved':True}
 if action=='checkpoint':return checkpoint()
 if action=='manifest':return manifest(bool(d.get('include_code',False)))
 if action=='data':
  m=manifest(bool(d.get('include_code',False)));assert m['identity']==d['identity'],'publication source changed'
  rows=[]
  for rel in d['paths']:
   assert rel in m['files'];p=ROOT/rel;b=p.read_bytes();meta=m['files'][rel]
   rows.append({'path':PREFIX+rel,'relative_path':rel,**meta,'content':base64.b64encode(b).decode() if meta['encoding']=='base64' else b.decode('utf-8')})
  return rows
 if action=='reconcile_private':
  pending=json.loads((ROOT/'operations/TRANSPORT_PENDING.json').read_text());t=d['transfer']
  from checkpoint import sha as stream_sha
  expected=pending.get('sha256')
  if not expected or stream_sha(Path(t['workspace_path']))!=expected:return {'resolved':False,'reason':'latest Library bytes do not match pending upload; no automatic retry'}
  identity=pending['contentIdentity'];original=ROOT/'backups'/f'checkpoint-{identity}.zip'
  if not original.exists():os.link(t['workspace_path'],original);fsync_dir(original.parent)
  library={'library_file_id':t['library_file_id'],'file_id':t['file_id'],'file_name':t['file_name'],'current_version_number':t['current_version_number']}
  private_ack({'meta':{'identity':identity,'sha256':expected,'path':str(original)},'library':library,'readback_path':t['workspace_path']})
  main('phase',{'status':'complete','phase':'private_reconciled','contentIdentity':identity});return {'resolved':True,'remote_mutations':0}
 if action=='reconcile_public':
  pending=json.loads((ROOT/'operations/TRANSPORT_PENDING.json').read_text());assert pending['phase']=='public_commit' and d['commit']==pending['commit'] and d['tree']==pending['tree']
  current=manifest(bool(pending.get('includeCode',True)));assert current['identity']==pending['publicIdentity'],'local projection changed; independent reconciliation needed'
  assert all(d['files'].get(k)==v['git_sha'] for k,v in current['files'].items()),'remote content differs from pending projection'
  main('public_ack',{'identity':current['identity'],'include_code':current['include_code'],'commit':d['commit'],'tree':d['tree']})
  main('phase',{'status':'complete','phase':'public_reconciled','commit':d['commit']});return {'resolved':True,'remote_mutations':0}
 if action=='private_ack':return private_ack(d)
 if action=='public_ack':
  current=manifest(d['include_code']);assert current['identity']==d['identity'],'public evidence changed during sync'
  state={'identity':d['identity'],'files':current['files'],'verified_head':d['commit'],'verified_tree':d['tree'],'include_code':d['include_code'],'status':'tree_verified'}
  replace(ROOT/'operations/PUBLICATION_STATE.json',state);g=json.loads((ROOT/'operations/GITHUB_STATE.json').read_text());g.update(verified_head=d['commit'],verified_tree=d['tree'],status='programme_content_tree_verified');replace(ROOT/'operations/GITHUB_STATE.json',g);return {'acknowledged':True,'files':len(current['files'])}
 if action=='barrier':
  private=json.loads((ROOT/'operations/CHECKPOINT_STATE.json').read_text());public=json.loads((ROOT/'operations/PUBLICATION_STATE.json').read_text())
  assert private['status']=='private_backup_readback_verified' and public['status']=='tree_verified'
  path=ROOT/'operations/PERSISTENCE_STATE.json';state=json.loads(path.read_text()) if path.exists() else {'worlds':{}}
  if d.get('world') is not None:
   w=int(d['world']);manifest_path=ROOT/f'receipts/science/{w}/manifest.json';h=sha(manifest_path)
   assert public['files'][str(manifest_path.relative_to(ROOT))]['sha256']==h
   with zipfile.ZipFile(locate(private['readback_path'])) as z:assert hashlib.sha256(z.read(str(manifest_path.relative_to(ROOT)))).hexdigest()==h
   state['worlds'][str(w)]={'manifest_sha256':h,'private_readback_verified':True,'github_tree_verified':True,'private_version':private['version'],'public_commit':public['verified_head']}
  if d.get('key'):
   key=d['key'];ledger=json.loads((ROOT/'operations/RUN_LEDGER.json').read_text());a=next(a for a in reversed(ledger['attempts']) if a['key']==key and a['status']=='completed');relp=a['receipt'];relp=relp+'/manifest.json' if (ROOT/relp).is_dir() else relp;h=sha(ROOT/relp)
   assert h==a['receipt_sha256'] and public['files'][relp]['sha256']==h
   with zipfile.ZipFile(locate(private['readback_path'])) as z:assert hashlib.sha256(z.read(relp)).hexdigest()==h
   state.setdefault('jobs',{})[key]={'manifest_sha256':h,'private_readback_verified':True,'github_tree_verified':True,'private_version':private['version'],'public_commit':public['verified_head']}
  if d.get('prelaunch'):
   assert public['include_code'] is True and (ROOT/'audits/CYCLE3_ACCEPTED.json').exists()
   state.update(prelaunch_verified=True,prelaunch_source_digest=digest_map(hashes()),private_version=private['version'],public_commit=public['verified_head'])
  replace(path,state);return state
 raise ValueError(action)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action');a=p.parse_args();raw=sys.stdin.read();d=json.loads(raw) if raw.strip() else {};print(json.dumps(main(a.action,d),separators=(',',':')))
