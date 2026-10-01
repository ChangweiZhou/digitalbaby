"""Prepare immutable GitHub archive batches; never publishes or acknowledges them.

The controller sends the exact Git-object bytes via the connector, verifies all
returned object IDs and non-force branch commit, then calls acknowledge.
"""
import argparse,base64,datetime,hashlib,json,os,subprocess,sys,uuid
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];REPO=ROOT.parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT/'ops'))
import recovery_local as local
MANIFEST='minifly_a_v3/results/recovery/archive/manifest.json'

def git(*args,env=None,binary=False):
 p=subprocess.run(['git',*args],cwd=REPO,env=env,capture_output=True,check=True)
 return p.stdout if binary else p.stdout.decode().strip()
def h(raw):return hashlib.sha256(raw).hexdigest()
def blob(raw):return hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
def jraw(d):return (json.dumps(d,sort_keys=True,indent=1)+'\n').encode()
def reconstructed(name):
 a,w=Path(name).parent.name,int(Path(name).name.split('.')[0])
 return (190054<=w<=190058 and (a,w)!=('R3',190054)) or(w==190059 and a in ('R1','R0','R0_signed'))

def prepare(parent):
 env=local.RecoveryEnv(); tip,cp=env.tip();ack=env.backup()
 if parent!=ack['commit']:
  git('merge-base','--is-ancestor',ack['commit'],parent)
  changed=git('diff','--name-only',ack['commit'],parent,'--','minifly_a_v3/results/science','minifly_a_v3/results/recovery/archive')
  if changed:raise AssertionError('Remote archive changed since ACK; reconcile before preparing')
 archive=json.loads(git('show',parent+':'+MANIFEST)); existing={x['original_path']:x for x in archive['receipts']}
 queue=ROOT/'scratch/publisher'/('batch-'+uuid.uuid4().hex);queue.mkdir(parents=True)
 files=[]
 def add(path,raw):
  p=queue/(str(len(files))+'.blob');p.write_bytes(raw);sha=git('hash-object','-w',str(p))
  assert sha==blob(raw)
  previous_entry=git('ls-tree',parent,'--',path).split()
  if previous_entry and previous_entry[2]==sha:return
  files.append({'path':path,'sha':sha,'sha256':h(raw),'bytes':len(raw),'local_path':str(p),'encoding':'utf-8'})
 for name,expected in cp['receipt_sha256'].items():
  if ack['receipt_sha256'].get(name)==expected:continue
  raw=(ROOT/name).read_bytes();assert h(raw)==expected
  target='minifly_a_v3/'+name
  if target in existing:
   assert existing[target]['original_sha256']==expected
   continue
  a=Path(name).parent.name;w=int(Path(name).name.split('.')[0]);chunks=[]
  for i,start in enumerate(range(0,len(raw),98304)):
   part=raw[start:start+98304];encoded=base64.b64encode(part)+b'\n';path=f'minifly_a_v3/results/recovery/archive/chunks/{a}/{w}/part-{i:03d}.b64'
   chunks.append({'path':path,'size':len(encoded),'sha256':h(encoded),'git_blob_sha':blob(encoded),'decoded_size':len(part),'decoded_sha256':h(part)});add(path,encoded)
  r={'arm':a,'world':w,'schema':'MINIFLY-A3-CLAUDE-RECEIPT-v1','lock_digest':cp['lock_digest'],'original_path':target,'original_size':len(raw),'original_sha256':expected,'original_git_blob_sha':blob(raw),'chunk_encoding':'base64 ASCII with final LF; independently encoded contiguous original-byte slices','chunks':chunks,'reconstruction_after_loss':reconstructed(name)}
  archive['receipts'].append(r);existing[target]=r
 original_cpraw=(ROOT/tip['path']).read_bytes();assert h(original_cpraw)==tip['sha256']
 accounting_path=ROOT/'results/recovery/host_loss_20261001/RECOVERY_ACCOUNTING.json'
 accounting=json.loads(accounting_path.read_text()) if accounting_path.exists() else {}
 counters={k:cp['ledger_snapshot'][k] for k in ('active_wall_s','worker_s')}
 if accounting:
  counters['active_wall_s']=max(counters['active_wall_s'],accounting['last_observed_active_wall_s']+accounting['additional_wall_reservation_s'])
  counters['worker_s']=max(counters['worker_s'],accounting['last_observed_worker_s']+accounting['additional_worker_reservation_s'])
 public_checkpoint={'schema':'MINIFLY-A3-PUBLIC-RECEIPT-CHECKPOINT-v1','created_at_utc':cp['created_at_utc'],'sequence':cp['sequence'],'lock_digest':cp['lock_digest'],'validated_receipts':cp['validated_receipts'],'receipt_sha256':cp['receipt_sha256'],'cumulative_resource_counters':counters,'counter_definition':'Worker-seconds include original lost computation and conservative interruption reservations; not a sum of surviving receipt durations.','checkpoint_stop_category':(cp['ledger_snapshot'].get('failure') or {}).get('type'),'runtime':json.loads((ROOT/'SOURCE_LOCK.json').read_text())['environment'],'reconstruction':{'original_receipts_restored':690,'last_validated_before_host_loss':757,'accepted_receipts_lost':67,'replacement_bytes_are_not_original_bytes':True,'authorized_at_utc':'2026-10-01T21:33:56Z'},'previous_remote_checkpoint':archive.get('latest_checkpoint')}
 cpraw=jraw(public_checkpoint)
 stamp=cp['created_at_utc'].replace(':','_').replace('+','_')
 cppath=f'minifly_a_v3/results/recovery/archive/checkpoints/{stamp}.{cp["sequence"]:06d}.json';add(cppath,cpraw)
 archive.update({'latest_checkpoint':{'path':cppath,'sha256':h(cpraw),'validated_receipts':cp['validated_receipts'],'sequence':cp['sequence']},'receipt_sha256':cp['receipt_sha256'],'total_receipts_covered':cp['validated_receipts'],'original_survivor_receipts':690,'native_receipt_files_at_base_commit':689,'lock_digest':cp['lock_digest'],'source_lock_unchanged':True,'reconstruction_authorization_at_utc':'2026-10-01T21:33:56Z','reconstruction_note':'67 accepted receipts lost with previous host; missing records regenerated at unchanged world-arm seeds/code. Archive entries mark reconstructions; no byte equality to unavailable originals is claimed.'})
 add(MANIFEST,jraw(archive))
 for name in ('ops/recovery_local.py','ops/test_recovery_local.py','ops/HOST_LOSS_RECOVERY_AMENDMENT.md','ops/prepare_archive.py','scratch/supervision/recover_20261001T2140.py'):
  add('minifly_a_v3/'+name,(ROOT/name).read_bytes())
 envgit=dict(os.environ,GIT_INDEX_FILE=str(queue/'index'));git('read-tree',parent,env=envgit)
 for f in files:git('update-index','--add','--cacheinfo','100644,'+f['sha']+','+f['path'],env=envgit)
 tree=git('write-tree',env=envgit);parenttree=git('rev-parse',parent+'^{tree}')
 req={'schema':'MINIFLY-A3-ARCHIVE-PUBLISH-REQUEST-v1','parent':parent,'base_tree':parenttree,'expected_tree':tree,'checkpoint_path':tip['path'],'checkpoint_sha256':tip['sha256'],'receipt_count':cp['validated_receipts'],'receipt_sha256':cp['receipt_sha256'],'lock_digest':cp['lock_digest'],'manifest_sha256':h(jraw(archive)),'files':files,'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
 local.write_json(queue/'request.json',req,True)
 return {'request':str(queue/'request.json'),'files':len(files),'receipt_count':req['receipt_count'],'expected_tree':tree,'parent':parent}

def acknowledge(request,commit):
 req=json.loads(Path(request).read_text());branch='refs/heads/claude/charming-clarke-u9g850'
 actual=git('ls-remote','origin',branch).split()[0];assert actual==commit
 git('fetch','--no-tags','origin',branch)
 assert git('rev-parse',commit+'^{tree}')==req['expected_tree']
 assert git('rev-list','--parents','-n','1',commit).split()==[commit,req['parent']]
 for f in req['files']:
  raw=git('show',commit+':'+f['path'],binary=True);assert h(raw)==f['sha256'] and blob(raw)==f['sha']
 manifest=json.loads(git('show',commit+':'+MANIFEST));assert manifest['receipt_sha256']==req['receipt_sha256']
 for record in manifest['receipts']:
  chunks=[]
  for c in record['chunks']:
   raw=git('show',commit+':'+c['path'],binary=True);assert h(raw)==c['sha256'];part=base64.b64decode(raw);assert h(part)==c['decoded_sha256'];chunks.append(part)
  raw=b''.join(chunks);assert h(raw)==record['original_sha256'] and blob(raw)==record['original_git_blob_sha']
 ack={'schema':'MINIFLY-A3-VERIFIED-BACKUP-v1','commit':commit,'lock_digest':req['lock_digest'],'receipt_sha256':req['receipt_sha256'],'verified_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'verification':'Remote branch exact commit; fetched exact tree and single parent; every uploaded file hash; all archive chunks reconstructed and every original receipt SHA256/Git blobSHA checked','checkpoint_path':req['checkpoint_path'],'checkpoint_sha256':req['checkpoint_sha256'],'manifest_sha256':req['manifest_sha256']}
 local.write_json(ROOT/'results/recovery/BACKUP_STATUS.json',ack)
 local.write_json(Path(request).parent/'VERIFIED.json',ack,True)
 return {'verified_remote_receipts':len(ack['receipt_sha256']),'commit':commit}

if __name__=='__main__':
 p=argparse.ArgumentParser();sp=p.add_subparsers(dest='mode',required=True);a=sp.add_parser('prepare');a.add_argument('parent');a=sp.add_parser('acknowledge');a.add_argument('request');a.add_argument('commit');args=p.parse_args()
 print(json.dumps(prepare(args.parent) if args.mode=='prepare' else acknowledge(args.request,args.commit)),flush=True)
