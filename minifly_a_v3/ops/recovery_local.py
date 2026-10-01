"""Local-first recovery transport; no locked scientific implementation is changed.

All accepted results are immutable. Github publication is independent. At eight
accepted but not remotely verified receipts, dispatch stops for infrastructure
recovery; in-flight workers finish under the frozen live budget checks.
"""
from __future__ import annotations
import datetime,fcntl,hashlib,json,os,sys,time,uuid
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import drive_science as frozen


def sha(raw): return hashlib.sha256(raw).hexdigest()
def fsyncdir(p):
 fd=os.open(p,os.O_RDONLY|os.O_DIRECTORY)
 try:os.fsync(fd)
 finally:os.close(fd)
def write_json(p,d,immutable=False):
 p.parent.mkdir(parents=True,exist_ok=True); t=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
 try:
  with t.open('x') as f:json.dump(d,f,sort_keys=True,indent=1,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
  if immutable:os.link(t,p)
  else:os.replace(t,p)
  fsyncdir(p.parent)
 finally:t.unlink(missing_ok=True)

class RecoveryEnv(frozen.Env):
 def __init__(self,root=ROOT):
  self.root=Path(root);super().__init__(self.root/'results/science',self.root/'scratch/jobs')
  self.queue=self.root/'results/recovery/local_queue';self.ack=self.root/'results/recovery/BACKUP_STATUS.json'
  self.validated={};self.cache=set();self.bootstrapping=False
 def validate(self,arm,world,raw,locked,deps):
  h=sha(raw); key=(arm,world,h,locked['lock_digest'],tuple(sorted((a,sha(b)) for a,b in deps.items())))
  if key not in self.cache:super().validate(arm,world,raw,locked,deps);self.cache.add(key)
  self.validated[(arm,world)]=h
 def disk(self):return sum(p.stat().st_size for p in (self.root/'results').rglob('*') if p.is_file())
 def backup(self):
  d=json.loads(self.ack.read_text())
  if d.get('schema')!='MINIFLY-A3-VERIFIED-BACKUP-v1' or not d.get('commit') or not d.get('verified_at_utc'):raise AssertionError('Missing verified remote backup identity')
  if d['lock_digest']!=json.loads((self.root/'SOURCE_LOCK.json').read_text())['lock_digest']:raise AssertionError('Backup lock mismatch')
  for name,h in d['receipt_sha256'].items():
   p=self.root/name
   if not name.startswith('results/science/') or not p.is_file() or sha(p.read_bytes())!=h:raise AssertionError('Backed-up receipt changed or missing: '+name)
  return d
 def tip(self):
  records=[]
  if self.queue.exists():
   for p in self.queue.glob('*.json'):
    raw=p.read_bytes(); d=json.loads(raw)
    if d['schema']!='MINIFLY-A3-LOCAL-CHECKPOINT-v2':raise AssertionError('Bad checkpoint schema')
    records.append((d['sequence'],p,sha(raw),d))
  records.sort();previous=None;prior=None
  for sequence,p,h,d in records:
   if sequence!=(1 if previous is None else previous['sequence']+1) or d['previous_checkpoint']!=previous:raise AssertionError('Broken checkpoint chain')
   if d['validated_receipts']!=len(d['receipt_sha256']):raise AssertionError('Checkpoint count mismatch')
   if prior and (prior['lock_digest']!=d['lock_digest'] or any(d['receipt_sha256'].get(k)!=v for k,v in prior['receipt_sha256'].items())):raise AssertionError('Checkpoint receipt replacement')
   previous={'path':p.relative_to(self.root).as_posix(),'sha256':h,'sequence':sequence};prior=d
  if previous is None and not self.bootstrapping:raise AssertionError('Local journal requires explicit validated bootstrap')
  return previous,prior
 def verify_remote(self):
  # Frozen hook name retained only for compatibility. Actual network verification
  # is performed by the independent publisher before this ACK can be written.
  self.backup();self.tip()
 def persist(self,message):
  self.verify_remote();previous,prior=self.tip(); roster={}
  for (a,w),h in sorted(self.validated.items()):
   p=frozen.receipt_path(self.sci,a,w)
   with p.open('rb') as f:
    if sha(f.read())!=h:raise AssertionError('Validated receipt changed')
    os.fsync(f.fileno())
   fsyncdir(p.parent);roster[p.relative_to(self.root).as_posix()]=h
  if prior and any(roster.get(k)!=v for k,v in prior['receipt_sha256'].items()):raise AssertionError('Checkpoint omitted prior receipts')
  ledger=json.loads((self.sci/'RUN_LEDGER.json').read_text());status=json.loads((self.sci/'RUN_STATUS.json').read_text())
  if len(roster)!=status['validated_receipts']:raise AssertionError('Status count mismatch')
  if prior and any(ledger[k]<prior['ledger_snapshot'][k] for k in ('active_wall_s','worker_s')):raise AssertionError('Cumulative counters decreased')
  d={'schema':'MINIFLY-A3-LOCAL-CHECKPOINT-v2','sequence':1 if previous is None else previous['sequence']+1,'previous_checkpoint':previous,'created_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'lock_digest':ledger['lock_digest'],'validated_receipts':len(roster),'receipt_sha256':roster,'ledger_snapshot':ledger,'status_snapshot':status,'message':message,'adapter_sha256':sha(Path(__file__).read_bytes()),'amendment_sha256':sha((self.root/'ops/HOST_LOSS_RECOVERY_AMENDMENT.md').read_bytes()),'off_host_backup_confirmed':False}
  p=self.queue/(str(time.time_ns())+'-'+uuid.uuid4().hex+'.json');write_json(p,d,True)
  backed=self.backup()['receipt_sha256'];pending={k:v for k,v in roster.items() if backed.get(k)!=v}
  write_json(self.root/'results/recovery/LOCAL_STATUS.json',{'checkpoint':p.relative_to(self.root).as_posix(),'checkpoint_sha256':sha(p.read_bytes()),'local_validated_receipts':len(roster),'verified_remote_receipts':len(backed),'unbacked_receipts':len(pending)})
  print(json.dumps({'local_checkpoint':p.name,'validated':len(roster),'verified_remote':len(backed),'unbacked':len(pending)}),flush=True)
  if len(pending)>=8:raise RuntimeError('Durability backlog reached8 accepted receipts; stop dispatch until verified remote backup catches up')

class RecoveryDriver(frozen.Driver):
 def persist(self,locked,done):self.save();return super().persist(locked,done)
 def fail(self,kind,detail):
  if kind=='infrastructure' and str(detail.get('error','')).startswith(('AssertionError(','JSONDecodeError(','KeyError(')):kind='integrity'
  super().fail(kind,detail)
 def bootstrap(self):
  locked=self.env.lock();self.ledger=json.loads(self.ledger_path.read_text())
  if self.ledger['lock_digest']!=locked['lock_digest']:raise AssertionError('Ledger lock mismatch')
  if self.ledger.get('failure',{} ) and self.ledger['failure']['type'] in frozen.FINAL_FAILURES:raise SystemExit('Final budget stop')
  done=self.validate_existing(locked);self.status(locked,done,{})
  self.env.bootstrapping=True
  try:self.env.persist('Validated recovery bootstrap; local checkpoint and remote coverage tracked separately')
  finally:self.env.bootstrapping=False
  return done
