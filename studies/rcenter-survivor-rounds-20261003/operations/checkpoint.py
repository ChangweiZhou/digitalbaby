# SPDX-License-Identifier: GPL-3.0-or-later
"""Streaming, semantic-change checkpoints. No payload archive is retained in RAM."""
import hashlib,json,os,shutil,sys,tempfile,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'source/runtime'))
from durable import fsync_dir
CHUNK=1024*1024
STATE_NAMES=('RUN_LEDGER.json','RECOVERY_JOURNAL.json','CHECKPOINT_STATE.json','GITHUB_STATE.json','PUBLICATION_STATE.json','PERSISTENCE_STATE.json','TRANSPORT_PENDING.json','RESERVATION_ACK.json','HOST_WALL.json','RECOVERY_ACCEPTED.json')
SEMANTIC_NAMES={'RUN_LEDGER.json','RECOVERY_JOURNAL.json','RECOVERY_ACCEPTED.json'}
def stream_hash(f):
 h=hashlib.sha256()
 while True:
  b=f.read(CHUNK)
  if not b:break
  h.update(b)
 return h.hexdigest()
def sha(p):
 with Path(p).open('rb') as f:return stream_hash(f)
def canonical(d):return json.dumps(d,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def scientific_files():
 files=[]
 for group in ('source','tests','protocol','audits','receipts','results','history'):
  files.extend(p for p in (ROOT/group).rglob('*') if p.is_file() and '__pycache__' not in p.parts and '.tmp' not in p.suffixes)
 if (ROOT/'README.md').exists():files.append(ROOT/'README.md')
 files.extend(ROOT/'operations'/x for x in ('supervise.py','checkpoint.py','transport.py','controller.js','runner.js','host_session.py') if (ROOT/'operations'/x).exists())
 for p in files:assert p.resolve().is_relative_to(ROOT.resolve()),'checkpoint symlink outside study'
 return sorted(set(files))
def state_files():
 out=[ROOT/'operations'/n for n in STATE_NAMES if (ROOT/'operations'/n).is_file()]
 # Exact historical acceptance inputs and bounded worker/quarantine evidence are private recovery dependencies.
 for a in (ROOT/'audits').glob('CYCLE*_ACCEPTED.json'):
  for name in json.loads(a.read_text()).get('input_hashes',{}):
   q=ROOT/name;assert q.resolve().is_relative_to(ROOT.resolve())
   if name.startswith('operations/') and q.is_file():out.append(q)
 for group in ('attempts','quarantine'):
  out.extend(p for p in (ROOT/'operations'/group).rglob('*') if p.is_file() and '.tmp' not in p.suffixes)
 for group in ('journal','transport_journal','host_events','recovery_journal','RECOVERY_TRANSITIONS'):
  out.extend(p for p in (ROOT/'operations'/group).rglob('*.json') if p.is_file())
 return sorted(set(out))
def verify_zip(path):
 with zipfile.ZipFile(path) as z:
  assert z.testzip() is None;m=json.loads(z.read('CHECKPOINT_MANIFEST.json'))
  assert len(z.namelist())==len(set(z.namelist()))
  assert set(z.namelist())==set(m['all_payload_sha256'])|{'CHECKPOINT_MANIFEST.json'}
  for name,h in m['all_payload_sha256'].items():
   with z.open(name) as f:assert stream_hash(f)==h
  return m

def build():
 assert shutil.disk_usage(ROOT).free>=3*1024**3,'disk reserve before checkpoint'
 base=ROOT/'backups';base.mkdir(exist_ok=True);fsync_dir(base.parent)
 scientific=scientific_files();states=state_files()
 state_bytes=sum(p.stat().st_size for p in states);assert state_bytes<=64*1024**2,'metadata staging cap'
 current_backup=sum(p.stat().st_size for p in base.rglob('*') if p.is_file());reserve=128*1024**2
 assert current_backup+reserve<=2*1024**3 and shutil.disk_usage(ROOT).free-reserve>=3*1024**3,'checkpoint temporary-space reserve'
 # Mutable small metadata is snapshotted through one opened inode; later keeper heartbeats cannot race it.
 stage=Path(tempfile.mkdtemp(prefix='.checkpoint-stage-',dir=base));snap={}
 try:
  for p in states:
   q=stage/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True)
   with p.open('rb') as src,q.open('xb') as dst:shutil.copyfileobj(src,dst,CHUNK)
   snap[str(p.relative_to(ROOT))]=q
  scientific_map={str(p.relative_to(ROOT)):sha(p) for p in scientific}
  semantic={k:sha(v) for k,v in snap.items() if Path(k).name in SEMANTIC_NAMES or k.startswith(('operations/journal/','operations/recovery_journal/','operations/attempts/','operations/quarantine/','operations/RECOVERY_TRANSITIONS/')) or k.endswith('.log')}
  # Host recovery/finish events affect conservative accounting, but normal heartbeat/transport acknowledgements do not self-trigger.
  for k,v in snap.items():
   if k.startswith('operations/host_events/'):
    row=json.loads(v.read_text())
    if row.get('event') in ('recovered_unknown_session','host_closed'):semantic[k]=sha(v)
  logical={'scientific':scientific_map,'accounting':semantic};identity=hashlib.sha256(canonical(logical)).hexdigest();dest=base/f'checkpoint-{identity}.zip'
  if dest.exists():
   m=verify_zip(dest);assert m['content_identity']==identity
   return {'changed':False,'identity':identity,'path':str(dest),'sha256':sha(dest),'bytes':dest.stat().st_size,'files':len(m['all_payload_sha256'])}
  files={str(p.relative_to(ROOT)):p for p in scientific};files.update(snap)
  all_hashes={k:sha(p) for k,p in files.items()}
  manifest={'schema':'RC-SURVIVOR-CHECKPOINT-v2','content_identity':identity,'logical_identity':logical,'scientific_files':scientific_map,'all_payload_sha256':all_hashes}
  tmp=dest.with_suffix('.tmp')
  with zipfile.ZipFile(tmp,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
   for name,p in sorted(files.items()):
    info=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16
    with p.open('rb') as src,z.open(info,'w') as dst:
     while True:
      block=src.read(CHUNK)
      if not block:break
      dst.write(block)
      assert tmp.stat().st_size<=49*1024**2,'archive streaming cap'
   info=zipfile.ZipInfo('CHECKPOINT_MANIFEST.json',date_time=(1980,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;z.writestr(info,canonical(manifest))
  verify_zip(tmp)
  assert tmp.stat().st_size<=48*1024**2,'compressed checkpoint cap'
  with tmp.open('rb') as f:os.fsync(f.fileno())
  try:os.link(tmp,dest)
  finally:tmp.unlink()
  fsync_dir(base)
  assert shutil.disk_usage(ROOT).free>=3*1024**3,'disk reserve after checkpoint'
  return {'changed':True,'identity':identity,'path':str(dest),'sha256':sha(dest),'bytes':dest.stat().st_size,'files':len(all_hashes)}
 finally:
  # These metadata staging copies are generated duplicates; originals and any completed ZIP remain intact.
  shutil.rmtree(stage);fsync_dir(base)
if __name__=='__main__':print(json.dumps(build()))
