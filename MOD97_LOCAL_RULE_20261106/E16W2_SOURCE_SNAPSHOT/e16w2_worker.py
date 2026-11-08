"""Exact teacher-batch recording/replay wrapper. Scientific source remains separately frozen."""
import os,sys,gzip,pickle,hashlib,json
from pathlib import Path
import exp8_e16w2 as E
import e14_fast_inference as F
E.X.respond=F.respond
R=Path(__file__).resolve().parent;O=E.OUT
role,seed=sys.argv[1],int(sys.argv[2]);stream=O/f'e16w2_batches_{seed}.pkl.gz'
orig=E.X.update;h=hashlib.sha256();rows=[];count=0
mode='rb' if role=='D_matched' else 'wb'
fh=gzip.open(stream,mode) if role in ('C_uni','D_matched') else None
# Named batch tuple is shared across both updates; no independent random draws.
def update(m,opt,items,local,store=None):
 global count
 flag=store is not None
 if role=='D_matched':
  recorded,write=pickle.load(fh)
  assert [p for p,t in items]==[p for p,t in recorded],('prompt mismatch',count)
  assert write==flag
  items=recorded;local=False
 elif role=='C_uni':pickle.dump((items,flag),fh,protocol=4)
 encoded=json.dumps([items,flag],separators=(',',':')).encode();h.update(len(encoded).to_bytes(8,'little'));h.update(encoded)
 comp=sum(p[0]==99 for p,t in items);mod97=sum(p[0] in (114,116,120,110) for p,t in items)
 rows.append((comp,mod97,len(items)-comp-mod97));count+=1
 return orig(m,opt,items,local,store)
E.X.update=update
try:E.life('D' if role=='D_ceiling' else 'C',seed)
finally:
 if fh:fh.close()
# Stream must contain exactly all used batches, without extra items.
if role=='D_matched':
 with gzip.open(stream,'rb') as check:
  for _ in range(count):pickle.load(check)
  try:pickle.load(check)
  except EOFError:pass
  else:raise AssertionError('extra teacher batch')
budgets=[E.STAGE_UPDATES]*4+[E.COMPOSE_UPDATES]+[E.STAGE_UPDATES]*4
names=['add97','sub97','max97','min97','compose','add73','sub73','max73','min73'];exposures=[];offset=0
for name,n in zip(names,budgets):
 part=rows[offset:offset+n];assert len(part)==n;offset+=n
 if role=='C_share' and name.endswith('73'):assert all(v[0]==E.BATCH//4 for v in part)
 exposures.append(dict(task=name,updates=n,composition=sum(v[0] for v in part),mod97=sum(v[1] for v in part),mod73=sum(v[2] for v in part)))
assert offset==count
receipt=O/f'{E.RUN_PREFIX}e8_life_{"D" if role=="D_ceiling" else "C"}{E.TAG}_{seed}.json'
meta=dict(role=role,seed=seed,training_batches=count,batch_sha256=h.hexdigest(),exposures=exposures,receipt=receipt.name,matched_learning_rule='BP' if role=='D_matched' else role,decay=os.environ['E8_WD'])
if role in ('C_uni','D_matched'):meta['teacher_stream_sha256']=hashlib.sha256(stream.read_bytes()).hexdigest()
(O/f'E16W2_BATCHES_{role}_{seed}.json').write_text(json.dumps(meta,indent=2)+'\n')
if role=='D_matched':
 teacher=json.loads((O/f'E16W2_BATCHES_C_uni_{seed}.json').read_text());assert meta['batch_sha256']==teacher['batch_sha256'] and meta['training_batches']==teacher['training_batches']
print(json.dumps(meta),flush=True)
