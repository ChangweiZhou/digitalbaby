"""Native V176 bridge, literal sensory cache, and bounded technical qualification."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import sys,time,json,hashlib,resource,pickle,importlib.util
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent;ROOT=H.parent
sys.path.insert(0,str(ROOT/'minifly_writer_interference_dev_v1_20261027'))
import writerlib as W
from collections import Counter
W.H=H;W.COUNTS=Counter();W.TIMES=Counter();W.ASSETS={};W.ASSET_IDS={};W.TIER='technical'
W.budget=lambda:None
MODULE_START=time.process_time();W.cpu=lambda:time.process_time()-MODULE_START
F=W.F
def sha(b):return hashlib.sha256(b).hexdigest()
def save(n,v):
 p=H/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2)+'\n')
def rng(s):return np.random.default_rng(int.from_bytes(hashlib.sha256(s.encode()).digest()[:16],'big'))
def adapter():
 g=rng('V176|bridge');hi=g.permutation(512);ri=g.permutation(88);a=np.zeros((88,512),np.float32)
 for j,h in enumerate(hi):a[ri[j%88],h]=.5;a[ri[(j+512)%88],h]=.5
 return a
class Model(W.Arm):
 def __init__(self,asset=None):super().__init__();self.asset=asset;self.bridge=adapter()
 def encode(self,p):
  x,u=F.FastMemory.encode(self,p)
  if self.asset is None:return x,u
  wh,b=self.asset;h=np.maximum(0,wh[np.flatnonzero(x)].sum(0)+b);ph=self.bridge@h
  return F.FastMemory.encode(self,F.BC._norm(ph))
 def clone(self):
  q=super().clone();q.asset=self.asset;q.bridge=self.bridge;return q
def sparse_codes(m,r):
 # Value stores do not affect FE-PERM or original TopK. Exact parent record clocks.
 fe=m._new_front();t=m.time+2*F.DELTA;fe.advance(t);ix=[]
 for i in range(len(r)+1):
  view=fe.clone();view.advance(t+F.DELTA);x=F.FastMemory.encode(m,view.read())[0];ix.append(np.flatnonzero(x).astype(np.int16))
  if i<len(r):t+=F.DELTA;fe.feed(r[i],t)
 return ix
def literal_codes(m,r):
 q=m.clone();q._control(True);q._control(False);out=[]
 for i in range(len(r)+1):
  f=q.fe.clone();f.advance(q.time+F.DELTA);out.append(np.flatnonzero(F.FastMemory.encode(q,f.read())[0]).astype(np.int16))
  if i<len(r):q._event(r[i])
 return out
def qualify_export():
 start=time.process_time();m=Model();a=adapter();assert np.all((a>0).sum(0)==2)
 degree=(a>0).sum(1);assert np.sum(degree==12)==56 and np.sum(degree==11)==32
 np.savez(H/'local/bridge.npz',A=a)
 samples=[bytes(range(128)),bytes(range(128,256)),b'generic byte trace',b'']
 parity=[]
 for r in samples:
  aa=sparse_codes(m,r);bb=literal_codes(m,r);assert all(np.array_equal(x,y) for x,y in zip(aa,bb));parity.append(len(aa))
 # Fixed engineering H; no text/labels enter initialization or bridge.
 wh=rng('V176|technical|H-init').normal(0,1/np.sqrt(184),(5177,512)).astype(np.float32);b=np.zeros(512,np.float32)
 mh=Model((wh,b));mh._prompt(b'engineering');f=mh.fe.clone();f.advance(mh.time+F.DELTA);x0=F.FastMemory.encode(mh,f.read())[0]
 expected=F.FastMemory.encode(mh,F.BC._norm(a@np.maximum(0,wh[np.flatnonzero(x0)].sum(0)+b)))[0]
 actual=mh.encode(f.read())[0];assert np.array_equal(actual,expected)
 zero=Model((np.zeros_like(wh),b));assert zero.encode(np.ones(88))[0].sum()==0
 row=dict(id='technical',cohort='technical',prompt=b'engineering'.hex(),answer=b'x'.hex())
 plain=W.Arm();raw=Model();plain.train_row(row);raw.train_row(row);assert W.digest(plain)==W.digest(raw)
 off=mh.clone();before=W.synstate(off);tt=time.process_time();off.train_row(row,write=False);native_record_cpu=time.process_time()-tt
 assert off.n==2 and not np.shares_memory(off.theta,mh.theta) and off.asset is mh.asset
 # A blocked teacher still advances the physical clocks and passive stores.
 assert off.time>mh.time and all(s.brain_t==off.time for s in off.bank)
 # Zero value stores remain zero under passive decay; verify nonzero stores separately.
 decay=mh.clone()
 for s in decay.bank:s.fly.m.fast[:]=.01
 v0=W.synstate(decay).copy();decay.train_row(dict(row,id='technical-passive-decay'),write=False)
 assert not np.array_equal(v0,W.synstate(decay))
 old=W.digest(mh);tt=time.process_time();p=W.panel(mh,[row]*4,'technical');panel_cpu=(time.process_time()-tt)/4;assert old==W.digest(mh)
 checkpoint=W.checkpoint(mh,'technical.pkl');restored=W.restore(checkpoint);assert W.digest(restored)==old
 assert np.array_equal(W.querycode(restored),W.querycode(mh))
 # Writer's prospective encoding and actual target-time encoding must coincide.
 q=mh.clone();pre=W.querycode(q);q.teach_once(120,np.zeros(16),False);assert np.array_equal(pre,q._code())
 receipt=dict(status='PASS',bridge_hash=sha(a.tobytes()),parity_positions=parity,all_bytes_checked=256,raw_writer_parity=True,zero_H_zero_code=True,original_x0_input=True,prospective_writer_parity=True,frozen_assets_shared=True,roundtrip=True,read_only_panels=True,blocked_teacher_passive_clock=True,technical_training_records=4,technical_targets=8,native_record_CPU=native_record_cpu,own_TF_pair_CPU=panel_cpu,CPU=time.process_time()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,counts=dict(W.COUNTS))
 save('NATIVE_QUALIFICATION.json',receipt)
 # Source assets use exact pinned preprocessing, never natural-text value updates.
 cp=H/'assets/source/minifly_v148_pretrain_probe_dev_20261025/corpus.py';sp=importlib.util.spec_from_file_location('cp',cp);c=importlib.util.module_from_spec(sp);sp.loader.exec_module(c)
 manifest=json.loads((H/'local/DATA_MANIFEST.json').read_text());exports={};tt=time.process_time()
 for pair,entry in manifest.items():
  rec=c.records((H/'assets/corpora'/entry['A']).read_bytes());cut=next((i for i,r in enumerate(rec) if r.startswith(b'End of Project Gutenberg')),len(rec));splits=c.split(rec[:cut],40,5,10)
  for split in ('train','calib'):
   ids=[];ys=[];record_ids=[];t=time.process_time()
   for ri,r in enumerate(splits[split]):
    cc=sparse_codes(m,r)
    for x,y in zip(cc,[*r,256]):
     pad=np.full(184,-1,np.int16);pad[:len(x)]=x;ids.append(pad);ys.append(y);record_ids.append(ri)
   path=H/f'local/{pair}_{split}.npz';np.savez(path,idx=np.array(ids),y=np.array(ys,np.int64),record_id=np.array(record_ids,np.int32))
   exports[f'{pair}_{split}']=dict(targets=len(ys),records=len(splits[split]),sha256=sha(path.read_bytes()),CPU=time.process_time()-t)
   save('EXPORT_PROGRESS.json',exports)
 save('EXPORT.json',dict(status='PASS',splits=exports,CPU=time.process_time()-tt,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
 if receipt['peak_RSS_bytes']>1073741824:raise RuntimeError('RSS_CAP')
 print(json.dumps(receipt))
def forecast_benchmark():
 start=time.process_time();manifest=json.loads((H/'local/DATA_MANIFEST.json').read_text());all_lengths=[len(bytes.fromhex(r['prompt'])) for e in manifest.values() for rows in e['rows'].values() for r in rows]
 # Labels and scores never select engineering contexts. Benchmark actual roster lengths.
 wh=rng('V176|forecast|H-init').normal(0,1/np.sqrt(184),(5177,512)).astype(np.float32);b=np.zeros(512,np.float32);m=Model((wh,b));bench=[]
 for length in sorted({0,int(np.ceil(np.mean(all_lengths))),max(all_lengths)}):
  row=dict(id=f'forecast:{length}',cohort='technical',prompt=bytes((i*37+11)%256 for i in range(length)).hex(),answer='78')
  tt=time.process_time();m.train_row(row);train_cpu=time.process_time()-tt
  tt=time.process_time();W.panel(m,[row]*4,f'forecast_{length}');pair_cpu=(time.process_time()-tt)/4
  bench.append(dict(prompt_length=length,train_record_CPU=train_cpu,own_TF_pair_CPU=pair_cpu))
 save('ROSTER_BENCHMARK.json',dict(status='COMPLETE',mean_prompt_length=float(np.mean(all_lengths)),max_prompt_length=max(all_lengths),benchmarks=bench,CPU=time.process_time()-start,counts=dict(W.COUNTS),peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
 print(json.dumps(bench))
if __name__=='__main__':
 try:(forecast_benchmark if len(sys.argv)>1 and sys.argv[1]=='benchmark' else qualify_export)()
 except BaseException as e:
  save('TECHNICAL_FAILURE.json',dict(error=repr(e),counts=dict(W.COUNTS)))
  raise
