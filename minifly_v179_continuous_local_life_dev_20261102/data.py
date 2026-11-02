"""Train-only edit selection; literal fixed native sensory export. No model scores."""
import util as U
import sys,importlib.util,collections,pickle,time
import numpy as np
P=U.ROOT/'minifly_v176_feature_deployment_dev_20261101'
sys.path.insert(0,str(P));import native_stage as N
s=importlib.util.spec_from_file_location('source_corpus',P/'assets/source/minifly_v148_pretrain_probe_dev_20261025/corpus.py');C=importlib.util.module_from_spec(s);s.loader.exec_module(C)
WORLDS={'W1':('chesterton-thursday.txt','edgeworth-parents.txt'),'W2':('austen-emma.txt','chesterton-ball.txt')}
def hashname(s):return U.digest(s.encode())
def edits(world,train):
 counts=collections.defaultdict(collections.Counter);freq=collections.Counter(b for r in train for b in r if b!=10)
 for r in train:
  for t in range(2,len(r)):
   if r[t]!=10:counts[r[t-2:t]][r[t]]+=1
 keys=sorted([k for k,v in counts.items() if sum(v.values())>=32 and max(v.values())/sum(v.values())>=.6],key=lambda k:(-sum(counts[k].values()),U.digest(b'V179|'+world.encode()+b'|edit|'+k)))[:4]
 if len(keys)!=4:raise RuntimeError('INVALID_CORRECTION_FIXTURE_KEYS')
 top=sorted(freq,key=lambda b:(-freq[b],b))[:16];out={}
 for k in keys:
  old=min(counts[k],key=lambda b:(-counts[k][b],b));new=min([b for b in top if b!=old],key=lambda b:U.digest(b'V179|'+world.encode()+b'|replacement|'+k+bytes([b])))
  out[k]=(old,new)
 return out

def correct(r,rule):
 a=bytearray(r);changed=np.zeros(len(r)+1,bool)
 for t in range(2,len(r)):
  if r[t-2:t] in rule and r[t]==rule[r[t-2:t]][0]:a[t]=rule[r[t-2:t]][1];changed[t]=True
 return bytes(a),changed

def main():
 auth=U.load('AUTHORIZATION.json');audit=json.loads((P/'DATA_AUDIT.json').read_text());worlds={};meta={};edmeta={};export={};m=N.Model();pdir=U.H/'local';pdir.mkdir(exist_ok=True)
 for w,(an,bn) in WORLDS.items():
  sp={}
  for role,fn in [('A',an),('B',bn)]:
   raw=(P/'assets/corpora'/fn).read_bytes();assert U.digest(raw)==audit['corpora'][fn]['sha256'];rs=C.records(raw);cut=next((i for i,r in enumerate(rs) if r.startswith(b'End of Project Gutenberg')),len(rs));q=C.split(rs[:cut],*( (40,5,10) if role=='A' else (10,5,10)))
   for k in ('train','calib','held'):
    rec=q[k];h=U.digest(b''.join(len(r).to_bytes(4,'big')+r for r in rec));assert h==audit['corpora'][fn]['splits'][k]['sha256'];sp[role+'_'+k]=rec
   meta[w+role]=dict(corpus=fn,sha256=U.digest(raw),offsets=q['offsets'])
  rule=edits(w,sp['A_train']);ids=[i for i,r in enumerate(sp['A_train']) if correct(r,rule)[1].any()];ids=sorted(ids,key=lambda i:hashname(f'V179|{w}|revision-record|{i}'))[:64]
  if len(ids)!=64:raise RuntimeError('INVALID_CORRECTION_FIXTURE_RECORDS')
  sp['NOEDIT_train']=[sp['A_train'][i] for i in ids];sp['EDITED_train']=[correct(r,rule)[0] for r in sp['NOEDIT_train']]
  for k in ('calib','held'):sp['EDITED_'+k]=[correct(r,rule)[0] for r in sp['A_'+k]]
  changed=[];stable=[];old=[]
  for r in sp['A_held']:
   new,ch=correct(r,rule);changed.extend(ch);old.extend([*r,256]);stable.extend([not ch[t] and r[max(0,t-16):t]==new[max(0,t-16):t] for t in range(len(r)+1)])
  enough=sum(changed)>=24 and sum(stable)>=128
  edmeta[w]=dict(keys=[dict(key=k.hex(),old=v[0],new=v[1]) for k,v in rule.items()],revision_record_ids=ids,changed_held=int(sum(changed)),stable_held=int(sum(stable)),admitted=bool(enough))
  np.savez(pdir/f'{w}_subsets.npz',changed=np.array(changed),stable=np.array(stable),old=np.array(old))
  worlds[w]=sp
 # Freeze all edits before any learner score or fitting.
 U.save('DATA_AUDIT.json',dict(status='PASS',corpora=meta,edits=edmeta,selection='train-only; original-byte noncascading transform',sha_key_bytes='raw pair; world ASCII; record ID decimal',held_used_only_for_coverage=True))
 with (pdir/'records.pkl').open('wb') as f:pickle.dump(worlds,f,protocol=5)
 # Technical future-target independence, full byte legality, and clock parity.
 for r in [bytes(range(256)),b'ordinary prefix',b'']:
  aa=N.sparse_codes(m,r);bb=N.literal_codes(m,r);assert all(np.array_equal(x,y) for x,y in zip(aa,bb))
 a=N.sparse_codes(m,b'abcd');b=N.sparse_codes(m,b'abcx');assert all(np.array_equal(x,y) for x,y in zip(a[:4],b[:4]))
 cache={};parity=0;positions=0
 for w,sp in worlds.items():
  for name,rs in sp.items():
   t=time.process_time();idx=[];yy=[];nn=[]
   for ri,r in enumerate(rs):
    # Immutable sensory values only; no neural weights or labels are cached.
    if r not in cache:cache[r]=N.sparse_codes(m,r)
    cc=cache[r]
    for x,y in zip(cc,[*r,256]):
     row=np.full(185,5178,np.int16);row[:len(x) if len(x) else 1]=x if len(x) else 5177;idx.append(row);yy.append(y)
    nn.append(len(r)+1)
    if ri%64==0:U.save('ACTIVE.json',dict(stage='CYCLE1_SENSORY_EXPORT',world=w,split=name,record=ri,CPU=time.process_time()));U.budget()
   idx=np.array(idx);yy=np.array(yy,np.int64);n=np.array(nn,np.int32)
   # Every original A train/calibration position is compared with authenticated V176 cache.
   if name in ('A_train','A_calib'):
    ref=np.load(P/f"local/{'P1' if w=='W1' else 'P2'}_{name[2:]}.npz");rr=ref['idx'];norm=np.full_like(idx,5178)
    for j,row in enumerate(rr):
     x=row[row>=0];norm[j,:len(x) if len(x) else 1]=x if len(x) else 5177
    assert np.array_equal(idx,norm) and np.array_equal(yy,ref['y']);parity+=len(yy)
   path=pdir/f'{w}_{name}.npz';np.savez(path,idx=idx,y=yy,n=n);positions+=len(yy);export[w+'_'+name]=dict(records=len(rs),targets=len(yy),sha256=U.sha(path),CPU=time.process_time()-t)
   U.save('EXPORT.json',dict(status='RUNNING',splits=export))
 U.save('EXPORT.json',dict(status='PASS',splits=export,all_position_parent_parity=parity,full_byte_literal_parity=True,future_target_independence=True,empty_state_column=5177,padding_column=5178,total_positions=positions,cached_records=len(cache),CPU=time.process_time(),original_graph_hash=U.digest(m.bank[0].fly.m.B.data.tobytes()+m.bank[0].fly.m.B.indices.tobytes()+m.bank[0].fly.m.B.indptr.tobytes())))
 U.finish_process('data_preflight');print('DATA_EXPORT_PASS',positions,flush=True)
if __name__=='__main__':
 import json
 try:main()
 except BaseException as e:U.save('DATA_FAILURE.json',dict(error=repr(e),CPU=time.process_time()));raise
