"""Evaluator-owned arithmetic data. Never imported by core, geometry or native engine."""
from v4_common import np

def dataset(seed,train_n=128,test_n=64):
 rng=np.random.default_rng(seed);seen=set();out=[]
 while len(out)<train_n+test_n:
  lengths=rng.integers(2,5,2);a=int(rng.integers(10**(lengths[0]-1),10**lengths[0]));b=int(rng.integers(10**(lengths[1]-1),10**lengths[1]))
  identity=tuple(sorted((a,b)))
  if identity in seen:continue
  seen.add(identity);out.append((f'{a}+{b}='.encode(),str(a+b).encode()))
 return out[:train_n],out[train_n:]

def score(model,records):
 before=model.digest();cpu=__import__('time').process_time();rows=[]
 for p,a in records:
  o=model.generate(p);got=bytes.fromhex(o['payload']);rows.append(dict(prompt=p.hex(),want=a.hex(),**o,exact=bool(got==a and o['ended'] and not o['truncated'])))
 assert model.digest()==before,'generation mutated persistent state'
 return dict(exact=sum(r['exact'] for r in rows)/max(1,len(rows)),rows=rows,evaluation_CPU_seconds=__import__('time').process_time()-cpu)
