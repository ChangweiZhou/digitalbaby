"""Evaluator only. Not imported by core.py."""
import numpy as np,collections
WORLDS=[202610236101,202610236102];ARMS=['LEARN','FROZEN','NO_OLD','SH_OLD']
def key(p):
 a,b=bytes(p)[:-1].split(b'+');return tuple(sorted((int(a),int(b))))
def fact(a,b):return dict(prompt=list(f'{a}+{b}='.encode()),answer=list(str(a+b).encode()),operands=[a,b])
def flip(r):return fact(*r['operands'][::-1])|{'answer':r['answer']}
def build(seed):
 rng=np.random.default_rng(seed);used=set()
 def cohort(n):
  out=[]
  for d in range(1,5):
   for e in range(1,5):
    for _ in range(n):
     for attempt in range(100000):
      a,b=[int(rng.integers(0 if L==1 else 10**(L-1),10**L)) for L in (d,e)];k=tuple(sorted((a,b)))
      if k not in used:used.add(k);out.append(fact(a,b));break
     else:raise RuntimeError('MANIFEST_SAMPLING_CAP')
  return out
 held=cohort(4);old=cohort(8);new=cohort(2);sh=[dict(r) for r in old];single=[]
 for L in sorted({len(r['answer']) for r in old}):
  ix=[i for i,r in enumerate(old) if len(r['answer'])==L];ix=list(rng.permutation(ix))
  if len(ix)==1:single+=ix
  for j,i in enumerate(ix):sh[i]=old[i]|{'answer':old[ix[(j+1)%len(ix)]]['answer']}
 def stream(base,passes):return [dict(index=int(i),reverse=bool(pas%2)) for pas in range(passes) for i in rng.permutation(len(base))]
 return dict(seed=seed,held=held,old=old,new=new,shuffled=sh,shuffle_singletons=single,old_order=stream(old,32),new_order=stream(new,8))
def records(d,phase,sh=False):
 base=d['shuffled' if sh and phase=='old' else phase]
 return [(flip(base[x['index']]) if x['reverse'] else base[x['index']]) for x in d[phase+'_order']]
def orientations(rs):return [q for r in rs for q in (r,flip(r))] # Equal orbit weights even for self-pairs.
def countermodels(exposures,probes):
 raw={};canonical={};freq=collections.Counter();bylen=collections.defaultdict(collections.Counter);suffix=[collections.defaultdict(collections.Counter) for _ in range(5)]
 for r in exposures:
  p,a=bytes(r['prompt']),bytes(r['answer']);raw[p]=a;canonical[key(p)]=a;freq[a]+=1;bylen[len(p)][a]+=1
  for k in range(5):suffix[k][p[-k:] if k else b''][a]+=1
 def top(c):return min(c,key=lambda a:(-c[a],a)) if c else None
 out={k:0 for k in ['raw_lookup','canonical_lookup','frequency','prompt_length_frequency',*[f'suffix{k}' for k in range(5)]]}
 for r in probes:
  p,a=bytes(r['prompt']),bytes(r['answer']);out['raw_lookup']+=raw.get(p)==a;out['canonical_lookup']+=canonical.get(key(p))==a;out['frequency']+=top(freq)==a;out['prompt_length_frequency']+=top(bylen[len(p)])==a
  for k in range(5):out[f'suffix{k}']+=top(suffix[k][p[-k:] if k else b''])==a
 return {k:v/len(probes) for k,v in out.items()}
