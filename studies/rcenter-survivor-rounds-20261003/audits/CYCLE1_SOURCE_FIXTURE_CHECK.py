"""Independent pure finite fixture/source-identity check; never imports learner/vendor."""
import hashlib, itertools, json
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
P=list(itertools.permutations(range(4)))
latin=[]
for rows in itertools.product(P,repeat=4):
 if all(len({rows[i][j] for i in range(4)})==4 for j in range(4)):
  latin.append(tuple(x for row in rows for x in row))
assert len(latin)==576
indices=range(16)
lookup={}
for mask in P:
 held=tuple(4*i+mask[i] for i in range(4))
 taught=tuple(i for i in indices if i not in held)
 partial={}
 for sq in latin: partial.setdefault(tuple(sq[i] for i in taught),[]).append(sq)
 lookup[held]=(taught,partial)
count=0;completions=Counter();unique=set()
for L,R,O in itertools.product(P,repeat=3):
 sq=tuple(O[L[a]^R[b]] for a in range(4) for b in range(4))
 held=tuple(4*a+R.index((0,2,3,1)[L[a]]) for a in range(4))
 taught,partial=lookup[held]
 consistent=partial[tuple(sq[i] for i in taught)]
 assert all(tuple(c[i] for i in held)==tuple(sq[i] for i in held) for c in consistent)
 assert len(consistent)==1
 assert Counter(sq[i] for i in held)==dict.fromkeys(range(4),1)
 assert Counter(sq[i] for i in taught)==dict.fromkeys(range(4),3)
 assert Counter(i//4 for i in taught)==dict.fromkeys(range(4),3)
 assert Counter(i%4 for i in taught)==dict.fromkeys(range(4),3)
 # A row-only missing-symbol strategy attains 100%, a scope limitation.
 for i in held: assert {sq[j] for j in taught if j//4==i//4}==set(range(4))-{sq[i]}
 completions[len(consistent)]+=1;unique.add((sq,held));count+=1
manifest=json.loads((ROOT/'source/BASELINE_PROJECTION_MANIFEST.json').read_text())
bad=[]
for e in manifest['members']:
 p=ROOT/'source/baseline'/e['path']
 if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=e['sha256'] or p.stat().st_size!=e['size']:bad.append(e['path'])
assert not bad
out={'schema':'rcenter-cycle1-pure-source-fixture-check-v1','native_learner_executed':False,'library_imports':['Python standard library only'], 'latin_squares':len(latin),'relabeling_triples_checked':count,'distinct_table_holdout_pairs':len(unique),'consistent_completion_count_distribution':dict(completions),'heldout_row_column_outcome_counts_each':1,'taught_row_column_outcome_counts_each':3,'row_only_missing_symbol_solver_accuracy':1.0,'verified_projection_members':len(manifest['members']),'source_mismatches':bad,'design_sha256':hashlib.sha256((ROOT/'protocol/ROUND1_DESIGN_CYCLE1.md').read_bytes()).hexdigest(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
print(json.dumps(out,indent=2,sort_keys=True))
