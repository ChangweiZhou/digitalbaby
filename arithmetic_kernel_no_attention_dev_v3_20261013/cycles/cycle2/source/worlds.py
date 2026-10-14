"""Evaluator/data generator only. Semantic fields never enter learner calls."""
import importlib.util, itertools, random, hashlib, json
from pathlib import Path
path=Path(__file__).resolve().parents[1]/'arithmetic_end_to_end_design_v1_20261013/validate_design.py'
spec=importlib.util.spec_from_file_location('data_validator',path);V=importlib.util.module_from_spec(spec);spec.loader.exec_module(V)
OPS=('+','-','*','/')

def make_world(seed):
    rng=random.Random(seed);pairs=list(itertools.combinations_with_replacement(range(16),2));eligible=[p for p in pairs if min(p)>=2]
    train=[];held=[];held_keys=set()
    for kind in ('SUM','PRODUCT'):
        picked=set(rng.sample(eligible,32));held_keys|={(kind,*p) for p in picked}
        for a,b in pairs:
            (held if (a,b) in picked else train).extend(V.rows(kind,a,b))
    assert {V.family(*r) for r in train}.isdisjoint({V.family(*r) for r in held})
    groups={}
    for r1 in train:
        op1,a,b,m=r1
        for r2 in train:
            op2,x,z,y=r2;trees=[]
            if x==m:trees.append((op2,(op1,a,b),z))
            if z==m:trees.append((op2,x,(op1,a,b)))
            for tree in trees:
                key=repr(V.canon(tree));g=groups.setdefault(key,{'pair':op1+op2,'answer':y,'forms':set(),'parts':set()})
                assert g['answer']==y
                g['forms'].add(V.render(tree)+'=');g['parts'].add((r1,r2))
    pool={a+b:sorted(k for k,g in groups.items() if g['pair']==a+b) for a,b in itertools.product(OPS,repeat=2)}
    trkeys=[];hokeys=[]
    for pair,keys in sorted(pool.items()):
        order=keys.copy();rng.shuffle(order);assert len(order)>=48
        trkeys+=order[:32];hokeys+=order[32:48]
    permutation=list(range(256));random.Random(seed+918273).shuffle(permutation)
    def encode(text):return bytes(permutation[b] for b in text.encode()).hex()
    def binary(r):
        op,a,b,y=r
        return {'prompt':encode(f'{a}{op}{b}='),'reply':encode(str(y)),'operation':op,'group':repr(V.family(*r))}
    def mixed(key):
        g=groups[key];forms=sorted(g['forms']);form=forms[rng.randrange(len(forms))]
        assert all(V.family(*r) not in held_keys for part in g['parts'] for r in part)
        return {'prompt':encode(form),'reply':encode(str(g['answer'])),'operation':g['pair'],'group':key}
    world={'seed':seed,'permutation':permutation,'binary_train':[binary(r) for r in train], 'binary_held':[binary(r) for r in held],
           'mixed_train':[mixed(k) for k in trkeys],'mixed_held':[mixed(k) for k in hokeys]}
    for a,b in [('binary_train','binary_held'),('mixed_train','mixed_held')]:
        assert {r['prompt'] for r in world[a]}.isdisjoint({r['prompt'] for r in world[b]})
        assert {r['group'] for r in world[a]}.isdisjoint({r['group'] for r in world[b]})
    world['data_digest']=hashlib.sha256(json.dumps(world,sort_keys=True).encode()).hexdigest()
    return world

def schedule(world,stage,n):
    keys=OPS if stage=='binary' else tuple(a+b for a,b in itertools.product(OPS,repeat=2))
    assert n%len(keys)==0
    rng=random.Random(world['seed']+(1 if stage=='binary' else 2));pool={k:[] for k in keys};index={k:0 for k in keys}
    for row in world[stage+'_train']:pool[row['operation']].append(row)
    for rows in pool.values():rng.shuffle(rows)
    out=[]
    for _ in range(n//len(keys)):
        for key in keys:
            j=index[key]
            if j and j%len(pool[key])==0:rng.shuffle(pool[key])
            row=dict(pool[key][j%len(pool[key])]);index[key]+=1;out.append(row)
    return out

def shuffle_targets(records,seed):
    rng=random.Random(seed);out=[dict(r) for r in records]
    # No operation/numeric label enters learner; preserve payload length distribution.
    for length in sorted({len(bytes.fromhex(r['reply'])) for r in out}):
        ids=[i for i,r in enumerate(out) if len(bytes.fromhex(r['reply']))==length]
        answers=[out[i]['reply'] for i in ids];rng.shuffle(answers)
        for i,a in zip(ids,answers):out[i]['reply']=a
    return out

def probes(rows,per_grouping=8):
    grouped={}
    for r in rows:grouped.setdefault(r['operation'],[]).append(r)
    # Technical probe slice only. Formal scoring must use full prospectively fixed groups.
    return [r for k in sorted(grouped) for r in sorted(grouped[k],key=lambda r:(r['group'],r['prompt']))[:per_grouping]]
