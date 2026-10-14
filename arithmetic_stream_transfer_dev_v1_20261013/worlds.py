"""Evaluator-only streams; arithmetic appears only in rendered data and test labels."""
import hashlib,itertools,json,random
PASSES=(1,4,16,64,256)
def seed(label):
    if not label.startswith('STREAM_TRANSFER_V1_20261013/TECH/'):
        raise PermissionError('technical namespace only; DEV/CONFIRM not admitted')
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:8],'big')
def canonical_digest(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def make_world(label):
    value=seed(label);rng=random.Random(value)
    counting=[]
    for rep in range(3):
        starts=list(range(50));rng.shuffle(starts)
        for k in starts[:50 if rep<2 else 28]:
            counting.append({'prompt_hex':','.join(str(k+i) for i in range(4)).encode().hex(),
                'answer_hex':(',%d,%d'%(k+4,k+5)).encode().hex()})
    all_orbits=list(itertools.combinations_with_replacement(range(10),2))
    eligible=list(itertools.combinations_with_replacement(range(2,10),2))
    tasks={}
    for op in ('+','*'):
        held=set(rng.sample(eligible,20));train=[o for o in all_orbits if o not in held]
        def response(a,b):return str(a+b if op=='+' else a*b).encode().hex()
        def question(a,b):return ('%d%s%d='%(a,op,b)).encode().hex()
        taught=[{'prompt_hex':question(a,b),'answer_hex':response(a,b),'operation':op,'orbit':[min(a,b),max(a,b)]}
                for x,y in train for a,b in sorted({(x,y),(y,x)})]
        test=[{'prompt_hex':question(a,b),'answer_hex':response(a,b),'operation':op,'orbit':[x,y]}
              for x,y in sorted(held) for a,b in sorted({(x,y),(y,x)})]
        records=[];shuffled=[];fixed=0
        for pass_n in range(1,257):
            groups={}
            for orbit in train:groups.setdefault(len(bytes.fromhex(response(*orbit))),[]).append(orbit)
            assignments={}
            for group in groups.values():
                answers=[response(*o) for o in group];rng.shuffle(answers)
                assignments.update(zip(group,answers))
            order=[(a,b,tuple(o)) for o in train for a,b in (o,o[::-1])];rng.shuffle(order)
            for a,b,o in order:
                r={'prompt_hex':question(a,b),'answer_hex':response(a,b),'operation':op,'orbit':list(o),'pass':pass_n}
                records.append(r);shuffled.append(assignments[o]);fixed+=assignments[o]==r['answer_hex']
        tasks[op]={'train_orbits':[list(o) for o in train],'held_orbits':[list(o) for o in sorted(held)],
                   'taught':taught,'held':test,'records':records,'shuffled_answers':shuffled,'shuffle_equal_label_records':fixed}
    prompts={x['prompt_hex'] for x in counting}|{x['prompt_hex'] for task in tasks.values() for x in task['taught']+task['held']}
    pairs=[];seen=set()
    while len(pairs)<8:
        key=bytes(rng.randrange(256) for _ in range(6)).hex()
        if key in prompts or key in seen:continue
        seen.add(key);target=bytes(rng.randrange(256) for _ in range(rng.randrange(1,4))).hex()
        pairs.append({'prompt_hex':key,'answer_hex':target})
    cohort=[]
    for _ in range(4):
        indices=list(range(8));rng.shuffle(indices);cohort.extend(pairs[i].copy() for i in indices)
    plan={'label':label,'seed':value,'technical':True,'counting':counting,'tasks':tasks,
          'cohort_pairs':pairs,'cohort_records':cohort}
    plan['stream_digest']=canonical_digest(plan)
    return plan
