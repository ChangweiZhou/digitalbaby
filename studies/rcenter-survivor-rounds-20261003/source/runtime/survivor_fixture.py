# SPDX-License-Identifier: GPL-3.0-or-later
"""Evaluator-owned relabeled V4 composition; never imported by the learner."""
import hashlib,itertools,json,random
ALPHABET=b'0123'
DT=30.0/14
RECORD_SECONDS=165.0
DAY=86400.0
OFFICIAL_WORLDS=tuple(range(310001,310065))
DEVELOPMENT_WORLDS=(310000,310101,310102)
PILOT_WORLD=310200

def rng(world,key):
    return random.Random(int.from_bytes(hashlib.sha256(f'RC-SURVIVOR-R1-v1|{world}|{key}'.encode()).digest()[:16],'big'))
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(x):return hashlib.sha256(canonical(x)).hexdigest()
def latin_squares():
    perms=list(itertools.permutations(range(4)))
    def visit(rows):
        if len(rows)==4:
            yield tuple(x for row in rows for x in row);return
        for row in perms:
            if all(row[j] not in [r[j] for r in rows] for j in range(4)):
                yield from visit(rows+[row])
    return tuple(visit([]))
def identifiable(table,taught,held):
    matches=[q for q in latin_squares() if all(q[i]==table[i] for i in taught)]
    return {'class':'all_576_order_four_latin_squares','consistent_tables':len(matches),
            'heldout_unique':bool(matches) and all(all(q[i]==table[i] for i in held) for q in matches)}
def make_world(world):
    L=list(range(4));R=list(range(4));O=list(range(4))
    for k,p in [('L',L),('R',R),('O',O)]:rng(world,k).shuffle(p)
    table=[O[L[a]^R[b]] for a in range(4) for b in range(4)]
    mul2=(0,2,3,1)
    held=[4*a+b for a in range(4) for b in range(4) if R[b]==mul2[L[a]]]
    taught=[i for i in range(16) if i not in held]
    old_cues=[(b' '*8+bytes((a,43,b,61))).hex() for a in b'0123' for b in b'0123']
    new_cues=[(b' '*8+bytes((a,43,b,61))).hex() for a in b'4567' for b in b'4567']
    new_outcomes=list(ALPHABET)*4;rng(world,'new-map').shuffle(new_outcomes)
    sets={'old':{'cues':[old_cues[i] for i in taught],'outcomes':[ALPHABET[table[i]] for i in taught]},
          'heldout':{'cues':[old_cues[i] for i in held],'outcomes':[ALPHABET[table[i]] for i in held]},
          'new':{'cues':new_cues,'outcomes':new_outcomes}}
    events=[]
    for stage,passes in [('old',16),('new',12)]:
        for epoch in range(passes):
            order=list(range(len(sets[stage]['cues'])));rng(world,f'{stage}-pass-{epoch}').shuffle(order)
            for i in order:
                at=len(events)*RECORD_SECONDS+(DAY if stage=='new' else 0)
                events.append({'cue_hex':sets[stage]['cues'][i],'outcome':sets[stage]['outcomes'][i],
                               'at':at,'stage':stage,'item':i})
    proof=identifiable(table,taught,held)
    assert proof['heldout_unique'] and sorted(sets['heldout']['outcomes'])==list(ALPHABET)
    d={'world':int(world),'sets':sets,'events':events,'old_end':192*RECORD_SECONDS,
       'new_start':192*RECORD_SECONDS+DAY,'new_end':384*RECORD_SECONDS+DAY,
       'final':384*RECORD_SECONDS+2*DAY,'identifiability':proof,
       'evaluator_only':{'L':L,'R':R,'O':O,'table':table,'held_indices':held,'taught_indices':taught}}
    d['sha256']=digest(d);return d
