"""Regenerate Codex's fresh QA1 stories (minifly_qa1_frozen_generalization/qualify.py materialize(), seed 2030411, 12 batches)
and verify the SHA-256 digests recorded in its OFFICIAL_PLAN.json. Writes original.txt / verb.txt / prefix.txt here."""
import numpy as np, hashlib
PEOPLE=('Mary','John','Daniel','Sandra')
PLACES=('kitchen','garden','office','hallway','bathroom','bedroom')
TEMPLATES={
    'original':('moved to the','travelled to the','went to the','journeyed to the','went back to the'),
    'verb':('strolled into the','walked into the','headed for the','wandered to the','hurried into the'),
    'prefix':('strolled into that','walked straight into','headed for this','wandered into our','hurried toward that')}
EXPECT={'original':'3c87c2a7089d77fa68329b1c225abfc178ac4871d12c4a37e3cd4ad018893276',
        'verb':'1e2a12a864d49463e313fc867d88ef5ab3573eaf52f31e48d1aa6791ef20b19e',
        'prefix':'3d85981e7f3e60169f7fd134a4e527544d4b1a0907340586d17766dd354ade77'}
def materialize(seed,batches=1):
    rng=np.random.default_rng(seed); common=[]
    for _ in range(20*batches):
        story=[]; latest={}
        for _ in range(5):
            for _ in range(2):
                p=str(rng.choice(PEOPLE))
                a=str(rng.choice([place for place in PLACES if place!=latest.get(p,(None,))[0]]))
                variant=int(rng.integers(5)); story.append((p,a,variant)); latest[p]=(a,len(story))
            p=str(rng.choice(sorted(latest))); a,index=latest[p]; story.append((p,a,index))
        common.append(story)
    def render(view):
        rows=[]
        for story in common:
            for i,(person,place,x) in enumerate(story,1):
                rows.append(f'{i} {person} {TEMPLATES[view][x]} {place}.' if i%3 else f'{i} Where is {person}? \t{place}\t{x}')
        return '\n'.join(rows)+'\n'
    return {v:render(v) for v in TEMPLATES}
for v,text in materialize(2030411,12).items():
    h=hashlib.sha256(text.encode()).hexdigest(); open(f'{v}.txt','w').write(text)
    print(v, 'sha OK' if h==EXPECT[v] else 'SHA MISMATCH '+h, text.count('Where'))
