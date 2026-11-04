"""MiniWorld v1: people move, pick up and drop objects. Questions about people and objects. One life, fixed learner.
Learner (no positional roles, no supplied operations per verb):
  store: every event -> bag of its words (key = byte n-grams of the event) PLUS an 'enrichment' bag: for each word of the
         event, the words recalled for that word at event time (generic "what did I know about each participant then").
  answer: candidates tagged d (words of the recalled event), e (its enrichment, i.e. event-time knowledge),
          h (words recalled now via each word of the recalled event: second hop).
  choice: naive-Bayes evidence from feedback over features: identity, slot context, and (tag x word of the recalled event)
          and (tag x via-word slot context). Random tie-breaking."""
import numpy as np, random, os, json
E=os.environ.get
SYL=['ka','lo','mi','ru','te','sa','no','vi','da','pe','zu','ho','bi','fe','go','ra','li','mu','to','ne']
MOVE_TR=['{N} went to the {L}.','{N} moved to the {L}.','{N} is now in the {L}.','The {L} is where {N} went.','{N} ran into the {L}.','Now {N} is in the {L}.','{N} walked over to the {L}.','{N} arrived at the {L}.']
MOVE_TE=['{N} strolled into the {L}.','At the {L} one finds {N}.','{N} has reached the {L}.']
GET=['{N} got the {O}.','{N} picked up the {O}.','{N} took the {O}.','{N} grabbed the {O}.']
DROP=['{N} dropped the {O}.','{N} put down the {O}.','{N} left the {O}.','{N} discarded the {O}.']
def word(r): return ''.join(r.choice(SYL) for _ in range(r.choice([2,3])))
def world(r,pool):
    return (list({word(r).capitalize() for _ in range(6)})[:4], list({word(r) for _ in range(9)})[:6], list({word(r) for _ in range(5)})[:3], r.sample(pool,2))
def stream(r,w,nq):
    names,places,objs,tpl=w; loc={n:r.choice(places) for n in names}; hold={}; oloc={}; out=[(  's',r.choice(tpl).format(N=n,L=loc[n])) for n in names]; q=0
    while q<nq:
        n=r.choice(names); x=r.random(); held=[o for o in objs if hold.get(o)==n]; free=[o for o in objs if o not in hold]
        if x<0.5:
            l=r.choice([p for p in places if p!=loc[n]]); loc[n]=l; out.append(('s',r.choice(tpl).format(N=n,L=l)))
        elif x<0.75 and free:
            o=r.choice(free); hold[o]=n; oloc.pop(o,None); out.append(('s',r.choice(GET).format(N=n,O=o)))
        elif held:
            o=r.choice(held); del hold[o]; oloc[o]=loc[n]; out.append(('s',r.choice(DROP).format(N=n,O=o)))
        else: continue
        if r.random()<0.4:
            known=[o for o in objs if o in hold or o in oloc]
            if known and r.random()<0.6:
                o=r.choice(known); out.append(('q',(f'Where is the {o}?',loc[hold[o]] if o in hold else oloc[o],'obj'))); q+=1
            else:
                m=r.choice(names); out.append(('q',(f'Where is {m}?',loc[m],'per'))); q+=1
    return out
