"""Count-based discovery of state-update laws on the MiniWorld text stream (no backprop, no attention, no QA labels
beyond the answer tokens that appear in the stream itself). Token level (scaffold: whitespace words, lines = events).
Learned from the stream:
  * signature of a line = its frequent (function) words with rare tokens replaced by slots -> event type
  * slot types: for each (signature, slot), counts of how often that slot's token is later observed as an answer
  * per-signature operation choice for the entity a question asks about, among generic candidate operations:
      SNAP : the place-typed token bound at that event time to the event's other participants (snapshot)
      PTR  : follow the other participant's CURRENT place (pointer, resolved at question time)
      CO   : a place-typed token in the same line (direct co-mention)
    each operation's reliability per signature is counted from answer tokens observed in the stream.
State store is NON-DESTRUCTIVE per attribute: an entity's place changes only through place-typed slots.
Output: a probability over candidate answers -> answer bits (vs chance log2 6 = 2.58)."""
import re, math, collections, json, sys, random
lines=open(sys.argv[1]).read().split('\n'); MODE=sys.argv[2] if len(sys.argv)>2 else 'full'
DF=collections.Counter(); NL=[0]
WIN=collections.defaultdict(set)
def rare(w): return len(WIN[w])<3   # dispersion: function words recur across many stretches of the life
SLOTANS=collections.defaultdict(lambda:[0,0])     # (sig,slot) -> [answer hits, trials]
OPREL=collections.defaultdict(lambda:[0,0])       # (sig, op) -> [correct, trials]
PLACE={}        # entity -> (place token, time)
LAST={}         # entity -> (sig, line tokens, slot index of entity, snapshot place of co-participants, time)
ANS=set(); TOP=collections.defaultdict(list); LI=[0]; import os; TESTFROM=int(os.environ.get('TESTFROM','0')); DIAG=collections.defaultdict(list); T=[0]; bits=collections.defaultdict(list); rng=random.Random(0)
def sig_of(ws):
    sl=[]; s=[]
    for i,w in enumerate(ws):
        if rare(w): s.append('_'); sl.append(i)
        else: s.append(w.lower())
    return ' '.join(s), sl
def placeprob(sig,j):
    h,n=SLOTANS[(sig,j)]; return (h+0.2)/(n+1)
def event(ws):
    T[0]+=1; sig,sl=sig_of(ws)
    ptype={i:placeprob(sig,k) for k,i in enumerate(sl)}
    places=[ws[i] for i in sl if ptype[i]>0.5]
    if MODE=='shufsig': sig=rng.choice(list({k[0] for k in OPREL}) or [sig])
    snap={}
    for i in sl:
        if ws[i] in PLACE: snap[ws[i]]=PLACE[ws[i]][0]
    for k,i in enumerate(sl):
        e=ws[i]
        if ptype[i]>0.5: continue
        if places and MODE!='noupd': PLACE[e]=(places[-1],T[0])
        others=[ws[j] for j in sl if j!=i and ptype[j]<=0.5]
        snapplace=next((snap[o] for o in others if o in snap),None)
        LAST[e]=(sig,others,places[-1] if places else None,snapplace)
    return sig,sl
def ask(e):
    c={}
    if e not in LAST: return c
    sig,others,co,snapplace=LAST[e]
    ops={'CO':co,'SNAP':snapplace,'PTR':next((PLACE[o][0] for o in others if o in PLACE),None),'OWN':PLACE.get(e,(None,))[0]}
    for op,v in ops.items():
        if v is None: continue
        if MODE.endswith('only') and op not in (MODE[:-4].upper(),'CO','OWN'): continue
        h,n=OPREL[(sig,op)]; r=(h+1)/(n+2)
        c.setdefault(v,[]).append((r,op,sig))
    return c
for ln in lines:
    LI[0]+=1
    if not ln.strip(): continue
    if '?' in ln:
        q,a=ln.split('? '); qw=q.split(); e=qw[-1]
        c=ask(e)
        # probability: candidate scored by best op reliability; mixture with uniform over known answers
        recentp={ws[i] for (sg,ws,sl) in RECENT[-60:] for k,i in enumerate(sl) if placeprob(sg,k)>0.5}; known=recentp or {a}; sc={v:max(r for r,_,_ in lst) for v,lst in c.items()}
        Z=sum(sc.values()); pv=(0.85*sc.get(a,0)/Z if Z else 0)+0.15/max(len(known),1)*(a in known) + (0 if a in known else 0.15*1e-3)
        kind='obj' if 'the' in qw else 'per'; top=max(sc,key=sc.get) if sc else None; DIAG[kind].append((a in c, top==a))
        if LI[0]>=TESTFROM: bits[(kind,'test')].append(-math.log2(max(pv,1e-6))); TOP[kind].append(top==a)
        # learning from the observed answer token (part of the stream)
        for v,lst in c.items():
            for r,op,sig in lst: OPREL[(sig,op)][0]+=(v==a); OPREL[(sig,op)][1]+=1
        for e2,(sig2,others,co,sp) in [(e,LAST[e])] if e in LAST else []: pass
        ANS.add(a)
        # slot-type learning: which slots of recent lines held this answer token
        for (sig,ws,sl) in RECENT[-12:]:
            for k,i in enumerate(sl):
                if ws[i]==a: SLOTANS[(sig,k)][0]+=1
        T[0]+=1
        for w in set(qw): DF[w]+=1; WIN[w].add(NL[0]//300)
        NL[0]+=1
        continue
    ws=re.findall(r"[A-Za-z]+",ln)
    for w in set(ws): DF[w]+=1; WIN[w].add(NL[0]//300)
    NL[0]+=1
    sig,sl=event(ws)
    RECENT=globals().setdefault('RECENT',[]); RECENT.append((sig,ws,sl))
    for k,i in enumerate(sl): SLOTANS[(sig,k)][1]+=0.1
print(json.dumps({'mode':MODE,'test_top1':{k:round(sum(v)/len(v),3) for k,v in TOP.items()},'avail_top1':{k:(round(sum(x for x,_ in v)/len(v),2),round(sum(y for _,y in v)/len(v),2)) for k,v in DIAG.items()},**{f'{k[0]}_{k[1]}':round(sum(v)/len(v),2) for k,v in sorted(bits.items())},'n':{f'{k[0]}_{k[1]}':len(v) for k,v in bits.items()}}))
if MODE=='full':
    for (sg,op),(h,n) in sorted(OPREL.items(),key=lambda z:-z[1][1])[:16]: print(f'{sg:35s} {op:5s} {h:4d}/{n:4d}')
