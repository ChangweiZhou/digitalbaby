"""MiniWorld v0 + forward-transfer assay. One continuous life, no resets, fixed learner (round-15 mechanism, online).
Worlds: invented names/places (random syllables), 3 statement templates per world from a pool; questions 'Where is N?'.
Learner answers each question BEFORE feedback; scored on that answer. Feedback: true | type (random place of current world) | none."""
import numpy as np, random, os, json, sys
E=os.environ.get
SYL=['ka','lo','mi','ru','te','sa','no','vi','da','pe','zu','ho','bi','fe','go','ra','li','mu','to','ne']
POOL_TRAIN=['{N} went to the {L}.','{N} moved to the {L}.','{N} is now in the {L}.','The {L} is where {N} went.',
 '{N} ran into the {L}.','Now {N} is in the {L}.','In the {L} you can see {N}.','{N} walked over to the {L}.',
 'The {L} now has {N} in it.','{N} arrived at the {L}.','{N} hurried to the {L}.','It is the {L} where {N} stands.']
POOL_TEST=['{N} strolled into the {L}.','The {L} is now where {N} is.','At the {L} one finds {N}.','{N} has reached the {L}.']
def word(r): return ''.join(r.choice(SYL) for _ in range(r.choice([2,3]))).capitalize()
def world(r,pool):
    names=list({word(r) for _ in range(6)})[:4]; places=list({word(r).lower() for _ in range(9)})[:6]
    return names,places,r.sample(pool,3)
def stream(r,w,nq):
    names,places,tpl=w; loc={}; out=[]
    while sum(1 for k,_ in out if k=='q')<nq:
        n=r.choice(names); l=r.choice([p for p in places if p!=loc.get(n)]); loc[n]=l
        out.append(('s',r.choice(tpl).format(N=n,L=l)))
        if r.random()<0.4 and loc: q=r.choice(sorted(loc)); out.append(('q',(f'Where is {q}?',loc[q])))
    return out
class Learner:
    N=1<<16; V=4096
    def __init__(s,seed):
        s.M=np.zeros((s.N,s.V),np.float32); s.lex={}; s.words=[]; s.H=np.zeros(s.V); s.T=np.zeros(s.V)
        s.CF={}; s.CH={}; s.CT={}; s.DF={}; s.NQ=0; s.rng=random.Random(seed)
    def wid(s,w):
        if w not in s.lex: s.lex[w]=len(s.words); s.words.append(w)
        return s.lex[w]
    def units(s,h,w=False):
        d={}
        for g in [h[i:i+k] for k in (2,3,4) for i in range(len(h)-k+1)]:
            wt=1.0 if w else np.log((1+s.NQ)/(1+s.DF.get(g,0)))
            for j in range(4): d[(hash(g)*2654435761+j*40503)%s.N]=wt
        return np.fromiter(d,np.int64),np.fromiter(d.values(),float)
    def see(s,x):
        ws=x.rstrip('.').split(); b=np.zeros(s.V,np.float32)
        for w in ws: b[s.wid(w)]=1/len(ws)
        ix,_=s.units(x.encode(),True); s.M[ix]=b
        if E('NOCTX')=='1': return
        for i,w in enumerate(ws): s.CF[w]=[('p',ws[i-1].lower() if i else '^'),('n',ws[i+1].lower() if i+1<len(ws) else '$')]
    def ask(s,q):
        ix,v=s.units(q.rstrip('?').encode()); bag=v@s.M[ix]; cand=np.nonzero(bag>0)[0]
        if len(cand)==0: return None,cand
        p0=(s.H.sum()+1)/(s.T.sum()+2); lo=lambda r: np.log(r/(1-r))
        def rel(c):
            fs=[(s.H[c]+2*p0)/(s.T[c]+2)]+[(s.CH.get(f,0)+2*p0)/(s.CT.get(f,0)+2) for f in s.CF.get(s.words[c],[])]
            return 1/(1+np.exp(-(lo(p0)+sum(lo(r)-lo(p0) for r in fs))))
        sc={c:bag[c]*rel(c) for c in cand}; best=max(sc.values()); ties=[c for c in sc if sc[c]>=best-1e-12]
        return s.words[s.rng.choice(ties)],cand
    def feedback(s,q,cand,a):
        if a is not None:
            for c in cand:
                ok=s.words[c]==a; s.T[c]+=1; s.H[c]+=ok
                for f in s.CF.get(s.words[c],[]): s.CT[f]=s.CT.get(f,0)+1; s.CH[f]=s.CH.get(f,0)+ok
        for g in {q[i:i+k] for k in (2,3,4) for i in range(len(q)-k+1)}: s.DF[g]=s.DF.get(g,0)+1
        s.NQ+=1
def live(L,r,w,nq,fb,rec=None):
    for k,x in stream(r,w,nq):
        if k=='s': L.see(x)
        else:
            q,a=x; out,cand=L.ask(q)
            if rec is not None: rec.append(out==a)
            f={'true':a,'type':r.choice(w[1]),'none':None}[fb]
            L.feedback(q.rstrip('?').encode(),cand,f)
K=int(E('K','30')); NQ=int(E('NQ','40')); NT=int(E('NT','40')); seeds=range(int(E('S0','0')),int(E('S0','0'))+int(E('NS','10')))
res={}
for fb in E('FBS','true,type,none').split(','):
    A=[];B=[]
    for sd in seeds:
        r=random.Random(sd); past=[world(r,POOL_TRAIN) for _ in range(K)]; test=world(random.Random(10_000+sd),POOL_TEST)
        La=Learner(sd); 
        for w in past: live(La,r,w,NQ,fb)
        ra=[]; live(La,random.Random(20_000+sd),test,NT,fb,ra)
        Lb=Learner(sd); rb=[]; live(Lb,random.Random(20_000+sd),test,NT,fb,rb)
        A.append(ra[:NT]); B.append(rb[:NT])
    A=np.array(A,float); B=np.array(B,float); d=(A-B).sum(1)   # transfer area in correct answers over NT questions
    res[fb]=dict(acc_experienced=round(A.mean(),3),acc_fresh=round(B.mean(),3),
                 first10_exp=round(A[:,:10].mean(),3),first10_fresh=round(B[:,:10].mean(),3),
                 transfer_area=round(d.mean(),2),ci95=round(1.96*d.std(ddof=1)/np.sqrt(len(d)),2))
    print(fb,json.dumps(res[fb]),flush=True)
