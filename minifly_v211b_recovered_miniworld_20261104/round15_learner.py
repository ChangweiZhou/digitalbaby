"""Round 14: QA1 with scaffolds removed: no answer vocabulary, no write gate, no positional roles, no ENDW.
Every sentence is stored whole: key = byte n-grams of the sentence, value = bag of its words (replacement write).
Question read: QIDF-weighted -> bag of words of the best-matching recent event.
Answer choice: learned word-level answer prior from feedback, counted take-the-best style: (hits+1)/(trials+2).
FB=true|random|none controls the feedback used for the prior (and IDF stats are unsupervised in all cases)."""
import numpy as np, os, json, random
E=os.environ.get; N=1<<16; V=1024; FB=E('FB','true'); rng=random.Random(7)
LOCS=['kitchen','garden','office','hallway','bathroom','bedroom']
def parse(fn):
    out=[]
    for line in open(fn):
        line=line.rstrip('\n'); i=line.index(' '); n=int(line[:i]); t=line[i+1:]
        if n==1: out.append(('story',None))
        out.append(('q',tuple(t.split('\t')[:2])) if '\t' in t else ('s',t))
    return out
TIES=[]; TIE=E('TIE','random'); rng2=random.Random(11); CTX=E('CTX','0')=='1'; COMB=E('COMB','max'); CF={}
DF={}; NQ=[0]; DW={}; NS=[0]; EIDF=E('EIDF','0')=='1'; BASE=E('BASE','1')=='1'
def grams(h): return [h[i:i+k] for k in (2,3,4) for i in range(len(h)-k+1)]
def units(h,w=False):
    d={}
    for g in grams(h):
        wt=1.0 if w else np.log((1+NQ[0])/(1+DF.get(g,0)))
        for j in range(4): d[(hash(g)*2654435761+j*40503)%N]=wt
    return np.fromiter(d,np.int64),np.fromiter(d.values(),float)
def run(train,evalf,valid=True):
    M=np.zeros((N,V),np.float32); CH={}; CT={}; lex={}; words=[]; H=np.zeros(V); T=np.zeros(V)
    def wid(w):
        if w not in lex: lex[w]=len(words); words.append(w)
        return lex[w]
    tr=parse(train)
    idx=[i for i,(k,_) in enumerate(tr) if k=='story']; cut=idx[int(len(idx)*.8)]
    va=parse(evalf); iv=[i for i,(k,_) in enumerate(va) if k=='story']; cv=iv[int(len(iv)*.8)]
    res=[]
    for items,test in ((tr[:cut],False),(va[cv:],True)):
        for k,x in items:
            if k=='s':
                ws=x.rstrip('.').split(); b=np.zeros(V,np.float32); NS[0]+=1
                for w in set(ws): DW[w]=DW.get(w,0)+1
                for w in ws: b[wid(w)]=1/len(ws)
                ix,_=units(x.encode(),True); M[ix]=b
                for i,w in enumerate(ws): CF[w]=[('prev',ws[i-1] if i else '^'),('next',ws[i+1] if i+1<len(ws) else '$')]
            elif k=='q':
                q,a=x; ix,v=units(q.rstrip('?').encode()); bag=v@M[ix]
                cand=np.nonzero(bag>0)[0]
                p0=(H.sum()+1)/(T.sum()+2); rel0=lambda c: max([(H[c]+2*p0)/(T[c]+2)]+([ (CH.get(f,0)+2*p0)/(CT.get(f,0)+2) for f in CF.get(words[c],[])] if CTX else [])); lo=lambda r: np.log(r/(1-r))
                def rel(c):
                    if not (CTX and COMB=='nb'): return rel0(c)
                    fs=[(H[c]+2*p0)/(T[c]+2)]+[(CH.get(f,0)+2*p0)/(CT.get(f,0)+2) for f in CF.get(words[c],[])]
                    z=lo(p0)+sum(lo(r)-lo(p0) for r in fs); return 1/(1+np.exp(-z))
                sc={c: bag[c]*rel(c)*(np.log((1+NS[0])/(1+DW.get(words[c],0))) if EIDF else 1) for c in cand}
                if sc:
                    best=max(sc.values()); ties=[c for c in sc if sc[c]>=best-1e-12]; TIES.append(len(ties)>1)
                    out=words[rng2.choice(ties) if TIE=='random' else ties[0]]
                else: out=None
                if test: res.append(out==a)
                else:
                    fb={'true':a,'random':rng.choice(LOCS),'none':None}[FB]
                    if fb is not None:
                        for c in cand:
                            T[c]+=1; H[c]+= words[c]==fb
                            for f in CF.get(words[c],[]): CT[f]=CT.get(f,0)+1; CH[f]=CH.get(f,0)+(words[c]==fb)
                    for g in set(grams(q.rstrip('?').encode())): DF[g]=DF.get(g,0)+1
                    NQ[0]+=1
    return round(float(np.mean(res)),3), round(float(np.mean(TIES)),3)
tr='/tmp/babi/qa1_single-supporting-fact_train.txt'
fam=E('FAM','base'); ev=tr if fam=='base' else f'/tmp/babi/qa1_train_{fam}.txt'
a,t=run(tr,ev); print(json.dumps({'fam':fam,'fb':FB,'ctx':CTX,'tie':TIE,'comb':COMB,'seed':E('PYTHONHASHSEED'),'acc':a,'tie_rate':t}))
