"""GLN context mixer + generic episodic word-association experts (no QA logic; every byte is predicted the same way).
Expert H1: candidates = words associated with the rare words of the current line (decaying per-word co-occurrence bags).
Expert H2: two hops (associates of associates). Each expert turns its candidate-word weights, filtered by the prefix of
the word being typed, into a next-byte distribution. The GLN mixer learns from prediction loss alone when to trust them.
REC=0 disables experts; REC=shuf uses bags of a random other word (control)."""
import numpy as np, sys, math, time, os, random, json
E=os.environ.get; REC=E('REC','1'); LAM=float(E('LAM','0.7'))
data=open(sys.argv[1],'rb').read()
TB=1<<22
def st(p): p=min(max(p,1e-4),1-1e-4); return math.log(p/(1-p))
def sq(x): return 1/(1+math.exp(-max(-30,min(30,x))))
ORD=[1,2,3,4,5,6]
P=np.full(TB,0.5); C=np.zeros(TB,np.uint8)
BAG={}; DF={}; NL=[0]; rnd=random.Random(3)
def idf(w): return math.log((1+NL[0])/(1+DF.get(w,0)))
def end_line(words):
    NL[0]+=1
    for w in set(words): DF[w]=DF.get(w,0)+1
    for w in set(words):
        b=BAG.setdefault(w,{})
        for k in list(b): b[k]*=LAM
        for u in set(words):
            if u!=w: b[u]=b.get(u,0)+1
def cands(cues):
    h1={}; 
    for c in cues:
        src=c if REC!='shuf' else (rnd.choice(list(BAG)) if BAG else c)
        wc=idf(c)
        for u,v in BAG.get(src,{}).items():
            if u in cues: continue
            h1[u]=h1.get(u,0)+wc*v*idf(u)
    h2={}
    top=sorted(h1.items(),key=lambda z:-z[1])[:6]
    for u,v in top:
        for x,y in BAG.get(u,{}).items():
            if x in cues or x==u: continue
            h2[x]=h2.get(x,0)+v*y*idf(x)
    return h1,h2
def byte_dist(cw,prefix):
    d=np.zeros(257); 
    for w,v in cw.items():
        if w.startswith(prefix):
            nxt=w[len(prefix)] if len(w)>len(prefix) else 256
            d[nxt]+=v
    return d
GATES1=[lambda h,c0,e: c0, lambda h,c0,e: (h[-1]<<8|c0), lambda h,c0,e: ((h[-2]<<16)|(h[-1]<<8)|c0)&0xFFFFF, lambda h,c0,e: (e<<12)|(min(len(WORD[0]),15)<<8)|c0]
W1=[dict() for _ in GATES1]; W2={}; WORD=[b'']; LINE=[]
lr=0.02; BITS=np.zeros(len(data)); t0=time.time(); hist=bytearray(b'\0'*8)
H1=H2={}; dist1=dist2=None
for i,byte in enumerate(data):
    if REC!='0':
        d1=byte_dist(H1,WORD[0]); d2=byte_dist(H2,WORD[0])
        # word end: if partial word complete in a candidate, the next byte is a delimiter -> map 256 to space/newline mass
    c0=1
    ctxh=[hash((k,bytes(hist[-k:]))) for k in ORD]+[hash(('w',WORD[0])), hash(('s1',bytes(hist[-2:-1]))), hash(('s2',bytes(hist[-3:-2]),bytes(hist[-1:])))]
    for b in range(7,-1,-1):
        bit=(byte>>b)&1
        idx=[(h*2654435761+c0*40503)%TB for h in ctxh]
        xs=[st(P[j]) for j in idx]; econf=0
        if REC!='0':
            for d in (d1,d2):
                # P(bit=1 | bytes consistent with high bits so far); only real bytes (0..255)
                lo=(c0<<(b+1))-(1<<8) if False else None
                base=(c0^(1<<(7-b)))<<(b+1) if False else None
                pref=c0-(1<<(7-b)) if False else None
                # bytes whose top (7-b) bits equal c0's bits
                hi=c0&((1<<(7-b))-1); start=hi<<(b+1); m=d[start:start+(1<<(b+1))]; tot=m.sum()
                if tot>0:
                    p1=m[(1<<b):].sum()/tot; conf=1
                else: p1=0.5; conf=0
                xs+= [st(0.02+0.96*p1), float(conf)]; econf=econf*2+conf
        x=np.array(xs+[0.3])
        outs=[]; keys=[]
        for gate,W in zip(GATES1,W1):
            k=gate(hist,c0,econf); w=W.get(k)
            if w is None: w=W[k]=np.full(len(x),0.3/len(x)*8)
            p=sq(w@x); outs.append(p); keys.append((w,p))
        x2=np.array([st(p) for p in outs]+[0.3]); w2=W2.get((c0,econf))
        if w2 is None: w2=W2[(c0,econf)]=np.full(len(x2),1/len(outs))
        p=min(max(sq(w2@x2),1e-5),1-1e-5)
        BITS[i]+=-math.log2(p if bit else 1-p)
        for w,pn in keys: w+=lr*(bit-pn)*x
        w2+=lr*(bit-p)*x2
        for j in idx: C[j]=min(C[j]+1,60); P[j]+=(bit-P[j])/(C[j]+1.5)
        c0=(c0<<1)|bit
    hist.append(byte); del hist[:-8]
    ch=chr(byte)
    if ch.isalpha(): WORD[0]=WORD[0]+bytes([byte])
    else:
        if WORD[0]: LINE.append(WORD[0])
        WORD[0]=b''
        if REC!='0' and LINE: H1,H2=cands(set(LINE))
        if ch in '.\n':
            if LINE: end_line(LINE)
            LINE=[]; H1=H2={}
print(json.dumps({'file':os.path.basename(sys.argv[1]),'REC':REC,'bpb':round(BITS.sum()/len(data),4),'sec':round(time.time()-t0)}))
np.save(sys.argv[1]+f'.bits_{REC}.npy',BITS)
