import numpy as np, sys, json, os, re
N=65536; WIN=int(os.environ.get('WIN','24')); HABT=float(os.environ.get('HABT','20')); RATE=float(os.environ.get('RATE','1.0'))
STOREMODE=os.environ.get('STORE','all')   # 'all' bytes written to episodic store, or 'none'
def parse(fn):
    out=[]; story=[]
    for line in open(fn):
        line=line.rstrip('\n'); i=line.index(' '); n=int(line[:i]); txt=line[i+1:]
        if n==1: story=[]; out.append(('story',None))
        if '\t' in txt:
            q,a,sup=txt.split('\t'); out.append(('q',(q.strip(),a)))
        else: out.append(('s',txt))
    return out
CNT={}
SEG=os.environ.get('SEG','0')=='1'; UNIT=os.environ.get('UNIT','0')=='1'; CREDIT=os.environ.get('CREDIT','match'); GTH=float(os.environ.get('GTH','0.05')); OSC=float(os.environ.get('OSC','1')); MATCHN=int(os.environ.get('MATCHN','3')); QNOSTORE=os.environ.get('QNOSTORE','0')=='1'; STEPN=int(os.environ.get('STEPN','0')); VALID=os.environ.get('VALID','0')=='1'; FREEZE=os.environ.get('FREEZE','0')=='1'; EPOCHS=int(os.environ.get('EPOCHS','1')); FROZ=[False]; RATEP=float(os.environ.get('RATEP','0.03')); RNGG=np.random.default_rng(int(os.environ.get('PYTHONHASHSEED','0') or 0)+5); NWR=[0,0]; GORD=tuple(int(x) for x in os.environ.get('GORD','1,2,3,4,5').split(','))
def bag(h,update=False,per=4,step=0):
    h=bytes(h)[-WIN:]
    if SEG:
        hh=h[:-2]; cut=max(hh.rfind(b'.'),hh.rfind(b'?'),hh.rfind(b'\n'))
        if cut>=0 and cut<len(h)-1: h=h[cut+1:]
    gs=[]
    for k in (2,3,4):
        gs+=[hash((h[i:i+k],step))+k for i in range(len(h)-k+1)]
    d={}
    for g in gs:
        w=1/np.sqrt(1+CNT.get(g,0)/HABT) if HABT>0 else 1.0
        for j in range(per):
            u=(g*2654435761+j*40503)%N; d[u]=max(d.get(u,0),w)
    if update:
        for g in gs: CNT[g]=CNT.get(g,0)+1
    ix=np.fromiter(d.keys(),np.int64); v=np.fromiter(d.values(),float); return ix,v
def gate_ix(h):
    return np.unique(np.array([(hash((b'G',bytes(h[-k:])))*2654435761+j*40503)%N for k in GORD if len(h)>=k for j in range(2)],np.int64))
class Pred:   # hashed-suffix local predictor (round 8), probability-space delta rule
    def __init__(s): s.W=np.zeros((N,256),np.float32)
    def ix(s,h): return np.unique(np.array([(hash(bytes(h[-k:]))*2654435761+j*40503+k*97)%N for k in (1,2,3,4,5) if len(h)>=k for j in range(2)],np.int64))
    def p(s,h): q=np.maximum(s.W[s.ix(h)].sum(0),1e-3); return q/q.sum()
    def learn(s,h,y,rate=0.1):
        ix=s.ix(h); e=-s.W[ix].sum(0); e[y]+=1; s.W[ix]+=(rate*e/len(ix)).astype(np.float32)
def run(train_fn,test_fn,seed=0):
    STREAM=[]; FOLLOW=[0]; NWR[0]=0; NWR[1]=0; P=Pred(); M=np.zeros((N,256),np.float32); A={}; G=np.zeros(N); TAGS=[]
    hist=bytearray(b'\n'*8)
    def feed(b,store=True,learnpred=True):
        for y in b:
            if learnpred and not FROZ[0]: P.learn(hist,y)
            if store and STOREMODE in ('all','gated','oracle','rand'):
                NWR[1]+=1
                gw=1.0
                if STOREMODE=='rand': gw=1.0 if RNGG.random()<RATEP else 0.0
                if STOREMODE=='oracle': gw=OSC if bytes(hist[-4:])==b'the ' else 0.0
                if STOREMODE=='gated':
                    gix=gate_ix(hist); gw=float(np.clip(G[gix].mean(),0,1)); (TAGS.append if not FROZ[0] else (lambda z: None))((gix,len(STREAM),set(bag(hist)[0].tolist()) if CREDIT=='overlap' else None)); del TAGS[:-300]
                if gw<=GTH and FOLLOW[0]>0:
                    FOLLOW[0]-=1; st=STEPN-FOLLOW[0]
                    ix,v=bag(hist[:-st] if st>0 else hist,step=st); T=np.zeros(256,np.float32); T[y]=1; a=(RATE*v/v.max())[:,None]; M[ix]+=(a*(T[None,:]-M[ix])).astype(np.float32)
                if gw>GTH:
                    NWR[0]+=1; FOLLOW[0]=STEPN
                    ix,v=bag(hist,update=True)
                    if UNIT:   # per-unit local target write (no shared error -> no explaining-away by shared units)
                        T=np.zeros(256,np.float32); T[y]=1; a=(gw*RATE*v/v.max())[:,None]; M[ix]+=(a*(T[None,:]-M[ix])).astype(np.float32)
                    else:
                        s=v@M[ix]; e=-s; e[y]+=1; M[ix]+=(gw*RATE*np.outer(v/(v@v),e)).astype(np.float32)
            hist.append(y); del hist[:-64]; STREAM.append(y)
    def answer(maxlen=12):
        out=bytearray()
        ix,v=bag(hist); sm=v@M[ix]
        first=int(np.argmax(sm)) if STOREMODE in ('all','gated','oracle','rand') and sm.max()>0 else int(np.argmax(P.p(hist)))
        y=first
        for t in range(maxlen):
            out.append(y); hist.append(y)
            if y==10: break
            key=(t+1,bytes(out[-3:]))
            if t+1<=STEPN:
                ix2,v2=bag(hist[:-(t+1)],step=t+1); s2=v2@M[ix2]
                if s2.max()>0.3: y=int(np.argmax(s2)); continue
            y=int(np.argmax(A[key])) if key in A else int(np.argmax(P.p(hist)))
        del hist[-len(out):]
        return bytes(out)
    tr=parse(train_fn)
    if VALID:
        idx=[i for i,(k,_) in enumerate(tr) if k=='story']; cut=idx[int(len(idx)*0.8)]
        phases=tuple((tr[:cut],False) for _ in range(EPOCHS))+((tr[cut:],True),)
    else: phases=tuple((tr,False) for _ in range(EPOCHS))+((parse(test_fn),True),)
    for items,test in phases:
        FROZ[0]= FREEZE and test
        if FROZ[0]: TAGS.clear()
        res=[];lastloc=[];
        for kind,x in items:
            if kind=='story': lastloc={}; continue
            if kind=='s':
                feed((x+' ').encode()); m=re.match(r'(\w+) .* (\w+)\.$',x)
                if m: lastloc[m.group(1)]=m.group(2)
            else:
                q,a=x; feed((q+' ').encode(),store=not QNOSTORE)
                if test:
                    out=answer(); name=q.split()[-1].rstrip('?')
                    res.append((out==(a+'\n').encode(), lastloc.get(name)==a, out[:1]==a.encode()[:1]))
                ans=(a+'\n').encode()
                if STOREMODE=='gated' and not FROZ[0]:
                    qset=set(bag(hist)[0].tolist())
                    for gix,pos,cs in TAGS:
                        nxt=bytes(STREAM[pos:pos+MATCHN])
                        ok = nxt==ans[:MATCHN] and (cs is None or len(cs & qset)>0)
                        tgt=1.0 if ok else 0.0; G[gix]+=0.2*(tgt-G[gix].mean())
                    TAGS.clear()
                for t,y in enumerate(ans if not FROZ[0] else b''):           # teach cue on answer bytes: step-conjoined answer memory
                    key=(t,bytes(ans[:t][-3:])); A.setdefault(key,np.zeros(256)); A[key]*=0.9; A[key][y]+=1
                feed(ans,store=not QNOSTORE)   # environment supplies the answer afterwards (feedback)
    acc=np.mean([r[0] for r in res]); fb=np.mean([r[2] for r in res]); heur=np.mean([r[1] for r in res])
    return {'exact_answer_with_own_newline':round(float(acc),3),'oracle_parse_last_location_heuristic':round(float(heur),3),'first_byte_acc':round(float(fb),3),'n':len(res),'write_frac':round(NWR[0]/max(1,NWR[1]),4)}
print(json.dumps({'epochs':EPOCHS,'gord':GORD,'gth':GTH,'valid':VALID,'store':STOREMODE,'win':WIN,'habt':HABT,'rate':RATE,**run(sys.argv[1], sys.argv[2])}))