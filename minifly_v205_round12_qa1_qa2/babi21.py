"""Round-12 bAbI QA1 learner (byte stream, no backprop, no attention).
Derived from babi17.py; existing modes (STORE=gated/vocab) are byte-for-byte the same computation.
New in babi20:
  * TRAIN/TEST/VALFN env: file paths (VALID=1 -> train on first 80% stories of TRAIN, evaluate on last 20% of VALFN (default TRAIN)).
  * STORE=vocabsent : event-level binding. At sentence end, each word of the sentence that is in the learned answer
                      vocabulary is written keyed by the bag of the *whole sentence with that word masked* (order-free).
  * REL=1           : learned cue reliability. Each store unit u keeps R[u] (prior RELP0). Read votes are weighted by
                      v*R. After the answer is revealed, every unit that had content votes argmax(M[u]); R[u] moves toward
                      1 if that vote equals the revealed byte, else toward 0 (rate RELETA). Frozen at test when FREEZE=1.
                      RELC=1 subtracts the chance level (1/|answer first bytes|) so units at chance get weight 0.
  * QIDF=1          : unsupervised alternative: read weight log((1+Nq)/(1+df_q[u])) with df_q counted once per question.
"""
import numpy as np, sys, json, os, re
E = os.environ.get
N=65536; WIN=int(E('WIN','24')); HABT=float(E('HABT','20')); RATE=float(E('RATE','1.0'))
STOREMODE=E('STORE','all'); MODES=set(STOREMODE.split('+')); ENDW=E('ENDW','0')=='1'; MAXW=16
TRAIN=E('TRAIN','/tmp/babi/qa1_single-supporting-fact_train.txt'); TEST=E('TEST','/tmp/babi/qa1_single-supporting-fact_test.txt'); VALFN=E('VALFN',TRAIN)
SEG=E('SEG','0')=='1'; UNIT=E('UNIT','0')=='1'; CREDIT=E('CREDIT','match'); GTH=float(E('GTH','0.05')); OSC=float(E('OSC','1')); MATCHN=int(E('MATCHN','3')); QNOSTORE=E('QNOSTORE','0')=='1'; STEPN=int(E('STEPN','0')); VALID=E('VALID','0')=='1'; FREEZE=E('FREEZE','0')=='1'; EPOCHS=int(E('EPOCHS','1')); BIN=E('BIN','0')=='1'
REL=E('REL','0')=='1'; RELP0=float(E('RELP0','0.0')); RELETA=float(E('RELETA','0.1')); RELC=E('RELC','0')=='1'; QIDF=E('QIDF','0')=='1'; QIDFADAPT=E('QIDFADAPT','0')=='1'; NOGOLD=E('NOGOLD','0')=='1'; QIDFDECAY=float(E('QIDFDECAY','1'))
DUMP=E('DUMP','')
SREAD=None  # set below
VOCAB=set(); FROZ=[False]; RATEP=float(E('RATEP','0.03')); RNGG=np.random.default_rng(int(E('PYTHONHASHSEED','0') or 0)+5); NWR=[0,0]; GORD=tuple(int(x) for x in E('GORD','1,2,3,4,5').split(','))
CNT={}
SREAD=MAXW if ENDW else STEPN
def parse(fn):
    out=[]
    for line in open(fn):
        line=line.rstrip('\n'); i=line.index(' '); n=int(line[:i]); txt=line[i+1:]
        if n==1: out.append(('story',None))
        if '\t' in txt:
            q,a,sup=txt.split('\t'); out.append(('q',(q.strip(),a)))
        else: out.append(('s',txt))
    return out
def grams(h,step):
    gs=[]
    for k in (2,3,4): gs+=[hash((h[i:i+k],step))+k for i in range(len(h)-k+1)]
    return gs
def units(gs,per=4,habw=None):
    d={}
    for g in gs:
        w=habw(g) if habw else 1.0
        for j in range(per):
            u=(g*2654435761+j*40503)%N; d[u]=max(d.get(u,0),w)
    return np.fromiter(d.keys(),np.int64), np.fromiter(d.values(),float)
def bag(h,update=False,per=4,step=0):
    h=bytes(h)[-WIN:]
    if SEG:
        hh=h[:-2]; cut=max(hh.rfind(b'.'),hh.rfind(b'?'),hh.rfind(b'\n'))
        if cut>=0 and cut<len(h)-1: h=h[cut+1:]
    gs=grams(h,step)
    ix,v=units(gs,per,(lambda g: 1/np.sqrt(1+CNT.get(g,0)/HABT)) if HABT>0 else None)
    if update:
        for g in gs: CNT[g]=CNT.get(g,0)+1
    return ix,v
def bagb(h,step=0,per=4):   # bag over an explicit byte string (no window / segmentation)
    return units(grams(bytes(h),step),per)
def gate_ix(h):
    return np.unique(np.array([(hash((b'G',bytes(h[-k:])))*2654435761+j*40503)%N for k in GORD if len(h)>=k for j in range(2)],np.int64))
class Pred:   # hashed-suffix local predictor (round 8), probability-space delta rule
    def __init__(s): s.W=np.zeros((N,256),np.float32)
    def ix(s,h): return np.unique(np.array([(hash(bytes(h[-k:]))*2654435761+j*40503+k*97)%N for k in (1,2,3,4,5) if len(h)>=k for j in range(2)],np.int64))
    def p(s,h): q=np.maximum(s.W[s.ix(h)].sum(0),1e-3); return q/q.sum()
    def learn(s,h,y,rate=0.1):
        ix=s.ix(h); e=-s.W[ix].sum(0); e[y]+=1; s.W[ix]+=(rate*e/len(ix)).astype(np.float32)
def run():
    VOCAB.clear(); STREAM=[]; FOLLOW=[0]; NWR[0]=0; NWR[1]=0; P=Pred(); M=np.zeros((N,256),np.float32); A={}; G=np.zeros(N); TAGS=[]
    R=np.full(N,RELP0); DFQ=np.zeros(N); NQ=[0]; FIRSTS=set()
    hist=bytearray(b'\n'*8); SENT=bytearray()
    def wt(ix,v):   # read-side weighting of active units
        if REL:
            r=R[ix]
            if RELC and len(FIRSTS)>1: c=1/len(FIRSTS); r=np.maximum(0,(r-c)/(1-c))
            v=v*r
        if QIDF: v=v*np.log((1+NQ[0])/(1+DFQ[ix]))
        return v
    def wr(ix,v,y):
        T=np.zeros(256,np.float32); T[y]=1; a=(RATE*v/v.max())[:,None]; M[ix]+=(a*(T[None,:]-M[ix])).astype(np.float32)
    PEND=[None,0]
    def capture(w,keyfn):   # write a whole word step-tagged under a context key; ENDW appends a '\n' end-of-value marker
        seq=(w+b'\n') if ENDW else w[:STEPN+1]
        for st,yy in enumerate(seq[:MAXW+1]):
            ix,v=keyfn(st); wr(ix,v,yy)
        NWR[0]+=1
    def feed(b,store=True,learnpred=True):
        for y in b:
            if learnpred and not FROZ[0]: P.learn(hist,y)
            if store and MODES & {'all','gated','oracle','rand','gword'}:
                NWR[1]+=1
                gw=1.0
                if 'rand' in MODES: gw=1.0 if RNGG.random()<RATEP else 0.0
                if 'oracle' in MODES: gw=OSC if bytes(hist[-4:])==b'the ' else 0.0
                if MODES & {'gated','gword'}:
                    gix=gate_ix(hist); gw=float(np.clip(G[gix].mean(),0,1)); (TAGS.append if not FROZ[0] else (lambda z: None))((gix,len(STREAM),set(bag(hist)[0].tolist()) if CREDIT=='overlap' else None)); del TAGS[:-300]
                if 'gword' in MODES:   # gate-triggered retroactive word capture (gate marks a value start; word boundary marks its end)
                    if PEND[0] is not None and y in (32,46,63,10):
                        pre=PEND[0]; w=bytes(STREAM[PEND[1]:]); PEND[0]=None
                        if w: capture(w,lambda st: bag(pre,step=st))
                    elif gw>GTH and PEND[0] is None and y not in (32,46,63,10): PEND[0]=bytearray(hist); PEND[1]=len(STREAM)
                    gw=0.0; FOLLOW[0]=0
                if gw<=GTH and FOLLOW[0]>0:
                    FOLLOW[0]-=1; st=STEPN-FOLLOW[0]
                    ix,v=bag(hist[:-st] if st>0 else hist,step=st); wr(ix,v,y)
                if gw>GTH:
                    NWR[0]+=1; FOLLOW[0]=STEPN
                    ix,v=bag(hist,update=True)
                    if UNIT:
                        T=np.zeros(256,np.float32); T[y]=1; a=((1.0 if BIN else gw)*RATE*v/v.max())[:,None]; M[ix]+=(a*(T[None,:]-M[ix])).astype(np.float32)
                    else:
                        s=v@M[ix]; e=-s; e[y]+=1; M[ix]+=(gw*RATE*np.outer(v/(v@v),e)).astype(np.float32)
            if store and 'vocab' in MODES and y in (32,46,63,10):
                j=len(hist)-1
                while j>=0 and hist[j] not in (32,46,63,10): j-=1
                w=bytes(hist[j+1:])
                if w in VOCAB and j>=0:
                    pre=bytearray(hist[:j+1]); capture(w,lambda st: bag(pre,step=st))
            if store and 'vocabsent' in MODES and y in (46,63,10):
                words=bytes(SENT).split(b' ')
                for wi,w in enumerate(words):
                    if w in VOCAB:
                        key=b' '.join(words[:wi]+[b'\x01']+words[wi+1:]); capture(w,lambda st: bagb(key,step=st))
            if y in (46,63,10): SENT.clear()
            elif not (y==32 and len(SENT)==0): SENT.append(y)
            hist.append(y); del hist[:-64]; STREAM.append(y)
    def answer(maxlen=12):
        out=bytearray(); reads=[]
        ix,v=bag(hist); sm=wt(ix,v)@M[ix]; reads.append((ix,0))
        first=int(np.argmax(sm)) if STOREMODE!='none' and sm.max()>0 else int(np.argmax(P.p(hist)))
        y=first
        for t in range(maxlen):
            out.append(y); hist.append(y)
            if y==10: break
            key=(t+1,bytes(out[-3:]))
            if t+1<=SREAD:
                ix2,v2=bag(hist[:-(t+1)],step=t+1); s2=wt(ix2,v2)@M[ix2]; reads.append((ix2,t+1))
                if s2.max()>0.3: y=int(np.argmax(s2)); continue
            y=int(np.argmax(A[key])) if key in A else int(np.argmax(P.p(hist)))
        del hist[-len(out):]
        return bytes(out), reads
    def credit_rel(reads,ans):
        for ix,t in reads:
            if t>=len(ans): continue
            rows=M[ix]; has=rows.max(1)>0.05
            if not has.any(): continue
            u=ix[has]; vote=rows[has].argmax(1); R[u]+=RELETA*((vote==ans[t]).astype(float)-R[u])
    tr=parse(TRAIN)
    if VALID:
        idx=[i for i,(k,_) in enumerate(tr) if k=='story']; cut=idx[int(len(idx)*0.8)]
        va=parse(VALFN); idv=[i for i,(k,_) in enumerate(va) if k=='story']; cutv=idv[int(len(idv)*0.8)]
        phases=tuple((tr[:cut],False) for _ in range(EPOCHS))+((va[cutv:],True),)
    else: phases=tuple((tr,False) for _ in range(EPOCHS))+((parse(TEST),True),)
    dump=[]
    for items,test in phases:
        FROZ[0]= FREEZE and test
        if FROZ[0]: TAGS.clear()
        res=[]; lastloc={}
        for kind,x in items:
            if kind=='story': lastloc={}; continue
            if kind=='s':
                feed((x+' ').encode()); m=re.match(r'(\w+) .* (\w+)\.$',x)
                if m: lastloc[m.group(1)]=m.group(2)
            else:
                q,a=x; feed((q+' ').encode(),store=not QNOSTORE)
                ans=(a+'\n').encode()
                needreads = test or (REL and not FROZ[0])
                if needreads:
                    out,reads=answer()
                    if test:
                        name=q.split()[-1].rstrip('?')
                        res.append((out==ans, lastloc.get(name)==a, out[:1]==a.encode()[:1]))
                        if DUMP: dump.append({'q':q,'a':a,'out':out.decode('latin1')})
                    if REL and not FROZ[0]: credit_rel(reads,ans)
                if QIDF and (not FROZ[0] or QIDFADAPT):
                    if QIDFDECAY<1: DFQ*=QIDFDECAY; NQ[0]*=QIDFDECAY   # habituation: recency-weighted question statistics
                    for st in range(SREAD+1):   # count the step-tagged read units of this question too
                        qix,_=bag(hist,step=st); DFQ[qix]+=1
                    NQ[0]+=1
                if not FROZ[0]: VOCAB.add(a.encode()); FIRSTS.add(ans[0])
                if MODES & {'gated','gword'} and not FROZ[0]:
                    qset=set(bag(hist)[0].tolist())
                    for gix,pos,cs in TAGS:
                        nxt=bytes(STREAM[pos:pos+MATCHN])
                        ok = nxt==ans[:MATCHN] and (cs is None or len(cs & qset)>0)
                        tgt=1.0 if ok else 0.0; G[gix]+=0.2*(tgt-G[gix].mean())
                    TAGS.clear()
                for t,y in enumerate(ans if not FROZ[0] else b''):
                    key=(t,bytes(ans[:t][-3:])); A.setdefault(key,np.zeros(256)); A[key]*=0.9; A[key][y]+=1
                if test and NOGOLD: feed(b'\n',store=False)   # no gold answer bytes enter the history at evaluation
                else: feed(ans,store=not QNOSTORE)
    if DUMP: json.dump(dump,open(DUMP,'w'))
    acc=np.mean([r[0] for r in res]); fb=np.mean([r[2] for r in res]); heur=np.mean([r[1] for r in res])
    return {'exact_answer_with_own_newline':round(float(acc),3),'oracle_parse_last_location_heuristic':round(float(heur),3),'first_byte_acc':round(float(fb),3),'n':len(res),'writes':NWR[0]}
cfg={k:E(k) for k in ('STORE','REL','RELP0','RELETA','RELC','QIDF','QIDFADAPT','QIDFDECAY','ENDW','NOGOLD','FREEZE','VALID','PYTHONHASHSEED') if E(k) is not None}
cfg['train']=os.path.basename(TRAIN); cfg['eval']=os.path.basename(VALFN if VALID else TEST)
print(json.dumps({**cfg,**run()}))
