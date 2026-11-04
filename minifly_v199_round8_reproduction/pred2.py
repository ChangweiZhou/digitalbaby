import numpy as np, sys, json, time
MODE=sys.argv[1]; TEMP=float(sys.argv[2]); RATE=float(sys.argv[3])
TXT=open('/tmp/alice.txt','rb').read()[2000:160000].replace(b'\r',b'')
tr,te=TXT[:130000],TXT[130000:150000]
def bag(h,N,orders=(1,2,3,4,5),per=2):
    ix=[]
    for k in orders:
        if len(h)>=k:
            g=hash(bytes(h[-k:]))  # suffix k-gram (position-bound to the end: a context feature)
            ix+= [ (g*2654435761+j*40503+k*97)%N for j in range(per)]
    ix=np.unique(np.array(ix,dtype=np.int64)); return ix
def run(kind,N=65536,rate=None):
    rate=rate or RATE
    W=np.zeros((N,256),np.float32); L=0.; n=0
    for split,data in (('train',tr),('test',te)):
        h=bytearray(b' '*8)
        for y in data:
            ix=bag(h,N) if kind=='bag' else bag(h,N,orders=(1,))
            s=W[ix].sum(0)
            if MODE=='prob': p=np.maximum(s,1e-3); p=p/p.sum()
            else: z=np.exp((s-s.max())/TEMP); p=z/z.sum()
            if split=='test': L+=-np.log2(max(p[y],1e-12)); n+=1
            e=-s; e[y]+=1; W[ix]+=(rate*e/len(ix)).astype(np.float32)
            h.append(y); del h[:-8]
    return L/n
def markov(k):
    from collections import defaultdict
    C=defaultdict(lambda: np.ones(256)*0.05); 
    for i in range(k,len(tr)): C[tr[i-k:i]][tr[i]]+=1
    L=0.
    for i in range(k,len(te)):
        c=C[te[i-k:i]]; L+=-np.log2(c[te[i]]/c.sum())
    return L/(len(te)-k)
print(json.dumps({'markov1':round(markov(1),3),'markov2':round(markov(2),3),'markov3':round(markov(3),3),'bag_orders1-5':round(float(run('bag')),3),'mode':MODE,'temp':TEMP,'rate':RATE}))