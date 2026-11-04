import sys, json, os, numpy as np
META=float(os.environ.get('META','0')); WAVES=int(os.environ.get('WAVES','2')); RANDIMP=os.environ.get('RANDIMP','0')=='1'; REOPEN=float(os.environ.get('REOPEN','0')); NUNITS=int(os.environ.get('NUNITS','16384'))
sys.path.insert(0,'/tmp/rep'); import importlib.util
spec=importlib.util.spec_from_file_location('rep','/tmp/rep/rep.py'); R=importlib.util.module_from_spec(spec); spec.loader.exec_module(R)
WORDS=b'the a of and to in it was he she they we on at by for with as is said then'.split()
CYC=b'abcdefghijkl'; n=12
TR=[lambda k:b'the code of '+k+b' is ', lambda k:k+b' -> ']
HO=[lambda k:b'remember '+k+b': ', lambda k:b'what is '+k+b'? ']
def bag(h,N,per=4,W=16):
    h=bytes(h)[-W:]
    gs=[(h[i]*257+h[i+1]) for i in range(len(h)-1)]+[(h[i]*65599+h[i+1]*257+h[i+2]*7919+1) for i in range(len(h)-2)]
    if not gs: return np.zeros(0,int),np.zeros(0)
    ix=np.unique(np.concatenate([np.array([g*2654435761+j*40503 for j in range(per)])%N for g in gs])).astype(int)
    return ix,np.ones(len(ix))/np.sqrt(len(ix))
def build(seed,nf=int(os.environ.get('NF','64')),reps=12):
    r=np.random.default_rng(seed)
    keys=[];
    while len(keys)<WAVES*nf:
        k=bytes(r.choice(list(b'mnopqrstuvwxyz'),5).tolist())   # keys avoid the cycle letters
        if k not in keys: keys.append(k)
    val={i:bytes([int(r.choice(list(b'0123456789')))]) for i in range(WAVES*nf)}
    pairs=[(x,y) for x in range(n) for y in range(n) if x!=y and (y-x)%n!=6]
    test=[p for p in pairs if (p[1]-p[0])%n in (2,10)]; train=[p for p in pairs if p not in test]
    CORR=[i for w in range(WAVES-1) for i in range(w*nf,w*nf+nf//4)]; corr=CORR; newval={i:bytes([ord('0')+(int(val[i])+1+int(r.integers(9)))%10]) for i in CORR}
    S=bytearray(); T=[]
    def put(b,teachlast=False):
        S.extend(b); T.extend([0]*(len(b)-1)+[1 if teachlast else 0])
    def filler():
        put(b' '.join(r.choice(WORDS,r.integers(2,5)))+b'. ')
        s=r.integers(n); put(bytes(CYC[(s+i)%n] for i in range(r.integers(4,9)))+b' ')
    def fact(i,v,rep): put(TR[rep%2](keys[i])); put(v,True); put(b'. ')
    def rel():
        x,y=train[r.integers(len(train))]; put(b'cmp '+bytes([CYC[x],CYC[y]])+b'='); put(b'>' if (y-x)%n<6 else b'<',True); put(b'\n')
    phases={}
    for w in range(WAVES):
        ids=list(range(w*nf,(w+1)*nf))
        cids=[i for i in CORR if (w-1)*nf<=i<w*nf] if w>0 else []
        for rep in range(reps):
            for i in r.permutation(ids+cids):
                filler(); fact(i,newval[i] if i in cids else val[i],rep)
                if r.random()<0.3: rel()
        phases[w]=len(S)
    return bytes(S),T,keys,val,newval,corr,train,test,phases,nf
def run(seed,N=None):
    N=N or NUNITS
    S,T,keys,val,newval,corr,train,test,phases,nf=build(seed)
    Cb=R.Code(seed,96,4096,205); Wb=np.zeros((4096,256),np.float32)   # all-byte predictor
    Wm=np.zeros((N,256),np.float32); IMP=np.zeros(N); PERM=np.random.default_rng(seed+77).permutation(N) if RANDIMP else np.arange(N)                                      # fact memory (bag code), teach-gated
    E=np.eye(256); M=np.zeros((256,256)); M2=np.zeros((256,256))
    rr=np.random.default_rng(seed+1); P1=rr.normal(size=(2000,512)); P2=rr.normal(size=(2000,512))
    Wr=[np.zeros((2000,256)),np.zeros((2000,256))]; REL=np.array([0.5,0.5]); SEEN=set()
    nzv=lambda v: v/(np.linalg.norm(v)+1e-9)
    def relcode(x,y):
        z1=np.concatenate([nzv(M[x]),nzv(M[y])]); z2=np.concatenate([nzv(M2[x]),nzv(M2[y])])
        return np.argpartition(-(P1@z1),100)[:100], np.argpartition(-(P2@z2),100)[:100]
    def relread(x,y):
        i1,i2=relcode(x,y); w=REL**4+1e-9; w/=w.sum(); return w[0]*Wr[0][i1].sum(0)+w[1]*Wr[1][i2].sum(0)
    def is_rel(h): return len(h)>=7 and bytes(h[-7:-3])==b'cmp '
    L=0.;cnt=0
    for t in range(1,len(S)):
        h=S[max(0,t-16):t]; y=S[t]
        z=R.front('lag',bytes(h)[-12:]); ib,vb=Cb.enc(z); sb=vb@Wb[ib]
        L+=R.m.loss(sb,y); cnt+=1
        if vb@vb>1e-15: e=-sb; e[y]+=1; Wb[ib]+=(0.6*np.outer(vb/(vb@vb),e)).astype(np.float32)
        if t>=2: q=S[t-2]; p=S[t-1]; M[q]+=0.05*(E[q]+0.8*M[p]-M[q]); M2[q]+=0.05*(E[q]-0.8*M2[p]-M2[q])
        if T[t]:                                   # environment teach cue
            if is_rel(h):
                x,yy=h[-3],h[-2]; i1,i2=relcode(x,yy); key=(x,yy)
                for k,ix in enumerate((i1,i2)):
                    s=Wr[k][ix].sum(0)
                    if key not in SEEN: REL[k]+=0.1*(float(np.argmax(s)==y)-REL[k])
                    e=-s; e[y]+=1; Wr[k][ix]+=0.6*e/100
                SEEN.add(key)
            else:
                im,vm=bag(h,N); s=vm@Wm[im]; contra=max(0.0,float(s.max())-float(s[y])); IMP[im]*=np.exp(-REOPEN*contra); e=-s; e[y]+=1; sc=1.0/(1.0+IMP[PERM[im]]/META) if META>0 else 1.0; Wm[im]+=(0.6*np.outer(sc*vm/(vm@vm),e)).astype(np.float32); IMP[im]+=np.abs(vm)
    def fact_ans(prefix): im,vm=bag(prefix,N); return int(np.argmax(vm@Wm[im]))   # query cue -> memory read
    res={'whole_bpb_predictor':round(L/cnt,3)}
    nfw=nf
    for w in range(WAVES):
        ids=range(w*nfw,(w+1)*nfw); kept=[i for i in ids if i not in corr]; cc=[i for i in ids if i in corr]
        res[f'w{w}_kept_new']=round(float(np.mean([fact_ans(HO[j%2](keys[i]))==val[i][0] for j,i in enumerate(kept)])),3)
        if cc: res[f'w{w}_corr_new']=round(float(np.mean([fact_ans(HO[j%2](keys[i]))==newval[i][0] for j,i in enumerate(cc)])),3); res[f'w{w}_obsolete']=round(float(np.mean([fact_ans(HO[j%2](keys[i]))==val[i][0] for j,i in enumerate(cc)])),3)
    return res
    keepA=[i for i in range(nf) if i not in corr]
    for nm,tmpls in (('trained',TR),('new_templates',HO)):
        res['A_kept_'+nm]=round(float(np.mean([fact_ans(tp(keys[i]))==val[i][0] for i in keepA for tp in tmpls])),3)
        res['A_corrected_'+nm]=round(float(np.mean([fact_ans(tp(keys[i]))==newval[i][0] for i in corr for tp in tmpls])),3)
        res['A_corrected_obsolete_'+nm]=round(float(np.mean([fact_ans(tp(keys[i]))==val[i][0] for i in corr for tp in tmpls])),3)
        res['B_'+nm]=round(float(np.mean([fact_ans(tp(keys[i]))==val[i][0] for i in range(nf,2*nf) for tp in tmpls])),3)
    ans=lambda x,y: ord('>') if (y-x)%n<6 else ord('<')
    res['rel_train']=round(float(np.mean([np.argmax(relread(CYC[x],CYC[y]))==ans(x,y) for x,y in train])),3)
    res['rel_heldout_distance']=round(float(np.mean([np.argmax(relread(CYC[x],CYC[y]))==ans(x,y) for x,y in test])),3)
    res['REL']=REL.round(2).tolist()
    return res
print(json.dumps({'seed':int(sys.argv[1]),**run(int(sys.argv[1]))}),flush=True)
