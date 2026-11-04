import numpy as np, sys, json
N=65536
def bag(h,per=4,W=16):
    h=bytes(h)[-W:]; gs=[]
    for k in (2,3,4):
        gs+=[hash(h[i:i+k])+k for i in range(len(h)-k+1)]
    gs+=[hash(b'SUF'+h[-k:])+100*k for k in (1,2) if len(h)>=k]
    ix=np.unique(np.array([(g*2654435761+j*40503)%N for g in gs for j in range(per)],dtype=np.int64)); return ix
def run(seed,n1=128,reps=12,META=5):
    r=np.random.default_rng(seed); A=list(b'mnopqrstuvwxyz')
    keys=[];
    while len(keys)<2*n1:
        k=bytes(r.choice(A,5).tolist())
        if k not in keys: keys.append(k)
    first,second=keys[:n1],keys[n1:]
    dig={k:bytes([int(r.choice(list(b'0123456789')))]) for k in second}
    W=np.zeros((N,256),np.float32); IMP=np.zeros(N)
    TR=lambda k: b'the code of '+k+b' is '
    recs=[(TR(f),s+b'.') for f,s in zip(first,second)]+[(TR(s),dig[s]+b'.') for s in second]
    filler=[b'we said so. ',b'it was on the way. ',b'and then they ran. ']
    for rep in range(reps):
        for i in r.permutation(len(recs)):
            pre,val=recs[i]; ctx=bytearray(filler[r.integers(3)]+pre)
            for y in val:           # teach cue on every value byte (multi-byte answers)
                ix=bag(ctx); s=W[ix].sum(0); e=-s; e[y]+=1; sc=1/(1+IMP[ix]/META); W[ix]+=(0.6*e[None,:]*sc[:,None]/len(ix)).astype(np.float32); IMP[ix]+=1/len(ix)
                ctx.append(y)
    def gen(prefix,maxlen=6):
        ctx=bytearray(prefix); out=bytearray()
        for _ in range(maxlen):
            ix=bag(ctx); y=int(np.argmax(W[ix].sum(0))); out.append(y); ctx.append(y)
            if y==ord('.'): break
        return bytes(out)
    Q=QQ    # never-seen query template
    one1=np.mean([gen(Q(f))==s+b'.' for f,s in zip(first,second)])
    one2=np.mean([gen(Q(s))==dig[s]+b'.' for s in second])
    two=[]
    strict=[]
    raw_key=[]
    eos=[]
    for f,s in zip(first,second):
        generated=gen(Q(f))
        mid=generated[:-1]                 # own multi-byte answer (incl. own '.' EOS)
        two.append(gen(Q(mid))==dig[s]+b'.')
        complete=generated==s+b'.'
        strict.append(complete and two[-1])
        raw_key.append(generated[:len(s)]==s)
        eos.append(generated.endswith(b'.'))   # re-issue own output as next query
    return {'strict_two_hop':sum(strict)/len(strict),'key_prefix':sum(raw_key)/len(raw_key),'hop1_eos_rate':sum(eos)/len(eos),'seed':seed,'hop1_key_to_key_with_EOS':round(float(one1),3),'hop1_key_to_digit':round(float(one2),3),'two_hop_never_taught':round(float(np.mean(two)),3)}

for name, QQ in (
    ('trained',lambda k:b'the code of '+k+b' is '),
    ('tell_me_what_is',lambda k:b'tell me, what is '+k+b'? '),
    ('remember',lambda k:b'remember '+k+b': ')):
    print(json.dumps(dict(template=name,**run(int(sys.argv[1])))))

