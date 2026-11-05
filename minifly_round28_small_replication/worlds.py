import random
SYL=['ka','lo','mi','ru','te','sa','no','vi','da','pe','zu','ho','bi','fe','go','ra','li','mu','to','ne']
TPL=['{N} went to the {L}.','{N} moved to the {L}.','{N} is now in the {L}.','The {L} is where {N} went.','{N} ran into the {L}.','{N} arrived at the {L}.']
def word(r): return ''.join(r.choice(SYL) for _ in range(r.choice([2,3])))
def world_text(seed,nq=30):
    r=random.Random(seed); names=[];places=[]
    while len(names)<3:
        w=word(r).capitalize()
        if w not in names: names.append(w)
    while len(places)<4:
        w=word(r)
        if w not in places and w.capitalize() not in names: places.append(w)
    tpl=r.sample(TPL,2); loc={}; out=[]; ans=[]; q=0; pos=0
    while q<nq:
        n=r.choice(names); l=r.choice([p for p in places if p!=loc.get(n)]); loc[n]=l
        s=r.choice(tpl).format(N=n,L=l)+'\n'; out.append(s)
        if r.random()<0.5:
            m=r.choice(sorted(loc)); s='Where is %s? '%m; out.append(s); a=loc[m]+'\n'
            cur=sum(map(len,out)); ans.append((cur,len(a))); out.append(a); q+=1
    return ''.join(out).encode(), ans
