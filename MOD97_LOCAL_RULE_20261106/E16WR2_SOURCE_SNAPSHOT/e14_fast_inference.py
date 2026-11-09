"""Candidate inference-only optimization for the stateless E8/E7 window model."""
import torch as T
from torch.nn.functional import pad
import exp45 as X

@T.no_grad()
def last_logits(m,p):
    window=pad(p,(max(0,m.W-p.shape[1]),0),value=X.BOS)[:,-m.W:]
    u=m.book(window).flatten(1)
    return m.l3(T.tanh(m.l2(T.tanh(m.l1(u))))),window

@T.no_grad()
def respond(m,prompts,store=None,cap=7):
    out=[None]*len(prompts);groups={}
    for i,p in enumerate(prompts):groups.setdefault(len(p),[]).append(i)
    for _,idx in groups.items():
        for j in range(0,len(idx),256):
            ids=idx[j:j+256];p=T.tensor([[X.BOS,*prompts[i]] for i in ids]);gen=[[] for _ in ids];done=[False]*len(ids)
            for _ in range(cap):
                lg,window=last_logits(m,p)
                if store is not None:
                    mem=T.zeros(len(ids),257)
                    for b,key in enumerate(window.tolist()):
                        counts=store.mem.get(tuple(key))
                        if counts:mem[b,max(counts,key=counts.get)]=store.gain
                    lg=lg+mem
                yy=lg.argmax(1)
                for q in range(len(ids)):
                    if not done[q]:
                        gen[q].append(int(yy[q]))
                        if int(yy[q])==X.EOS:done[q]=True
                p=T.cat([p,yy[:,None]],1)
                if all(done):break
            for q,i in enumerate(ids):out[i]=gen[q]
    return out
