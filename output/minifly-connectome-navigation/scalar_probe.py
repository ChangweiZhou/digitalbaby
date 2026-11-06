"""Small independent scalar reward instrumentation check on archived mod97."""
from hashlib import sha256
import json
from pathlib import Path
from random import Random
import sys
import torch as T
from torch import nn
from torch.nn.functional import pad

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/"minifly-remaining-experiments"
sys.path.insert(0,str(SOURCE))
from fixtures import arithmetic,expand,mod_split

T.set_num_threads(1)
SEED=73129
CK=SOURCE/"official/mod97/73129/early-branches/none-final.pt"


class Model(nn.Module):
    def __init__(self,warm):
        super().__init__()
        T.manual_seed(SEED)
        self.book=nn.Embedding(258,8,padding_idx=257)
        self.shared=nn.Sequential(nn.Linear(64,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,257))
        if warm:
            state=T.load(CK,weights_only=True)["model"]
            self.load_state_dict({k:v for k,v in state.items() if k in self.state_dict()})
        self.requires_grad_(False)

    def hidden(self,p):
        u=self.book(pad(T.tensor(p[-8:]),(max(0,8-len(p)),0),value=257)).flatten()
        return self.shared[:-1](u)


def digest(params):
    return sha256(b"".join(p.numpy().tobytes() for p in params)).hexdigest()


@T.no_grad()
def trial(m,p,rng=None):
    context=[257,*p]
    head=m.shared[-1]
    out=[]
    e=T.zeros_like(head.weight)
    for _ in range(5):
        h=m.hidden(context)
        logits=head(h)
        if rng is not None:
            noise=T.randn(257,generator=rng)
            logits=logits+.50*noise
            e+=noise[:,None]*h
        y=int(logits.argmax())
        out.append(y)
        if y==256:break
        context.append(y)
    return out,e


def evaluate(m,panel):
    return sum(trial(m,r[:6])[0]==r[6:] for r in panel)


@T.no_grad()
def simulate(warm,mode,rows,probe,reference=None):
    m=Model(warm)
    head=m.shared[-1]
    fixed=[p for k,p in m.named_parameters() if k!="shared.4.weight"]
    before=digest(fixed);w0=digest([head.weight])
    rng=T.Generator().manual_seed(SEED+1)
    start=evaluate(m,probe)
    rewards=[];baseline=0.;writes=0
    for i,row in enumerate(rows):
        out,elig=trial(m,row[:6],rng)
        actual=int(out==row[6:])
        rewards.append(actual)
        r=reference[i] if reference is not None else actual
        if mode!="no_write":
            change=.0004*(r-baseline)*elig
            writes+=int(T.any(change!=0))
            head.weight.add_(change)
        baseline=.95*baseline+.05*r
    assert digest(fixed)==before
    assert (digest([head.weight])!=w0)==bool(writes)
    return {"pre_exact":start,"post_exact":evaluate(m,probe),"probe_n":len(probe),
            "earned_rewards":sum(rewards),"delivered_rewards":sum(reference if reference is not None else rewards),
            "write_events":writes},rewards


def main():
    train,monitor,_=map(expand,mod_split(SEED))
    rng=Random(SEED)
    rows=[arithmetic(p,97) for p in rng.sample(train,128)]
    probe=[arithmetic(p,97) for p in rng.sample(monitor,32)]
    out={}
    for warm in (False,True):
        data,rewards=simulate(warm,"aligned",rows,probe)
        shuffled=rewards.copy()
        rng.shuffle(shuffled)
        out["warm" if warm else "cold"]={"aligned":data,
             "no_write":simulate(warm,"no_write",rows,probe)[0],
             "yoked":simulate(warm,"yoked",rows,probe,shuffled)[0]}
    out.update(scope="engineering; monitor probe only",episodes_per_arm=128,
               warm_checkpoint_sha256=sha256(CK.read_bytes()).hexdigest())
    (ROOT/"SCALAR_PROBE.json").write_text(json.dumps(out,indent=2)+"\n")
    print(out)


if __name__=="__main__":main()
