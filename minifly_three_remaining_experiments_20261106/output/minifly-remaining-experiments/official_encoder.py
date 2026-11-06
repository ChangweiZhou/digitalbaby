"""Development-before-birth and frozen-encoder local life campaign."""
from hashlib import sha256
from pathlib import Path
from random import Random
import sys
import time

from campaign_common import RUN,SEEDS,lock,measures,read,sha,usage,verify,write
from fixtures import ACTORS,EOS,fixture
from models import Encoder,Local,Sensory,T,digest,step
from torch.nn.functional import cross_entropy

ROOT=RUN/"encoder"
BOOK=Path(__file__).resolve().parents[1]/"minifly-pretraining"/"books"/"2600.txt"
BIRTHS=("aligned","shuffled","random","stock")
DOSE=2048
B_TIMES=(0,128,512,2048)


def bank(seed,phase):
    a,p=fixture(seed+(phase=="B"),"A")
    if phase=="A":return a,p
    return ([(k,x.translate(ACTORS),y) for k,x,y in a],
            [(k,[x.translate(ACTORS) for x in xs],y) for k,xs,y in p])


def stream(records,seed,phase,law,consistent):
    rng=Random(f"{seed}:{phase}:official-life")
    for epoch in range(4):
        for i in rng.sample(range(len(records)),len(records)):
            k,p,ys=records[i]
            use=law if consistent else (law+(k%64>=32)+epoch)%2
            yield [*p,ys[use],EOS]


def windows(seed):
    text=BOOK.read_bytes()
    rng=Random(seed+473)
    start=rng.randrange(len(text)-4096*129)
    text=text[start:start+4096*129]
    prose=[list(text[i:i+129]) for i in range(0,len(text),129)]
    synth=[]
    for _ in range(4096):
        key=bytes(rng.sample(range(97,123),4))
        s=bytearray()
        while len(s)<129:
            s.extend(b"["+key+b"]"+bytes(rng.choices(range(65,91),k=rng.randint(2,7)))+key)
        synth.append(list(s[:129]))
    return prose,synth


def corpus_hash(groups):
    return [sha256(bytes(x for row in group for x in row)).hexdigest() for group in groups]


def prepare(archive):
    hashes={str(seed):corpus_hash(windows(seed)) for seed in SEEDS}
    banks={}
    for seed in SEEDS:
        banks[str(seed)]={}
        for phase in ("A","B"):
            records,probes=bank(seed,phase)
            exposed={p for _,p,_ in records}
            assert len(exposed)==512 and len(probes)==64
            assert all(x not in exposed for _,xs,_ in probes for x in xs[2:])
            banks[str(seed)][phase]=sha256(b"".join(p+bytes(ys) for _,p,ys in records)).hexdigest()
    lock(ROOT,Path(__file__).name,archive,dict(births=BIRTHS,book_sha256=sha(BOOK),corpus=hashes,
         feedback=banks,pretrain_targets=1048576,local_episodes=4096,checkpoints=B_TIMES,
         local_histories=48,planned_worker_rss_mib=1800,planned_worker_cpu_seconds=500,
         gates=dict(taught=52,birth_margin=.05,history_margin=.05)))


def develop(seed):
    data=verify(ROOT,Path(__file__).name)
    assert sha(BOOK)==data["book_sha256"]
    path=ROOT/f"birth-{seed}"
    path.mkdir(exist_ok=True)
    if (path/"COMPLETE.json").exists():return
    start=time.process_time()
    groups=windows(seed)
    assert corpus_hash(groups)==data["corpus"][str(seed)]
    groups=[T.tensor(g) for g in groups]
    for arm in ("aligned","shuffled"):
        T.manual_seed(seed)
        m=Encoder()
        opt=T.optim.Adam(m.parameters(),lr=.003)
        observed=[]
        for i in range(0,4096,16):
            for x in groups:
                inputs=x[i:i+16,:-1]
                target=x[:,1:]
                target=(target if arm=="aligned" else target.roll(1,0))[i:i+16]
                p=m.head(m.features(inputs))
                observed.append(step(opt,cross_entropy(p.flatten(0,1),target.flatten())))
            if (i+16)%512==0:print(seed,arm,(i+16)*256,flush=True)
        m.head=T.nn.Identity()
        T.save(m.state_dict(),path/f"{arm}.pt")
        write(path/f"{arm}.json",dict(targets=1048576,hash=digest(m),
              early_loss=sum(observed[:32])/32,late_loss=sum(observed[-32:])/32))
    write(path/"COMPLETE.json",dict(targets=2097152,**usage(start)))


def eval_keys(model,probes,law,which,seen=None):
    prompts=[ps[which] for _,ps,_ in probes]
    targets=[[ys[law],EOS] for _,_,ys in probes]
    outputs=[model.generate(p) for p in prompts]
    r=measures(prompts,targets,outputs)
    if seen is not None:
        r["keys_seen"]=sum(k in seen for k,_,_ in probes)
        r["exact_on_seen"]=sum(k in seen and o==y for (k,_,_),o,y in zip(probes,outputs,targets))
    return r


def run(archive,seed,arm):
    verify(ROOT,Path(__file__).name)
    assert seed in SEEDS and arm in BIRTHS
    path=ROOT/f"{seed}-{arm}"
    path.mkdir(exist_ok=True)
    if (path/"COMPLETE.json").exists():return
    start=time.process_time()
    sensor=Sensory(archive) if arm=="stock" else None
    T.manual_seed(seed)
    enc=None if arm=="stock" else Encoder()
    if enc is not None:
        enc.head=T.nn.Identity()
        if arm!="random":
            enc.load_state_dict(T.load(ROOT/f"birth-{seed}"/f"{arm}.pt",weights_only=True))
        enc.requires_grad_(False)
    before=digest(enc) if enc is not None else None
    a,pa=bank(seed,"A")
    b,pb=bank(seed,"B")
    keys={p:k for k,p,_ in b}
    for law in range(2):
        for consistent in (True,False):
            label=f"{law}-{'consistent' if consistent else 'mixed'}"
            if (path/f"{label}.json").exists():continue
            m=Local(enc,sensor)
            for row in stream(a,seed,"A",law,consistent):m.train(row)
            taught=eval_keys(m,pa,law,0)
            seen=set()
            checks=[eval_keys(m,pb,law,2,seen)]
            for i,row in enumerate(stream(b,seed,"B",law,True),1):
                m.train(row)
                seen.add(keys[bytes(row[:-2])])
                if i in B_TIMES:checks.append(eval_keys(m,pb,law,2,seen))
            scores=[r["exact"]/r["n"] for r in checks]
            auc=sum((hi-lo)*(scores[i]+scores[i+1])/2 for i,(lo,hi) in enumerate(zip(B_TIMES,B_TIMES[1:])))/DOSE
            assert enc is None or digest(enc)==before
            write(path/f"{label}.json",dict(law=law,consistent=consistent,encoder_hash=before,
                  taught_A=taught,B=checks,B_AUC=auc,after_B_A=eval_keys(m,pa,law,2)))
            print(seed,arm,label,taught["exact"],[r["exact"] for r in checks],flush=True)
    write(path/"COMPLETE.json",dict(episodes=4*4096,**usage(start)))


if __name__=="__main__":
    action=sys.argv[1]
    if action=="prepare":prepare(Path(sys.argv[2]))
    elif action=="develop":develop(int(sys.argv[2]))
    else:run(Path(sys.argv[2]),int(sys.argv[3]),sys.argv[4])
