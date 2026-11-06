"""Bounded engineering qualification for recurrent worked examples."""
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
from random import Random
import sys
import time

import torch as T
from torch import nn
from torch.nn.functional import cross_entropy

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / "minifly-ordered-front-end"))
from revised_panel import material, parse, held, truth, GRAPHS, alphabet_for, render
from benchmark import expected, structure_key

T.set_num_threads(1)
EOS = 256
ARMS = ("answer", "shuffled", "worked")
BLOCKS = (71021, 81043)
MODES = ("window", "recurrent")


def frame(prefix, order):
    links = [prefix[i:i+3] for i in range(24,48,3)]
    return prefix[:24] + b"".join(links[i] for i in order) + prefix[48:]


def work(prefix, active):
    q = parse(prefix)
    actors = [a for a in q.locations if a not in (q.mover,q.target)]
    reached = {q.mover}
    result = []
    for _ in range(2):
        frontier = {b for op,a,b in q.edges if a in reached and op == active} - reached
        result.extend(48 + (a in frontier) for a in actors)
        reached |= frontier
    return bytes(result)


def fixture(block, phase):
    rng = Random(f"{block}:{phase}")
    source = list(material(block, phase))
    rng.shuffle(source)
    groups = {}
    seen = set()
    for prefix, ys, stay, move in source:
        canonical = frame(prefix, sorted(range(8),key=lambda i:prefix[24+3*i:27+3*i]))
        if canonical in seen:
            continue
        seen.add(canonical)
        group = tuple(int(y == move) for y in ys)
        pool = groups.setdefault(group, [])
        if len(pool) < 64:
            orders = set()
            while len(orders) < 4:
                orders.add(tuple(rng.sample(range(8),8)))
            rendered = [frame(prefix, p) for p in sorted(orders)]
            pool.append((rendered,ys))
    assert sorted(map(len,groups.values())) == [64]*4
    keys = [(k, *row) for outcome in sorted(groups) for k,row in enumerate(groups[outcome],64*int(''.join(map(str,outcome)),2))]
    varied = []
    for k, ps, ys in keys:
        # All four renderings of this key retain the same actor-row order.
        rng = Random(f"actor-introduction-order:{block}:{phase}:{k}")
        rows = [ps[0][i:i+3] for i in range(0,24,3)]
        rng.shuffle(rows)
        intro = b"".join(rows)
        varied.append((k,[intro+p[24:] for p in ps],ys))
    keys = varied
    bank = [(k,p,y) for k,ps,y in keys for p in ps[:2]]
    probes = [row for row in keys if row[0]%4 == 0]
    return bank, probes


def sequence(bank, block, law, condition, phase, count):
    rng = Random(f"{block}:{phase}:stream")
    active = alphabet_for(phase)[law]
    traces = [work(p,active) for _,p,_ in bank]
    for offset in range(0,count,len(bank)):
        order = rng.sample(range(len(bank)),len(bank))
        perm = order[len(bank)//2:] + order[:len(bank)//2]
        # The alternate presentation is adjacent in bank, and a shuffle can
        # accidentally pair the two. Resample until all semantic keys differ.
        while any(bank[i][0] == bank[j][0] for i,j in zip(order,perm)):
            rng.shuffle(perm)
        for i,j in zip(order[:count-offset],perm):
            _,p,ys = bank[i]
            yield (list(p+b"?")+[ys[law],EOS], list(p+b"W"+traces[i if condition=="worked" else j])+[EOS])


class Model(nn.Module):
    def __init__(self, mode, seed):
        super().__init__()
        T.manual_seed(seed)
        self.mode = mode
        self.register_buffer("book",T.randn(258,8))
        self.book[257].zero_()
        self.proj = nn.Linear(576,32)
        self.cell = nn.GRU(32,32,batch_first=True)
        self.head = nn.Linear(32,257)

    def forward(self, xs):
        windows = self.book[nn.functional.pad(xs,(71,0),value=257).unfold(1,72,1)]
        u = self.proj(windows.flatten(2)).tanh()
        if self.mode=="recurrent":
            h,_ = self.cell(u)
        else:
            prev = nn.functional.pad(u[:,:-1],(0,0,1,0))
            h,_ = self.cell(u.flatten(0,1)[:,None],prev.flatten(0,1)[None])
            h = h.reshape_as(u)
        return self.head(h)

    def generate(self,prefix):
        data = list(prefix+b"?")
        out = []
        with T.no_grad():
            for _ in range(16):
                symbol = int(self(T.tensor([data]))[0,-1].argmax())
                out.append(symbol)
                if symbol==EOS:
                    break
                data.append(symbol)
        return out


def loss(model, row, condition):
    losses = []
    for i, symbols in enumerate(row):
        seq = T.tensor([[257,*symbols]])
        preds = model(seq[:,:-1])[0]
        err = cross_entropy(preds,seq[0,1:],reduction="none")
        weight = T.ones_like(err)
        weight[54:] = (4 if i==0 else (condition!="answer"))
        losses.append((err*weight).mean())
    return sum(losses)


def update(model,opt,row,arm):
    opt.zero_grad()
    value = loss(model,row,arm)
    assert T.isfinite(value)
    value.backward()
    assert all(T.isfinite(p.grad).all() for p in model.parameters())
    opt.step()
    return float(value)


def digest(model):
    return sha256(b"".join(x.detach().numpy().tobytes() for x in model.state_dict().values())).hexdigest()


def cycle1():
    total = 0
    unchanged = 0
    for block in BLOCKS:
        for phase in ("A","B"):
            bank, probes = fixture(block,phase)
            exposed = {p for _,p,_ in bank}
            assert len(exposed)==512 and len(probes)==64
            assert all(p not in exposed for _,ps,_ in probes for p in ps[2:])
            assert all(structure_key(parse(p)) in ((2,), (3,), (1,2)) for p in exposed)
            for law,active in enumerate(alphabet_for(phase)):
                assert all(ys[law]==expected(parse(p),active) for _,p,ys in bank)
                assert all(ys[law]==expected(parse(p),active) for _,ps,ys in probes for p in ps)
                assert all(p not in exposed and structure_key(parse(p)) not in ((2,),(3,),(1,2))
                           for pair in held(block,phase,law) for row in pair for p in [row.prefix])
                paired = list(sequence(bank,block,law,"worked",phase,512))
                shuffled = list(sequence(bank,block,law,"shuffled",phase,512))
                assert [p for p,_ in paired] == [p for p,_ in shuffled]
                assert Counter(tuple(w[54:]) for _,w in paired)==Counter(tuple(w[54:]) for _,w in shuffled)
                for (answer,a),(_,b) in zip(paired,shuffled):
                    assert len(answer)+len(a)==123 and len(a)==len(b)==67
                    assert set(a[54:-1])<= {48,49} and a[-1]==EOS
                    unchanged += a[54:]==b[54:]
                    total += 1
    oracle_checks = 0
    for graph in GRAPHS:
        moves=truth(graph)
        assert 0 < sum(moves) < 256
        for mask in range(256):
            p=render(graph,mask,b"XY",range(128,136),range(48,58),range(8))
            q=parse(p)
            for law,op in enumerate(b"XY"):
                assert expected(q,op)==(q.destination if moves[mask^(255*law)] else q.stay)
                oracle_checks += 1
    return {"trace_pairs_checked":total,"shuffled_identical_trace_count":unchanged,
            "held_presentation_pairs_per_fixture":64,"held_feedback_overlap":0,
            "full_mask_oracle_checks":oracle_checks}


def cycle2():
    bank,_ = fixture(BLOCKS[0],"A")
    row = next(sequence(bank,BLOCKS[0],0,"worked","A",1))
    models = [Model(m,42) for m in MODES]
    assert digest(models[0])==digest(models[1])
    changed = [row[0],row[1][:54]+[97]*12+[EOS]]
    counts = {}
    for model in models:
        count=sum(p.numel() for p in model.parameters())
        assert count==33281
        assert T.equal(loss(model,row,"answer"),loss(model,changed,"answer"))
        assert not T.equal(loss(model,row,"worked"),loss(model,changed,"worked"))
        inp=T.tensor([[257,*row[0]]])
        prefix=model(inp[:,:55])
        assert T.allclose(prefix,model(inp)[:,:55],atol=1e-6)
        before=digest(model)
        opt=T.optim.Adam(model.parameters(),lr=.003)
        update(model,opt,row,"worked")
        assert digest(model)!=before
        counts[model.mode]=count
    assert not T.equal(models[0](inp),models[1](inp))
    return {"scalars":counts,"identical_birth":True,"mask_and_causality_verified":True}


def assess(model,probes,law):
    hit = [0,0,0]
    for _,ps,ys in probes:
        target=[ys[law],EOS]
        outputs=[model.generate(p) for p in (ps[0],ps[2],ps[3])]
        hit[0]+=outputs[0]==target
        hit[1]+=outputs[1:]==[target,target]
        hit[2]+=sum(x[-1]==EOS for x in outputs)
    return dict(taught=hit[0],held_both=hit[1],eos=hit[2],keys=len(probes))


def cycle3(checkpoint):
    start=time.process_time()
    rows=[]
    for block in BLOCKS:
        bank,probes=fixture(block,"A")
        bank_b,_=fixture(block,"B")
        for mode in MODES:
            for law in range(2):
                for arm in ARMS:
                    m=Model(mode,block)
                    opt=T.optim.Adam(m.parameters(),lr=.003)
                    stream=list(sequence(bank,block,law,arm,"A",129))
                    for row in stream[:128]:
                        update(m,opt,row,arm)
                    seen={bytes(answer[:53]) for answer,_ in stream[:128]}
                    selected=[]
                    for k,ps,ys in probes:
                        exposed=next((p for p in ps[:2] if p in seen),None)
                        if exposed is not None:
                            selected.append((k,[exposed,ps[1],*ps[2:]],ys))
                    assert len(selected)>=8
                    selected=selected[:8]
                    before=digest(m)
                    metrics=assess(m,selected,law)
                    assert digest(m)==before
                    if (block,mode,law,arm)==(BLOCKS[0],"recurrent",0,"worked"):
                        T.save(dict(model=m.state_dict(),optimizer=opt.state_dict()),checkpoint)
                        clone=Model(mode,block)
                        clone.load_state_dict(m.state_dict())
                        clone_opt=T.optim.Adam(clone.parameters(),lr=.003)
                        from copy import deepcopy
                        clone_opt.load_state_dict(deepcopy(opt.state_dict()))
                        update(clone,clone_opt,stream[128],arm)
                        restart={"expected":digest(clone),"checkpoint_sha256":sha256(checkpoint.read_bytes()).hexdigest()}
                    for row in sequence(bank_b,block,law,arm,"B",16):
                        update(m,opt,row,arm)
                    assert digest(m)!=before
                    continued=digest(m)
                    after_b=assess(m,selected,law)
                    assert digest(m)==continued
                    rows.append(dict(block=block,mode=mode,law=law,arm=arm,after_A=metrics,after_B=after_b))
    return {"histories":rows,"events":len(rows)*144,"delivered_symbols":len(rows)*144*123,
            "restart":restart,"cpu_seconds":time.process_time()-start}


def main():
    action=sys.argv[1]
    path=ROOT/f"CYCLE{3 if action=='resume' else action}.json"
    if action=="1":
        r=cycle1()
    elif action=="2":
        r=cycle2()
    elif action=="3":
        r=cycle3(Path(sys.argv[2]))
    else:
        r=json.loads(path.read_text())
        checkpoint=Path(sys.argv[2])
        assert sha256(checkpoint.read_bytes()).hexdigest()==r["restart"]["checkpoint_sha256"]
        state=T.load(checkpoint,weights_only=True)
        m=Model("recurrent",BLOCKS[0])
        opt=T.optim.Adam(m.parameters(),lr=.003)
        m.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        bank,_=fixture(BLOCKS[0],"A")
        row=list(sequence(bank,BLOCKS[0],0,"worked","A",129))[-1]
        update(m,opt,row,"worked")
        r["restart"]["actual"]=digest(m)
        assert r["restart"]["actual"]==r["restart"]["expected"]
    r["status"]="RESTART_PENDING" if action=="3" else "ENGINEERING_PASS"
    r["source_sha256"]=sha256(Path(__file__).read_bytes()).hexdigest()
    path.write_text(json.dumps(r,indent=2)+"\n")
    print(action,r["status"],flush=True)


if __name__=="__main__":
    main()
