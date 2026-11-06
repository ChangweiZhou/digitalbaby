"""Run the three bounded engineering cycles and verify fresh-process restart."""
import copy
from hashlib import sha256
import json
from pathlib import Path
import sys
import time

from fixtures import (EOS,SEED,arithmetic,counting,cycle1,development_windows,expand,
                      graph_bank,graph_stream,mod_split)
from models import (Attractor,Encoder,Local,Modular,Sensory,T,digest,generate,
                    loss,pretrain,step,topology)

ROOT=Path(__file__).resolve().parent
PAIR=[(a,b) for a in range(10) for b in range(10) if min(a,b)<=4]
HELD=[(a,b) for a in range(5,10) for b in range(5,10)]


def build(kind,sensor,rewired=False):
    T.manual_seed(SEED)
    return Modular(sensor) if kind=="mod97" else Attractor(sensor,rewired)


def clock_update(model,opt,row,arm):
    if arm in ("none","periodic","smooth"):
        return step(opt,loss(model,row))
    return step(opt,sum(loss(model,[r],6 if i==2 and arm=="answer" else None)
                        for i,r in enumerate(row)))


def fingerprint(x):
    return sha256(x.detach().numpy().tobytes()).hexdigest()


def cycle2(sensor):
    x=[*b"07+11=18",EOS]
    m=build("mod97",sensor)
    changed=build("mod97",sensor)
    changed.load_state_dict(m.state_dict())
    before={k:v.clone() for k,v in changed.state_dict().items()}
    changed.shrink(.98)
    for k,v in changed.state_dict().items():
        applies=k=="book.weight" or k.startswith("shared.") and k.endswith(".weight")
        assert T.equal(v,before[k]*(.98 if applies else 1))
    smooth=build("mod97",sensor)
    for _ in range(128): smooth.shrink(.98**(1/128))
    assert all(T.allclose(a,b,atol=3e-6,rtol=5e-6) for a,b in zip(changed.parameters(),smooth.parameters()))
    ring=build("attractor",sensor)
    rewired=build("attractor",sensor,True)
    assert all(T.equal(a,b) for a,b in zip(ring.parameters(),rewired.parameters()))
    assert topology(False)[1]!=topology(True)[1]
    local=ring.persists()
    assert local["distinct_supports"]>=4
    aux=next(counting(PAIR,"worked",1))[2]
    other=aux[:6]+[46]*(len(aux)-7)+[EOS]
    assert T.equal(loss(ring,[aux],6),loss(ring,[other],6))
    assert not T.equal(loss(ring,[aux]),loss(ring,[other]))
    for model in (m,ring):
        inp=T.tensor([[257,*x[:-1]]])
        short=model(inp[:,:7])
        assert T.allclose(short,model(inp)[:,:7],atol=1e-6)
    enc=Encoder()
    l=Local(enc,sensor)
    row=next(graph_stream(graph_bank("A")[0],"A",0,True,1))
    assert len(l.features(row[:-1]))==55
    assert len(Local(None,sensor).features(row[:-1]))==55
    h=digest(enc)
    l.train(row)
    assert digest(enc)==h and T.any(l.w!=0)
    return {"frozen_local_encoder":True,"pulse_scope_and_nominal_factor":True,"aux_mask_verified":True,
            "causal_prefix_and_parameter_parity":True,"ring":local,"rewired":rewired.persists(),
            "triangles":{"ring":topology(False)[1],"rewired":topology(True)[1]}}


def graph_score(model,probes,law):
    return sum(model.generate(ps[2])==[ys[law],EOS] for _,ps,ys in probes[:8])


def arith_score(model,pairs,mod=0):
    before=digest(model)
    rows=[arithmetic(p,mod) for p in pairs]
    got=[generate(model,r[:6]) for r in rows]
    assert digest(model)==before
    return {"exact":sum(y==r[6:] for r,y in zip(rows,got)),
            "first_byte":sum(y[0]==r[6] for r,y in zip(rows,got)),
            "eos":sum(y[-1]==EOS for y in got),"n":len(rows)}


def save_model(path,model,opt,row,arm):
    T.save({"model":model.state_dict(),"opt":opt.state_dict(),"row":row,"arm":arm},path)
    clock_update(model,opt,row,arm)
    return {"file":path.name,"sha256":sha256(path.read_bytes()).hexdigest(),"expected":digest(model)}


def local_experiment(sensor,scratch):
    windows=development_windows()
    encs=pretrain(windows)
    assert len({digest(x) for x in encs.values()})==3
    encs["stock"]=None
    bank,pa=graph_bank("A")
    b,pb=graph_bank("B")
    start=time.process_time()
    results={}
    restart=None
    for name,enc in encs.items():
        results[name]=[]
        frozen=digest(enc) if enc else None
        for law in range(2):
            for consistent in (True,False):
                m=Local(enc,sensor)
                for row in graph_stream(bank,"A",law,consistent,64): m.train(row)
                prior=fingerprint(m.w)
                b0=graph_score(m,pb,law)
                assert fingerprint(m.w)==prior
                for row in graph_stream(b,"B",law,True,32): m.train(row)
                results[name].append(dict(law=law,consistent=consistent,B0=b0,
                                          B32=graph_score(m,pb,law),A_post=graph_score(m,pa,law)))
                if name=="aligned" and law==0 and consistent:
                    path=scratch/"encoder.pt"
                    T.save(dict(encoder=enc.state_dict(),readout=m.w,row=row),path)
                    m.train(row)
                    restart=dict(file=path.name,sha256=sha256(path.read_bytes()).hexdigest(),expected=fingerprint(m.w))
        assert not enc or frozen==digest(enc)
    return {"target_sites_per_trained_birth":4096,"histories":16,"records":1536,
            "development_window_sha256":[sha256(bytes(y for x in group for y in x)).hexdigest() for group in windows],
            "B_probe_n":8,"metrics":results,"local_cpu_seconds":time.process_time()-start,"restart":restart}


def modular_experiment(sensor,scratch):
    train,_,sealed=map(expand,mod_split())
    import random
    rng=random.Random(SEED)
    sample=rng.sample(train,128)
    batches=[[arithmetic(p,97) for p in rng.choices(sample,k=64)] for _ in range(144)]
    m=build("mod97",sensor)
    opt=T.optim.Adam(m.parameters(),lr=.003)
    start=time.process_time()
    for rows in batches[:16]: clock_update(m,opt,rows,"none")
    checkpoint=copy.deepcopy(m.state_dict()),copy.deepcopy(opt.state_dict())
    frozen=digest(m)
    results={}
    for arm in ("none","periodic","smooth"):
        m=build("mod97",sensor)
        opt=T.optim.Adam(m.parameters(),lr=.003)
        m.load_state_dict(checkpoint[0])
        opt.load_state_dict(copy.deepcopy(checkpoint[1]))
        assert digest(m)==frozen
        for i,rows in enumerate(batches[16:],1):
            clock_update(m,opt,rows,arm)
            if arm=="smooth": m.shrink(.98**(1/128))
            elif arm=="periodic" and i%128==0: m.shrink(.98)
        results[arm]={"taught":arith_score(m,sample[:12],97),"held":arith_score(m,sealed[:12],97),
                      "shared_norm":float(m.shared[0].weight.detach().norm())}
        if arm=="periodic": restart=save_model(scratch/"mod97.pt",m,opt,batches[-1],"none")
    return {"common_updates":16,"branch_updates_each":128,"batch":64,"training_records":25600,
            "forced_engineering_fork":True,"scores":results,"cpu_seconds":time.process_time()-start,"restart":restart}


def attractor_experiment(sensor,scratch):
    results={}
    start=time.process_time()
    for rewire in (False,True):
        name="rewired" if rewire else "ring"
        results[name]={}
        for arm in ("answer","shuffled","worked"):
            m=build("attractor",sensor,rewire)
            opt=T.optim.Adam(m.parameters(),lr=.003)
            for row in counting(PAIR,arm,75): clock_update(m,opt,row,arm)
            results[name][arm]={"taught":arith_score(m,PAIR[::6]),"held":arith_score(m,HELD),
                                "post_update_state":m.persists()}
            if not rewire and arm=="worked": restart=save_model(scratch/"attractor.pt",m,opt,row,arm)
    return {"histories":6,"training_groups":450,"scores":results,
            "cpu_seconds":time.process_time()-start,"restart":restart}


def resume(sensor,scratch,data):
    for name in ("encoder","mod97","attractor"):
        check=data[name]["restart"]
        path=scratch/check["file"]
        assert sha256(path.read_bytes()).hexdigest()==check["sha256"]
        state=T.load(path,weights_only=True)
        if name=="encoder":
            enc=Encoder()
            enc.head=T.nn.Identity()
            enc.load_state_dict(state["encoder"])
            m=Local(enc,sensor)
            m.w=state["readout"]
            m.train(state["row"])
            actual=fingerprint(m.w)
        else:
            m=build(name,sensor)
            opt=T.optim.Adam(m.parameters(),lr=.003)
            m.load_state_dict(state["model"])
            opt.load_state_dict(state["opt"])
            clock_update(m,opt,state["row"],state["arm"])
            actual=digest(m)
        assert check["expected"]==actual
        check["actual"]=actual


def main():
    mode=sys.argv[1]
    path=ROOT/f"CYCLE{3 if mode=='resume' else mode}.json"
    start=time.process_time()
    if mode=="1":
        result=cycle1()
    else:
        sensor=Sensory(Path(sys.argv[2]))
        if mode=="2": result=cycle2(sensor)
        else:
            scratch=Path(sys.argv[3])
            if mode=="3":
                result={}
                for name,fn in (("encoder",local_experiment),("mod97",modular_experiment),
                                ("attractor",attractor_experiment)):
                    before=(sensor.new,sensor.seconds)
                    result[name]=fn(sensor,scratch)
                    result[name]["uncached_feature_batches"]=sensor.new-before[0]
                    result[name]["uncached_feature_cpu_seconds"]=sensor.seconds-before[1]
                    print(name,"engineering complete",flush=True)
            else:
                result=json.loads(path.read_text())
                resume(sensor,scratch,result)
    result["status"]="RESTART_PENDING" if mode=="3" else "ENGINEERING_PASS"
    result["receipt_cpu_seconds"]=time.process_time()-start
    result["source_sha256"]={p.name:sha256(p.read_bytes()).hexdigest() for p in ROOT.glob("*.py")}
    path.write_text(json.dumps(result,indent=2)+"\n")
    print(mode,result["status"],flush=True)


if __name__=="__main__":
    main()
